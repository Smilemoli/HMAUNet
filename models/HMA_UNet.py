import torch
import torch.nn as nn
import torch.nn.functional as F
from .components.encoder import HybridEncoder, hybrid_encoder_tiny, hybrid_encoder_small, hybrid_encoder_base
from .components.hma_module import HMABottleneck, hma_bottleneck_tiny, hma_bottleneck_small, hma_bottleneck_base
from .components.decoder import ResNetDecoder, OutputHead, resnet_decoder_tiny, resnet_decoder_small, resnet_decoder_base


class HMAUNet(nn.Module):
    """
    HMA-UNet: Hierarchical Mamba Aggregator U-Net
    
    一个创新的医学图像分割网络，结合了以下核心设计：
    1. 混合编码器: 浅层ConvNeXt + 深层Mamba (重编码器设计)
    2. HMA瓶颈层: 层级式Mamba聚合器，实现深度多尺度上下文建模
    3. CSFG跳跃连接: 跨尺度融合门，智能选择"哪种信息重要"
    4. ResNet解码器: 轻量级但高效的特征重建 (轻解码器设计)
    
    Args:
        in_channels (int): 输入图像通道数，默认3 (RGB)
        num_classes (int): 分割类别数
        base_channels (int): 基础通道数 C
        encoder_depths (list): 编码器各阶段块数 [2,2,2,2]
        encoder_drop_path_rate (float): 编码器DropPath概率
        bottleneck_num_levels (int): HMA瓶颈层金字塔层数
        bottleneck_drop_path_rate (float): HMA瓶颈层DropPath概率
        csfg_reduction_ratio (int): CSFG模块通道压缩比例
        decoder_num_res_blocks (int): 解码器每阶段残差块数
        use_transpose_conv (bool): 解码器是否使用转置卷积
        d_state (int): Mamba状态空间维度
        dropout (float): 最终输出dropout概率
        use_checkpoint (bool): 是否使用梯度检查点节省内存
    """
    
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        base_channels=32,
        encoder_depths=[2, 2, 2, 2],
        encoder_drop_path_rate=0.1,
        bottleneck_num_levels=3,
        bottleneck_drop_path_rate=0.2,
        csfg_reduction_ratio=8,
        decoder_num_res_blocks=2,
        use_transpose_conv=True,
        d_state=16,
        dropout=0.1,
        use_checkpoint=False
    ):
        super().__init__()
        
        # 保存配置参数
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.use_checkpoint = use_checkpoint
        
        # 计算各阶段通道数
        # 编码器输出: [2C, 4C, 8C, 8C]
        self.encoder_channels = [
            2 * base_channels,  # Stage 1: 2C
            4 * base_channels,  # Stage 2: 4C  
            8 * base_channels,  # Stage 3: 8C
            8 * base_channels   # Stage 4: 8C
        ]
        
        # 1. 混合式编码器 (ConvNeXt + Mamba)
        self.encoder = HybridEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            depths=encoder_depths,
            drop_path_rate=encoder_drop_path_rate,
            d_state=d_state
        )
        
        # 2. HMA瓶颈层 (层级式Mamba聚合器)
        self.bottleneck = HMABottleneck(
            in_channels=self.encoder_channels[3],  # 8C
            out_channels=self.encoder_channels[3],  # 8C (保持不变)
            d_state=d_state,
            num_levels=bottleneck_num_levels,
            drop_path_rate=bottleneck_drop_path_rate,
            use_checkpoint=use_checkpoint
        )
        
        # 3. ResNet解码器 (带CSFG跳跃连接)
        self.decoder = ResNetDecoder(
            base_channels=base_channels,
            encoder_channels=self.encoder_channels,
            reduction_ratio=csfg_reduction_ratio,
            use_transpose_conv=use_transpose_conv,
            num_res_blocks=decoder_num_res_blocks
        )
        
        # 4. 输出头
        self.output_head = OutputHead(
            in_channels=base_channels,  # C
            num_classes=num_classes,
            dropout=dropout
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        HMA-UNet前向传播
        
        Args:
            x: 输入图像 (B, in_channels, H, W)
            
        Returns:
            分割预测 (B, num_classes, H, W)
        """
        # 保存输入尺寸信息
        input_size = x.shape[2:]
        
        # 1. 混合式编码器: 浅层ConvNeXt + 深层Mamba
        encoder_features = self.encoder(x)
        # encoder_features: [x_enc1, x_enc2, x_enc3, x_enc4]
        # x_enc1: (B, 2C, H/2, W/2)   - Stage 1 输出
        # x_enc2: (B, 4C, H/4, W/4)   - Stage 2 输出
        # x_enc3: (B, 8C, H/8, W/8)   - Stage 3 输出  
        # x_enc4: (B, 8C, H/16, W/16) - Stage 4 输出
        
        x_enc1, x_enc2, x_enc3, x_enc4 = encoder_features
        
        # 获取stem输出用于最后的跳跃连接
        # 注意：需要从编码器获取stem输出
        x_stem = self._get_stem_features(x)  # (B, C, H/2, W/2)
        
        # 2. HMA瓶颈层: 层级式Mamba聚合
        bottleneck_features = self.bottleneck(x_enc4)  # (B, 8C, H/16, W/16)
        
        # 3. ResNet解码器: 轻量级特征重建 + CSFG智能跳跃连接
        decoder_features = [x_enc1, x_enc2, x_enc3, x_stem]
        decoded_features = self.decoder(bottleneck_features, decoder_features)  # (B, C, H, W)
        
        # 4. 输出头: 生成最终分割结果
        output = self.output_head(decoded_features)  # (B, num_classes, H, W)
        
        # 确保输出尺寸与输入匹配
        if output.shape[2:] != input_size:
            output = F.interpolate(
                output, size=input_size, mode='bilinear', align_corners=True
            )
        
        return output
    
    def _get_stem_features(self, x):
        """获取编码器stem层的输出特征"""
        # 通过编码器的stem层获取初始特征
        return self.encoder.stem(x)  # (B, C, H/2, W/2)
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_name': 'HMA-UNet',
            'input_channels': self.in_channels,
            'num_classes': self.num_classes,
            'base_channels': self.base_channels,
            'encoder_channels': self.encoder_channels,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def get_feature_maps(self, x):
        """获取中间特征图，用于可视化和分析"""
        with torch.no_grad():
            # 编码器特征
            encoder_features = self.encoder(x)
            x_enc1, x_enc2, x_enc3, x_enc4 = encoder_features
            
            # Stem特征
            x_stem = self._get_stem_features(x)
            
            # 瓶颈层特征
            bottleneck_features = self.bottleneck(x_enc4)
            
            # 解码器特征
            decoder_features = [x_enc1, x_enc2, x_enc3, x_stem]
            decoded_features = self.decoder(bottleneck_features, decoder_features)
            
            return {
                'stem': x_stem,
                'encoder': {
                    'stage1': x_enc1,
                    'stage2': x_enc2, 
                    'stage3': x_enc3,
                    'stage4': x_enc4
                },
                'bottleneck': bottleneck_features,
                'decoder': decoded_features
            }


def create_hma_unet(
    config='tiny',
    in_channels=3,
    num_classes=1,
    **kwargs
):
    """
    创建HMA-UNet模型的工厂函数
    
    Args:
        config (str): 模型配置 ['tiny', 'small', 'base']
        in_channels (int): 输入通道数
        num_classes (int): 分割类别数
        **kwargs: 其他配置参数
    
    Returns:
        HMAUNet: 配置好的HMA-UNet模型
    """
    
    if config == 'tiny':
        return hma_unet_tiny(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif config == 'small':
        return hma_unet_small(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif config == 'base':
        return hma_unet_base(in_channels=in_channels, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown config: {config}. Choose from ['tiny', 'small', 'base']")


def hma_unet_tiny(in_channels=3, num_classes=1, **kwargs):
    """轻量级HMA-UNet配置 - 适合快速原型和资源受限环境"""
    return HMAUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=32,
        encoder_depths=[2, 2, 2, 2],
        encoder_drop_path_rate=0.1,
        bottleneck_num_levels=3,
        bottleneck_drop_path_rate=0.15,
        csfg_reduction_ratio=8,
        decoder_num_res_blocks=2,
        use_transpose_conv=True,
        d_state=16,
        dropout=0.1,
        use_checkpoint=False,
        **kwargs
    )


def hma_unet_small(in_channels=3, num_classes=1, **kwargs):
    """小型HMA-UNet配置 - 平衡性能和效率"""
    return HMAUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=48,
        encoder_depths=[2, 2, 4, 2],
        encoder_drop_path_rate=0.15,
        bottleneck_num_levels=3,
        bottleneck_drop_path_rate=0.2,
        csfg_reduction_ratio=6,
        decoder_num_res_blocks=3,
        use_transpose_conv=True,
        d_state=16,
        dropout=0.1,
        use_checkpoint=False,
        **kwargs
    )


def hma_unet_base(in_channels=3, num_classes=1, **kwargs):
    """基础HMA-UNet配置 - 追求最佳性能"""
    return HMAUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=64,
        encoder_depths=[3, 3, 6, 3],
        encoder_drop_path_rate=0.2,
        bottleneck_num_levels=4,
        bottleneck_drop_path_rate=0.25,
        csfg_reduction_ratio=4,
        decoder_num_res_blocks=3,
        use_transpose_conv=True,
        d_state=24,
        dropout=0.1,
        use_checkpoint=False,
        **kwargs
    )


def hma_unet_large(in_channels=3, num_classes=1, **kwargs):
    """大型HMA-UNet配置 - 复杂任务的最大容量"""
    return HMAUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=96,
        encoder_depths=[3, 3, 9, 3],
        encoder_drop_path_rate=0.3,
        bottleneck_num_levels=4,
        bottleneck_drop_path_rate=0.3,
        csfg_reduction_ratio=4,
        decoder_num_res_blocks=4,
        use_transpose_conv=True,
        d_state=32,
        dropout=0.15,
        use_checkpoint=True,
        **kwargs
    )


# 模型测试函数
def test_hma_unet():
    """测试HMA-UNet模型的前向传播"""
    print("Testing HMA-UNet models...")
    
    # 测试输入
    batch_size = 2
    input_size = (256, 256)
    in_channels = 3
    num_classes = 1
    
    x = torch.randn(batch_size, in_channels, *input_size)
    
    # 测试不同配置
    configs = ['tiny', 'small', 'base']
    
    for config in configs:
        print(f"\nTesting {config} configuration:")
        
        try:
            model = create_hma_unet(
                config=config,
                in_channels=in_channels,
                num_classes=num_classes
            )
            
            # 前向传播
            with torch.no_grad():
                output = model(x)
            
            # 打印模型信息
            model_info = model.get_model_info()
            print(f"  Model: {model_info['model_name']}")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Total parameters: {model_info['total_params']:,}")
            print(f"  Base channels: {model_info['base_channels']}")
            
            # 验证输出形状
            expected_shape = (batch_size, num_classes, *input_size)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            print(f"  ✓ Output shape correct")
            
        except Exception as e:
            print(f"  ✗ Error testing {config}: {e}")
    
    print("\nHMA-UNet testing completed!")


if __name__ == "__main__":
    test_hma_unet()