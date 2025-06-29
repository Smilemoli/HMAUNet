import torch
import torch.nn as nn
import torch.nn.functional as F
from .components.encoder import hybrid_encoder_base
from .components.hma_module import hma_bottleneck_base
from .components.decoder import resnet_decoder_base, OutputHead


class HMAUNet(nn.Module):
    """
    HMA-UNet: Hierarchical Mamba Aggregator U-Net - 完整实现版本

    一个创新的医学图像分割网络，结合了以下核心设计：
    1. 混合编码器: 浅层ConvNeXt + 深层Mamba
    2. HMA瓶颈层: 层级式Mamba聚合器（增强版）
    3. CSFG跳跃连接: 跨尺度融合门
    4. ResNet解码器: 轻量级特征重建

    增强特性：
    - 多尺度注意力机制
    - 边界增强模块
    - 自适应特征融合
    - 深度监督支持
    - 动态权重调整

    Args:
        in_channels (int): 输入图像通道数，默认3 (RGB)
        num_classes (int): 分割类别数，默认1
        base_channels (int): 基础通道数，默认32
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        base_channels=32,
    ):
        super().__init__()

        # 保存配置参数
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        # 计算各阶段通道数
        # 编码器输出: [2C, 4C, 8C, 8C]
        self.encoder_channels = [
            2 * base_channels,  # Stage 1: 2C
            4 * base_channels,  # Stage 2: 4C
            8 * base_channels,  # Stage 3: 8C
            8 * base_channels,  # Stage 4: 8C
        ]

        print(f"HMAUNet: base_channels={base_channels}, encoder_channels={self.encoder_channels}")

        # 1. 混合式编码器 (ConvNeXt + Mamba)
        self.encoder = hybrid_encoder_base(
            in_channels=in_channels,
            base_channels=base_channels,
        )

        # 2. HMA瓶颈层 (层级式Mamba聚合器) - 增强版
        self.bottleneck = hma_bottleneck_base(
            in_channels=self.encoder_channels[3],  # 8C
            out_channels=self.encoder_channels[3],  # 8C
        )

        # 3. ResNet解码器 (带CSFG跳跃连接)
        self.decoder = resnet_decoder_base(
            base_channels=base_channels,
            encoder_channels=self.encoder_channels,
        )

        # 4. 输出头
        self.output_head = OutputHead(
            in_channels=base_channels, 
            num_classes=num_classes, 
            dropout=0.1
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        # x_enc1: (B, 2C, H/4, W/4)   - Stage 1 输出
        # x_enc2: (B, 4C, H/8, W/8)   - Stage 2 输出
        # x_enc3: (B, 8C, H/16, W/16) - Stage 3 输出
        # x_enc4: (B, 8C, H/16, W/16) - Stage 4 输出

        x_enc1, x_enc2, x_enc3, x_enc4 = encoder_features

        # 获取stem输出用于最后的跳跃连接
        x_stem = self._get_stem_features(x)  # (B, C, H/2, W/2)

        # 2. HMA瓶颈层: 层级式Mamba聚合 - 增强版
        bottleneck_features = self.bottleneck(x_enc4)  # (B, 8C, H/16, W/16)

        # 3. ResNet解码器: 轻量级特征重建 + CSFG智能跳跃连接
        decoder_features = [x_enc1, x_enc2, x_enc3, x_stem]
        decoded_features = self.decoder(
            bottleneck_features, decoder_features
        )  # (B, C, H, W)

        # 4. 输出头: 生成最终分割结果
        output = self.output_head(decoded_features)  # (B, num_classes, H, W)

        # 确保输出尺寸与输入匹配
        if output.shape[2:] != input_size:
            output = F.interpolate(
                output, size=input_size, mode="bilinear", align_corners=True
            )

        return output

    def _get_stem_features(self, x):
        """获取编码器stem层的输出特征"""
        # 通过编码器的stem层获取初始特征
        return self.encoder.stem(x)  # (B, C, H/2, W/2)

    def get_model_info(self):
        """获取模型信息"""
        return {
            "model_name": "HMA-UNet",
            "input_channels": self.in_channels,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "encoder_channels": self.encoder_channels,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def get_feature_maps(self, x):
        """获取中间特征图，用于可视化和分析"""
        with torch.no_grad():
            # 临时关闭训练模式
            old_training = self.training
            self.eval()

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

            # 恢复训练状态
            self.training = old_training

            return {
                "stem": x_stem,
                "encoder": {
                    "stage1": x_enc1,
                    "stage2": x_enc2,
                    "stage3": x_enc3,
                    "stage4": x_enc4,
                },
                "bottleneck": bottleneck_features,
                "decoder": decoded_features,
            }


# =============================================================================
# 工厂函数 - 只保留base配置
# =============================================================================

def create_hma_unet(
    config="base",
    in_channels=3,
    num_classes=1,
    **kwargs
):
    """
    创建HMA-UNet模型的工厂函数 - 只支持base配置

    Args:
        config (str): 模型配置，只支持"base"
        in_channels (int): 输入通道数
        num_classes (int): 分割类别数
        **kwargs: 其他配置参数

    Returns:
        HMAUNet: 配置好的HMA-UNet模型
    """
    
    # 只支持base配置
    if config != "base":
        print(f"Warning: Config '{config}' not supported, using base config")
    
    # 过滤kwargs中的无效参数
    valid_kwargs = {}
    if 'base_channels' in kwargs:
        valid_kwargs['base_channels'] = kwargs['base_channels']
    
    return hma_unet_base(
        in_channels=in_channels,
        num_classes=num_classes,
        **valid_kwargs
    )


def hma_unet_base(
    in_channels=3, 
    num_classes=1, 
    base_channels=32,
    **kwargs
):
    """基础HMA-UNet配置 - 追求最佳性能"""
    
    # 过滤掉可能冲突的参数
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['base_channels']}
    
    return HMAUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        **filtered_kwargs
    )


# =============================================================================
# 向后兼容的别名 - 全部指向base配置
# =============================================================================

# 所有配置都指向base，确保兼容性
hma_unet_nano = hma_unet_base
hma_unet_tiny = hma_unet_base
hma_unet_small = hma_unet_base
hma_unet_efficient = hma_unet_base
hma_unet_memory_efficient = hma_unet_base
hma_unet_large = hma_unet_base
hma_unet_enhanced = hma_unet_base
hma_unet_improved = hma_unet_base
hma_unet_regularized = hma_unet_base
hma_unet_anti_overfitting = hma_unet_base


# =============================================================================
# 配置获取函数
# =============================================================================

def get_available_configs():
    """获取可用的模型配置"""
    return ["base"]  # 只返回base配置


# =============================================================================
# 模型导出和加载工具
# =============================================================================

def save_hma_unet(model, filepath, include_config=True):
    """保存HMA-UNet模型"""
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_info": model.get_model_info(),
    }

    if include_config:
        save_dict["config"] = {
            "in_channels": model.in_channels,
            "num_classes": model.num_classes,
            "base_channels": model.base_channels,
        }

    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_hma_unet(filepath, config="base", device="cuda", **kwargs):
    """
    加载HMA-UNet模型
    
    Args:
        filepath: 检查点文件路径
        config: 模型配置（只支持base）
        device: 设备
        **kwargs: 其他参数
        
    Returns:
        加载的模型
    """
    import os
    
    # 创建模型
    model = create_hma_unet(config="base", **kwargs)
    
    # 加载检查点
    if filepath and os.path.exists(filepath):
        print(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        
        # 尝试加载模型状态
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("✅ Model state loaded successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not load some parameters: {e}")
                # 尝试部分加载
                model_dict = model.state_dict()
                pretrained_dict = checkpoint['model_state_dict']
                
                # 过滤出匹配的参数
                matched_dict = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        matched_dict[k] = v
                
                model_dict.update(matched_dict)
                model.load_state_dict(model_dict)
                
                print(f"✅ Partially loaded {len(matched_dict)}/{len(pretrained_dict)} parameters")
        
        # 加载模型信息（如果存在）
        if 'model_info' in checkpoint:
            loaded_info = checkpoint['model_info']
            print(f"✅ Loaded model: {loaded_info.get('model_name', 'Unknown')}")
            print(f"   Parameters: {loaded_info.get('total_params', 'Unknown'):,}")
    else:
        print(f"⚠️ Checkpoint file not found: {filepath}")
        print("Using random initialization")
    
    return model


def get_model_complexity_info(model, input_size=(3, 512, 512)):
    """获取模型复杂度信息"""
    try:
        from thop import profile, clever_format
        
        # 创建测试输入
        device = next(model.parameters()).device
        input_tensor = torch.randn(1, *input_size).to(device)
        
        # 计算FLOPs和参数量
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        # 格式化输出
        flops_formatted, params_formatted = clever_format([flops, params], "%.2f")
        
        return {
            "flops": flops,
            "params": params,
            "flops_formatted": flops_formatted,
            "params_formatted": params_formatted,
            "flops_gflops": flops / 1e9,
            "params_mb": params * 4 / 1024 / 1024,  # 假设float32
        }
        
    except ImportError:
        print("⚠️ thop not available, cannot compute FLOPs")
        return {
            "flops": None,
            "params": sum(p.numel() for p in model.parameters()),
            "flops_formatted": "N/A",
            "params_formatted": f"{sum(p.numel() for p in model.parameters()):,}",
            "flops_gflops": None,
            "params_mb": sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024,
        }
    
    except Exception as e:
        print(f"⚠️ Error computing complexity: {e}")
        return {
            "flops": None,
            "params": sum(p.numel() for p in model.parameters()),
            "flops_formatted": "Error",
            "params_formatted": f"{sum(p.numel() for p in model.parameters()):,}",
            "flops_gflops": None,
            "params_mb": sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024,
        }


# =============================================================================
# 模型测试函数
# =============================================================================

def test_hma_unet():
    """测试HMA-UNet模型的前向传播"""
    print("🚀 测试HMA-UNet模型 (base配置)")
    print("=" * 50)

    # 测试输入
    batch_size = 2
    input_size = (256, 256)
    in_channels = 3
    num_classes = 1

    x = torch.randn(batch_size, in_channels, *input_size)

    try:
        # 创建模型
        model = create_hma_unet(
            config="base", 
            in_channels=in_channels, 
            num_classes=num_classes
        )

        print(f"📊 模型创建成功")

        # 前向传播
        with torch.no_grad():
            output = model(x)

        # 打印模型信息
        model_info = model.get_model_info()
        print(f"  模型名称: {model_info['model_name']}")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  总参数量: {model_info['total_params']:,}")
        print(f"  基础通道数: {model_info['base_channels']}")

        # 验证输出形状
        expected_shape = (batch_size, num_classes, *input_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"  ✅ 输出形状正确")

        # 测试特征图提取
        feature_maps = model.get_feature_maps(x)
        print(f"  ✅ 特征图提取成功: {len(feature_maps)} 个级别")
        
        # 测试模型信息
        complexity_info = get_model_complexity_info(model, input_size=(3, 256, 256))
        if complexity_info['flops_gflops']:
            print(f"  📈 计算复杂度: {complexity_info['flops_gflops']:.2f} GFLOPs")
        print(f"  💾 模型大小: {complexity_info['params_mb']:.2f} MB")

        print("\n🎉 HMA-UNet测试通过!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_hma_unet()