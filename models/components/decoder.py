import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.ResNet_blocks import ResidualBlock, ConvBlock, TransposeConvBlock
from .csfg_module import CSFGSkipConnection


class DecoderStage(nn.Module):
    """
    单个解码器阶段 - 修复尺寸匹配问题
    """
    
    def __init__(
        self,
        deep_channels,
        skip_channels,
        out_channels,
        reduction_ratio=8,
        use_transpose_conv=True,
        num_res_blocks=2
    ):
        super().__init__()
        
        self.deep_channels = deep_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        
        # 1. 上采样层 - 确保正确的上采样
        if use_transpose_conv:
            # 使用转置卷积进行上采样，同时调整通道数
            self.upsample = TransposeConvBlock(
                in_channels=deep_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                norm_layer=nn.BatchNorm2d,
                activation=nn.ReLU(inplace=True)
            )
        else:
            # 使用双线性插值 + 卷积
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvBlock(
                    in_channels=deep_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_layer=nn.BatchNorm2d,
                    activation=nn.ReLU(inplace=True)
                )
            )
        
        # 2. CSFG智能跳跃连接
        self.csfg_skip = CSFGSkipConnection(
            enc_channels=skip_channels,
            dec_channels=out_channels,
            out_channels=skip_channels,  # 保持跳跃连接特征的通道数
            reduction_ratio=reduction_ratio,
            use_residual=True
        )
        
        # 3. 特征融合后的处理
        concat_channels = out_channels + skip_channels
        
        # 首先通过一个卷积层调整通道数
        self.channel_adjust = ConvBlock(
            in_channels=concat_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU(inplace=True)
        )
        
        # 然后使用多个ResNet残差块进行特征提炼
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                norm_layer=nn.BatchNorm2d,
                activation=nn.ReLU(inplace=True),
                dropout=0.1 if i == 0 else 0.05
            )
            for i in range(num_res_blocks)
        ])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重 - ResNet风格"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_deep, x_skip):
        """
        解码器阶段前向传播 - 修复尺寸匹配
        
        Args:
            x_deep: 深层特征 (B, deep_channels, H, W)
            x_skip: 跳跃连接特征 (B, skip_channels, H, W) - 注意：同样的空间尺寸
            
        Returns:
            输出特征 (B, out_channels, 2H, 2W)
        """
        # 1. 上采样深层特征到与跳跃连接相同的尺寸
        g_up = self.upsample(x_deep)  # (B, out_channels, 2H, 2W)
        
        # 确保跳跃连接特征与上采样特征的空间尺寸匹配
        target_size = g_up.shape[2:]  # 获取上采样后的尺寸
        if x_skip.shape[2:] != target_size:
            # 如果尺寸不匹配，调整跳跃连接特征的尺寸
            x_skip = F.interpolate(
                x_skip, 
                size=target_size, 
                mode='bilinear', 
                align_corners=True
            )
        
        # 2. CSFG智能跳跃连接融合
        x_fused = self.csfg_skip(x_skip, g_up)  # (B, skip_channels, 2H, 2W)
        
        # 3. 特征合并
        concat_features = torch.cat([g_up, x_fused], dim=1)  # (B, out_channels + skip_channels, 2H, 2W)
        
        # 4. 通道调整
        x = self.channel_adjust(concat_features)  # (B, out_channels, 2H, 2W)
        
        # 5. ResNet残差块提炼
        for res_block in self.res_blocks:
            x = res_block(x)
        
        return x


class ResNetDecoder(nn.Module):
    """
    基于ResNet的解码器 - 修复尺寸匹配问题
    
    正确的数据流：
    - Stage 4: (B,8C,H/16,W/16) → (B,4C,H/8,W/8)   与 x_enc3(B,8C,H/16,W/16) 融合
    - Stage 3: (B,4C,H/8,W/8) → (B,2C,H/4,W/4)     与 x_enc2(B,4C,H/8,W/8) 融合  
    - Stage 2: (B,2C,H/4,W/4) → (B,C,H/2,W/2)      与 x_enc1(B,2C,H/4,W/4) 融合
    - Stage 1: (B,C,H/2,W/2) → (B,C,H,W)          与 x_stem(B,C,H/2,W/2) 融合
    """
    
    def __init__(
        self,
        base_channels=32,
        encoder_channels=[64, 128, 256, 256],  # [2C, 4C, 8C, 8C]
        reduction_ratio=8,
        use_transpose_conv=True,
        num_res_blocks=2
    ):
        super().__init__()
        
        self.base_channels = base_channels
        self.encoder_channels = encoder_channels
        self.num_res_blocks = num_res_blocks
        
        # 解码器各阶段的输出通道数
        self.decoder_channels = [
            4 * base_channels,  # Stage 4 output: 4C
            2 * base_channels,  # Stage 3 output: 2C
            base_channels,      # Stage 2 output: C
            base_channels       # Stage 1 output: C
        ]
        
        # 瓶颈层输入通道数 (8C)
        bottleneck_channels = encoder_channels[3]  # 8C = 256
        
        # 解码器阶段 4: (8C,H/16) → (4C,H/8) 与 x_enc3(8C,H/16) 融合
        self.stage4 = DecoderStage(
            deep_channels=bottleneck_channels,     # 8C (from HMA bottleneck)
            skip_channels=encoder_channels[2],     # 8C (from encoder stage 3) 
            out_channels=self.decoder_channels[0], # 4C
            reduction_ratio=reduction_ratio,
            use_transpose_conv=use_transpose_conv,
            num_res_blocks=num_res_blocks
        )
        
        # 解码器阶段 3: (4C,H/8) → (2C,H/4) 与 x_enc2(4C,H/8) 融合
        self.stage3 = DecoderStage(
            deep_channels=self.decoder_channels[0], # 4C
            skip_channels=encoder_channels[1],      # 4C (from encoder stage 2)
            out_channels=self.decoder_channels[1],  # 2C
            reduction_ratio=reduction_ratio,
            use_transpose_conv=use_transpose_conv,
            num_res_blocks=num_res_blocks
        )
        
        # 解码器阶段 2: (2C,H/4) → (C,H/2) 与 x_enc1(2C,H/4) 融合
        self.stage2 = DecoderStage(
            deep_channels=self.decoder_channels[1], # 2C
            skip_channels=encoder_channels[0],      # 2C (from encoder stage 1)
            out_channels=self.decoder_channels[2],  # C
            reduction_ratio=reduction_ratio,
            use_transpose_conv=use_transpose_conv,
            num_res_blocks=num_res_blocks
        )
        
        # 解码器阶段 1: (C,H/2) → (C,H) 与 x_stem(C,H/2) 融合
        self.stage1 = DecoderStage(
            deep_channels=self.decoder_channels[2], # C
            skip_channels=base_channels,            # C (from encoder stem)
            out_channels=self.decoder_channels[3],  # C
            reduction_ratio=reduction_ratio,
            use_transpose_conv=use_transpose_conv,
            num_res_blocks=num_res_blocks
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重 - ResNet风格"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, bottleneck_features, encoder_features):
        """
        ResNet解码器前向传播
        
        Args:
            bottleneck_features: HMA瓶颈层输出 (B, 8C, H/16, W/16)
            encoder_features: 编码器特征列表 [x_enc1, x_enc2, x_enc3, x_stem]
                - x_enc1: (B, 2C, H/4, W/4)
                - x_enc2: (B, 4C, H/8, W/8)
                - x_enc3: (B, 8C, H/16, W/16)
                - x_stem: (B, C, H/2, W/2)
            
        Returns:
            最终解码特征 (B, C, H, W)
        """
        x_enc1, x_enc2, x_enc3, x_stem = encoder_features
        
        # 解码器阶段 4: H/16 → H/8
        # bottleneck_features (B,8C,H/16,W/16) 上采样到 (B,4C,H/8,W/8)
        # 与 x_enc3 (B,8C,H/16,W/16) 融合
        x_dec4 = self.stage4(bottleneck_features, x_enc3)  # (B, 4C, H/8, W/8)
        
        # 解码器阶段 3: H/8 → H/4  
        # x_dec4 (B,4C,H/8,W/8) 上采样到 (B,2C,H/4,W/4)
        # 与 x_enc2 (B,4C,H/8,W/8) 融合
        x_dec3 = self.stage3(x_dec4, x_enc2)  # (B, 2C, H/4, W/4)
        
        # 解码器阶段 2: H/4 → H/2
        # x_dec3 (B,2C,H/4,W/4) 上采样到 (B,C,H/2,W/2)
        # 与 x_enc1 (B,2C,H/4,W/4) 融合
        x_dec2 = self.stage2(x_dec3, x_enc1)  # (B, C, H/2, W/2)
        
        # 解码器阶段 1: H/2 → H
        # x_dec2 (B,C,H/2,W/2) 上采样到 (B,C,H,W)
        # 与 x_stem (B,C,H/2,W/2) 融合
        x_dec1 = self.stage1(x_dec2, x_stem)  # (B, C, H, W)
        
        return x_dec1
    
    def get_feature_channels(self):
        """获取各阶段特征通道信息"""
        return {
            'stage4': self.decoder_channels[0],  # 4C
            'stage3': self.decoder_channels[1],  # 2C
            'stage2': self.decoder_channels[2],  # C
            'stage1': self.decoder_channels[3],  # C
        }


class OutputHead(nn.Module):
    """
    输出头，将解码器特征转换为最终分割结果
    """
    
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super().__init__()
        
        self.output_conv = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        输出头前向传播
        
        Args:
            x: 解码器特征 (B, in_channels, H, W)
            
        Returns:
            分割logits (B, num_classes, H, W)
        """
        return self.output_conv(x)


def create_resnet_decoder(
    base_channels=32,
    encoder_channels=[64, 128, 256, 256],
    reduction_ratio=8,
    use_transpose_conv=True,
    num_res_blocks=2
):
    """创建基于ResNet的解码器的工厂函数"""
    return ResNetDecoder(
        base_channels=base_channels,
        encoder_channels=encoder_channels,
        reduction_ratio=reduction_ratio,
        use_transpose_conv=use_transpose_conv,
        num_res_blocks=num_res_blocks
    )


# 预定义配置
def resnet_decoder_tiny(base_channels=32, **kwargs):
    """轻量级ResNet解码器配置"""
    encoder_channels = [2*base_channels, 4*base_channels, 8*base_channels, 8*base_channels]
    return create_resnet_decoder(
        base_channels=base_channels,
        encoder_channels=encoder_channels,
        reduction_ratio=8,
        use_transpose_conv=True,
        num_res_blocks=2,
        **kwargs
    )


def resnet_decoder_small(base_channels=48, **kwargs):
    """小型ResNet解码器配置"""
    encoder_channels = [2*base_channels, 4*base_channels, 8*base_channels, 8*base_channels]
    return create_resnet_decoder(
        base_channels=base_channels,
        encoder_channels=encoder_channels,
        reduction_ratio=6,
        use_transpose_conv=True,
        num_res_blocks=3,
        **kwargs
    )


def resnet_decoder_base(base_channels=64, **kwargs):
    """基础ResNet解码器配置"""
    encoder_channels = [2*base_channels, 4*base_channels, 8*base_channels, 8*base_channels]
    return create_resnet_decoder(
        base_channels=base_channels,
        encoder_channels=encoder_channels,
        reduction_ratio=4,
        use_transpose_conv=True,
        num_res_blocks=3,
        **kwargs
    )


# 向后兼容的别名
HybridDecoder = ResNetDecoder
create_hybrid_decoder = create_resnet_decoder
hybrid_decoder_tiny = resnet_decoder_tiny
hybrid_decoder_small = resnet_decoder_small
hybrid_decoder_base = resnet_decoder_base