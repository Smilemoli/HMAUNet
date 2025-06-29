import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.ResNet_blocks import ResidualBlock, ConvBlock, TransposeConvBlock
from .csfg_module import EnhancedCSFGSkipConnection, csfg_base


class SpatialAttention(nn.Module):
    """空间注意力机制 (Spatial Attention Module, SAM)"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_spatial = torch.cat([avg_out, max_out], dim=1)
        x_spatial = self.conv(x_spatial)
        return x * self.sigmoid(x_spatial)


class DecoderStage(nn.Module):
    """
    单个解码器阶段 - 内置空间注意力机制
    """

    def __init__(
        self,
        deep_channels,
        skip_channels,
        out_channels,
        reduction_ratio=8,
        use_transpose_conv=True,
        num_res_blocks=2,
    ):
        super().__init__()

        self.deep_channels = deep_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels

        # 1. 上采样层
        if use_transpose_conv:
            self.upsample = TransposeConvBlock(
                in_channels=deep_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                norm_layer=nn.BatchNorm2d,
                activation=nn.ReLU(inplace=True),
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                ConvBlock(
                    in_channels=deep_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_layer=nn.BatchNorm2d,
                    activation=nn.ReLU(inplace=True),
                ),
            )

        # 2. 修复版CSFG跳跃连接 - 只使用base配置
        self.csfg_skip = csfg_base(
            enc_channels=skip_channels,
            dec_channels=out_channels,
            out_channels=skip_channels,
        )

        # 3. 特征融合后的处理
        concat_channels = out_channels + skip_channels

        # 轻量级通道调整
        self.channel_adjust = ConvBlock(
            in_channels=concat_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU(inplace=True),
        )

        # 精简的ResNet残差块
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_layer=nn.BatchNorm2d,
                    activation=nn.ReLU(inplace=True),
                    dropout=0.05 if i == 0 else 0.0,
                )
                for i in range(num_res_blocks)
            ]
        )

        # 4. 空间注意力机制
        self.spatial_attention_upsample = SpatialAttention(kernel_size=7)
        self.spatial_attention_fusion = SpatialAttention(kernel_size=7)

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_deep, x_skip):
        """
        解码器阶段前向传播 - 内置空间注意力
        
        Args:
            x_deep: 来自更深层的特征 (B, deep_channels, H, W)
            x_skip: 跳跃连接特征 (B, skip_channels, 2H, 2W)
            
        Returns:
            融合后的特征 (B, out_channels, 2H, 2W)
        """
        # 1. 上采样
        x_up = self.upsample(x_deep)  # (B, out_channels, 2H, 2W)
        
        # 2. 上采样空间注意力
        x_up = self.spatial_attention_upsample(x_up)

        # 3. 尺寸对齐
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(
                x_up, size=x_skip.shape[2:], mode="bilinear", align_corners=True
            )

        # 4. CSFG智能跳跃连接
        x_skip_fused = self.csfg_skip(x_skip, x_up)

        # 5. 特征拼接和融合
        concat_features = torch.cat([x_up, x_skip_fused], dim=1)
        x = self.channel_adjust(concat_features)

        # 6. 空间注意力融合
        x = self.spatial_attention_fusion(x)

        # 7. 残差块处理
        for res_block in self.res_blocks:
            x = res_block(x)

        return x


class ResNetDecoder(nn.Module):
    """
    基于ResNet的解码器 - 内置空间注意力机制
    """

    def __init__(
        self,
        base_channels=32,
        encoder_channels=[64, 128, 256, 256],
        reduction_ratio=8,
        use_transpose_conv=True,
        num_res_blocks=2,
        use_checkpoint=False,
    ):
        super().__init__()

        self.base_channels = base_channels
        self.encoder_channels = encoder_channels
        self.num_res_blocks = num_res_blocks
        self.use_checkpoint = use_checkpoint

        # 解码器各阶段的输出通道数
        self.decoder_channels = [
            4 * base_channels,  # Stage 4 output: 4C
            2 * base_channels,  # Stage 3 output: 2C
            base_channels,  # Stage 2 output: C
            base_channels,  # Stage 1 output: C
        ]

        # 瓶颈层输入通道数 (8C)
        bottleneck_channels = encoder_channels[3]

        # 解码器阶段 4: (8C,H/16) → (4C,H/8) 与 x_enc3(8C,H/16) 融合
        self.stage4 = DecoderStage(
            deep_channels=bottleneck_channels,
            skip_channels=encoder_channels[2],
            out_channels=self.decoder_channels[0],
            reduction_ratio=reduction_ratio,
            use_transpose_conv=use_transpose_conv,
            num_res_blocks=num_res_blocks,
        )

        # 解码器阶段 3: (4C,H/8) → (2C,H/4) 与 x_enc2(4C,H/8) 融合
        self.stage3 = DecoderStage(
            deep_channels=self.decoder_channels[0],
            skip_channels=encoder_channels[1],
            out_channels=self.decoder_channels[1],
            reduction_ratio=reduction_ratio,
            use_transpose_conv=use_transpose_conv,
            num_res_blocks=num_res_blocks,
        )

        # 解码器阶段 2: (2C,H/4) → (C,H/2) 与 x_enc1(2C,H/4) 融合
        self.stage2 = DecoderStage(
            deep_channels=self.decoder_channels[1],
            skip_channels=encoder_channels[0],
            out_channels=self.decoder_channels[2],
            reduction_ratio=reduction_ratio,
            use_transpose_conv=use_transpose_conv,
            num_res_blocks=num_res_blocks,
        )

        # 解码器阶段 1: (C,H/2) → (C,H) 与 x_stem(C,H/2) 融合
        self.stage1 = DecoderStage(
            deep_channels=self.decoder_channels[2],
            skip_channels=base_channels,
            out_channels=self.decoder_channels[3],
            reduction_ratio=reduction_ratio,
            use_transpose_conv=use_transpose_conv,
            num_res_blocks=num_res_blocks,
        )

        # 最终上采样到原始尺寸
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBlock(
                in_channels=self.decoder_channels[3],
                out_channels=self.decoder_channels[3],
                kernel_size=3,
                padding=1,
                norm_layer=nn.BatchNorm2d,
                activation=nn.ReLU(inplace=True),
            ),
        )

        # 全局空间注意力机制
        self.global_spatial_attention = SpatialAttention(kernel_size=7)

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _apply_stage_with_checkpoint(self, stage, x_deep, x_skip):
        """带梯度检查点的阶段应用"""
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(stage, x_deep, x_skip, use_reentrant=False)
        else:
            return stage(x_deep, x_skip)

    def forward(self, bottleneck_features, encoder_features):
        """
        ResNet解码器前向传播 - 内置空间注意力版本

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

        # 解码器阶段 4: H/16 → H/8 (内置空间注意力)
        x_dec4 = self._apply_stage_with_checkpoint(
            self.stage4, bottleneck_features, x_enc3
        )

        # 解码器阶段 3: H/8 → H/4 (内置空间注意力)
        x_dec3 = self._apply_stage_with_checkpoint(
            self.stage3, x_dec4, x_enc2
        )

        # 解码器阶段 2: H/4 → H/2 (内置空间注意力)
        x_dec2 = self._apply_stage_with_checkpoint(
            self.stage2, x_dec3, x_enc1
        )

        # 解码器阶段 1: H/2 → H (内置空间注意力)
        x_dec1 = self._apply_stage_with_checkpoint(
            self.stage1, x_dec2, x_stem
        )

        # 最终上采样到原始尺寸: H → 2H
        x_final = self.final_upsample(x_dec1)

        # 全局空间注意力增强
        x_final = self.global_spatial_attention(x_final)

        return x_final

    def get_feature_channels(self):
        """获取各阶段特征通道信息"""
        return {
            "stage4": self.decoder_channels[0],
            "stage3": self.decoder_channels[1],
            "stage2": self.decoder_channels[2],
            "stage1": self.decoder_channels[3],
        }



class OutputHead(nn.Module):
    """
    修复版输出头 - 解决激活死亡和性能问题
    
    主要修复：
    1. 添加特征稳定化层
    2. 使用渐进式输出策略
    3. 添加空间注意力机制
    4. 优化激活函数和初始化
    5. 添加特征增强
    """

    def __init__(self, in_channels, num_classes, dropout=0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        print(f"FixedOutputHead: in_channels={in_channels}, num_classes={num_classes}, dropout={dropout}")
        
        # 1. 特征预处理和稳定化
        self.feature_stabilizer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 空间注意力机制
        self.output_spatial_attention = SpatialAttention(kernel_size=7)
        
        # 3. 特征增强层
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 4. 渐进式输出策略
        self.intermediate_conv = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 5. 最终输出层 - 使用更稳定的配置
        self.final_dropout = nn.Dropout2d(p=max(dropout, 0.05))  # 确保最小dropout
        self.output_conv = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1, bias=True)
        
        # 6. 可选的深度监督输出
        self.aux_conv = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        """终极修复版权重初始化 - 防止梯度死亡"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(m.weight, gain=0.0001)
                
                if m.bias is not None:
                    # 输出层使用小的正偏置，有助于训练稳定性
                    if m == self.output_conv:
                        nn.init.constant_(m.bias, 0.01)  # 小正偏置
                    else:
                        nn.init.constant_(m.bias, 0.0001)
                            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_aux=False):
        """
        修复版前向传播
        
        Args:
            x: 解码器输出特征 (B, in_channels, H, W)
            return_aux: 是否返回辅助输出用于深度监督
            
        Returns:
            主输出 (B, num_classes, H, W)
            可选的辅助输出 (B, num_classes, H, W) if return_aux=True
        """
        # 保存原始输入用于残差连接
        identity = x
        
        # 1. 特征稳定化处理
        x = self.feature_stabilizer(x)
        
        # 残差连接增强稳定性
        x = x + identity
        
        # 2. 空间注意力增强
        x = self.output_spatial_attention(x)
        
        # 3. 特征增强
        x_enhanced = self.feature_enhancer(x)
        
        # 4. 辅助输出用于深度监督（可选）
        aux_output = None
        if return_aux:
            aux_output = self.aux_conv(x_enhanced)
        
        # 5. 渐进式特征处理
        x_intermediate = self.intermediate_conv(x_enhanced)
        
        # 6. 最终输出
        x_final = self.final_dropout(x_intermediate)
        main_output = self.output_conv(x_final)
        
        if return_aux and aux_output is not None:
            return main_output, aux_output
        else:
            return main_output


# =============================================================================
# 工厂函数 - 只保留base配置
# =============================================================================


def create_resnet_decoder(
    base_channels=32,
    encoder_channels=[64, 128, 256, 256],
    reduction_ratio=8,
    use_transpose_conv=True,
    num_res_blocks=2,
    use_checkpoint=False,
):
    """创建内置空间注意力的ResNet解码器"""
    return ResNetDecoder(
        base_channels=base_channels,
        encoder_channels=encoder_channels,
        reduction_ratio=reduction_ratio,
        use_transpose_conv=use_transpose_conv,
        num_res_blocks=num_res_blocks,
        use_checkpoint=use_checkpoint,
    )


def resnet_decoder_base(**kwargs):
    """基础ResNet解码器配置 - 质量优先 + 内置空间注意力"""
    base_channels = kwargs.get("base_channels", 32)
    encoder_channels = kwargs.get(
        "encoder_channels",
        [2 * base_channels, 4 * base_channels, 8 * base_channels, 8 * base_channels],
    )

    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in [
            "base_channels",
            "encoder_channels",
            "reduction_ratio",
            "use_transpose_conv",
            "num_res_blocks",
            "use_checkpoint",
        ]
    }

    return create_resnet_decoder(
        base_channels=base_channels,
        encoder_channels=encoder_channels,
        reduction_ratio=8,
        use_transpose_conv=True,
        num_res_blocks=2,
        use_checkpoint=False,
        **filtered_kwargs,
    )


# 向后兼容的别名
HybridDecoder = ResNetDecoder
create_hybrid_decoder = create_resnet_decoder
hybrid_decoder_base = resnet_decoder_base