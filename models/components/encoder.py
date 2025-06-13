import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.convnext_blocks import ConvNeXtV2Block
from ..backbones.vss_blocks import VSSBlock
from functools import partial


class DownsampleLayer(nn.Module):
    """Downsampling layer between encoder stages."""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class ConvNeXtStage(nn.Module):
    """ConvNeXt stage for shallow layers (Stage 1-2)."""

    def __init__(self, dim, depth=2, drop_path_rate=0.0):
        super().__init__()
        # Build drop path rates for this stage
        if isinstance(drop_path_rate, (list, tuple)):
            drop_path_rates = drop_path_rate
        else:
            drop_path_rates = [drop_path_rate] * depth

        self.blocks = nn.ModuleList(
            [
                ConvNeXtV2Block(dim=dim, drop_path=drop_path_rates[i])
                for i in range(depth)
            ]
        )

    def forward(self, x):
        # ConvNeXt expects (N, C, H, W)
        for block in self.blocks:
            x = block(x)
        return x


class VSSStage(nn.Module):
    """VSS stage for deep layers (Stage 3-4)."""

    def __init__(self, dim, depth=2, drop_path_rate=0.0, d_state=16):
        super().__init__()
        # Build drop path rates for this stage
        if isinstance(drop_path_rate, (list, tuple)):
            drop_path_rates = drop_path_rate
        else:
            drop_path_rates = [drop_path_rate] * depth

        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path_rates[i],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    d_state=d_state,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        # Convert from (N, C, H, W) to (N, H, W, C) for VSS blocks
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)

        for block in self.blocks:
            x = block(x)

        # Convert back to (N, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, C, H, W)
        return x


class HybridEncoder(nn.Module):
    """
    混合式编码器：浅层CNN(ConvNeXt) + 深层Mamba(VSS)

    Architecture:
    - Stem: H -> H/2, 3 -> C
    - Stage 1: H/2 -> H/4, C -> 2C  
    - Stage 2: H/4 -> H/8, 2C -> 4C
    - Stage 3: H/8 -> H/16, 4C -> 8C
    - Stage 4: H/16 (no downsample), 8C -> 8C
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=32,  # C in the architecture
        depths=[2, 2, 2, 2],  # Number of blocks in each stage
        drop_path_rate=0.1,
        d_state=16,  # For VSS blocks
    ):
        super().__init__()

        self.base_channels = base_channels
        self.num_stages = 4

        # Calculate channels for each stage
        # [C, 2C, 4C, 8C, 8C] - note Stage 4 keeps 8C
        self.stage_channels = [
            base_channels,      # Stem output
            2 * base_channels,  # Stage 1 output
            4 * base_channels,  # Stage 2 output
            8 * base_channels,  # Stage 3 output
            8 * base_channels,  # Stage 4 output
        ]

        # Build drop path rates for all stages
        drop_path_rates = self._build_drop_path_rates(depths, drop_path_rate)

        # Initial stem layer - H -> H/2
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.stage_channels[0],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.stage_channels[0]),
            nn.GELU(),
        )

        # Stage 1: ConvNeXt (C -> 2C, H/2 -> H/4)
        self.stage1 = ConvNeXtStage(
            dim=self.stage_channels[0],
            depth=depths[0],
            drop_path_rate=drop_path_rates[0],
        )
        self.downsample1 = DownsampleLayer(
            self.stage_channels[0], self.stage_channels[1]
        )

        # Stage 2: ConvNeXt (2C -> 4C, H/4 -> H/8)
        self.stage2 = ConvNeXtStage(
            dim=self.stage_channels[1],
            depth=depths[1],
            drop_path_rate=drop_path_rates[1],
        )
        self.downsample2 = DownsampleLayer(
            self.stage_channels[1], self.stage_channels[2]
        )

        # Stage 3: VSS (4C -> 8C, H/8 -> H/16)
        self.stage3 = VSSStage(
            dim=self.stage_channels[2],
            depth=depths[2],
            drop_path_rate=drop_path_rates[2],
            d_state=d_state,
        )
        self.downsample3 = DownsampleLayer(
            self.stage_channels[2], self.stage_channels[3]
        )

        # Stage 4: VSS (8C -> 8C, H/16 -> H/16) - No downsampling
        self.stage4 = VSSStage(
            dim=self.stage_channels[3],
            depth=depths[3],
            drop_path_rate=drop_path_rates[3],
            d_state=d_state,
        )
        # No downsample after stage 4 - it goes to bottleneck

        self._initialize_weights()

    def _build_drop_path_rates(self, depths, drop_path_rate):
        """Build drop path rates for all stages."""
        total_depth = sum(depths)
        drop_path_rates = []

        depth_idx = 0
        for stage_idx, depth in enumerate(depths):
            stage_rates = []
            for block_idx in range(depth):
                rate = (
                    drop_path_rate * depth_idx / (total_depth - 1)
                    if total_depth > 1
                    else 0.0
                )
                stage_rates.append(rate)
                depth_idx += 1
            drop_path_rates.append(stage_rates)

        return drop_path_rates

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through hybrid encoder.
        
        Input: (B, 3, H, W)
        Returns:
        - x_enc1: (B, 2C, H/4, W/4)   - Stage 1 output
        - x_enc2: (B, 4C, H/8, W/8)   - Stage 2 output  
        - x_enc3: (B, 8C, H/16, W/16) - Stage 3 output
        - x_enc4: (B, 8C, H/16, W/16) - Stage 4 output
        """
        # Initial processing - stem reduces by 2x
        x = self.stem(x)  # (B, C, H/2, W/2)

        # Stage 1: ConvNeXt
        x = self.stage1(x)  # (B, C, H/2, W/2)
        x_enc1 = self.downsample1(x)  # (B, 2C, H/4, W/4)

        # Stage 2: ConvNeXt
        x = self.stage2(x_enc1)  # (B, 2C, H/4, W/4)
        x_enc2 = self.downsample2(x)  # (B, 4C, H/8, W/8)

        # Stage 3: VSS
        x = self.stage3(x_enc2)  # (B, 4C, H/8, W/8)
        x_enc3 = self.downsample3(x)  # (B, 8C, H/16, W/16)

        # Stage 4: VSS - 保持在H/16分辨率，不下采样
        x_enc4 = self.stage4(x_enc3)  # (B, 8C, H/16, W/16)

        return [x_enc1, x_enc2, x_enc3, x_enc4]

    def get_feature_channels(self):
        """Get channel numbers for each output feature map."""
        return {
            "enc1": self.stage_channels[1],  # 2C
            "enc2": self.stage_channels[2],  # 4C
            "enc3": self.stage_channels[3],  # 8C
            "enc4": self.stage_channels[4],  # 8C
        }


def create_hybrid_encoder(
    in_channels=3, base_channels=32, depths=[2, 2, 2, 2], drop_path_rate=0.1, d_state=16
):
    """Create hybrid encoder with specified configuration."""
    return HybridEncoder(
        in_channels=in_channels,
        base_channels=base_channels,
        depths=depths,
        drop_path_rate=drop_path_rate,
        d_state=d_state,
    )


# Predefined configurations
def hybrid_encoder_tiny(in_channels=3, **kwargs):
    """Tiny configuration."""
    return create_hybrid_encoder(
        in_channels=in_channels,
        base_channels=32,
        depths=[2, 2, 2, 2],
        drop_path_rate=0.1,
        **kwargs
    )


def hybrid_encoder_small(in_channels=3, **kwargs):
    """Small configuration."""
    return create_hybrid_encoder(
        in_channels=in_channels,
        base_channels=48,
        depths=[2, 2, 4, 2],
        drop_path_rate=0.2,
        **kwargs
    )


def hybrid_encoder_base(in_channels=3, **kwargs):
    """Base configuration."""
    return create_hybrid_encoder(
        in_channels=in_channels,
        base_channels=64,
        depths=[3, 3, 6, 3],
        drop_path_rate=0.3,
        **kwargs
    )