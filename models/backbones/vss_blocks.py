import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from typing import Optional, Callable

# 如果没有安装mamba_ssm，我们提供一个简化的实现
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    print("Warning: mamba_ssm not found. Using simplified implementation.")


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SimplifiedSS2D(nn.Module):
    """
    简化的SS2D模块，避免复杂的selective scan操作
    使用标准的注意力机制来近似Mamba的功能
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 深度卷积
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=3,
            padding=1,
        )
        self.act = nn.SiLU()

        # 简化的"状态空间"处理 - 使用多头注意力近似
        self.num_heads = 8
        self.head_dim = self.d_inner // self.num_heads
        
        # 查询、键、值投影
        self.qkv = nn.Linear(self.d_inner, self.d_inner * 3, bias=False)
        self.proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        
        # 输出处理
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        # 输入投影
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)  # (B, H, W, d_inner), (B, H, W, d_inner)

        # 深度卷积（需要转换维度）
        x_conv = x_inner.permute(0, 3, 1, 2).contiguous()  # (B, d_inner, H, W)
        x_conv = self.conv2d(x_conv)  # (B, d_inner, H, W)
        x_conv = self.act(x_conv)
        x_conv = x_conv.permute(0, 2, 3, 1).contiguous()  # (B, H, W, d_inner)

        # 简化的"状态空间"处理 - 使用注意力机制
        # 将空间维度展平
        x_flat = x_conv.view(B, H * W, self.d_inner)  # (B, N, d_inner)
        
        # 多头注意力
        qkv = self.qkv(x_flat).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力权重
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        # 应用注意力权重
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H * W, self.d_inner)
        x_attn = self.proj(x_attn)
        
        # 重新整形回空间维度
        x_attn = x_attn.view(B, H, W, self.d_inner)

        # 输出处理
        y = self.out_norm(x_attn)
        y = y * F.silu(z)  # 门控机制
        out = self.out_proj(y)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        # 使用简化的SS2D模块
        self.self_attention = SimplifiedSS2D(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            **kwargs
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input
        x = self.ln_1(x)
        x = self.self_attention(x)
        x = self.drop_path(x)
        x = input + x
        return x


class VSSLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = (
            nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        )
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # 简化的上采样实现
        x = F.interpolate(x.permute(0, 3, 1, 2), scale_factor=2, mode='bilinear', align_corners=True)
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H_new, W_new, C = x.shape
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x