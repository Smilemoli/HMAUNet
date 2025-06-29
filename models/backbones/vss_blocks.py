import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from typing import Optional, Callable

# 使用官方mamba_ssm
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    raise ImportError("mamba_ssm is required. Please install it using: pip install mamba-ssm")


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
    """2D Layer Normalization"""
    def __init__(self, num_channels, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if self.weight is not None:
            x = x * self.weight[:, None, None]
        if self.bias is not None:
            x = x + self.bias[:, None, None]
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    从1D位置生成Sin-Cos位置嵌入
    embed_dim: 输出维度
    pos: 形状 (*) 的网格
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)
    
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    从网格生成2D Sine-Cosine位置嵌入
    embed_dim: 输出维度
    grid: np.array形状 (2, H, W)
    """
    assert embed_dim % 2 == 0
    
    # 使用半数维度用于sin，半数用于cos
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    2D Sine-Cosine位置嵌入
    embed_dim: 嵌入维度
    grid_size: int或元组，图像网格尺寸
    cls_token: 是否为分类token添加位置嵌入
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # 2 (H, W)
    grid = np.stack(grid, axis=0)  # 2 (H, W)
    grid = grid.reshape([2, 1, grid_size, grid_size])  # 2 (1, H, W)
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class SS2D(nn.Module):
    """
    官方Mamba的2D扩展模块 - 只支持CUDA
    将2D特征图转换为序列，使用Mamba处理，再转换回2D
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm is required. Please install it using: pip install mamba-ssm")
        
        # 检查CUDA可用性
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for mamba_ssm but not available")
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # 使用官方Mamba模块
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=use_fast_path,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )

        # 添加缓存优化
        self.cached_shapes = {}
        
        # 添加位置编码支持
        self.use_pos_embed = kwargs.get('use_pos_embed', False)
        self.pos_embed = None

    def _compute_scan_plan(self, H, W):
        """计算扫描路径优化计划"""
        # 简单的扫描顺序优化，可以根据MambaVision进一步增强
        scan_plan = {
            'height': H,
            'width': W,
            # 可添加更多信息用于优化扫描模式
        }
        return scan_plan
        
    def _get_pos_embed(self, H, W):
        """获取形状对应的位置编码"""
        if self.use_pos_embed:
            # 生成2D位置编码
            pos_embed = get_2d_sincos_pos_embed(self.d_model, H)
            return torch.from_numpy(pos_embed).float().to(self.mamba.in_proj.weight.device)
        return None

    def forward(self, x: torch.Tensor, **kwargs):
        """
        x: (B, H, W, C)
        """
        # 确保输入在CUDA设备上
        if not x.is_cuda:
            raise RuntimeError("Input tensor must be on CUDA device for mamba_ssm")
        
        B, H, W, C = x.shape
        
        # 使用形状缓存优化重复计算
        shape_key = (H, W)
        if shape_key not in self.cached_shapes:
            self.cached_shapes[shape_key] = self._compute_scan_plan(H, W)
            
            # 如果使用位置编码，也准备好它
            if self.use_pos_embed:
                pos_embed = self._get_pos_embed(H, W)
                self.cached_shapes[shape_key]['pos_embed'] = pos_embed
        
        scan_plan = self.cached_shapes[shape_key]
        
        # 将2D特征图重塑为序列
        x_seq = x.reshape(B, H * W, C)  # (B, L, C) where L = H * W
        
        # 添加位置编码（如果启用）
        if self.use_pos_embed and 'pos_embed' in scan_plan:
            pos_embed = scan_plan['pos_embed']
            x_seq = x_seq + pos_embed.unsqueeze(0)
        
        # 使用官方Mamba处理序列
        y = self.mamba(x_seq)
        
        # 重新整形回2D
        y = y.view(B, H, W, C)
        
        return y


class VSSBlock(nn.Module):
    """Vision State Space Block using official Mamba - CUDA only"""
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        mlp_ratio: float = 0.0,  # 添加MLP比例选项
        init_scale: float = 1.0,  # 添加初始化尺度参数
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.hidden_dim = hidden_dim
        
        # 使用官方Mamba的SS2D模块
        self.self_attention = SS2D(
            d_model=hidden_dim, 
            d_state=d_state,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        
        # 添加可选的MLP层
        self.has_mlp = mlp_ratio > 0
        if self.has_mlp:
            self.ln_2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, hidden_dim)
            )
        
        # 应用初始化尺度
        self.init_scale = init_scale
        self._init_weights()

    def _init_weights(self):
        """添加MambaVision使用的权重初始化方法"""
        if self.init_scale != 1.0:
            with torch.no_grad():
                # 将注意力输出缩放以改进训练初期稳定性
                if hasattr(self.self_attention.mamba, 'out_proj'):
                    self.self_attention.mamba.out_proj.weight.mul_(self.init_scale)
                    if self.self_attention.mamba.out_proj.bias is not None:
                        self.self_attention.mamba.out_proj.bias.mul_(self.init_scale)
                
                # 如果有MLP，也缩放其输出
                if self.has_mlp:
                    nn.init.xavier_uniform_(self.mlp[-1].weight, gain=self.init_scale)
                    if self.mlp[-1].bias is not None:
                        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, input: torch.Tensor):
        """
        input: (B, H, W, C)
        """
        # 确保输入在CUDA设备上
        if not input.is_cuda:
            raise RuntimeError("Input tensor must be on CUDA device for VSSBlock")
        
        x = input
        # 应用第一个规范化和Mamba状态空间模型
        x = x + self.drop_path(self.self_attention(self.ln_1(x)))
        
        # 应用可选的MLP
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
            
        return x


class VSSLayer(nn.Module):
    """Layer of multiple VSSBlocks - CUDA only"""
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
        mlp_ratio=0.0,  # 添加MLP比例
        init_scale=1.0,  # 添加初始化尺度
        use_pos_embed=False,  # 是否使用位置编码
        pos_embed_type='sincos',  # 位置编码类型
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # 使用深度特定的初始化尺度，类似MambaVision
        denom = depth ** 0.5  # 使用深度的平方根来缩放初始化
        
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
                    mlp_ratio=mlp_ratio,
                    init_scale=init_scale / denom,  # 按深度调整初始化尺度
                    use_pos_embed=use_pos_embed and (i == 0),  # 通常只在第一层添加位置编码
                    **kwargs
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """前向传播"""
        # 确保输入在CUDA设备上
        if not x.is_cuda:
            raise RuntimeError("Input tensor must be on CUDA device for VSSLayer")
        
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer"""
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
    """Patch Expanding Layer"""
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # 改进的上采样实现
        x = F.interpolate(x.permute(0, 3, 1, 2), scale_factor=2, mode='bilinear', align_corners=True)
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H_new, W_new, C = x.shape
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


# 序列优化工具函数
def optimize_scan_pattern(H, W, block_size=None):
    """
    优化2D扫描模式，提高处理效率
    
    Args:
        H, W: 特征图高度和宽度
        block_size: 分块大小，如果为None则自适应选择
    
    Returns:
        优化后的扫描索引
    """
    if block_size is None:
        # 自适应块大小
        block_size = min(32, max(16, min(H, W) // 8))
    
    indices = torch.arange(H * W).reshape(H, W)
    blocks_h = (H + block_size - 1) // block_size
    blocks_w = (W + block_size - 1) // block_size
    
    block_indices = []
    for i in range(blocks_h):
        for j in range(blocks_w):
            h_start = i * block_size
            h_end = min(h_start + block_size, H)
            w_start = j * block_size
            w_end = min(w_start + block_size, W)
            block = indices[h_start:h_end, w_start:w_end].reshape(-1)
            block_indices.append(block)
    
    return torch.cat(block_indices)