import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.vss_blocks import VSSBlock
from ..backbones.ResNet_blocks import ResidualBlock
from functools import partial


class HMABottleneck(nn.Module):
    """
    层级式Mamba聚合器瓶颈层 (Hierarchical Mamba Aggregator Bottleneck)
    
    这是HMA-UNet的核心瓶颈层，集成了完整的层级式Mamba聚合功能。
    它接收编码器Stage 4的输出，通过层级式Mamba聚合进行深度多尺度上下文建模，
    然后输出给解码器进行逐级上采样。
    
    完整流程:
    1. 接收编码器输出 x_enc4 (B, 8C, H/16, W/16)
    2. 初始特征提炼: 2个串联的VSSBlock
    3. 内部金字塔生成: 创建多尺度特征 [L0, L1, L2]
    4. 尺度专属Mamba处理: 每个尺度独立的VSSBlock
    5. 自下而上层级融合: 从最深层开始逐级融合
    6. 输出融合特征 F0 给解码器 (B, 8C, H/16, W/16)
    
    Args:
        in_channels (int): 编码器输出通道数 (通常为8C)
        out_channels (int): 输出通道数，如果None则与in_channels相同
        d_state (int): Mamba状态空间维度
        num_levels (int): 金字塔层数，默认3层
        drop_path_rate (float): DropPath概率
        use_checkpoint (bool): 是否使用梯度检查点节省内存
    """
    
    def __init__(
        self,
        in_channels,
        out_channels=None,
        d_state=16,
        num_levels=3,
        drop_path_rate=0.1,
        use_checkpoint=False
    ):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.use_checkpoint = use_checkpoint
        
        # 输入通道调整（如果需要）
        self.input_projection = None
        if in_channels != out_channels:
            self.input_projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        
        # 1. 初始特征提炼 - 2个串联的VSSBlock
        self.initial_refinement = nn.ModuleList([
            VSSBlock(
                hidden_dim=out_channels,
                drop_path=drop_path_rate * 0.5,  # 前面的块使用较低的drop_path
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                d_state=d_state
            ),
            VSSBlock(
                hidden_dim=out_channels,
                drop_path=drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                d_state=d_state
            )
        ])
        
        # 2. 内部金字塔生成 - 下采样卷积层
        self.pyramid_downsample = nn.ModuleList()
        for i in range(num_levels - 1):  # 需要num_levels-1个下采样层
            self.pyramid_downsample.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, 
                             padding=1, bias=False, groups=out_channels),  # 深度可分离卷积
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),  # 点卷积
                    nn.BatchNorm2d(out_channels),
                    nn.GELU()
                )
            )
        
        # 3. 尺度专属Mamba处理 - 每个尺度独立的VSSBlock
        self.scale_specific_mamba = nn.ModuleList()
        for i in range(num_levels):
            # 为每个尺度创建独立的VSSBlock，参数不共享
            self.scale_specific_mamba.append(
                VSSBlock(
                    hidden_dim=out_channels,
                    drop_path=drop_path_rate * (1.0 + i * 0.1),  # 更深层使用更高的drop_path
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    d_state=d_state,
                )
            )
        
        # 4. 自下而上融合 - 残差块用于平滑融合
        self.fusion_blocks = nn.ModuleList()
        for i in range(num_levels - 1):  # 需要num_levels-1个融合块
            self.fusion_blocks.append(
                ResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_layer=nn.BatchNorm2d,
                    activation=nn.GELU(),
                    dropout=0.1
                )
            )
        
        # 5. 最终输出处理
        self.output_processing = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # 残差连接准备
        self.residual_projection = None
        if in_channels != out_channels:
            self.residual_projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _apply_mamba_block(self, mamba_block, x):
        """应用Mamba块，处理维度转换"""
        # Mamba块期望输入格式为 (B, H, W, C)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, C, H, W) -> (B, H, W, C)
        
        if self.use_checkpoint:
            x = torch.utils.checkpoint.checkpoint(mamba_block, x)
        else:
            x = mamba_block(x)
        
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, H, W, C) -> (B, C, H, W)
        return x
    
    def forward(self, x_enc4):
        """
        HMA瓶颈层前向传播
        
        Args:
            x_enc4: 编码器Stage 4输出 (B, in_channels, H/16, W/16)
            
        Returns:
            解码器输入特征 (B, out_channels, H/16, W/16)
        """
        # 保存原始输入用于残差连接
        identity = x_enc4
        
        # 输入通道调整
        if self.input_projection is not None:
            x = self.input_projection(x_enc4)
        else:
            x = x_enc4
        
        # 1. 初始特征提炼 - 通过2个VSSBlock进行深度依赖建模
        x_refined = x
        for refine_block in self.initial_refinement:
            x_refined = self._apply_mamba_block(refine_block, x_refined)
        
        # 2. 内部金字塔生成
        pyramid_features = [x_refined]  # L0 = x_refined (H/16, W/16)
        
        current_feature = x_refined
        for downsample_layer in self.pyramid_downsample:
            current_feature = downsample_layer(current_feature)
            pyramid_features.append(current_feature)
        
        # 此时 pyramid_features = [L0, L1, L2, ...]
        # L0: (B, C, H/16, W/16)   - 原始尺度
        # L1: (B, C, H/32, W/32)   - 下采样1次
        # L2: (B, C, H/64, W/64)   - 下采样2次
        
        # 3. 尺度专属Mamba处理 - 每个尺度独立处理
        processed_features = []
        for i, (feature, mamba_block) in enumerate(zip(pyramid_features, self.scale_specific_mamba)):
            processed_feature = self._apply_mamba_block(mamba_block, feature)
            processed_features.append(processed_feature)
        
        # 此时 processed_features = [P0, P1, P2, ...]
        
        # 4. 自下而上的层级式融合
        # 从最深层开始融合 (P2 -> P1 -> P0)
        fused_feature = processed_features[-1]  # 从最深层开始
        
        for i in range(len(processed_features) - 2, -1, -1):
            # 上采样当前融合特征到目标尺度
            target_size = processed_features[i].shape[2:]
            upsampled = F.interpolate(
                fused_feature, 
                size=target_size, 
                mode='bilinear', 
                align_corners=True
            )
            
            # 与当前层特征相加
            added_feature = processed_features[i] + upsampled
            
            # 通过残差块平滑融合
            fusion_idx = len(processed_features) - 2 - i
            fused_feature = self.fusion_blocks[fusion_idx](added_feature)
        
        # 5. 最终输出处理
        output = self.output_processing(fused_feature)  # F0
        
        # 6. 残差连接
        if self.residual_projection is not None:
            identity = self.residual_projection(identity)
        
        # 添加残差连接
        output = output + identity
        
        return output
    
    def get_feature_info(self):
        """获取特征信息"""
        return {
            'input_channels': self.in_channels,
            'output_channels': self.out_channels,
            'num_levels': self.num_levels,
            'pyramid_scales': [f'1/{2**i}' for i in range(self.num_levels)]
        }


def create_hma_bottleneck(
    in_channels,
    out_channels=None,
    d_state=16,
    num_levels=3,
    drop_path_rate=0.1,
    use_checkpoint=False
):
    """创建HMA瓶颈层的工厂函数"""
    return HMABottleneck(
        in_channels=in_channels,
        out_channels=out_channels,
        d_state=d_state,
        num_levels=num_levels,
        drop_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint
    )


# 预定义配置
def hma_bottleneck_tiny(in_channels, out_channels=None, **kwargs):
    """轻量级HMA瓶颈层配置"""
    return create_hma_bottleneck(
        in_channels=in_channels,
        out_channels=out_channels,
        d_state=16,
        num_levels=3,
        drop_path_rate=0.1,
        **kwargs
    )


def hma_bottleneck_small(in_channels, out_channels=None, **kwargs):
    """小型HMA瓶颈层配置"""
    return create_hma_bottleneck(
        in_channels=in_channels,
        out_channels=out_channels,
        d_state=16,
        num_levels=3,
        drop_path_rate=0.15,
        **kwargs
    )


def hma_bottleneck_base(in_channels, out_channels=None, **kwargs):
    """基础HMA瓶颈层配置"""
    return create_hma_bottleneck(
        in_channels=in_channels,
        out_channels=out_channels,
        d_state=24,
        num_levels=4,
        drop_path_rate=0.2,
        **kwargs
    )


# 向后兼容的别名
HMAModule = HMABottleneck
create_hma_module = create_hma_bottleneck
hma_tiny = hma_bottleneck_tiny
hma_small = hma_bottleneck_small
hma_base = hma_bottleneck_base