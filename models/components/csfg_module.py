import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.ResNet_blocks import ConvBlock, ResidualBlock


class CSFGModule(nn.Module):
    """
    跨尺度融合门 (Cross-Scale Fusion Gate, CSFG)
    """
    
    def __init__(
        self,
        enc_channels,
        dec_channels,
        reduction_ratio=8,
        use_residual=True
    ):
        super().__init__()
        
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.use_residual = use_residual
        
        # 1. 编码器特征的多尺度分解
        # 细节分支: 3x3卷积，捕获精细纹理和边缘
        self.detail_branch = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels),
            nn.GELU(),
            nn.Conv2d(enc_channels, enc_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels)
        )
        
        # 局部区域分支: 5x5卷积，捕获局部形状信息
        self.local_branch = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(enc_channels),
            nn.GELU(),
            nn.Conv2d(enc_channels, enc_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels)
        )
        
        # 上下文分支: 空洞卷积，捕获更大范围的上下文信息
        self.context_branch = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, kernel_size=3, padding=2, 
                     dilation=2, bias=False),
            nn.BatchNorm2d(enc_channels),
            nn.GELU(),
            nn.Conv2d(enc_channels, enc_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels)
        )
        
        # 2. 解码器语义引导生成 - 移除归一化层
        reduced_channels = max(dec_channels // reduction_ratio, 8)
        
        self.semantic_guidance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(dec_channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),  # 使用ReLU替代GELU，更稳定
            nn.Conv2d(reduced_channels, 3, kernel_size=1, bias=True),  # 输出3个权重
            nn.Softmax(dim=1)  # 确保权重和为1
        )
        
        # 3. 特征融合后的后处理
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels),
            nn.GELU()
        )
        
        # 4. 残差连接的投影层（如果需要）
        self.residual_projection = None
        if use_residual:
            self.residual_projection = nn.Sequential(
                nn.Conv2d(enc_channels, enc_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(enc_channels)
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 特殊初始化：语义引导的最后一层偏置，使初始权重相等
        if hasattr(self.semantic_guidance[-2], 'bias') and self.semantic_guidance[-2].bias is not None:
            nn.init.constant_(self.semantic_guidance[-2].bias, 0.33)  # 1/3 for each scale
    
    def forward(self, x_enc, g_up):
        """
        CSFG前向传播
        
        Args:
            x_enc: 编码器特征 (B, enc_channels, H, W)
            g_up: 解码器上采样特征 (B, dec_channels, H, W)
            
        Returns:
            融合后的特征 x_fused (B, enc_channels, H, W)
        """
        # 1. 编码器特征的多尺度分解
        f_detail = self.detail_branch(x_enc)     # 细节信息 (3x3卷积)
        f_local = self.local_branch(x_enc)       # 局部信息 (5x5卷积)
        f_context = self.context_branch(x_enc)   # 上下文信息 (空洞卷积)
        
        # 2. 解码器语义引导生成
        # 从g_up中提取语义指令，生成三个尺度的权重
        semantic_weights = self.semantic_guidance(g_up)  # (B, 3, 1, 1)
        
        # 分离三个权重
        alpha_detail = semantic_weights[:, 0:1, :, :]    # (B, 1, 1, 1)
        alpha_local = semantic_weights[:, 1:2, :, :]     # (B, 1, 1, 1)
        alpha_context = semantic_weights[:, 2:3, :, :]   # (B, 1, 1, 1)
        
        # 3. 动态加权融合
        # 根据语义权重，自适应地融合不同尺度的特征
        x_fused = (alpha_detail * f_detail + 
                   alpha_local * f_local + 
                   alpha_context * f_context)
        
        # 4. 融合后处理
        x_fused = self.fusion_conv(x_fused)
        
        # 5. 残差连接（可选）
        if self.use_residual and self.residual_projection is not None:
            residual = self.residual_projection(x_enc)
            x_fused = x_fused + residual
        
        return x_fused
    
    def get_attention_weights(self, x_enc, g_up):
        """
        获取注意力权重，用于可视化和分析
        """
        with torch.no_grad():
            semantic_weights = self.semantic_guidance(g_up)  # (B, 3, 1, 1)
            
            return {
                'detail_weight': semantic_weights[:, 0, 0, 0].cpu().numpy(),
                'local_weight': semantic_weights[:, 1, 0, 0].cpu().numpy(),
                'context_weight': semantic_weights[:, 2, 0, 0].cpu().numpy(),
                'weights_tensor': semantic_weights
            }


class CSFGSkipConnection(nn.Module):
    """
    完整的CSFG跳跃连接模块
    """
    
    def __init__(
        self,
        enc_channels,
        dec_channels,
        out_channels=None,
        reduction_ratio=8,
        use_residual=True
    ):
        super().__init__()
        
        if out_channels is None:
            out_channels = enc_channels
        
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.out_channels = out_channels
        
        # CSFG核心模块
        self.csfg = CSFGModule(
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            reduction_ratio=reduction_ratio,
            use_residual=use_residual
        )
        
        # 融合后的特征处理
        concat_channels = enc_channels + dec_channels
        self.post_fusion = nn.Sequential(
            # 第一个卷积块：融合特征
            ConvBlock(
                in_channels=concat_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                norm_layer=nn.BatchNorm2d,
                activation=nn.GELU()
            ),
            # 残差块：进一步精炼特征
            ResidualBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                norm_layer=nn.BatchNorm2d,
                activation=nn.GELU(),
                dropout=0.1
            )
        )
    
    def forward(self, x_enc, g_up):
        """
        完整的CSFG跳跃连接前向传播
        """
        # 1. CSFG智能融合
        x_fused = self.csfg(x_enc, g_up)
        
        # 2. 与解码器特征拼接
        concat_features = torch.cat([x_fused, g_up], dim=1)
        
        # 3. 后续特征处理
        output = self.post_fusion(concat_features)
        
        return output


def create_csfg_module(
    enc_channels,
    dec_channels,
    reduction_ratio=8,
    use_residual=True
):
    """创建CSFG模块的工厂函数"""
    return CSFGModule(
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        reduction_ratio=reduction_ratio,
        use_residual=use_residual
    )


def create_csfg_skip_connection(
    enc_channels,
    dec_channels,
    out_channels=None,
    reduction_ratio=8,
    use_residual=True
):
    """创建完整CSFG跳跃连接的工厂函数"""
    return CSFGSkipConnection(
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        out_channels=out_channels,
        reduction_ratio=reduction_ratio,
        use_residual=use_residual
    )


# 预定义配置
def csfg_tiny(enc_channels, dec_channels, **kwargs):
    """轻量级CSFG配置"""
    return create_csfg_skip_connection(
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        reduction_ratio=8,
        use_residual=True,
        **kwargs
    )


def csfg_small(enc_channels, dec_channels, **kwargs):
    """小型CSFG配置"""
    return create_csfg_skip_connection(
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        reduction_ratio=6,
        use_residual=True,
        **kwargs
    )


def csfg_base(enc_channels, dec_channels, **kwargs):
    """基础CSFG配置"""
    return create_csfg_skip_connection(
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        reduction_ratio=4,
        use_residual=True,
        **kwargs
    )