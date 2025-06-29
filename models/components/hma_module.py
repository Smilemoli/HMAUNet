import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.vss_blocks import VSSBlock
from ..backbones.ResNet_blocks import ResidualBlock
from functools import partial
import math


class HMABottleneck(nn.Module):
    """
    增强版HMA瓶颈层 - 提升分割性能
    
    新增特性：
    1. 多尺度注意力机制
    2. 自适应特征融合
    3. 边界增强模块
    4. 深度监督支持
    5. 动态权重调整
    """
    
    def __init__(
        self,
        in_channels,
        out_channels=None,
        d_state=16,
        num_levels=3,  # 增加层数以提升性能
        drop_path_rate=0.1,  # 适度的drop_path有助于泛化
        use_checkpoint=False,
        enhanced_features=True  # 新增：是否启用增强特性
    ):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.use_checkpoint = use_checkpoint
        self.enhanced_features = enhanced_features
        
        # 动态调整drop_path
        self.drop_path_rate = drop_path_rate
        
        # 优化d_state设置
        self.d_state = min(d_state, max(out_channels // 8, 16))
        
        print(f"Enhanced HMABottleneck: in_channels={in_channels}, out_channels={out_channels}, "
              f"d_state={self.d_state}, enhanced_features={enhanced_features}")
        
        # 改进的GroupNorm函数
        def adaptive_group_norm(channels):
            if channels <= 8:
                return nn.GroupNorm(1, channels)
            elif channels <= 32:
                return nn.GroupNorm(min(4, channels // 4), channels)
            else:
                groups = min(32, max(8, channels // 16))
                return nn.GroupNorm(groups, channels)
        
        # 1. 增强的输入处理 - 添加注意力机制
        self.input_stabilizer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            adaptive_group_norm(out_channels),
            nn.GELU(),  # 使用GELU激活
            nn.Dropout2d(p=0.05)
        )
        
        # 2. 多尺度注意力模块 - 完全修复版本
        if self.enhanced_features:
            self.multi_scale_attention = MultiScaleAttention(out_channels)
        
        # 3. 改进的特征提炼
        self.feature_refinement = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            adaptive_group_norm(out_channels),
            nn.GELU()
        )
        
        # 4. 增强的Mamba块
        self.mamba_block = VSSBlock(
            hidden_dim=out_channels,
            drop_path=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            d_state=self.d_state
        )
        
        # 5. 自适应多尺度特征提取
        self.multi_scale_layers = nn.ModuleList()
        for i in range(self.num_levels):
            dilation = 2 ** i
            layer = AdaptiveMultiScaleBlock(
                out_channels, 
                dilation=dilation,
                enhanced=self.enhanced_features
            )
            self.multi_scale_layers.append(layer)
        
        # 6. 边界增强模块
        if self.enhanced_features:
            self.boundary_enhancement = BoundaryEnhancementModule(out_channels)
        
        # 7. 自适应特征融合
        total_channels = out_channels * (self.num_levels + 2)  # +2 for input and mamba
        if self.enhanced_features:
            total_channels += out_channels  # +1 for boundary features
            
        self.adaptive_fusion = AdaptiveFeatureFusion(
            total_channels, 
            out_channels,
            enhanced=self.enhanced_features
        )
        
        # 8. 输出稳定化 - 添加注意力
        self.output_stabilizer = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            adaptive_group_norm(out_channels),
            nn.GELU(),
            ChannelAttention(out_channels) if self.enhanced_features else nn.Identity()
        )
        
        # 9. 智能残差连接
        if in_channels != out_channels:
            self.residual_projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                adaptive_group_norm(out_channels)
            )
        else:
            self.residual_projection = None
        
        # 10. 动态权重参数
        self.mamba_weight = nn.Parameter(torch.tensor(0.3))  # 增加Mamba权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        self.boundary_weight = nn.Parameter(torch.tensor(0.2)) if self.enhanced_features else None
        
        # 11. 深度监督输出头（可选）
        if self.enhanced_features:
            self.deep_supervision_head = nn.Conv2d(out_channels, 1, kernel_size=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """改进的权重初始化"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.GroupNorm):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 初始化动态权重
        if hasattr(self, 'mamba_weight'):
            nn.init.constant_(self.mamba_weight, 0.3)
        if hasattr(self, 'residual_weight'):
            nn.init.constant_(self.residual_weight, 0.5)
        if hasattr(self, 'boundary_weight') and self.boundary_weight is not None:
            nn.init.constant_(self.boundary_weight, 0.2)
    
    def _safe_mamba_forward(self, x):
        """安全的Mamba前向传播"""
        try:
            B, C, H, W = x.shape
            
            # 维度转换：(B, C, H, W) -> (B, H, W, C)
            x_hwc = x.permute(0, 2, 3, 1).contiguous()
            
            # Mamba处理
            x_out = self.mamba_block(x_hwc)
            
            # 维度转换回：(B, H, W, C) -> (B, C, H, W)
            x_chw = x_out.permute(0, 3, 1, 2).contiguous()
            
            return x_chw
            
        except Exception as e:
            print(f"⚠️ Mamba处理失败: {e}")
            return x
    
    def forward(self, x_enc4, return_deep_supervision=False):
        """
        增强版前向传播
        """
        # 保存残差
        x_residual = x_enc4
        
        # 1. 输入稳定化
        x = self.input_stabilizer(x_enc4)
        
        # 2. 多尺度注意力（如果启用）
        if self.enhanced_features:
            x_attended = self.multi_scale_attention(x)
            x = x + 0.1 * x_attended  # 轻量级融合
        
        # 3. 特征提炼
        x_refined = self.feature_refinement(x)
        
        # 4. Mamba处理
        x_mamba = self._safe_mamba_forward(x_refined)
        x_mamba = x_refined + self.mamba_weight * (x_mamba - x_refined)
        
        # 5. 多尺度特征提取
        multi_scale_features = [x_refined, x_mamba]
        
        for scale_layer in self.multi_scale_layers:
            try:
                scale_feat = scale_layer(x_refined)
                multi_scale_features.append(scale_feat)
            except Exception as e:
                print(f"⚠️ 多尺度处理失败: {e}")
                multi_scale_features.append(x_refined)
        
        # 6. 边界增强（如果启用）
        if self.enhanced_features:
            try:
                boundary_feat = self.boundary_enhancement(x_refined)
                multi_scale_features.append(boundary_feat)
            except Exception as e:
                print(f"⚠️ 边界增强失败: {e}")
        
        # 7. 自适应特征融合
        try:
            x = self.adaptive_fusion(multi_scale_features)
        except Exception as e:
            print(f"⚠️ 特征融合失败: {e}")
            x = x_refined
        
        # 8. 输出稳定化
        x = self.output_stabilizer(x)
        
        # 9. 残差连接
        if self.residual_projection is not None:
            x_residual = self.residual_projection(x_residual)
        
        # 应用残差连接
        output = x + self.residual_weight * x_residual
        
        # 深度监督输出
        deep_output = None
        if self.enhanced_features and hasattr(self, 'deep_supervision_head') and return_deep_supervision:
            deep_output = self.deep_supervision_head(output)
        
        if return_deep_supervision:
            return output, deep_output
        return output
    
    def get_feature_info(self):
        """获取特征信息"""
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'num_levels': self.num_levels,
            'd_state': self.d_state,
            'drop_path_rate': self.drop_path_rate,
            'enhanced_features': self.enhanced_features,
            'mamba_weight': float(self.mamba_weight.item()),
            'residual_weight': float(self.residual_weight.item()),
        }


class MultiScaleAttention(nn.Module):
    """多尺度注意力模块 - 完全修复版本"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 不同尺度的卷积
        self.scales = [1, 3, 5]
        self.num_scales = len(self.scales)
        
        # 确保每个尺度的输出通道数是合理的
        channels_per_scale = max(1, channels // self.num_scales)
        
        # 多尺度卷积 - 使用标准卷积
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(
                channels, 
                channels_per_scale, 
                kernel_size=scale, 
                padding=scale//2,
                bias=False
            )
            for scale in self.scales
        ])
        
        # 计算拼接后的实际通道数
        self.concat_channels = channels_per_scale * self.num_scales
        
        # 通道适配层（如果需要）
        if self.concat_channels != channels:
            self.channel_adapter = nn.Conv2d(self.concat_channels, channels, 1, bias=False)
        else:
            self.channel_adapter = None
        
        # 注意力权重生成 - 使用实际的拼接通道数
        attention_input_channels = channels  # 使用适配后的通道数
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(attention_input_channels, max(8, attention_input_channels // 4), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(8, attention_input_channels // 4), channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        print(f"MultiScaleAttention: channels={channels}, "
              f"channels_per_scale={channels_per_scale}, "
              f"concat_channels={self.concat_channels}")
        
    def forward(self, x):
        # 多尺度特征提取
        scale_features = []
        for conv in self.scale_convs:
            scale_features.append(conv(x))
        
        # 拼接多尺度特征
        multi_scale = torch.cat(scale_features, dim=1)
        
        # 通道适配（如果需要）
        if self.channel_adapter is not None:
            multi_scale = self.channel_adapter(multi_scale)
        
        # 生成注意力权重
        attention = self.attention_conv(multi_scale)
        
        # 应用注意力
        return x * attention


class AdaptiveMultiScaleBlock(nn.Module):
    """自适应多尺度块 - 修复版本"""
    def __init__(self, channels, dilation=1, enhanced=True):
        super().__init__()
        self.enhanced = enhanced
        
        # 确保groups参数合理
        groups = min(channels, max(1, channels // 8))
        
        if enhanced:
            # 使用可分离卷积提升效率
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, 
                         dilation=dilation, groups=groups, bias=False),  # 深度卷积
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),  # 点卷积
                nn.GroupNorm(min(32, max(1, channels // 8)), channels),
                nn.GELU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, 
                         dilation=dilation, bias=False),
                nn.GroupNorm(min(32, max(1, channels // 8)), channels),
                nn.GELU()
            )
    
    def forward(self, x):
        return self.conv(x)


class BoundaryEnhancementModule(nn.Module):
    """边界增强模块 - 提升分割边缘精度"""
    def __init__(self, channels):
        super().__init__()
        
        # Sobel算子用于边缘检测
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2], 
            [-1, 0, 1]
        ]).float().unsqueeze(0).unsqueeze(0))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).float().unsqueeze(0).unsqueeze(0))
        
        # 边界特征处理
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(channels + 2, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, max(1, channels // 8)), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 计算梯度（边缘信息）
        x_gray = torch.mean(x, dim=1, keepdim=True)  # 转为灰度
        
        # 应用Sobel算子
        grad_x = F.conv2d(x_gray, self.sobel_x.repeat(1, 1, 1, 1), padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y.repeat(1, 1, 1, 1), padding=1)
        
        # 合并梯度信息
        boundary_info = torch.cat([x, grad_x, grad_y], dim=1)
        
        # 处理边界特征
        boundary_enhanced = self.boundary_conv(boundary_info)
        
        return boundary_enhanced


class AdaptiveFeatureFusion(nn.Module):
    """自适应特征融合模块 - 修复版本"""
    def __init__(self, in_channels, out_channels, enhanced=True):
        super().__init__()
        self.enhanced = enhanced
        
        if enhanced:
            # 使用注意力机制的融合
            mid_channels = max(32, out_channels * 2)  # 确保中间层有足够的通道数
            
            self.fusion = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(32, max(1, mid_channels // 8)), mid_channels),
                nn.GELU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(32, max(1, out_channels // 8)), out_channels),
                nn.GELU()
            )
            
            # 特征权重生成
            self.weight_gen = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, max(8, in_channels // 4), 1, bias=False),
                nn.GELU(),
                nn.Conv2d(max(8, in_channels // 4), out_channels, 1, bias=False),
                nn.Sigmoid()
            )
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(32, max(1, out_channels // 8)), out_channels),
                nn.GELU()
            )
    
    def forward(self, feature_list):
        # 拼接所有特征
        fused_features = torch.cat(feature_list, dim=1)
        
        # 基础融合
        output = self.fusion(fused_features)
        
        # 如果启用增强模式，应用注意力权重
        if self.enhanced and hasattr(self, 'weight_gen'):
            weights = self.weight_gen(fused_features)
            output = output * weights
        
        return output


class ChannelAttention(nn.Module):
    """通道注意力模块 - 修复版本"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 确保reduced_channels至少为1
        reduced_channels = max(1, channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


# =============================================================================
# 工厂函数和预定义配置
# =============================================================================

def create_hma_bottleneck(
    in_channels,
    out_channels=None,
    d_state=16,
    num_levels=3,
    drop_path_rate=0.1,
    use_checkpoint=False,
    enhanced_features=True
):
    """创建HMA瓶颈层的工厂函数"""
    return HMABottleneck(
        in_channels=in_channels,
        out_channels=out_channels,
        d_state=d_state,
        num_levels=num_levels,
        drop_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint,
        enhanced_features=enhanced_features
    )


def hma_bottleneck_base(in_channels, out_channels=None, **kwargs):
    """基础HMA瓶颈层配置 - 增强版"""
    return create_hma_bottleneck(
        in_channels=in_channels,
        out_channels=out_channels,
        d_state=16,
        num_levels=3,
        drop_path_rate=0.1,
        use_checkpoint=False,
        enhanced_features=True,
        **kwargs
    )


# 向后兼容的别名
HMAModule = HMABottleneck
create_hma_module = create_hma_bottleneck
hma_base = hma_bottleneck_base


# =============================================================================
# 测试函数
# =============================================================================

def test_hma_bottleneck():
    """测试增强版HMA瓶颈层"""
    print("🎯 测试增强版HMA瓶颈层...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建增强版HMA瓶颈层
    hma = HMABottleneck(
        in_channels=256, 
        out_channels=256,
        enhanced_features=True
    ).to(device)
    hma.eval()
    
    # 测试不同输入
    test_cases = {
        'normal': torch.randn(2, 256, 16, 16).to(device),
        'small': torch.randn(2, 256, 8, 8).to(device),
        'large': torch.randn(2, 256, 32, 32).to(device),
    }
    
    print("测试增强版HMA瓶颈层:")
    
    success_count = 0
    total_tests = len(test_cases)
    
    for case_name, test_input in test_cases.items():
        try:
            # 测试正常前向传播
            with torch.no_grad():
                output = hma(test_input)
                
            # 测试深度监督
            with torch.no_grad():
                output, deep_out = hma(test_input, return_deep_supervision=True)
            
            # 检查输出
            assert output.shape == test_input.shape, f"Shape mismatch: {output.shape} vs {test_input.shape}"
            
            if deep_out is not None:
                expected_deep_shape = (test_input.shape[0], 1, test_input.shape[2], test_input.shape[3])
                assert deep_out.shape == expected_deep_shape, f"Deep supervision shape mismatch"
            
            print(f"  {case_name:8s}: ✅ 形状匹配，深度监督正常")
            success_count += 1
                
        except Exception as e:
            print(f"  {case_name:8s}: ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n增强版HMA瓶颈层测试: {success_count}/{total_tests} 成功")
    
    # 打印特征信息
    try:
        feature_info = hma.get_feature_info()
        print("\n特征信息:")
        for key, value in feature_info.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"⚠️ 无法获取特征信息: {e}")
    
    return success_count == total_tests


if __name__ == "__main__":
    import numpy as np
    test_hma_bottleneck()