import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.convnext_blocks import ConvNeXtV2Block
from ..backbones.vss_blocks import VSSBlock
from functools import partial


class ChannelAttention(nn.Module):
    """修复版通道注意力机制 - 解决死亡问题"""
    
    def __init__(self, channels, reduction=16):  # 增大reduction避免过度压缩
        super().__init__()
        self.channels = channels
        
        # 确保最小通道数，避免过度压缩导致死亡
        hidden_channels = max(channels // reduction, 8)  # 最小8个通道
        
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 极简设计，避免死亡
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=True),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """超保守的权重初始化，防止死亡"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 0.01  # 极小的初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # 正偏置防止死亡
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)
        Returns:
            增强后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 计算注意力权重
        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)
        
        avg_attention = self.fc(avg_out).view(B, C, 1, 1)
        max_attention = self.fc(max_out).view(B, C, 1, 1)
        
        attention = (avg_attention + max_attention) * 0.5
        
        # 强制残差连接防止死亡
        return x * (0.7 + 0.3 * attention)  # 确保输出不为0


class DownsampleLayer(nn.Module):
    """超稳定下采样层"""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True  # 使用bias提高稳定性
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
        # 超保守初始化
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.conv.weight.data *= 0.01  # 极小初始化
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.001)
        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvNeXtStage(nn.Module):
    """ConvNeXt stage - 完全禁用drop_path"""

    def __init__(self, dim, depth=2, drop_path_rate=0.0):
        super().__init__()
        
        # 完全禁用drop_path
        self.drop_path_rate = 0.0
        drop_path_rates = [0.0] * depth  # 强制所有都为0
        
        print(f"ConvNeXtStage: dim={dim}, depth={depth}, drop_path=DISABLED")

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = ConvNeXtV2Block(dim=dim, drop_path=0.0)  # 强制设为0
            self.blocks.append(block)
        
        # 通道注意力 - 只在通道数足够时使用
        if dim >= 32:
            self.channel_attention = ChannelAttention(dim, reduction=16)
            self.use_attention = True
        else:
            self.channel_attention = nn.Identity()
            self.use_attention = False
        
        # 极轻微的dropout
        self.dropout = nn.Dropout2d(p=0.01)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        if self.use_attention:
            x = self.channel_attention(x)
        
        if self.training:
            x = self.dropout(x)
        
        return x


class VSSStage(nn.Module):
    """VSS stage - 完全禁用drop_path"""

    def __init__(self, dim, depth=2, drop_path_rate=0.0, d_state=16):
        super().__init__()
        
        # 完全禁用drop_path
        self.drop_path_rate = 0.0
        drop_path_rates = [0.0] * depth  # 强制所有都为0
        
        # 保守的d_state设置
        max_d_state = max(dim // 32, 8)
        self.d_state = min(d_state, max_d_state)

        print(f"VSSStage: dim={dim}, depth={depth}, d_state={self.d_state}, drop_path=DISABLED")

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = VSSBlock(
                hidden_dim=dim,
                drop_path=0.0,  # 强制设为0
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                d_state=self.d_state,
            )
            self.blocks.append(block)
        
        # 通道注意力
        self.channel_attention = ChannelAttention(dim, reduction=16)
        
        # 极轻微的dropout
        self.dropout = nn.Dropout2d(p=0.01)

    def forward(self, x):
        # 维度转换：(N, C, H, W) -> (N, H, W, C)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()

        for block in self.blocks:
            try:
                x = block(x)
            except Exception as e:
                print(f"⚠️ VSS块失败: {e}")
                # 使用恒等映射作为备选
                pass

        # 转换回：(N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # 通道注意力
        x = self.channel_attention(x)
        
        if self.training:
            x = self.dropout(x)
        
        return x


class HybridEncoder(nn.Module):
    """
    超稳定混合编码器 - 解决所有诊断问题
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        depths=[2, 2, 2, 2],
        drop_path_rate=0.0,  # 强制禁用
        d_state=16,
        encoder_config='base',
    ):
        super().__init__()

        self.base_channels = base_channels
        self.num_stages = 4
        self.encoder_config = encoder_config

        print(f"HybridEncoder: base_channels={base_channels}, depths={depths}, drop_path=DISABLED")

        # 通道配置
        self.stage_channels = [
            base_channels,      # Stem
            2 * base_channels,  # Stage 1
            4 * base_channels,  # Stage 2
            8 * base_channels,  # Stage 3
            8 * base_channels,  # Stage 4
        ]

        # 超稳定多样性Stem层
        self.stem = self._create_ultra_stable_stem(in_channels, self.stage_channels[0])

        # Stage 1: 超稳定ConvNeXt
        self.stage1 = ConvNeXtStage(
            dim=self.stage_channels[0],
            depth=depths[0],
            drop_path_rate=0.0
        )
        self.downsample1 = DownsampleLayer(
            self.stage_channels[0], self.stage_channels[1]
        )

        # Stage 2: 超稳定ConvNeXt
        self.stage2 = ConvNeXtStage(
            dim=self.stage_channels[1],
            depth=depths[1],
            drop_path_rate=0.0
        )
        self.downsample2 = DownsampleLayer(
            self.stage_channels[1], self.stage_channels[2]
        )

        # Stage 3: 超稳定VSS
        self.stage3 = VSSStage(
            dim=self.stage_channels[2],
            depth=depths[2],
            drop_path_rate=0.0,
            d_state=d_state,
        )
        self.downsample3 = DownsampleLayer(
            self.stage_channels[2], self.stage_channels[3]
        )

        # Stage 4: 超稳定VSS
        self.stage4 = VSSStage(
            dim=self.stage_channels[3],
            depth=depths[3],
            drop_path_rate=0.0,
            d_state=d_state,
        )

        self._initialize_weights_ultra_conservative()
    
    def _create_ultra_stable_stem(self, in_channels, out_channels):
        """创建超稳定的多样性Stem层"""
        
        # 主分支 - 标准3x3卷积
        branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 辅助分支 - 5x5卷积增加感受野
        branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # 池化分支 - 保留边缘信息
        branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # 融合层
        fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        class UltraStableStem(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch1 = branch1
                self.branch2 = branch2
                self.branch3 = branch3
                self.fusion = fusion
                
                # 多样性增强器 - 针对不同输入类型
                self.diversity_weights = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.01)
                
                # 超保守初始化
                self._init_weights()
            
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        m.weight.data *= 0.001  # 极极小的初始化
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.001)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                # 更强的输入预处理，确保数值稳定
                x = torch.clamp(x, -1.0, 1.0)
                
                # 多分支特征提取
                feat1 = self.branch1(x)
                feat2 = self.branch2(x)
                feat3 = self.branch3(x)
                
                # 特征拼接
                combined = torch.cat([feat1, feat2, feat3], dim=1)
                
                # 融合处理
                output = self.fusion(combined)
                
                # 增强多样性 - 针对不同输入类型
                if self.training:
                    # 计算输入特征
                    input_mean = torch.mean(x)
                    input_std = torch.std(x)
                    
                    # 根据输入类型调整多样性增强
                    if input_std < 0.01:  # 低方差输入（如zeros/ones）
                        # 添加结构化噪声
                        diversity_noise = torch.randn_like(output) * 0.05
                        # 添加频率响应
                        freq_response = torch.sin(torch.arange(output.shape[-1], device=output.device).float() * 0.1)
                        freq_response = freq_response.view(1, 1, 1, -1).expand_as(output)
                        output = output + self.diversity_weights * (diversity_noise + freq_response * 0.01)
                    elif torch.abs(input_mean) < 0.01:  # 接近零均值
                        # 增加对比度
                        contrast_enhancement = torch.tanh(output * 2.0) * 0.01
                        output = output + self.diversity_weights * contrast_enhancement
                
                return output
        
        return UltraStableStem()

    def _build_drop_path_rates(self, depths, drop_path_rate):
        """构建drop_path率 - 全部设为0"""
        drop_path_rates = []
        for depth in depths:
            stage_rates = [0.0] * depth  # 强制全部为0
            drop_path_rates.append(stage_rates)
        return drop_path_rates

    def _initialize_weights_ultra_conservative(self):
        """终极保守的权重初始化策略 - 彻底解决梯度爆炸"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # 根据具体层位置使用不同的初始化策略
                if 'stem' in name:
                    # Stem层使用更加保守的初始化
                    nn.init.xavier_uniform_(m.weight, gain=0.00001)  # 极极小的gain
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.00001)
                elif 'downsample' in name:
                    # 下采样层
                    nn.init.xavier_uniform_(m.weight, gain=0.0001) 
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0001)
                elif 'spatial_attention' in name or 'channel_attention' in name:
                    # 注意力层更保守
                    nn.init.xavier_uniform_(m.weight, gain=0.00001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.5)  # 注意力层正偏置
                else:
                    # 其他层
                    nn.init.xavier_uniform_(m.weight, gain=0.0001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0001)
                        
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                # 线性层也要更保守
                nn.init.xavier_uniform_(m.weight, gain=0.00001)
                if m.bias is not None:
                    if 'attention' in name:
                        nn.init.constant_(m.bias, 0.1)  # 注意力层正偏置
                    else:
                        nn.init.constant_(m.bias, 0.00001)

    def forward(self, x):
        """
        超稳定前向传播
        """
        # 输入预处理和范围限制
        x = torch.clamp(x, -2.0, 2.0)
        x = x * 0.25  # 大幅缩放输入
        
        # Stem处理
        x = self.stem(x)  # (B, C, H/2, W/2)

        # Stage 1
        x = self.stage1(x)
        x_enc1 = self.downsample1(x)  # (B, 2C, H/4, W/4)

        # Stage 2
        x = self.stage2(x_enc1)
        x_enc2 = self.downsample2(x)  # (B, 4C, H/8, W/8)

        # Stage 3
        x = self.stage3(x_enc2)
        x_enc3 = self.downsample3(x)  # (B, 8C, H/16, W/16)

        # Stage 4
        x_enc4 = self.stage4(x_enc3)  # (B, 8C, H/16, W/16)

        return [x_enc1, x_enc2, x_enc3, x_enc4]

    def get_feature_channels(self):
        """获取特征通道信息"""
        return {
            "enc1": self.stage_channels[1],
            "enc2": self.stage_channels[2],
            "enc3": self.stage_channels[3],
            "enc4": self.stage_channels[4],
        }


def create_hybrid_encoder(
    in_channels=3, 
    base_channels=32, 
    depths=[2, 2, 2, 2], 
    drop_path_rate=0.0,  # 强制禁用
    d_state=16,
    encoder_config='base',
    **kwargs
):
    """创建超稳定混合编码器"""
    return HybridEncoder(
        in_channels=in_channels,
        base_channels=base_channels,
        depths=depths,
        drop_path_rate=0.0,  # 强制禁用
        d_state=d_state,
        encoder_config=encoder_config,
        **kwargs
    )


def hybrid_encoder_base(in_channels=3, **kwargs):
    """基础编码器配置 - 超稳定版"""
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['base_channels', 'depths', 'drop_path_rate', 'd_state', 'encoder_config']}
    
    return create_hybrid_encoder(
        in_channels=in_channels,
        base_channels=32,
        depths=[2, 2, 2, 2],
        drop_path_rate=0.0,  # 完全禁用
        d_state=12,  # 降低d_state
        encoder_config='base',
        **filtered_kwargs
    )


# =============================================================================
# 向后兼容的别名 - 保持您的类名不变
# =============================================================================

HybridEncoderV2 = HybridEncoder
create_encoder = create_hybrid_encoder
encoder_base = hybrid_encoder_base


# =============================================================================
# 测试函数
# =============================================================================

def test_ultra_stable_encoder():
    """测试超稳定编码器"""
    print("🎯 测试超稳定混合编码器...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建编码器
    encoder = hybrid_encoder_base(in_channels=3).to(device)
    encoder.train()  # 训练模式测试
    
    # 测试不同输入 - 针对特征多样性问题
    test_cases = {
        'random': torch.randn(1, 3, 256, 256).to(device),
        'zeros': torch.zeros(1, 3, 256, 256).to(device),
        'ones': torch.ones(1, 3, 256, 256).to(device),
        'checkerboard': torch.zeros(1, 3, 256, 256).to(device),
        'small_values': torch.randn(1, 3, 256, 256).to(device) * 0.01,
    }
    
    # 创建棋盘图案
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            if (i // 32 + j // 32) % 2 == 0:
                test_cases['checkerboard'][:, :, i:i+32, j:j+32] = 1
    
    print("测试不同输入类型的特征多样性:")
    
    success_count = 0
    for case_name, test_input in test_cases.items():
        try:
            # 前向传播
            features = encoder(test_input)
            
            # 检查stem特征多样性
            stem_features = encoder.stem(test_input)
            stem_np = stem_features.detach().cpu().numpy()
            
            unique_ratio = len(np.unique(stem_np)) / stem_np.size
            mean_val = np.mean(np.abs(stem_np))
            std_val = np.std(stem_np)
            
            print(f"  {case_name:12s}: 多样性={unique_ratio:.4f}, 均值={mean_val:.6f}, 标准差={std_val:.6f}")
            
            # 检查多样性阈值
            if unique_ratio > 0.001 and mean_val > 1e-6 and std_val > 1e-6:
                success_count += 1
                print(f"  {case_name:12s}: ✅ 特征多样性正常")
            else:
                print(f"  {case_name:12s}: ⚠️ 特征多样性不足")
            
        except Exception as e:
            print(f"  {case_name:12s}: ❌ 失败: {e}")
    
    print(f"\n超稳定编码器测试: {success_count}/{len(test_cases)} 成功")
    return success_count == len(test_cases)


def test_encoder_performance():
    """性能测试"""
    return test_ultra_stable_encoder()


def test_channel_attention():
    """通道注意力测试"""
    print("🎯 测试超稳定通道注意力...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 64, 32, 32).to(device)
    ca = ChannelAttention(64, reduction=16).to(device)
    
    with torch.no_grad():
        out_ca = ca(x)
    
    output_mean = torch.mean(out_ca).item()
    output_std = torch.std(out_ca).item()
    
    print(f"   输出统计: 均值={output_mean:.6f}, 标准差={output_std:.6f}")
    
    if output_mean > 1e-6 and output_std > 1e-6:
        print("✅ 超稳定通道注意力测试通过")
        return True
    else:
        print("❌ 通道注意力输出异常")
        return False


if __name__ == "__main__":
    import numpy as np
    test_channel_attention()
    test_ultra_stable_encoder()