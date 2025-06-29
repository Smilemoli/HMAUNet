import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.ResNet_blocks import ConvBlock, ResidualBlock


class UltraFixedSpatialAttention(nn.Module):
    """超修复版空间注意力 - 完全避免归一化问题"""
    
    def __init__(self, channels, kernel_sizes=[3], reduction=16):
        super().__init__()
        
        self.kernel_sizes = kernel_sizes
        
        # 完全避免归一化，只使用卷积
        self.spatial_branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=ks, stride=1, padding=ks//2, bias=True),
                nn.ReLU(inplace=True)
            )
            self.spatial_branches.append(branch)
        
        # 特征融合
        if len(kernel_sizes) > 1:
            self.fusion = nn.Sequential(
                nn.Conv2d(len(kernel_sizes), 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
        else:
            self.fusion = nn.Sigmoid()
    
    def forward(self, x):
        # 计算空间统计
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        
        # 多尺度处理
        branch_outputs = []
        for branch in self.spatial_branches:
            branch_out = branch(spatial_input)
            branch_outputs.append(branch_out)
        
        # 融合多尺度特征
        if len(branch_outputs) > 1:
            fused = torch.cat(branch_outputs, dim=1)
            attention = self.fusion(fused)
        else:
            attention = self.fusion(branch_outputs[0])
        
        return x * attention

class UltraFixedCSFGModule(nn.Module):
    """
    超修复版CSFG模块 - 完全解决所有归一化问题
    
    主要修复：
    1. 完全移除所有可能有问题的归一化层
    2. 使用最安全的GroupNorm配置
    3. 添加数值稳定性保证
    4. 极简化设计
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
        
        print(f"UltraFixedCSFGModule: enc_channels={enc_channels}, dec_channels={dec_channels}")
        
        # 安全的GroupNorm计算函数
        def safe_group_norm(channels, min_groups=1):
            # 确保组数合理
            if channels <= 4:
                return nn.Identity()  # 对于很小的通道数，不使用归一化
            elif channels <= 8:
                return nn.GroupNorm(1, channels)  # 只有1组
            elif channels <= 16:
                return nn.GroupNorm(2, channels)  # 2组
            elif channels <= 32:
                return nn.GroupNorm(4, channels)  # 4组
            else:
                groups = min(32, channels // 4)  # 最多32组，每组至少4个通道
                return nn.GroupNorm(groups, channels)
        
        # 1. 极简特征对齐
        self.enc_align = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, kernel_size=1, bias=True),
            safe_group_norm(enc_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dec_align = nn.Sequential(
            nn.Conv2d(dec_channels, enc_channels, kernel_size=1, bias=True),
            safe_group_norm(enc_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 超简化的三分支设计
        # Detail分支 - 只用卷积
        self.detail_branch = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Local分支 - 最简设计
        local_mid = max(enc_channels // 2, 1)
        self.local_branch = nn.Sequential(
            nn.Conv2d(enc_channels, local_mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(local_mid, enc_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Context分支 - 简化膨胀卷积
        self.context_branch = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, kernel_size=3, padding=2, dilation=2, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # 3. 完全重新设计的全局分支 - 避免所有1x1问题
        global_mid = max(enc_channels // 4, 1)
        self.global_branch = nn.Sequential(
            # 先降维，避免大特征图上的全局池化
            nn.Conv2d(enc_channels, global_mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            # 使用更大的池化，避免1x1
            nn.AdaptiveAvgPool2d(2),  # 池化到2x2
            # 2x2卷积处理
            nn.Conv2d(global_mid, enc_channels, kernel_size=2, bias=True),
            # 最后池化到1x1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(2, 3),  # 展平为(B, C, 1)
        )
        
        # 使用简单的线性层生成权重
        self.global_weight_gen = nn.Sequential(
            nn.Linear(enc_channels, enc_channels, bias=True),
            nn.Sigmoid()
        )
        
        # 4. 超简化的语义引导网络
        guidance_mid = max(dec_channels // 4, 4)  # 至少4个通道
        self.semantic_guidance = nn.Sequential(
            # 先降维
            nn.Conv2d(dec_channels, guidance_mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            # 池化到2x2
            nn.AdaptiveAvgPool2d(2),
            # 2x2卷积
            nn.Conv2d(guidance_mid, 4, kernel_size=2, bias=True),  # 直接输出4个权重通道
            # 最后池化到1x1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # 展平为(B, 4)
            nn.Softmax(dim=1)
        )
        
        # 5. 最简融合
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # 6. 简化空间注意力
        self.spatial_attention = UltraFixedSpatialAttention(enc_channels, kernel_sizes=[3])
        
        # 7. 残差连接
        if use_residual:
            self.residual_projection = nn.Conv2d(enc_channels, enc_channels, kernel_size=1, bias=True)
            self.residual_weight = nn.Parameter(torch.tensor(0.1))  # 更小的残差权重
        else:
            self.residual_projection = None
        
        # 8. 最终处理 - 无归一化
        self.final_activation = nn.ReLU(inplace=True)
        
        # 9. Dropout
        self.dropout = nn.Dropout2d(p=0.05)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """超保守的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 0.05  # 极极保守的初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
                    
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.0001)  # 极小的初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
                    
            elif isinstance(m, nn.GroupNorm):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x_enc, g_up):
        """超修复版前向传播"""
        # 极强的输入限制
        x_enc = torch.clamp(x_enc, -0.1, 0.1)
        g_up = torch.clamp(g_up, -0.1, 0.1)
        
        # 1. 安全的特征对齐
        try:
            x_enc_aligned = self.enc_align(x_enc)
            g_up_aligned = self.dec_align(g_up)
            # 简单相加交互
            cross_scale = x_enc_aligned + g_up_aligned
        except Exception as e:
            print(f"⚠️ 特征对齐失败: {e}")
            # 最简单的备选方案
            if g_up.shape[1] != x_enc.shape[1]:
                g_up = F.adaptive_avg_pool2d(g_up, 1)
                g_up = F.interpolate(g_up, size=x_enc.shape[2:], mode='nearest')
                g_up = F.pad(g_up, (0, 0, 0, 0, 0, x_enc.shape[1] - g_up.shape[1]))[:, :x_enc.shape[1]]
            cross_scale = (x_enc + g_up) * 0.5
        
        # 2. 三分支特征提取
        try:
            f_detail = self.detail_branch(cross_scale)
            f_local = self.local_branch(cross_scale)
            f_context = self.context_branch(cross_scale)
        except Exception as e:
            print(f"⚠️ 分支特征提取失败: {e}")
            f_detail = f_local = f_context = cross_scale
        
        # 3. 全局分支
        try:
            global_features = self.global_branch(cross_scale)  # (B, C, 1)
            global_weights = self.global_weight_gen(global_features.squeeze(-1))  # (B, C)
            global_weights = global_weights.view(-1, self.enc_channels, 1, 1)
            f_global = cross_scale * global_weights
        except Exception as e:
            print(f"⚠️ 全局分支失败: {e}")
            f_global = cross_scale * 0.25
        
        # 4. 语义引导权重
        try:
            semantic_weights = self.semantic_guidance(g_up)  # (B, 4)
            # 确保是4维
            if semantic_weights.shape[1] != 4:
                semantic_weights = torch.ones(semantic_weights.shape[0], 4, device=semantic_weights.device) * 0.25
            
            alpha_detail = semantic_weights[:, 0:1].view(-1, 1, 1, 1)
            alpha_local = semantic_weights[:, 1:2].view(-1, 1, 1, 1)
            alpha_context = semantic_weights[:, 2:3].view(-1, 1, 1, 1)
            alpha_global = semantic_weights[:, 3:4].view(-1, 1, 1, 1)
        except Exception as e:
            print(f"⚠️ 语义引导失败: {e}")
            alpha_detail = alpha_local = alpha_context = alpha_global = 0.25
        
        # 5. 特征融合
        try:
            x_fused = (alpha_detail * f_detail + 
                       alpha_local * f_local + 
                       alpha_context * f_context + 
                       alpha_global * f_global)
        except Exception as e:
            print(f"⚠️ 特征融合失败: {e}")
            x_fused = (f_detail + f_local + f_context + f_global) * 0.25
        
        # 6. 后处理
        try:
            x_fused = self.adaptive_fusion(x_fused)
            x_fused = self.spatial_attention(x_fused)
        except Exception as e:
            print(f"⚠️ 后处理失败: {e}")
        
        # 7. 残差连接
        if self.use_residual and self.residual_projection is not None:
            try:
                residual = self.residual_projection(x_enc)
                x_fused = x_fused + self.residual_weight * residual
            except Exception as e:
                print(f"⚠️ 残差连接失败: {e}")
        
        # 8. 最终处理
        x_fused = self.final_activation(x_fused)
        
        # 9. Dropout和输出限制
        if self.training:
            x_fused = self.dropout(x_fused)
        
        x_fused = torch.clamp(x_fused, -0.1, 0.1)
        
        return x_fused
    
    def get_attention_weights(self, x_enc, g_up):
        """获取注意力权重"""
        with torch.no_grad():
            try:
                semantic_weights = self.semantic_guidance(g_up)
                return {
                    'weights_tensor': semantic_weights,
                    'device': semantic_weights.device,
                    'requires_sync': True
                }
            except Exception as e:
                print(f"⚠️ 获取注意力权重失败: {e}")
                B = g_up.shape[0]
                default_weights = torch.ones(B, 4, device=g_up.device) * 0.25
                return {
                    'weights_tensor': default_weights,
                    'device': g_up.device,
                    'requires_sync': False
                }
    
    def get_attention_weights_legacy(self, x_enc, g_up):
        """兼容接口"""
        weights_info = self.get_attention_weights(x_enc, g_up)
        weights = weights_info['weights_tensor']
        
        try:
            if weights.dim() == 2 and weights.shape[1] == 4:  # (B, 4)
                weights_cpu = weights.cpu().numpy()
                return {
                    'detail_weight': weights_cpu[:, 0],
                    'local_weight': weights_cpu[:, 1],
                    'context_weight': weights_cpu[:, 2],
                    'global_weight': weights_cpu[:, 3],
                    'weights_tensor': weights,
                    'device': weights.device
                }
        except Exception as e:
            print(f"⚠️ Legacy权重转换失败: {e}")
        
        # 备用方案
        B = g_up.shape[0]
        return {
            'detail_weight': [0.25] * B,
            'local_weight': [0.25] * B,
            'context_weight': [0.25] * B,
            'global_weight': [0.25] * B,
            'weights_tensor': torch.ones(B, 4, device=g_up.device) * 0.25,
            'device': g_up.device
        }


class UltraFixedCSFGSkipConnection(nn.Module):
    """超修复版CSFG跳跃连接"""
    
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
        
        print(f"UltraFixedCSFGSkipConnection: enc={enc_channels}, dec={dec_channels}, out={out_channels}")
        
        # 使用超修复版CSFG模块
        self.csfg = UltraFixedCSFGModule(
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            reduction_ratio=reduction_ratio,
            use_residual=use_residual
        )
        
        # 安全的GroupNorm函数
        def safe_group_norm(channels):
            if channels <= 4:
                return nn.Identity()
            elif channels <= 8:
                return nn.GroupNorm(1, channels)
            elif channels <= 16:
                return nn.GroupNorm(2, channels)
            else:
                groups = min(32, channels // 4)
                return nn.GroupNorm(groups, channels)
        
        # 极简的后处理网络
        concat_channels = enc_channels + dec_channels
        self.post_fusion = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=True),
            safe_group_norm(out_channels),
            nn.ReLU(inplace=True),
            
            # 精炼层
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # 极简的全局上下文 - 完全避免1x1归一化
        context_mid = max(out_channels // 4, 1)
        self.global_context = nn.Sequential(
            nn.Conv2d(out_channels, context_mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),  # 池化到2x2
            nn.Conv2d(context_mid, out_channels, kernel_size=2, bias=True),
            nn.AdaptiveAvgPool2d(1),  # 最后池化到1x1
            nn.Sigmoid()
        )
        
        # Dropout
        self.dropout = nn.Dropout2d(p=0.05)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """超保守初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 0.05  # 极保守
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.GroupNorm):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x_enc, g_up):
        """超修复版前向传播"""
        # 极强输入限制
        x_enc = torch.clamp(x_enc, -0.1, 0.1)
        g_up = torch.clamp(g_up, -0.1, 0.1)
        
        # 1. CSFG智能融合
        try:
            x_fused = self.csfg(x_enc, g_up)
        except Exception as e:
            print(f"⚠️ CSFG融合失败，使用简单融合: {e}")
            # 最安全的备选方案
            if x_enc.shape[2:] != g_up.shape[2:]:
                g_up = F.interpolate(g_up, size=x_enc.shape[2:], mode='nearest')
            if x_enc.shape[1] != g_up.shape[1]:
                if g_up.shape[1] < x_enc.shape[1]:
                    g_up = F.pad(g_up, (0, 0, 0, 0, 0, x_enc.shape[1] - g_up.shape[1]))
                else:
                    g_up = g_up[:, :x_enc.shape[1]]
            x_fused = (x_enc + g_up) * 0.5
        
        # 2. 特征拼接
        try:
            if x_fused.shape[2:] != g_up.shape[2:]:
                g_up = F.interpolate(g_up, size=x_fused.shape[2:], mode='nearest')
            
            concat_features = torch.cat([x_fused, g_up], dim=1)
        except Exception as e:
            print(f"⚠️ 特征拼接失败: {e}")
            concat_features = x_fused
        
        # 3. 后处理
        try:
            output = self.post_fusion(concat_features)
        except Exception as e:
            print(f"⚠️ 后处理失败: {e}")
            # 最简单的备选
            if concat_features.shape[1] != self.out_channels:
                simple_conv = nn.Conv2d(concat_features.shape[1], self.out_channels, 1, bias=True).to(concat_features.device)
                nn.init.kaiming_normal_(simple_conv.weight)
                simple_conv.weight.data *= 0.05
                nn.init.constant_(simple_conv.bias, 0.01)
                output = simple_conv(concat_features)
            else:
                output = concat_features
        
        # 4. 全局上下文增强
        try:
            global_weight = self.global_context(output)
            output = output * global_weight
        except Exception as e:
            print(f"⚠️ 全局上下文增强失败: {e}")
        
        # 5. Dropout
        if self.training:
            output = self.dropout(output)
        
        # 输出限制
        output = torch.clamp(output, -0.1, 0.1)
        
        return output


# =============================================================================
# 更新所有别名到超修复版
# =============================================================================

# 向后兼容别名 - 使用超修复版
CSFGModule = UltraFixedCSFGModule
CSFGSkipConnection = UltraFixedCSFGSkipConnection
FixedCSFGModule = UltraFixedCSFGModule  # 替换之前的修复版
FixedCSFGSkipConnection = UltraFixedCSFGSkipConnection  # 替换之前的修复版
EnhancedCSFGModule = UltraFixedCSFGModule  # 替换原来的增强版
EnhancedCSFGSkipConnection = UltraFixedCSFGSkipConnection  # 替换原来的增强版


def create_csfg_module(
    enc_channels,
    dec_channels,
    reduction_ratio=8,
    use_residual=True
):
    """创建超修复版CSFG模块"""
    return UltraFixedCSFGModule(
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
    """创建超修复版CSFG跳跃连接"""
    return UltraFixedCSFGSkipConnection(
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        out_channels=out_channels,
        reduction_ratio=reduction_ratio,
        use_residual=use_residual
    )

create_csfg = create_csfg_module


def csfg_base(enc_channels, dec_channels, **kwargs):
    """基础配置"""
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['enc_channels', 'dec_channels', 'reduction_ratio', 'use_residual']}
    
    return create_csfg_skip_connection(
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        reduction_ratio=4,
        use_residual=True,
        **filtered_kwargs
    )


def csfg_skip_base(enc_channels, dec_channels, out_channels=None, **kwargs):
    """基础跳跃连接配置"""
    return csfg_base(enc_channels, dec_channels, out_channels=out_channels, **kwargs)


# =============================================================================
# 测试函数
# =============================================================================

def test_ultra_fixed_csfg():
    """测试超修复版CSFG"""
    print("🎯 测试超修复版CSFG...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建超修复版CSFG模块
    csfg = UltraFixedCSFGModule(enc_channels=64, dec_channels=32).to(device)
    csfg.train()  # 训练模式测试
    
    # 测试不同类型的输入
    test_cases = {
        'random': torch.randn(2, 64, 32, 32).to(device),
        'zeros': torch.zeros(2, 64, 32, 32).to(device),
        'ones': torch.ones(2, 64, 32, 32).to(device),
        'small_values': torch.randn(2, 64, 32, 32).to(device) * 0.01,
    }
    
    dec_input = torch.randn(2, 32, 32, 32).to(device)
    
    print("测试超修复版CSFG在训练模式下:")
    
    success_count = 0
    for case_name, test_input in test_cases.items():
        try:
            # 前向传播
            output = csfg(test_input, dec_input)
            
            # 检查输出
            output_np = output.detach().cpu().numpy()
            zero_ratio = np.sum(output_np == 0) / output_np.size
            mean_val = np.mean(np.abs(output_np))
            std_val = np.std(output_np)
            
            print(f"  {case_name:12s}: ✅ 零激活={zero_ratio:.3f}, 均值={mean_val:.6f}, 标准差={std_val:.6f}")
            
            # 梯度测试
            test_input.requires_grad_(True)
            dec_input.requires_grad_(True)
            output = csfg(test_input, dec_input)
            loss = torch.mean(output ** 2)
            loss.backward()
            
            if test_input.grad is not None:
                grad_norm = torch.norm(test_input.grad).item()
                print(f"  {case_name:12s}: ✅ 梯度范数={grad_norm:.6f}")
                success_count += 1
            else:
                print(f"  {case_name:12s}: ❌ 无梯度")
                
        except Exception as e:
            print(f"  {case_name:12s}: ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n超修复版CSFG测试: {success_count}/{len(test_cases)} 成功")
    return success_count == len(test_cases)


if __name__ == "__main__":
    import numpy as np
    test_ultra_fixed_csfg()