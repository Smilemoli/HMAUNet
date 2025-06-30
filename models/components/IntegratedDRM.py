import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Union, Dict, List
from ..HMA_UNet import HMAUNet


# ====================================集成扩散调度器=========================================
class IntegratedNoiseScheduler:
    """集成噪声调度器 - 针对端到端训练优化"""
    
    def __init__(self, timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.timesteps = timesteps
        
        # 余弦调度 - 更适合端到端训练
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(betas, 0.0001, 0.9999)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算去噪所需的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 后验分布方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def to(self, device):
        """移动到指定设备"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
    
    def add_noise(self, x_start, noise, timesteps):
        """添加噪声"""
        device = x_start.device
        timesteps = timesteps.to(device)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def denoise_step(self, model_output, timestep, sample):
        """单步去噪"""
        device = sample.device
        timestep = timestep.to(device)
        
        alpha_prod_t = self.alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        alpha_prod_t_prev = self.alphas_cumprod_prev[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # 预测原始样本
        pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        pred_original_sample = torch.clamp(pred_original_sample, 0, 1)
        
        # 计算前一步样本
        pred_sample_direction = torch.sqrt(beta_prod_t_prev) * model_output
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        prev_sample = torch.clamp(prev_sample, 0, 1)
        
        return prev_sample


# =============================================================================
# 集成网络组件
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        device = timestep.device
        half_dim = self.dim // 2
        
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        
        if emb.shape[-1] < self.dim:
            padding = self.dim - emb.shape[-1]
            emb = F.pad(emb, (0, padding))
        elif emb.shape[-1] > self.dim:
            emb = emb[:, :self.dim]
        
        return emb


class TimeConditionedConv(nn.Module):
    """时间条件卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, **kwargs):
        super().__init__()
        
        kernel_size = kwargs.get('kernel_size', 3)
        padding = kwargs.get('padding', 1)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm = nn.GroupNorm(min(8, max(1, out_channels//4)), out_channels)
        
    def forward(self, x, time_emb):
        h = self.conv(x)
        h = self.norm(h)
        
        time_cond = self.time_proj(time_emb)
        time_cond = time_cond[:, :, None, None]
        h = h + time_cond
        
        return F.silu(h)


class CrossScaleAttention(nn.Module):
    """跨尺度注意力 - 连接HMA-UNet特征与扩散特征"""
    
    def __init__(self, diffusion_dim: int, hma_dim: int):
        super().__init__()
        
        self.diffusion_dim = diffusion_dim
        self.hma_dim = hma_dim
        
        # 特征适配
        if hma_dim != diffusion_dim:
            self.feature_adapter = nn.Conv2d(hma_dim, diffusion_dim, kernel_size=1)
        else:
            self.feature_adapter = nn.Identity()
        
        # 跨模态注意力
        self.cross_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(diffusion_dim * 2, diffusion_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(diffusion_dim // 4, diffusion_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, diffusion_features, hma_features):
        """
        Args:
            diffusion_features: (B, C1, H, W) - 扩散模型特征
            hma_features: (B, C2, H', W') - HMA-UNet特征
        """
        B, C, H, W = diffusion_features.shape
        
        # 调整HMA特征的空间维度
        if hma_features.shape[2:] != (H, W):
            hma_features = F.interpolate(hma_features, size=(H, W), mode='bilinear', align_corners=False)
        
        # 通道适配
        hma_features = self.feature_adapter(hma_features)
        
        # 融合特征
        combined = torch.cat([diffusion_features, hma_features], dim=1)
        
        # 计算注意力权重
        attn_weights = self.cross_attn(combined)
        
        # 应用注意力
        enhanced_diffusion = diffusion_features * attn_weights
        enhanced_hma = hma_features * (1 - attn_weights)
        
        return enhanced_diffusion + enhanced_hma


# =============================================================================
# 集成扩散精炼网络
# =============================================================================

class IntegratedDiffusionUNet(nn.Module):
    """集成扩散U-Net - 与HMA-UNet协同工作"""
    
    def __init__(
        self,
        in_channels: int = 2,  # [noisy_mask, initial_mask]
        out_channels: int = 1,
        base_channels: int = 32,
        time_emb_dim: int = 128,
        hma_channels: List[int] = [64, 128, 256, 256]
    ):
        super().__init__()
        
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        
        # 时间嵌入
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        
        # 编码器
        encoder_channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        self.encoder = nn.ModuleList([
            TimeConditionedConv(in_channels, encoder_channels[0], time_emb_dim),
            TimeConditionedConv(encoder_channels[0], encoder_channels[1], time_emb_dim),
            TimeConditionedConv(encoder_channels[1], encoder_channels[2], time_emb_dim),
            TimeConditionedConv(encoder_channels[2], encoder_channels[3], time_emb_dim),
        ])
        
        # 下采样
        self.downsample = nn.ModuleList([
            nn.AvgPool2d(2) for _ in range(4)
        ])
        
        # 跨尺度注意力层 - 连接HMA-UNet
        self.cross_scale_attention = nn.ModuleList([
            CrossScaleAttention(encoder_channels[0], hma_channels[0]),
            CrossScaleAttention(encoder_channels[1], hma_channels[1]),
            CrossScaleAttention(encoder_channels[2], hma_channels[2]),
            CrossScaleAttention(encoder_channels[3], hma_channels[3]),
        ])
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            TimeConditionedConv(encoder_channels[3], encoder_channels[3], time_emb_dim),
            TimeConditionedConv(encoder_channels[3], encoder_channels[3], time_emb_dim),
        )
        
        # 解码器
        decoder_channels = [encoder_channels[2], encoder_channels[1], encoder_channels[0], encoder_channels[0]]
        self.decoder = nn.ModuleList([
            TimeConditionedConv(encoder_channels[3] + encoder_channels[3], decoder_channels[0], time_emb_dim),
            TimeConditionedConv(decoder_channels[0] + encoder_channels[2], decoder_channels[1], time_emb_dim),
            TimeConditionedConv(decoder_channels[1] + encoder_channels[1], decoder_channels[2], time_emb_dim),
            TimeConditionedConv(decoder_channels[2] + encoder_channels[0], decoder_channels[3], time_emb_dim),
        ])
        
        # 上采样
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) for _ in range(4)
        ])
        
        # 输出层
        self.output_conv = nn.Conv2d(decoder_channels[3], out_channels, kernel_size=1)
        
        print(f"🏗️ 集成扩散U-Net配置:")
        print(f"   编码器通道: {encoder_channels}")
        print(f"   解码器通道: {decoder_channels}")
        print(f"   HMA连接通道: {hma_channels}")
        
    def forward(self, x, time_emb, hma_features):
        """
        Args:
            x: (B, 2, H, W) - [noisy_mask, initial_mask]
            time_emb: (B, time_emb_dim)
            hma_features: Dict[str, torch.Tensor] - HMA-UNet特征
        """
        # 编码器路径
        encoder_features = []
        h = x
        
        hma_keys = ['encoder_stage1', 'encoder_stage2', 'encoder_stage3', 'encoder_stage4']
        
        for i, (enc_layer, down, cross_attn) in enumerate(zip(
            self.encoder, self.downsample, self.cross_scale_attention
        )):
            h = enc_layer(h, time_emb)
            
            # 跨尺度注意力融合HMA特征
            if hma_keys[i] in hma_features:
                h = cross_attn(h, hma_features[hma_keys[i]])
            
            encoder_features.append(h)
            h = down(h)
        
        # 瓶颈层
        for bottleneck_layer in self.bottleneck:
            h = bottleneck_layer(h, time_emb)
        
        # 解码器路径
        for i, (dec_layer, up) in enumerate(zip(self.decoder, self.upsample)):
            h = up(h)
            
            # 跳跃连接
            skip = encoder_features[-(i+1)]
            h = torch.cat([h, skip], dim=1)
            
            # 解码器卷积
            h = dec_layer(h, time_emb)
        
        # 输出
        output = self.output_conv(h)
        return output


# =============================================================================
# 主要集成模型
# =============================================================================

class IntegratedHMADRM(nn.Module):
    """集成HMA-UNet与DRM的端到端模型"""
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        timesteps: int = 1000,
        use_diffusion_training: bool = True,
        diffusion_probability: float = 0.5,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.use_diffusion_training = use_diffusion_training
        self.diffusion_probability = diffusion_probability
        self.time_emb_dim = base_channels * 4
        
        print(f"🔧 初始化集成HMA-DRM模型...")
        print(f"   目标设备: {device}")
        print(f"   扩散训练: {'启用' if use_diffusion_training else '禁用'}")
        print(f"   扩散概率: {diffusion_probability}")
        print(f"   时间步数: {timesteps}")
        
        # 1. HMA-UNet核心 (可训练) - 确保在正确设备上
        print("📦 创建HMA-UNet核心...")
        from ..HMA_UNet import create_hma_unet
        self.hma_unet = create_hma_unet(
            config="base",
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels
        )
        
        # 立即移动到目标设备
        self.hma_unet = self.hma_unet.to(self.device)
        print(f"✅ HMA-UNet已移动到设备: {next(self.hma_unet.parameters()).device}")
        
        # 2. 扩散调度器
        print("⏰ 创建扩散调度器...")
        self.noise_scheduler = IntegratedNoiseScheduler(timesteps=timesteps).to(self.device)
        
        # 3. 获取HMA-UNet通道配置 - 在模型移动到设备后
        print("🔍 检测HMA-UNet通道配置...")
        self.hma_channels = self._get_hma_channels()
        print(f"   HMA通道配置: {self.hma_channels}")
        
        # 4. 集成扩散精炼网络
        print("🌊 创建集成扩散精炼网络...")
        self.diffusion_unet = IntegratedDiffusionUNet(
            in_channels=2,  # [noisy_mask, initial_mask]
            out_channels=1,
            base_channels=base_channels,
            time_emb_dim=self.time_emb_dim,
            hma_channels=self.hma_channels
        )
        
        # 立即移动到目标设备
        self.diffusion_unet = self.diffusion_unet.to(self.device)
        print(f"✅ 扩散网络已移动到设备: {next(self.diffusion_unet.parameters()).device}")
        
        print("✅ 集成模型初始化完成")
        
    def _get_hma_channels(self) -> List[int]:
        """获取HMA-UNet的通道配置 - 修复设备同步"""
        # 确保模型在正确的设备上
        device = next(self.hma_unet.parameters()).device
        test_input = torch.randn(1, 3, 256, 256).to(device)
        
        print(f"🔍 通道检测 - 输入设备: {test_input.device}, 模型设备: {device}")
        
        with torch.no_grad():
            self.hma_unet.eval()
            try:
                encoder_features = self.hma_unet.encoder(test_input)
                x_enc1, x_enc2, x_enc3, x_enc4 = encoder_features
                
                channels = [
                    x_enc1.shape[1],  # encoder_stage1
                    x_enc2.shape[1],  # encoder_stage2
                    x_enc3.shape[1],  # encoder_stage3
                    x_enc4.shape[1],  # encoder_stage4
                ]
                
                print(f"✅ 通道检测成功: {channels}")
                print(f"   特征形状: enc1={x_enc1.shape}, enc2={x_enc2.shape}, enc3={x_enc3.shape}, enc4={x_enc4.shape}")
                
                return channels
                
            except Exception as e:
                print(f"❌ 通道检测失败: {e}")
                # 使用默认通道配置
                default_channels = [64, 128, 256, 256]  # base配置的默认通道
                print(f"⚠️ 使用默认通道配置: {default_channels}")
                return default_channels
    
    def extract_hma_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取HMA-UNet特征 - 确保设备一致性"""
        # 确保输入在正确的设备上
        images = images.to(next(self.hma_unet.parameters()).device)
        
        # 获取编码器特征
        encoder_features = self.hma_unet.encoder(images)
        x_enc1, x_enc2, x_enc3, x_enc4 = encoder_features
        
        # 获取瓶颈层特征
        bottleneck_features = self.hma_unet.bottleneck(x_enc4)
        
        return {
            'encoder_stage1': x_enc1,
            'encoder_stage2': x_enc2,
            'encoder_stage3': x_enc3,
            'encoder_stage4': x_enc4,
            'bottleneck': bottleneck_features
        }
    
    def get_initial_prediction(self, images: torch.Tensor) -> torch.Tensor:
        """获取HMA-UNet的初始预测 - 确保设备一致性"""
        images = images.to(next(self.hma_unet.parameters()).device)
        return torch.sigmoid(self.hma_unet(images))
    
    def forward(self, images: torch.Tensor, target_masks: torch.Tensor = None, mode: str = "train"):
        """
        前向传播
        
        Args:
            images: 输入图像 (B, 3, H, W)
            target_masks: 目标掩码 (B, 1, H, W) - 仅训练时需要
            mode: "train" 或 "inference"
        """
        # 确保输入在正确的设备上
        images = images.to(self.device)
        if target_masks is not None:
            target_masks = target_masks.to(self.device)
            
        if mode == "train":
            return self._forward_train(images, target_masks)
        else:
            return self._forward_inference(images)
    
    def _forward_train(self, images: torch.Tensor, target_masks: torch.Tensor):
        """训练时的前向传播"""
        device = images.device
        batch_size = images.shape[0]
        
        # 1. 提取HMA特征
        hma_features = self.extract_hma_features(images)
        
        # 2. 获取初始预测
        initial_prediction = self.get_initial_prediction(images)
        
        # 3. 决定是否使用扩散训练
        use_diffusion = self.use_diffusion_training and (torch.rand(1).item() < self.diffusion_probability)
        
        if use_diffusion:
            # 扩散训练路径
            # 随机采样时间步
            timesteps = torch.randint(0, self.timesteps // 2, (batch_size,), device=device, dtype=torch.long)
            
            # 生成噪声
            noise = torch.randn_like(target_masks)
            
            # 前向扩散
            noisy_masks = self.noise_scheduler.add_noise(target_masks, noise, timesteps)
            
            # 时间嵌入
            time_emb = self.diffusion_unet.time_embedding(timesteps.float())
            
            # 扩散网络输入
            diffusion_input = torch.cat([noisy_masks, initial_prediction], dim=1)
            
            # 预测噪声
            predicted_noise = self.diffusion_unet(diffusion_input, time_emb, hma_features)
            
            return {
                'initial_prediction': initial_prediction,
                'predicted_noise': predicted_noise,
                'target_noise': noise,
                'mode': 'diffusion'
            }
        else:
            # 直接分割训练路径
            return {
                'initial_prediction': initial_prediction,
                'mode': 'direct'
            }
    
    def _forward_inference(self, images: torch.Tensor, num_inference_steps: int = 50):
        """推理时的前向传播"""
        device = images.device
        batch_size = images.shape[0]
        
        # 1. 提取HMA特征
        hma_features = self.extract_hma_features(images)
        
        # 2. 获取初始预测
        initial_prediction = self.get_initial_prediction(images)
        
        if not self.use_diffusion_training:
            # 仅返回初始预测
            return initial_prediction
        
        # 3. 扩散精炼过程
        current_mask = initial_prediction + 0.05 * torch.randn_like(initial_prediction)
        current_mask = torch.clamp(current_mask, 0, 1)
        
        # 时间步序列
        max_timestep = self.timesteps // 4
        timesteps = torch.linspace(max_timestep-1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        # 逐步去噪
        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)
            time_emb = self.diffusion_unet.time_embedding(t_batch.float())
            
            # 扩散网络输入
            diffusion_input = torch.cat([current_mask, initial_prediction], dim=1)
            
            # 预测噪声
            with torch.no_grad():
                predicted_noise = self.diffusion_unet(diffusion_input, time_emb, hma_features)
            
            # 去噪步骤
            current_mask = self.noise_scheduler.denoise_step(predicted_noise, t_batch, current_mask)
            current_mask = torch.clamp(current_mask, 0, 1)
            
            # 渐进式混合
            alpha = 0.8 + 0.2 * (i / len(timesteps))
            current_mask = alpha * current_mask + (1 - alpha) * initial_prediction
        
        return current_mask
    
    def compute_loss(self, images: torch.Tensor, target_masks: torch.Tensor):
        """计算训练损失"""
        outputs = self.forward(images, target_masks, mode="train")
        
        if outputs['mode'] == 'diffusion':
            # 扩散损失
            noise_loss = F.mse_loss(outputs['predicted_noise'], outputs['target_noise'])
            initial_loss = F.binary_cross_entropy(outputs['initial_prediction'], target_masks)
            
            total_loss = noise_loss + 0.1 * initial_loss
            
            return {
                'total_loss': total_loss,
                'noise_loss': noise_loss,
                'initial_loss': initial_loss,
                'mode': 'diffusion'
            }
        else:
            # 直接分割损失
            initial_loss = F.binary_cross_entropy(outputs['initial_prediction'], target_masks)
            
            return {
                'total_loss': initial_loss,
                'initial_loss': initial_loss,
                'mode': 'direct'
            }
    
    def to(self, device):
        """重写to方法确保所有组件都移动到正确设备"""
        super().to(device)
        self.device = torch.device(device)
        
        # 移动所有子模块
        self.hma_unet = self.hma_unet.to(device)
        self.diffusion_unet = self.diffusion_unet.to(device)
        self.noise_scheduler = self.noise_scheduler.to(device)
        
        print(f"📱 模型已完全移动到设备: {device}")
        return self


# =============================================================================
# 工厂函数
# =============================================================================

def create_integrated_hma_drm(
    in_channels: int = 3,
    num_classes: int = 1,
    base_channels: int = 32,
    timesteps: int = 1000,
    use_diffusion_training: bool = True,
    diffusion_probability: float = 0.5,
    device: str = "cuda",
    **kwargs
) -> IntegratedHMADRM:
    """创建集成HMA-DRM模型的工厂函数"""
    
    print(f"🏭 创建集成HMA-DRM模型...")
    print(f"   目标设备: {device}")
    
    model = IntegratedHMADRM(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        timesteps=timesteps,
        use_diffusion_training=use_diffusion_training,
        diffusion_probability=diffusion_probability,
        device=device
    )
    
    # 确保模型在正确的设备上
    model = model.to(device)
    
    print(f"✅ 集成HMA-DRM模型创建完成，设备: {next(model.parameters()).device}")
    
    return model