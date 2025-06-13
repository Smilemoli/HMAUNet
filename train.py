import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from tqdm import tqdm
import logging
import warnings
import sys
import time
from pathlib import Path

# 🔥 完整的字体配置，解决减号显示问题
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial', 'Liberation Sans']
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['figure.max_open_warning'] = 0  # 关闭图像过多警告

# 🔥 设置matplotlib后端，避免GUI相关警告
matplotlib.use('Agg')  # 使用非交互式后端

# 🔥 额外的字体设置
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 🔥 更全面的警告过滤
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*does not have a glyph.*')
warnings.filterwarnings('ignore', message='.*Substituting symbol.*')

# 将导入检查移到函数内部，避免模块级别的重复打印
def _import_modules():
    """导入HMA-UNet相关模块"""
    try:
        from models.HMA_UNet import HMAUNet, create_hma_unet
        from models.loss import HMAUNetLoss, get_default_loss, get_boundary_focused_loss, get_lightweight_loss
        from data.dataset import create_dataloaders
        return {
            'HMAUNet': HMAUNet,
            'create_hma_unet': create_hma_unet,
            'HMAUNetLoss': HMAUNetLoss,
            'get_default_loss': get_default_loss,
            'get_boundary_focused_loss': get_boundary_focused_loss,
            'get_lightweight_loss': get_lightweight_loss,
            'create_dataloaders': create_dataloaders
        }
    except ImportError as e:
        raise ImportError(f"导入HMA-UNet模块失败: {e}")


class HMAUNetTrainer:
    """
    HMA-UNet训练器
    
    专门为HMA-UNet设计的训练流程，包含：
    1. 模型初始化和配置
    2. 数据加载和预处理（仅训练/验证集）
    3. 训练和验证循环
    4. 模型保存和可视化
    5. 指标计算和记录
    """
    
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 配置类，包含所有训练参数
        """
        self.config = config
        self.device = torch.device(config.device)
        self.current_epoch = 0
        
        # 在这里导入模块，避免重复
        self.modules = _import_modules()
        
        # 创建保存目录
        self.exp_name = f"HMAUNet_{config.model_config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = Path(config.save_dir) / self.exp_name
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.pred_dir = self.save_dir / "predictions"
        self.log_dir = self.save_dir / "logs"
        
        # 创建所有必要的目录
        for dir_path in [self.save_dir, self.checkpoint_dir, self.pred_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 实验名称: {self.exp_name}")
        print(f"📂 保存目录: {self.save_dir}")
        
        # 配置日志
        self._setup_logging()
        
        # 记录配置信息
        self._log_config()
        
        # 初始化数据加载器（仅训练/验证集）
        self._setup_dataloaders()
        
        # 初始化模型
        self._setup_model()
        
        # 初始化损失函数
        self._setup_loss()
        
        # 初始化优化器和调度器
        self._setup_optimizer()
        
        # 初始化指标记录
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []
        self.train_ious = []
        self.val_ious = []
        self.learning_rates = []
        
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.patience_counter = 0

    def _setup_logging(self):
        """配置日志系统"""
        log_file = self.log_dir / "train.log"
        
        # 清除现有的处理器
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 🔥 配置日志过滤器，过滤matplotlib字体警告
        class FontWarningFilter(logging.Filter):
            def filter(self, record):
                # 过滤字体相关的警告
                message = record.getMessage()
                font_warnings = [
                    'does not have a glyph',
                    'Substituting symbol',
                    'Font \'default\'',
                    'STIXGeneral'
                ]
                return not any(warning in message for warning in font_warnings)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # 给所有日志处理器添加过滤器
        font_filter = FontWarningFilter()
        for handler in logging.root.handlers:
            handler.addFilter(font_filter)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 80)
        self.logger.info("🚀 HMA-UNet 训练开始")
        self.logger.info("=" * 80)

    def _log_config(self):
        """记录配置信息"""
        self.logger.info("📋 训练配置:")
        config_dict = vars(self.config)
        for key, value in config_dict.items():
            self.logger.info(f"   {key}: {value}")

    def _setup_dataloaders(self):
        """初始化数据加载器（仅训练和验证集）"""
        self.logger.info("📊 初始化训练和验证数据加载器...")
        
        try:
            # 检查训练数据目录是否存在
            if not os.path.exists(self.config.train_img_dir):
                raise FileNotFoundError(f"训练图像目录不存在: {self.config.train_img_dir}")
            if not os.path.exists(self.config.train_mask_dir):
                self.logger.warning(f"训练标签目录不存在: {self.config.train_mask_dir}")
                self.logger.info("将自动创建空标签文件...")
            
            # 🔥 使用已导入的模块
            create_dataloaders = self.modules['create_dataloaders']
            self.train_loader, self.val_loader = create_dataloaders(
                img_dir=self.config.train_img_dir,
                mask_dir=self.config.train_mask_dir,
                batch_size=self.config.batch_size,
                image_size=self.config.image_size,
                num_workers=self.config.num_workers,
                use_fixed_split=True,
                split_ratio=self.config.split_ratio
            )
            
            self.logger.info(f"   ✅ 训练集批次数: {len(self.train_loader)}")
            self.logger.info(f"   ✅ 验证集批次数: {len(self.val_loader)}")
            self.logger.info(f"   ✅ 训练样本数: {len(self.train_loader.dataset)}")
            self.logger.info(f"   ✅ 验证样本数: {len(self.val_loader.dataset)}")
            
        except Exception as e:
            self.logger.error(f"❌ 数据加载器初始化失败: {e}")
            raise

    def _setup_model(self):
        """初始化模型"""
        self.logger.info("🏗️  初始化HMA-UNet模型...")
        
        try:
            # 🔥 使用已导入的模块
            create_hma_unet = self.modules['create_hma_unet']
            self.model = create_hma_unet(
                config=self.config.model_config,
                in_channels=self.config.in_channels,
                num_classes=self.config.num_classes
            ).to(self.device)
            
            # 获取模型信息
            try:
                model_info = self.model.get_model_info()
                self.logger.info(f"   ✅ 模型: {model_info['model_name']}")
                self.logger.info(f"   ✅ 配置: {self.config.model_config}")
                self.logger.info(f"   ✅ 总参数量: {model_info['total_params']:,}")
                self.logger.info(f"   ✅ 可训练参数: {model_info['trainable_params']:,}")
                self.logger.info(f"   ✅ 基础通道数: {model_info.get('base_channels', 'N/A')}")
            except:
                # 如果模型没有 get_model_info 方法，手动计算参数
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                self.logger.info(f"   ✅ 模型: HMA-UNet")
                self.logger.info(f"   ✅ 配置: {self.config.model_config}")
                self.logger.info(f"   ✅ 总参数量: {total_params:,}")
                self.logger.info(f"   ✅ 可训练参数: {trainable_params:,}")
            
            # 初始化权重
            if self.config.init_weights:
                self._init_weights()
                self.logger.info("   ✅ 权重初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _setup_loss(self):
        """初始化损失函数"""
        self.logger.info("📐 初始化损失函数...")
        
        loss_type = getattr(self.config, 'loss_type', 'default')
        
        try:
            # 🔥 使用已导入的模块
            if loss_type == 'default':
                self.criterion = self.modules['get_default_loss']()
            elif loss_type == 'boundary_focused':
                self.criterion = self.modules['get_boundary_focused_loss']()
            elif loss_type == 'lightweight':
                self.criterion = self.modules['get_lightweight_loss']()
            else:
                # 使用默认配置
                HMAUNetLoss = self.modules['HMAUNetLoss']
                self.criterion = HMAUNetLoss(
                    focal_weight=getattr(self.config, 'focal_weight', 0.3),
                    dice_weight=getattr(self.config, 'dice_weight', 0.3),
                    boundary_weight=getattr(self.config, 'boundary_weight', 0.2),
                    iou_weight=getattr(self.config, 'iou_weight', 0.2),
                    adaptive_weights=getattr(self.config, 'adaptive_weights', True)
                )
            
            self.criterion = self.criterion.to(self.device)
            self.logger.info(f"   ✅ 损失函数类型: {type(self.criterion).__name__}")
            
            # 显示损失权重
            if hasattr(self.criterion, 'get_weights'):
                weights = self.criterion.get_weights()
                self.logger.info("   ✅ 损失权重:")
                for name, weight in weights.items():
                    self.logger.info(f"      {name}: {weight:.3f}")
                    
        except Exception as e:
            self.logger.error(f"❌ 损失函数初始化失败: {e}")
            raise

    def _setup_optimizer(self):
        """初始化优化器和学习率调度器"""
        self.logger.info("⚙️  初始化优化器和调度器...")
        
        # 优化器
        if self.config.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas
            )
        elif self.config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")
        
        # 学习率调度器
        scheduler_type = getattr(self.config, 'scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif scheduler_type == 'onecycle':
            total_steps = len(self.train_loader) * self.config.epochs
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate * 10,
                total_steps=total_steps,
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1000
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"   ✅ 优化器: {type(self.optimizer).__name__}")
        self.logger.info(f"   ✅ 学习率: {self.config.learning_rate}")
        self.logger.info(f"   ✅ 权重衰减: {self.config.weight_decay}")
        if self.scheduler:
            self.logger.info(f"   ✅ 调度器: {type(self.scheduler).__name__}")

    def calculate_metrics(self, pred, target, threshold=0.5):
        """
        计算分割指标
        
        Args:
            pred: 预测概率 (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
            threshold: 二值化阈值
            
        Returns:
            dict: 包含各种指标的字典
        """
        # 二值化预测
        pred_binary = (pred > threshold).float()
        target_binary = (target > 0.5).float()
        
        # 计算基础指标
        tp = (pred_binary * target_binary).sum()
        fp = (pred_binary * (1 - target_binary)).sum()
        fn = ((1 - pred_binary) * target_binary).sum()
        tn = ((1 - pred_binary) * (1 - target_binary)).sum()
        
        # 避免除零
        eps = 1e-8
        
        # Dice系数
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        
        # IoU (Jaccard系数)
        iou = (tp + eps) / (tp + fp + fn + eps)
        
        # 精确率和召回率
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        
        # F1分数
        f1 = 2 * precision * recall / (precision + recall + eps)
        
        # 准确率
        accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)
        
        return {
            'dice': dice.item(),
            'iou': iou.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'accuracy': accuracy.item()
        }

    def save_predictions(self, images, masks, outputs, epoch, phase="train", max_samples=4):
        """保存预测结果可视化 - 修复字体问题版本"""
        try:
            # 创建保存目录
            save_folder = self.pred_dir / f"epoch_{epoch:03d}"
            save_folder.mkdir(exist_ok=True)
            
            # 修复Tensor梯度问题 - 使用detach()
            images = images.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            # 对于outputs，先sigmoid再detach
            with torch.no_grad():
                preds = torch.sigmoid(outputs).detach().cpu().numpy()
            preds_binary = (preds > 0.5).astype(np.float32)
            
            # 限制保存数量
            num_samples = min(max_samples, len(images))
            
            for i in range(num_samples):
                # 🔥 关闭matplotlib警告并设置字体
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    
                    # 原图
                    img = images[i].transpose(1, 2, 0)
                    if img.shape[2] == 3:  # RGB图像
                        # 反归一化显示
                        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        img = np.clip(img, 0, 1)
                    else:  # 灰度图像
                        img = img.squeeze()
                    
                    axes[0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
                    axes[0].axis("off")
                    
                    # 真实标签
                    axes[1].imshow(masks[i, 0], cmap="gray", vmin=0, vmax=1)
                    axes[1].set_title("Ground Truth", fontsize=12, fontweight='bold')
                    axes[1].axis("off")
                    
                    # 预测概率
                    axes[2].imshow(preds[i, 0], cmap="hot", vmin=0, vmax=1)
                    axes[2].set_title("Prediction Probability", fontsize=12, fontweight='bold')
                    axes[2].axis("off")
                    
                    # 二值化预测
                    axes[3].imshow(preds_binary[i, 0], cmap="gray", vmin=0, vmax=1)
                    axes[3].set_title("Binary Prediction", fontsize=12, fontweight='bold')
                    axes[3].axis("off")
                    
                    # 添加整体标题
                    fig.suptitle(f'Epoch {epoch+1} - {phase.capitalize()} Sample {i+1}', 
                               fontsize=14, fontweight='bold', y=0.95)
                    
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.85)  # 为总标题留出空间
                    
                    # 保存图片
                    save_path = save_folder / f"{phase}_sample_{i:02d}.png"
                    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
                    plt.close()  # 重要：关闭图形释放内存
                
        except Exception as e:
            self.logger.warning(f"保存预测结果时出错: {e}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = []
        
        # 创建进度条
        pbar = tqdm(
            self.train_loader,
            desc=f"训练 Epoch {epoch+1}/{self.config.epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            # 确保标签为浮点数且在[0,1]范围
            masks = (masks > 0.5).float()
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            if hasattr(self.criterion, 'forward') and 'total_loss' in str(self.criterion.forward.__code__.co_varnames):
                loss_dict = self.criterion(outputs, masks)
                loss = loss_dict['total_loss']
            else:
                loss = self.criterion(outputs, masks)
                loss_dict = {'total_loss': loss}
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # 更新参数
            self.optimizer.step()
            
            # 更新学习率（如果使用OneCycleLR）
            if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # 计算指标
            with torch.no_grad():
                pred_probs = torch.sigmoid(outputs)
                metrics = self.calculate_metrics(pred_probs, masks)
            
            epoch_losses.append(loss.item())
            epoch_metrics.append(metrics)
            
            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{metrics['dice']:.4f}",
                'IoU': f"{metrics['iou']:.4f}",
                'LR': f"{current_lr:.6f}"
            })
            
            # 保存第一个batch的预测结果
            if batch_idx == 0:
                self.save_predictions(images, masks, outputs, epoch, "train")
        
        # 计算平均指标
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {key: np.mean([m[key] for m in epoch_metrics]) for key in epoch_metrics[0].keys()}
        
        return avg_loss, avg_metrics

    def validate(self):
        """验证模型"""
        self.model.eval()
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="验证中", leave=False)
            
            for batch_idx, batch in enumerate(pbar):
                # 数据移到设备
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                # 确保标签为浮点数且在[0,1]范围
                masks = (masks > 0.5).float()
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                if hasattr(self.criterion, 'forward') and 'total_loss' in str(self.criterion.forward.__code__.co_varnames):
                    loss_dict = self.criterion(outputs, masks)
                    loss = loss_dict['total_loss']
                else:
                    loss = self.criterion(outputs, masks)
                
                # 计算指标
                pred_probs = torch.sigmoid(outputs)
                metrics = self.calculate_metrics(pred_probs, masks)
                
                val_losses.append(loss.item())
                val_metrics.append(metrics)
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Dice': f"{metrics['dice']:.4f}",
                    'IoU': f"{metrics['iou']:.4f}"
                })
                
                # 保存第一个batch的预测结果
                if batch_idx == 0:
                    self.save_predictions(images, masks, outputs, self.current_epoch, "val")
        
        # 计算平均指标
        avg_loss = np.mean(val_losses)
        avg_metrics = {key: np.mean([m[key] for m in val_metrics]) for key in val_metrics[0].keys()}
        
        return avg_loss, avg_metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_dice': self.best_dice,
            'best_iou': self.best_iou
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 保存最佳模型: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % self.config.save_freq == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch+1:03d}.pth"
            torch.save(checkpoint, epoch_path)

    def plot_training_curves(self):
        """绘制训练曲线 - 修复字体问题版本"""
        try:
            # 🔥 关闭matplotlib警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                epochs = range(1, len(self.train_losses) + 1)
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                # 损失曲线
                ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
                ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
                ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Dice曲线
                ax2.plot(epochs, self.train_dices, 'b-', label='Training Dice', linewidth=2)
                ax2.plot(epochs, self.val_dices, 'r-', label='Validation Dice', linewidth=2)
                ax2.set_title('Dice Score Curves', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Dice Score')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 学习率曲线
                ax3.plot(epochs, self.learning_rates, 'g-', linewidth=2)
                ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Learning Rate')
                # 🔥 改为线性坐标轴，避免科学计数法中的减号问题
                ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
                ax3.grid(True, alpha=0.3)
                
                # IoU曲线
                ax4.plot(epochs, self.train_ious, 'b-', label='Training IoU', linewidth=2)
                ax4.plot(epochs, self.val_ious, 'r-', label='Validation IoU', linewidth=2)
                ax4.set_title('IoU Score Curves', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('IoU Score')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                # 添加整体标题
                fig.suptitle(f'HMA-UNet Training Progress - Epoch {len(epochs)}', 
                            fontsize=16, fontweight='bold', y=0.98)
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.93)  # 为总标题留出空间
                
                # 保存图片
                plt.savefig(self.save_dir / "training_curves.png", dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()  # 重要：关闭图形释放内存
                
        except Exception as e:
            self.logger.warning(f"绘制训练曲线时出错: {e}")

    def train(self):
        """主训练循环"""
        self.logger.info("🎯 开始训练...")
        start_time = time.time()
        
        try:
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # 训练
                train_loss, train_metrics = self.train_epoch(epoch)
                
                # 验证
                val_loss, val_metrics = self.validate()
                
                # 记录指标
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_dices.append(train_metrics['dice'])
                self.val_dices.append(val_metrics['dice'])
                self.train_ious.append(train_metrics['iou'])
                self.val_ious.append(val_metrics['iou'])
                self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
                
                # 更新学习率（非OneCycleLR）
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['dice'])
                    elif not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                        self.scheduler.step()
                
                # 检查是否为最佳模型
                is_best = False
                if val_metrics['dice'] > self.best_dice:
                    self.best_dice = val_metrics['dice']
                    self.best_iou = val_metrics['iou']
                    is_best = True
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # 计算epoch时间
                epoch_time = time.time() - epoch_start_time
                
                # 记录日志
                self.logger.info(
                    f"Epoch {epoch+1:03d}/{self.config.epochs} | "
                    f"Time: {epoch_time:.1f}s | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Train Dice: {train_metrics['dice']:.4f} | "
                    f"Val Dice: {val_metrics['dice']:.4f} | "
                    f"Train IoU: {train_metrics['iou']:.4f} | "
                    f"Val IoU: {val_metrics['iou']:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
                
                if is_best:
                    self.logger.info(f"🎉 新的最佳模型! Dice: {self.best_dice:.4f}, IoU: {self.best_iou:.4f}")
                
                # 保存检查点
                self.save_checkpoint(epoch, val_metrics, is_best)
                
                # 早停检查
                if hasattr(self.config, 'early_stopping_patience'):
                    if self.patience_counter >= self.config.early_stopping_patience:
                        self.logger.info(f"⏹️  早停触发，连续 {self.patience_counter} 个epoch无改善")
                        break
                
                # 定期绘制训练曲线
                if (epoch + 1) % 10 == 0:
                    self.plot_training_curves()
        
        except KeyboardInterrupt:
            self.logger.info("⚠️  训练被用户中断")
        except Exception as e:
            self.logger.error(f"❌ 训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 最终绘制训练曲线
            self.plot_training_curves()
            
            # 计算总训练时间
            total_time = time.time() - start_time
            self.logger.info("=" * 80)
            self.logger.info(f"🏁 训练完成!")
            self.logger.info(f"⏱️  总训练时间: {total_time/3600:.2f} 小时")
            self.logger.info(f"🥇 最佳Dice: {self.best_dice:.4f}")
            self.logger.info(f"🥇 最佳IoU: {self.best_iou:.4f}")
            self.logger.info(f"📂 模型保存在: {self.checkpoint_dir}")
            self.logger.info("=" * 80)


class TrainingConfig:
    """训练配置类"""
    
    def __init__(self):
        # 基础配置
        self.model_config = "tiny"  # 模型配置 ['tiny', 'small', 'base']
        self.in_channels = 3
        self.num_classes = 1
        
        # 数据配置
        self.train_img_dir = "data/train/images"
        self.train_mask_dir = "data/train/labels"
        
        # 训练配置
        self.batch_size = 8
        self.epochs = 150
        self.image_size = (256, 256)
        self.split_ratio = 0.8  # 训练/验证划分比例
        self.num_workers = 4
        
        # 优化器配置
        self.optimizer = "AdamW"
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.betas = (0.9, 0.999)
        self.scheduler = "cosine"  # ['cosine', 'onecycle', 'plateau', None]
        
        # 损失函数配置
        self.loss_type = "default"  # ['default', 'boundary_focused', 'lightweight']
        self.focal_weight = 0.3
        self.dice_weight = 0.3
        self.boundary_weight = 0.2
        self.iou_weight = 0.2
        self.adaptive_weights = True
        
        # 其他配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = "checkpoints"
        self.save_freq = 10  # 每多少个epoch保存一次模型
        self.init_weights = True
        self.grad_clip = 1.0  # 梯度裁剪
        self.early_stopping_patience = 30  # 早停耐心值


def test_imports():
    """测试导入是否正常"""
    print("🧪 测试模块导入...")
    
    try:
        # 🔥 只测试一次，避免重复打印
        modules = _import_modules()
        
        # 测试模型创建
        try:
            test_model = modules['create_hma_unet'](config="tiny", in_channels=3, num_classes=1)
            print("✅ HMA-UNet模型创建成功")
            print(f"   模型类型: {type(test_model).__name__}")
            
            # 测试前向传播
            with torch.no_grad():
                test_input = torch.randn(1, 3, 256, 256)
                test_output = test_model(test_input)
                print(f"   测试输入: {test_input.shape}")
                print(f"   测试输出: {test_output.shape}")
            
        except Exception as e:
            print(f"⚠️ 模型测试失败: {e}")
        
        print("🎉 所有模块导入测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🚀 HMA-UNet 训练程序启动")
    print("=" * 80)
    
    # 首先测试导入
    if not test_imports():
        print("❌ 模块导入失败，请检查文件路径和依赖")
        return
    
    print("=" * 80)
    
    # 创建配置
    config = TrainingConfig()
    
    # 显示配置信息
    print("📋 数据集使用说明:")
    print(f"   🏃 训练/验证: {config.train_img_dir} (按 {config.split_ratio:.0%} 比例划分)")
    print(f"   ✅ 仅使用训练数据，确保训练过程的纯净性")
    
    print(f"\n📋 当前配置:")
    print(f"   模型配置: {config.model_config}")
    print(f"   批次大小: {config.batch_size}")
    print(f"   训练轮数: {config.epochs}")
    print(f"   图像尺寸: {config.image_size}")
    print(f"   设备: {config.device}")
    
    try:
        # 创建训练器
        trainer = HMAUNetTrainer(config)
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()