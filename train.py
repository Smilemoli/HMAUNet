import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import logging
import argparse
from models.HMA_UNet import create_hma_unet
from data.dataset import create_train_val_dataloaders, save_split_info
from models.loss import CombinedLoss
from vision import TrainingVisualizer
import glob


class HMAUNetTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.current_epoch = 0

        # 创建保存目录 - 固定使用base配置
        self.exp_name = f"HMA_UNet_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = os.path.join(config.save_dir, self.exp_name)
        self.pred_dir = os.path.join(self.save_dir, "predictions")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)

        # 初始化数据加载器
        self.train_loader, self.val_loader = create_train_val_dataloaders(
            train_img_dir=config.train_img_dir,
            train_mask_dir=config.train_mask_dir,
            batch_size=config.batch_size,
            img_size=config.img_size,
            num_workers=config.num_workers,
            split_ratio=config.split_ratio,
            pin_memory=getattr(config, 'pin_memory', True)
        )
        
        # 显示数据集统计信息
        self._print_dataset_info()
        
        # 保存数据集划分信息
        self._save_dataset_split_info()

        # 初始化HMA-UNet模型 - 强制使用base配置
        print(f"🚀 初始化HMA-UNet模型 (配置: base)")
        self.model = create_hma_unet(
            config="base",  # 强制使用base
            in_channels=3,
            num_classes=1
        ).to(self.device)
        
        # 打印模型信息
        self._print_model_info()
        self._init_weights()

        # 使用组合损失
        self.criterion = CombinedLoss()

        # 优化器 - 使用base配置的学习率
        base_lr = 1e-4  # base配置的固定学习率
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=base_lr, 
            weight_decay=config.weight_decay, 
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        total_steps = len(self.train_loader) * config.epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=base_lr * 10,  # 峰值学习率
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4,
        )

        # 配置日志和可视化
        self._setup_logging()
        self.visualizer = TrainingVisualizer(self.save_dir)
        
        # 修复训练历史记录结构
        self.train_history = {
            'train_losses': [],
            'train_dices': [],  # 添加训练Dice记录
            'val_losses': [],
            'val_dices': [],
            'lr_history': [],
            'val_epochs': [],  # 记录进行验证的epoch
            'best_dice': 0.0,
            'best_epoch': 0
        }

    def _print_model_info(self):
        """打印模型信息 - 适配base配置"""
        model_info = self.model.get_model_info()
        print("=" * 60)
        print("HMA-UNet 模型信息 (Base配置)")
        print("=" * 60)
        print(f"模型名称: {model_info['model_name']}")
        print(f"模型配置: base (唯一可用配置)")
        print(f"基础通道数: {model_info['base_channels']}")
        print(f"输入通道数: {model_info['input_channels']}")
        print(f"输出类别数: {model_info['num_classes']}")
        print(f"总参数量: {model_info['total_params']:,}")
        print(f"可训练参数: {model_info['trainable_params']:,}")
        print("=" * 60)

    def _print_dataset_info(self):
        """打印数据集统计信息"""
        train_dataset_size = len(self.train_loader.dataset)
        val_dataset_size = len(self.val_loader.dataset)
        total_images = train_dataset_size + val_dataset_size
        
        print("=" * 60)
        print("数据集统计信息")
        print("=" * 60)
        print(f"训练集图片数量: {train_dataset_size:,}")
        print(f"验证集图片数量: {val_dataset_size:,}")
        print(f"总图片数量: {total_images:,}")
        print(f"训练/验证 比例: {train_dataset_size/total_images:.1%} / {val_dataset_size/total_images:.1%}")
        print(f"训练批次数量: {len(self.train_loader):,}")
        print(f"验证批次数量: {len(self.val_loader):,}")
        print(f"批次大小: {self.config.batch_size}")
        print(f"图像尺寸: {self.config.img_size}x{self.config.img_size}")
        print("=" * 60)

    def _save_dataset_split_info(self):
        """保存数据集划分信息"""
        try:
            # 获取训练集和验证集的文件列表
            train_files = self.train_loader.dataset.image_files
            val_files = self.val_loader.dataset.image_files
            split_info_path = os.path.join(self.save_dir, "dataset_split_info.txt")
            save_split_info(train_files, val_files, split_info_path)
        except Exception as e:
            print(f"保存数据集划分信息失败: {e}")

    def _setup_logging(self):
        """配置日志系统"""
        # 创建日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # 配置根日志记录器
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 清除现有的处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 文件处理器
        file_handler = logging.FileHandler(
            os.path.join(self.save_dir, "train.log"), 
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 记录训练开始信息
        logging.info(f"HMA-UNet模型训练开始 (配置: base)")
        logging.info(f"保存路径: {self.save_dir}")
        logging.info(f"目标epoch数: {self.config.epochs}")
        logging.info(f"训练设备: {self.device}")

    def _init_weights(self):
        """初始化模型权重 - HMA-UNet已有自己的初始化"""
        # HMA-UNet模型在__init__中已经调用了_initialize_weights()
        # 这里可以添加额外的初始化策略
        pass

    def save_predictions(self, images, masks, outputs, epoch, phase="train"):
        """保存预测结果可视化"""
        try:
            # 转换为numpy格式用于可视化
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()

            save_folder = os.path.join(self.pred_dir, f"epoch_{epoch}")
            os.makedirs(save_folder, exist_ok=True)

            for i in range(min(2, len(images_np))):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # 处理图像显示
                if images_np[i].shape[0] == 3:  # RGB图像
                    img_display = np.transpose(images_np[i], (1, 2, 0))
                    img_display = np.clip(img_display, 0, 1)
                else:
                    img_display = images_np[i, 0]

                axes[0].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                axes[1].imshow(masks_np[i, 0], cmap="gray")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(preds_np[i, 0], cmap="gray")
                axes[2].set_title("Prediction")
                axes[2].axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(save_folder, f"{phase}_{i}.png"), dpi=150, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logging.warning(f"保存预测结果失败: {e}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_dice = 0
        batch_count = 0
        dice_calculation_freq = 10  # 每10个batch计算一次dice

        with tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.config.epochs} [base]",
            leave=False,
            ncols=120
        ) as pbar:
            for batch_idx, batch in enumerate(pbar):
                # 处理数据解包
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device)
                else:
                    images, masks = batch
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                
                masks = (masks > 0.5).float()

                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.criterion(outputs, masks)
                loss.backward()

                # 梯度裁剪 - 针对修复后的模型使用更严格的裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                batch_count += 1

                # 减少Dice计算频率
                if batch_idx % dice_calculation_freq == 0:
                    with torch.no_grad():
                        dice_loss = self.criterion.dice(torch.sigmoid(outputs), masks)
                        dice_score = 1 - dice_loss.item()
                        epoch_dice += dice_score
                
                # 更新进度条
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "LR": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    "Config": "base"
                })

                # 保存预测结果（特定条件下）
                if (epoch % 10 == 0 or epoch == self.config.epochs - 1) and batch_idx == len(self.train_loader) - 1:
                    self.save_predictions(images, masks, outputs, epoch, "train")

        avg_loss = epoch_loss / batch_count
        avg_dice = epoch_dice / (batch_count // dice_calculation_freq + 1)
        return avg_loss, avg_dice

    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0
        val_dice = 0
        batch_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="验证中", leave=False)):
                # 处理数据解包
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device)
                else:
                    images, masks = batch
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                
                masks = (masks > 0.5).float()

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice_loss = self.criterion.dice(torch.sigmoid(outputs), masks)

                val_loss += loss.item()
                val_dice += 1 - dice_loss.item()
                batch_count += 1

                # 保存验证预测结果
                if (self.current_epoch % 10 == 0 or self.current_epoch == self.config.epochs - 1) and batch_idx == len(self.val_loader) - 1:
                    self.save_predictions(images, masks, outputs, self.current_epoch, "val")

        return val_loss / batch_count, val_dice / batch_count

    def train(self):
        """主训练循环 - 修复数据记录问题"""
        best_dice = 0
        best_epoch = 0
        validation_freq = getattr(self.config, 'validation_freq', 5)
        
        logging.info("开始训练循环...")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_loss, train_dice = self.train_epoch(epoch)
            
            # 记录学习率 - 每个epoch都记录
            current_lr = self.scheduler.get_last_lr()[0]
            self.train_history['lr_history'].append(current_lr)
            
            # 验证
            if epoch % validation_freq == 0 or epoch == self.config.epochs - 1:
                val_loss, val_dice = self.validate()
                
                # 更新历史记录 - 只在验证时记录
                self.train_history['train_losses'].append(train_loss)
                self.train_history['train_dices'].append(train_dice)
                self.train_history['val_losses'].append(val_loss)
                self.train_history['val_dices'].append(val_dice)
                self.train_history['val_epochs'].append(epoch + 1)  # 记录验证的epoch
                
                # 记录详细信息
                log_msg = (
                    f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
                    f"LR: {current_lr:.2e}"
                )
                logging.info(log_msg)
                
                # 保存最佳模型
                if val_dice > best_dice:
                    best_dice = val_dice
                    best_epoch = epoch + 1
                    self.train_history['best_dice'] = best_dice
                    self.train_history['best_epoch'] = best_epoch
                    
                    # 保存模型 - 固定使用base配置
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "model_info": self.model.get_model_info(),
                        "config": {
                            'model_config': 'base',  # 固定为base
                            'in_channels': 3,
                            'num_classes': 1
                        },
                        "val_dice": val_dice,
                        "train_dice": train_dice,
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                        "train_history": self.train_history
                    }
                    
                    model_path = os.path.join(self.save_dir, f"HMA_UNet_base_best_model.pth")
                    torch.save(checkpoint, model_path)
                    logging.info(f"保存最佳模型 (Epoch {epoch+1}, Val Dice: {val_dice:.4f})")
                
                # 保存训练曲线和预测样本 - 传入修复后的数据
                try:
                    # 创建用于可视化的数据结构
                    viz_data = {
                        'epochs': self.train_history['val_epochs'],  # 使用验证epoch
                        'train_losses': self.train_history['train_losses'],
                        'train_dices': self.train_history['train_dices'],
                        'val_losses': self.train_history['val_losses'],
                        'val_dices': self.train_history['val_dices'],
                        'lr_history': self.train_history['lr_history'][:len(self.train_history['val_epochs'])],  # 截取对应长度
                        'best_dice': self.train_history['best_dice'],
                        'best_epoch': self.train_history['best_epoch']
                    }
                    
                    self.visualizer.save_training_curves(viz_data)
                    
                    if hasattr(self.visualizer, 'save_prediction_samples'):
                        self.visualizer.save_prediction_samples(
                            self.model, self.val_loader, self.device, epoch
                        )
                except Exception as e:
                    logging.warning(f"保存可视化失败: {e}")
                    
            else:
                # 仅记录训练信息
                log_msg = (
                    f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
                    f"LR: {current_lr:.2e}"
                )
                logging.info(log_msg)
        
        # 训练完成处理
        self._finalize_training(best_dice, best_epoch)

    def _finalize_training(self, best_dice, best_epoch):
        """完成训练后的处理"""
        # 保存最终的训练摘要
        model_info = self.model.get_model_info()
        
        # 创建最终的可视化数据
        final_viz_data = {
            'epochs': self.train_history['val_epochs'],
            'train_losses': self.train_history['train_losses'],
            'train_dices': self.train_history['train_dices'],
            'val_losses': self.train_history['val_losses'],
            'val_dices': self.train_history['val_dices'],
            'lr_history': self.train_history['lr_history'][:len(self.train_history['val_epochs'])],
            'best_dice': self.train_history['best_dice'],
            'best_epoch': self.train_history['best_epoch']
        }
        
        try:
            self.visualizer.save_metrics_summary(final_viz_data, model_info)
        except Exception as e:
            logging.warning(f"保存训练摘要失败: {e}")
        
        # 创建预测对比和动画
        if hasattr(self.visualizer, 'create_prediction_animation'):
            try:
                self.visualizer.create_prediction_animation()
            except Exception as e:
                logging.warning(f"创建预测动画失败: {e}")
        
        # 训练完成总结
        logging.info("=" * 60)
        logging.info("训练完成!")
        logging.info(f"模型配置: base")
        logging.info(f"最佳验证Dice: {best_dice:.4f} (Epoch {best_epoch})")
        logging.info(f"总参数量: {model_info['total_params']:,}")
        logging.info(f"模型保存路径: {os.path.join(self.save_dir, f'HMA_UNet_base_best_model.pth')}")
        logging.info(f"训练日志: {os.path.join(self.save_dir, 'train.log')}")
        logging.info("=" * 60)


def parse_args():
    """解析命令行参数 - 简化版本"""
    parser = argparse.ArgumentParser(description='HMA-UNet 训练脚本 (仅支持base配置)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--img_size', type=int, default=512, help='图像尺寸')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='训练验证划分比例')
    parser.add_argument('--validation_freq', type=int, default=5, help='验证频率')
    
    # 数据路径
    parser.add_argument('--train_img_dir', type=str, default='./data/train/images/', 
                        help='训练图像目录')
    parser.add_argument('--train_mask_dir', type=str, default='./data/train/labels/', 
                        help='训练标签目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/HMA_UNet/', 
                        help='模型保存目录')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='auto', 
                        help='训练设备 (auto/cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作进程数')
    parser.add_argument('--pin_memory', action='store_true', default=True, 
                        help='使用固定内存')
    
    return parser.parse_args()


class Config:
    """配置类 - 简化版本"""
    def __init__(self, args):
        # 从命令行参数初始化
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        # 自动设备选择
        if self.device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 批次大小自动调整（针对可能的显存限制）
        if self.batch_size > 4:
            print(f"⚠️ 大批次大小可能导致显存不足，建议使用 <= 4")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    config = Config(args)
    
    # 打印配置信息
    print("=" * 60)
    print("HMA-UNet 训练配置 (仅支持base)")
    print("=" * 60)
    print(f"模型配置: base (唯一可用)")
    print(f"训练轮数: {config.epochs}")
    print(f"批次大小: {config.batch_size}")
    print(f"图像尺寸: {config.img_size}")
    print(f"设备: {config.device}")
    print("=" * 60)
    
    # 测试数据集功能
    print("🧪 测试数据集功能...")
    try:
        from data.dataset import test_dataset
        test_dataset()
    except Exception as e:
        print(f"数据集测试失败: {e}")
        print("继续进行训练...")
    
    # 确保保存目录存在
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 开始训练
    trainer = HMAUNetTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()