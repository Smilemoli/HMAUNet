#!/usr/bin/env python3
"""
HMA-UNet 训练可视化模块 - 修复字体版本
包含所有训练过程中的可视化功能
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import torch
import torch.nn.functional as F
from datetime import datetime
import logging

# 设置matplotlib - 修复字体问题
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 使用系统中实际存在的字体
import matplotlib.font_manager as fm
available_fonts = [f.name for f in fm.fontManager.ttflist]

# 按优先级选择可用字体
preferred_fonts = ['DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
selected_font = 'sans-serif'  # 默认字体

for font in preferred_fonts:
    if font in available_fonts or font == 'sans-serif':
        selected_font = font
        break

plt.rcParams['font.family'] = [selected_font]
plt.rcParams['axes.unicode_minus'] = False

# 设置警告过滤
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='findfont: Font family*')

print(f"Using font: {selected_font}")


def safe_text_for_plot(text: str) -> str:
    """将包含emoji的文本转换为matplotlib安全的文本"""
    emoji_map = {
        '📊': 'Stats',
        '🏆': 'Best',
        '📈': 'Up',
        '📉': 'Down', 
        '⏰': 'Time',
        '🎯': 'Target',
        '📚': 'LR',
        '🖼️': 'Images',
        '✅': '[OK]',
        '❌': '[ERR]',
        '⚠️': '[WARN]',
        '🎨': 'Visual',
        '🔍': 'Compare',
        '🎬': 'Animation',
        '🐧': 'Linux',
        '🚀': 'Start',
        '🏁': 'End',
        '🛑': 'Stop',
        '⏹️': 'Stop',
        '💾': 'Save'
    }
    
    result = text
    for emoji, replacement in emoji_map.items():
        result = result.replace(emoji, replacement)
    
    return result


def safe_unpack_batch_vision(batch_data):
    """安全解包批次数据 - vision模块专用版本"""
    if isinstance(batch_data, dict):
        # 处理字典格式的批次数据
        if 'image' in batch_data and 'mask' in batch_data:
            return batch_data['image'], batch_data['mask']
        else:
            available_keys = list(batch_data.keys())
            raise ValueError(f"字典中缺少必要的键 'image' 或 'mask'。可用键: {available_keys}")
    elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
        return batch_data[0], batch_data[1]
    else:
        raise ValueError(f"不支持的批次数据格式: {type(batch_data)}，期望dict、list或tuple")


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, experiment_dir: str, logger: Optional[logging.Logger] = None):
        """
        初始化可视化器
        
        Args:
            experiment_dir: 实验目录路径
            logger: 日志记录器
        """
        self.experiment_dir = Path(experiment_dir)
        self.logger = logger if logger else logging.getLogger(__name__)
        
        # 创建可视化目录结构
        self.vis_dirs = {
            'curves': self.experiment_dir / 'visualizations' / 'curves',
            'predictions': self.experiment_dir / 'visualizations' / 'predictions',
            'comparisons': self.experiment_dir / 'visualizations' / 'comparisons',
            'animations': self.experiment_dir / 'visualizations' / 'animations',
            'metrics': self.experiment_dir / 'visualizations' / 'metrics'
        }
        
        # 创建所有目录
        for vis_dir in self.vis_dirs.values():
            vis_dir.mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"[Visual] 可视化器初始化完成: {self.experiment_dir}")
        self.logger.info(f"[Visual] 使用字体: {selected_font}")
    
    def save_training_curves(self, metrics_data: Dict[str, List]) -> None:
        """
        保存训练曲线
        
        Args:
            metrics_data: 包含训练指标的字典
        """
        try:
            if not metrics_data.get('train_losses'):
                self.logger.warning("[WARN] 没有训练数据，跳过曲线保存")
                return
            
            self.logger.info("[Visual] 开始保存训练曲线...")
            
            epochs = metrics_data.get('epochs', list(range(1, len(metrics_data['train_losses']) + 1)))
            
            # 创建综合训练曲线图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Training Progress Dashboard', fontsize=16, fontweight='bold')
            
            # 1. 损失曲线
            ax1 = axes[0, 0]
            ax1.plot(epochs, metrics_data['train_losses'], 'b-', label='Train Loss', linewidth=2)
            if metrics_data.get('val_losses'):
                ax1.plot(epochs, metrics_data['val_losses'], 'r-', label='Val Loss', linewidth=2)
            ax1.set_title('Training and Validation Loss', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Dice分数曲线
            ax2 = axes[0, 1]
            if metrics_data.get('val_dices'):
                ax2.plot(epochs, metrics_data['val_dices'], 'g-', label='Val Dice', linewidth=2)
                best_dice = max(metrics_data['val_dices'])
                best_epoch = metrics_data['val_dices'].index(best_dice) + 1
                ax2.axhline(y=best_dice, color='r', linestyle='--', 
                           label=f'Best: {best_dice:.4f} (Epoch {best_epoch})', linewidth=2)
                ax2.scatter([best_epoch], [best_dice], color='red', s=100, zorder=5)
            ax2.set_title('Validation Dice Score', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Dice Score')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 学习率曲线
            ax3 = axes[1, 0]
            if metrics_data.get('lr_history'):
                ax3.plot(epochs, metrics_data['lr_history'], 'orange', label='Learning Rate', linewidth=2)
                ax3.set_yscale('log')
            ax3.set_title('Learning Rate Schedule', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 训练摘要 - 使用安全文本
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # 创建训练摘要文本 - 移除emoji
            summary_text = self._create_training_summary_safe(metrics_data)
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            plt.tight_layout()
            
            # 保存图像
            curves_path = self.vis_dirs['curves'] / 'training_curves.png'
            plt.savefig(curves_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 保存独立的损失曲线
            self._save_individual_curves(metrics_data)
            
            self.logger.info(f"[OK] 训练曲线已保存: {curves_path}")
            
        except Exception as e:
            self.logger.error(f"[ERR] 保存训练曲线失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
    
    def save_prediction_samples(self, model, data_loader, device, epoch: int, 
                               num_samples: int = 8) -> None:
        """
        保存预测样本可视化 - 修复版本
        
        Args:
            model: 训练的模型
            data_loader: 验证数据加载器
            device: 设备
            epoch: 当前epoch
            num_samples: 要保存的样本数量
        """
        try:
            # 创建epoch目录
            epoch_dir = self.vis_dirs['predictions'] / f'epoch_{epoch:03d}'
            epoch_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"[Images] 保存第 {epoch} 轮预测样本...")
            
            model.eval()
            samples_saved = 0
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(data_loader):
                    if samples_saved >= num_samples:
                        break
                    
                    # 解包数据 - 使用修复版本
                    try:
                        images, masks = safe_unpack_batch_vision(batch_data)
                        self.logger.debug(f"[OK] 成功解包批次 {batch_idx}: images={images.shape}, masks={masks.shape}")
                    except Exception as e:
                        self.logger.error(f"[ERR] 解包批次 {batch_idx} 失败: {e}")
                        self.logger.error(f"批次数据类型: {type(batch_data)}")
                        if isinstance(batch_data, dict):
                            self.logger.error(f"字典键: {list(batch_data.keys())}")
                        continue
                    
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    # 预测
                    predictions = model(images)
                    predictions = torch.sigmoid(predictions)
                    
                    # 保存每个样本
                    batch_size = images.size(0)
                    for i in range(min(batch_size, num_samples - samples_saved)):
                        self._save_single_prediction(
                            images[i], masks[i], predictions[i], 
                            epoch_dir, f'sample_{samples_saved:03d}', epoch
                        )
                        samples_saved += 1
                        
                        if samples_saved >= num_samples:
                            break
            
            self.logger.info(f"[OK] 已保存 {samples_saved} 个预测样本到: {epoch_dir}")
            
        except Exception as e:
            self.logger.error(f"[ERR] 保存预测样本失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
    
    def create_prediction_comparison(self, model, data_loader, device, 
                                   epochs_to_compare: List[int]) -> None:
        """
        创建不同epoch之间的预测对比 - 修复版本
        
        Args:
            model: 当前模型
            data_loader: 数据加载器
            device: 设备
            epochs_to_compare: 要对比的epoch列表
        """
        try:
            self.logger.info("[Compare] 创建预测对比图...")
            
            # 获取样本数据
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(data_loader))
                
                # 使用修复版本解包数据
                try:
                    images, masks = safe_unpack_batch_vision(sample_batch)
                    images = images[:4].to(device)  # 取前4个样本
                    masks = masks[:4]
                    
                    current_preds = torch.sigmoid(model(images))
                    
                    # 创建对比网格
                    self._create_epoch_comparison_grid(images, masks, current_preds, epochs_to_compare)
                    
                except Exception as e:
                    self.logger.error(f"[ERR] 创建预测对比时解包数据失败: {e}")
                    return
            
        except Exception as e:
            self.logger.error(f"[ERR] 创建预测对比失败: {e}")
    
    def create_prediction_animation(self) -> None:
        """创建预测演化动画"""
        try:
            predictions_dir = self.vis_dirs['predictions']
            if not predictions_dir.exists():
                return
            
            # 查找所有epoch目录
            epoch_dirs = sorted([d for d in predictions_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('epoch_')])
            
            if len(epoch_dirs) < 3:
                self.logger.info("[WARN] 预测样本不足，跳过动画创建")
                return
            
            self.logger.info("[Animation] 创建预测演化动画...")
            
            try:
                from PIL import Image
                
                # 为每个样本创建GIF
                sample_files = list(epoch_dirs[0].glob('sample_*_grid.png'))
                
                for sample_file in sample_files[:3]:  # 只为前3个样本创建GIF
                    sample_name = sample_file.stem.replace('_grid', '')
                    images = []
                    
                    for epoch_dir in epoch_dirs:
                        img_path = epoch_dir / f'{sample_name}_grid.png'
                        if img_path.exists():
                            images.append(Image.open(img_path))
                    
                    if len(images) >= 3:
                        gif_path = self.vis_dirs['animations'] / f'{sample_name}_evolution.gif'
                        images[0].save(
                            gif_path,
                            save_all=True,
                            append_images=images[1:],
                            duration=800,  # 每帧800ms
                            loop=0
                        )
                        self.logger.info(f"[OK] 预测演化GIF已创建: {gif_path}")
                
            except ImportError:
                self.logger.warning("[WARN] PIL未安装，跳过GIF创建")
        
        except Exception as e:
            self.logger.error(f"[ERR] 创建预测动画失败: {e}")
    
    def save_metrics_summary(self, metrics_data: Dict[str, Any], 
                           model_info: Optional[Dict] = None) -> None:
        """
        保存训练指标摘要
        
        Args:
            metrics_data: 训练指标数据
            model_info: 模型信息
        """
        try:
            # 保存JSON格式
            summary_data = {
                'training_summary': {
                    'total_epochs': len(metrics_data.get('train_losses', [])),
                    'best_dice': metrics_data.get('best_dice', 0.0),
                    'best_epoch': metrics_data.get('best_epoch', 0),
                    'final_train_loss': metrics_data.get('train_losses', [0])[-1],
                    'final_val_loss': metrics_data.get('val_losses', [0])[-1],
                    'final_val_dice': metrics_data.get('val_dices', [0])[-1],
                },
                'metrics_data': metrics_data,
                'model_info': model_info,
                'timestamp': datetime.now().isoformat()
            }
            
            summary_path = self.vis_dirs['metrics'] / 'training_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            # 保存CSV格式
            self._save_metrics_csv(metrics_data)
            
            self.logger.info(f"[OK] 训练摘要已保存: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"[ERR] 保存训练摘要失败: {e}")
    
    def _create_training_summary_safe(self, metrics_data: Dict) -> str:
        """创建训练摘要文本 - 安全版本（无emoji）"""
        try:
            total_epochs = len(metrics_data.get('train_losses', []))
            best_dice = metrics_data.get('best_dice', 0.0)
            best_epoch = metrics_data.get('best_epoch', 0)
            
            summary = f"""Training Summary
════════════════
[Stats] Total Epochs: {total_epochs}
[Best] Best Dice Score: {best_dice:.4f}
[Up] Best Epoch: {best_epoch}

[Down] Final Metrics:
   Train Loss: {metrics_data.get('train_losses', [0])[-1]:.4f}
   Val Loss: {metrics_data.get('val_losses', [0])[-1]:.4f}
   Val Dice: {metrics_data.get('val_dices', [0])[-1]:.4f}

[Time] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            return summary
        except:
            return "Training Summary\n(Error generating summary)"
    
    def _save_single_prediction(self, image: torch.Tensor, mask: torch.Tensor, 
                               prediction: torch.Tensor, save_dir: Path, 
                               filename: str, epoch: int) -> None:
        """保存单个预测结果的可视化"""
        try:
            # 转换为numpy格式
            image_np = self._tensor_to_numpy(image)
            mask_np = self._tensor_to_numpy(mask)
            pred_np = self._tensor_to_numpy(prediction)
            
            # 创建二值化预测
            pred_binary = (pred_np > 0.5).astype(np.float32)
            
            # 计算指标
            dice_score = self._calculate_dice_score(mask_np, pred_binary)
            iou_score = self._calculate_iou_score(mask_np, pred_binary)
            
            # 创建可视化网格
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Epoch {epoch} - {filename}\nDice: {dice_score:.4f}, IoU: {iou_score:.4f}', 
                        fontsize=16, fontweight='bold')
            
            # 原始图像
            axes[0, 0].imshow(image_np, cmap='gray' if len(image_np.shape) == 2 else None)
            axes[0, 0].set_title('Original Image', fontweight='bold')
            axes[0, 0].axis('off')
            
            # 真实标签
            axes[0, 1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title('Ground Truth', fontweight='bold')
            axes[0, 1].axis('off')
            
            # 预测概率图
            im1 = axes[0, 2].imshow(pred_np, cmap='jet', vmin=0, vmax=1)
            axes[0, 2].set_title('Prediction (Probability)', fontweight='bold')
            axes[0, 2].axis('off')
            plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # 二值化预测
            axes[1, 0].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
            axes[1, 0].set_title('Prediction (Binary)', fontweight='bold')
            axes[1, 0].axis('off')
            
            # 叠加显示
            axes[1, 1].imshow(image_np, cmap='gray' if len(image_np.shape) == 2 else None)
            if mask_np.sum() > 0:
                mask_overlay = np.ma.masked_where(mask_np == 0, mask_np)
                axes[1, 1].imshow(mask_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
            if pred_binary.sum() > 0:
                pred_overlay = np.ma.masked_where(pred_binary == 0, pred_binary)
                axes[1, 1].imshow(pred_overlay, cmap='Blues', alpha=0.4, vmin=0, vmax=1)
            axes[1, 1].set_title('GT (Red) + Pred (Blue)', fontweight='bold')
            axes[1, 1].axis('off')
            
            # 误差分析
            self._create_error_map(axes[1, 2], mask_np, pred_binary)
            
            plt.tight_layout()
            
            # 保存图像
            save_path = save_dir / f'{filename}_grid.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.debug(f"[OK] 预测样本已保存: {save_path}")
            
        except Exception as e:
            self.logger.error(f"[ERR] 保存单个预测失败 {filename}: {e}")
    
    def _create_error_map(self, ax, mask_np: np.ndarray, pred_binary: np.ndarray) -> None:
        """创建误差分析图"""
        # 误差图 (绿色=正确, 红色=假阳性, 蓝色=假阴性, 灰色=背景)
        error_map = np.zeros((*mask_np.shape, 3))
        
        # 真阴性 (背景正确)
        true_negative = (mask_np == 0) & (pred_binary == 0)
        error_map[true_negative] = [0.5, 0.5, 0.5]  # 灰色
        
        # 真阳性 (前景正确)
        true_positive = (mask_np == 1) & (pred_binary == 1)
        error_map[true_positive] = [0, 1, 0]  # 绿色
        
        # 假阳性 (预测为正，实际为负)
        false_positive = (pred_binary == 1) & (mask_np == 0)
        error_map[false_positive] = [1, 0, 0]  # 红色
        
        # 假阴性 (预测为负，实际为正)
        false_negative = (pred_binary == 0) & (mask_np == 1)
        error_map[false_negative] = [0, 0, 1]  # 蓝色
        
        ax.imshow(error_map)
        ax.set_title('Error Analysis\n(Green=TP, Red=FP, Blue=FN, Gray=TN)', fontweight='bold')
        ax.axis('off')
        
        # 添加统计信息
        tp = np.sum(true_positive)
        fp = np.sum(false_positive)
        fn = np.sum(false_negative)
        tn = np.sum(true_negative)
        
        stats_text = f'TP: {tp}\nFP: {fp}\nFN: {fn}\nTN: {tn}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=8)
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将tensor转换为numpy数组，适合可视化"""
        np_array = tensor.cpu().numpy()
        
        # 处理不同的张量形状
        if len(np_array.shape) == 3:  # (C, H, W)
            if np_array.shape[0] == 3:  # RGB
                np_array = np.transpose(np_array, (1, 2, 0))
                # 归一化到[0,1]
                np_array = (np_array - np_array.min()) / (np_array.max() - np_array.min() + 1e-8)
            elif np_array.shape[0] == 1:  # 单通道
                np_array = np_array[0]
        elif len(np_array.shape) == 2:  # (H, W)
            pass
        else:
            raise ValueError(f"不支持的张量形状: {np_array.shape}")
        
        return np_array
    
    def _calculate_dice_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算Dice分数"""
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)
        if union == 0:
            return 1.0
        return 2.0 * intersection / union
    
    def _calculate_iou_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算IoU分数"""
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        if union == 0:
            return 1.0
        return intersection / union
    
    def _save_individual_curves(self, metrics_data: Dict) -> None:
        """保存独立的训练曲线"""
        try:
            epochs = metrics_data.get('epochs', list(range(1, len(metrics_data['train_losses']) + 1)))
            
            # 损失曲线
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, metrics_data['train_losses'], 'b-', label='Train Loss', linewidth=2)
            if metrics_data.get('val_losses'):
                plt.plot(epochs, metrics_data['val_losses'], 'r-', label='Val Loss', linewidth=2)
            plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.vis_dirs['curves'] / 'loss_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Dice曲线
            if metrics_data.get('val_dices'):
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, metrics_data['val_dices'], 'g-', label='Val Dice', linewidth=2)
                best_dice = max(metrics_data['val_dices'])
                best_epoch = metrics_data['val_dices'].index(best_dice) + 1
                plt.axhline(y=best_dice, color='r', linestyle='--', 
                           label=f'Best: {best_dice:.4f}', linewidth=2)
                plt.scatter([best_epoch], [best_dice], color='red', s=100, zorder=5)
                plt.title('Validation Dice Score', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch')
                plt.ylabel('Dice Score')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.vis_dirs['curves'] / 'dice_curve.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"保存独立曲线失败: {e}")
    
    def _save_metrics_csv(self, metrics_data: Dict) -> None:
        """保存CSV格式的指标数据"""
        try:
            import pandas as pd
            
            df_data = {
                'epoch': metrics_data.get('epochs', list(range(1, len(metrics_data['train_losses']) + 1))),
                'train_loss': metrics_data.get('train_losses', []),
                'val_loss': metrics_data.get('val_losses', []),
                'val_dice': metrics_data.get('val_dices', []),
                'learning_rate': metrics_data.get('lr_history', [])
            }
            
            # 确保所有列表长度一致
            max_len = max(len(v) for v in df_data.values() if isinstance(v, list))
            for key, value in df_data.items():
                if isinstance(value, list) and len(value) < max_len:
                    df_data[key].extend([0] * (max_len - len(value)))
            
            df = pd.DataFrame(df_data)
            csv_path = self.vis_dirs['metrics'] / 'training_metrics.csv'
            df.to_csv(csv_path, index=False)
            
            self.logger.info(f"[OK] CSV格式指标已保存: {csv_path}")
            
        except ImportError:
            self.logger.warning("[WARN] pandas未安装，跳过CSV保存")
        except Exception as e:
            self.logger.error(f"保存CSV失败: {e}")
    
    def _create_epoch_comparison_grid(self, images: torch.Tensor, masks: torch.Tensor, 
                                    current_preds: torch.Tensor, epochs: List[int]) -> None:
        """创建不同epoch的预测对比网格"""
        try:
            num_samples = min(4, images.size(0))
            fig, axes = plt.subplots(num_samples, len(epochs) + 2, 
                                   figsize=(4 * (len(epochs) + 2), 4 * num_samples))
            
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_samples):
                # 原图
                image_np = self._tensor_to_numpy(images[i])
                axes[i, 0].imshow(image_np, cmap='gray' if len(image_np.shape) == 2 else None)
                axes[i, 0].set_title('Original' if i == 0 else '')
                axes[i, 0].axis('off')
                
                # 真实标签
                mask_np = self._tensor_to_numpy(masks[i])
                axes[i, 1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
                axes[i, 1].set_title('Ground Truth' if i == 0 else '')
                axes[i, 1].axis('off')
                
                # 不同epoch的预测
                for j, epoch in enumerate(epochs):
                    col_idx = j + 2
                    if j == len(epochs) - 1:  # 最后一个是当前预测
                        pred_np = self._tensor_to_numpy(current_preds[i])
                        pred_binary = (pred_np > 0.5).astype(np.float32)
                        dice = self._calculate_dice_score(mask_np, pred_binary)
                        axes[i, col_idx].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
                        axes[i, col_idx].set_title(f'Epoch {epoch}\nDice: {dice:.3f}' if i == 0 else '')
                    else:
                        # 尝试加载历史预测结果
                        axes[i, col_idx].text(0.5, 0.5, f'Epoch {epoch}\n(Not Available)', 
                                            ha='center', va='center', transform=axes[i, col_idx].transAxes)
                        axes[i, col_idx].set_title(f'Epoch {epoch}' if i == 0 else '')
                    axes[i, col_idx].axis('off')
            
            plt.suptitle('Prediction Evolution Comparison', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            comparison_path = self.vis_dirs['comparisons'] / f'epoch_comparison_{epochs[-1]:03d}.png'
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"创建epoch对比失败: {e}")


def create_visualizer(experiment_dir: str, logger: Optional[logging.Logger] = None) -> TrainingVisualizer:
    """
    创建训练可视化器的工厂函数
    
    Args:
        experiment_dir: 实验目录
        logger: 日志记录器
        
    Returns:
        TrainingVisualizer实例
    """
    return TrainingVisualizer(experiment_dir, logger)