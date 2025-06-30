import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
import os
from datetime import datetime
from tqdm import tqdm

# 导入模块
from models.components.IntegratedDRM import create_integrated_hma_drm
from data.dataset import create_train_val_dataloaders


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='集成HMA-DRM训练脚本')
    
    # 数据参数
    parser.add_argument('--train_img_dir', type=str, required=True, help='训练图像目录')
    parser.add_argument('--train_mask_dir', type=str, required=True, help='训练掩码目录')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--img_size', type=int, default=512, help='图像大小')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='训练验证分割比例')
    
    # 模型参数
    parser.add_argument('--base_channels', type=int, default=32, help='基础通道数')
    parser.add_argument('--timesteps', type=int, default=1000, help='扩散时间步数')
    parser.add_argument('--diffusion_probability', type=float, default=0.5, help='扩散训练概率')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/IntegratedHMADRM', help='保存目录')
    parser.add_argument('--val_frequency', type=int, default=5, help='验证频率')
    parser.add_argument('--save_frequency', type=int, default=10, help='保存频率')
    
    return parser.parse_args()


def setup_logging(save_dir: str):
    """设置日志"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算Dice系数"""
    pred_binary = (pred > 0.5).float()
    intersection = (pred_binary * target).sum()
    dice = (2 * intersection) / (pred_binary.sum() + target.sum() + 1e-8)
    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算IoU"""
    pred_binary = (pred > 0.5).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-8)
    return iou.item()


def train_epoch(model, train_loader, optimizer, device, epoch, logger):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    diffusion_count = 0
    direct_count = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # 获取数据
            if isinstance(batch, dict):
                images = batch['image'].to(device, non_blocking=True)
                masks = batch['mask'].to(device, non_blocking=True)
            else:
                images, masks = batch
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
            
            masks = (masks > 0.5).float()
            
            # 前向传播
            optimizer.zero_grad()
            loss_dict = model.compute_loss(images, masks)
            
            # 反向传播
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 统计
            total_loss += loss_dict['total_loss'].item()
            
            # 计算指标（使用初始预测）
            with torch.no_grad():
                outputs = model.forward(images, masks, mode="train")
                pred_masks = outputs['initial_prediction']
                
                dice = compute_dice(pred_masks, masks)
                iou = compute_iou(pred_masks, masks)
                
                total_dice += dice
                total_iou += iou
            
            # 统计模式
            if loss_dict['mode'] == 'diffusion':
                diffusion_count += 1
            else:
                direct_count += 1
            
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'Dice': f"{dice:.4f}",
                'Mode': loss_dict['mode'][:4]
            })
            
            # 定期日志
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Loss={loss_dict['total_loss'].item():.4f}, "
                    f"Mode={loss_dict['mode']}"
                )
                
        except Exception as e:
            logger.error(f"训练批次 {batch_idx} 失败: {e}")
            continue
    
    # 计算平均指标
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches
    else:
        avg_loss = avg_dice = avg_iou = 0.0
    
    logger.info(f"Epoch {epoch} 训练完成:")
    logger.info(f"  平均损失: {avg_loss:.4f}")
    logger.info(f"  平均Dice: {avg_dice:.4f}")
    logger.info(f"  平均IoU: {avg_iou:.4f}")
    logger.info(f"  扩散批次: {diffusion_count}, 直接批次: {direct_count}")
    
    return {
        'train_loss': avg_loss,
        'train_dice': avg_dice,
        'train_iou': avg_iou,
        'diffusion_ratio': diffusion_count / max(num_batches, 1)
    }


def validate(model, val_loader, device, logger):
    """验证"""
    model.eval()
    
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            try:
                # 获取数据
                if isinstance(batch, dict):
                    images = batch['image'].to(device, non_blocking=True)
                    masks = batch['mask'].to(device, non_blocking=True)
                else:
                    images, masks = batch
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                
                masks = (masks > 0.5).float()
                
                # 推理
                refined_masks = model.forward(images, mode="inference")
                
                # 计算指标
                dice = compute_dice(refined_masks, masks)
                iou = compute_iou(refined_masks, masks)
                
                batch_size = images.shape[0]
                total_dice += dice * batch_size
                total_iou += iou * batch_size
                total_samples += batch_size
                
            except Exception as e:
                logger.error(f"验证批次失败: {e}")
                continue
    
    if total_samples > 0:
        avg_dice = total_dice / total_samples
        avg_iou = total_iou / total_samples
    else:
        avg_dice = avg_iou = 0.0
    
    logger.info(f"验证结果: Dice={avg_dice:.4f}, IoU={avg_iou:.4f}")
    
    return {
        'val_dice': avg_dice,
        'val_iou': avg_iou
    }


def save_checkpoint(model, optimizer, epoch, metrics, save_dir, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'model_config': {
            'base_channels': model.hma_unet.base_channels,
            'timesteps': model.timesteps,
            'use_diffusion_training': model.use_diffusion_training,
            'diffusion_probability': model.diffusion_probability
        }
    }
    
    # 保存最新检查点
    latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"💾 保存最佳模型: {best_path}")
    
    print(f"💾 保存检查点: {latest_path}")


def main():
    """主函数"""
    args = parse_args()
    logger = setup_logging(args.save_dir)
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("集成HMA-DRM训练配置")
    logger.info("=" * 60)
    logger.info(f"基础通道数: {args.base_channels}")
    logger.info(f"扩散时间步: {args.timesteps}")
    logger.info(f"扩散概率: {args.diffusion_probability}")
    logger.info(f"训练轮数: {args.num_epochs}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"设备: {args.device}")
    logger.info("=" * 60)
    
    # 创建数据加载器
    train_loader, val_loader = create_train_val_dataloaders(
        train_img_dir=args.train_img_dir,
        train_mask_dir=args.train_mask_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        split_ratio=args.split_ratio
    )
    
    # 创建模型
    try:
        model = create_integrated_hma_drm(
            base_channels=args.base_channels,
            timesteps=args.timesteps,
            diffusion_probability=args.diffusion_probability,
            device=args.device
        )
        
        # 再次确保模型在正确设备上
        model = model.to(args.device)
        
        # 验证设备状态
        model_device = next(model.parameters()).device
        print(f"✅ 模型创建成功，当前设备: {model_device}")
        
        if str(model_device) != args.device:
            logger.warning(f"设备不匹配! 预期: {args.device}, 实际: {model_device}")
            model = model.to(args.device)
            print(f"🔄 强制移动到设备: {args.device}")
        
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01
    )
    
    # 训练循环
    best_dice = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        try:
            # 训练
            train_metrics = train_epoch(model, train_loader, optimizer, args.device, epoch, logger)
            scheduler.step()
            
            # 验证
            if epoch % args.val_frequency == 0:
                val_metrics = validate(model, val_loader, args.device, logger)
                
                # 检查最佳模型
                current_dice = val_metrics['val_dice']
                is_best = current_dice > best_dice
                if is_best:
                    best_dice = current_dice
                
                # 合并指标
                all_metrics = {**train_metrics, **val_metrics}
                
                # 保存检查点
                save_checkpoint(model, optimizer, epoch, all_metrics, args.save_dir, is_best)
                
                if is_best:
                    logger.info(f"🎉 新的最佳模型! Dice: {current_dice:.4f}")
            
            # 定期保存
            elif epoch % args.save_frequency == 0:
                save_checkpoint(model, optimizer, epoch, train_metrics, args.save_dir)
                
        except Exception as e:
            logger.error(f"Epoch {epoch} 失败: {e}")
            continue
    
    logger.info(f"✅ 训练完成! 最佳Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()