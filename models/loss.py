import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class DiceLoss(nn.Module):
    """
    Dice损失函数 - 专注于分割准确性
    
    适用于医学图像分割，特别是处理类别不平衡问题
    """
    def __init__(self, smooth: float = 1e-6, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测logits (B, C, H, W) 或 (B, 1, H, W)
            target: 真实标签 (B, C, H, W) 或 (B, 1, H, W)
        """
        # 确保预测值在[0,1]范围内
        if pred.dtype != torch.bool:
            pred = torch.sigmoid(pred)
        
        pred = pred.contiguous()
        target = target.contiguous()
        
        # 计算intersection和union
        intersection = (pred * target).sum(dim=(2, 3))
        total = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        # 计算Dice系数
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        
        # 应用类别权重
        if self.weight is not None:
            dice = dice * self.weight.to(dice.device)
        
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal损失函数 - 专注于困难样本
    
    解决前景背景极度不平衡的问题，让模型关注困难的边界像素
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测logits (B, C, H, W)
            target: 真实标签 (B, C, H, W)，值为0或1
        """
        # 计算BCE损失（不reduction）
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算概率
        pred_sigmoid = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        
        # 计算动态权重
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt).pow(self.gamma)
        
        # 应用focal权重
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class IoULoss(nn.Module):
    """
    IoU损失函数 - 直接优化IoU指标
    
    更直接地优化最终评估指标
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 确保预测值在[0,1]范围内
        if pred.dtype != torch.bool:
            pred = torch.sigmoid(pred)
        
        pred = pred.contiguous()
        target = target.contiguous()
        
        # 计算IoU
        intersection = (pred * target).sum(dim=(2, 3))
        total = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou.mean()


class BoundaryLoss(nn.Module):
    """
    边界损失函数 - 专注于边界精度
    
    使用距离变换来强调边界像素的重要性，提高分割边界质量
    """
    def __init__(self, theta0: float = 3, theta: float = 5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def compute_distance_map(self, mask: torch.Tensor) -> torch.Tensor:
        """
        计算距离变换图
        
        Args:
            mask: 二值掩码 (B, 1, H, W)
        
        Returns:
            距离图 (B, 1, H, W)
        """
        batch_size = mask.size(0)
        distance_maps = []
        
        # 转换为numpy处理
        masks_np = mask.detach().cpu().numpy()
        
        for i in range(batch_size):
            # 提取单个mask并确保是uint8类型
            curr_mask = (masks_np[i, 0] > 0.5).astype(np.uint8)
            
            if curr_mask.max() == 0:  # 全零mask
                # 创建全为最大距离的map
                h, w = curr_mask.shape
                dist_map = np.ones((h, w), dtype=np.float32) * max(h, w)
            else:
                # 计算到边界的距离
                # 首先找到边界
                kernel = np.ones((3, 3), np.uint8)
                boundary = cv2.morphologyEx(curr_mask, cv2.MORPH_GRADIENT, kernel)
                
                if boundary.max() > 0:
                    # 计算到边界的距离
                    dist_map = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, 3)
                else:
                    # 如果没有边界，使用掩码的距离变换
                    dist_map = cv2.distanceTransform(1 - curr_mask, cv2.DIST_L2, 3)
            
            # 归一化到[0,1]范围
            if dist_map.max() > 0:
                dist_map = dist_map / dist_map.max()
            
            distance_maps.append(dist_map)
        
        # 转换回tensor
        distance_tensor = torch.from_numpy(np.stack(distance_maps)).unsqueeze(1).float()
        return distance_tensor.to(mask.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测logits (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
        """
        # 获取预测概率
        pred_sigmoid = torch.sigmoid(pred)
        
        # 计算距离图
        dist_map = self.compute_distance_map(target)
        
        # 计算边界敏感的损失
        # 对于预测为前景但真实为背景的像素，使用距离加权
        # 对于预测为背景但真实为前景的像素，同样使用距离加权
        pos_loss = target * (1 - pred_sigmoid) * (1 + dist_map)
        neg_loss = (1 - target) * pred_sigmoid * (1 + dist_map)
        
        boundary_loss = (pos_loss + neg_loss).mean()
        
        return boundary_loss


class TverskyLoss(nn.Module):
    """
    Tversky损失函数 - 可调节的Dice损失
    
    通过alpha和beta参数调节对假阳性和假阴性的敏感度
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # 假阳性权重
        self.beta = beta    # 假阴性权重
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 确保预测值在[0,1]范围内
        if pred.dtype != torch.bool:
            pred = torch.sigmoid(pred)
        
        pred = pred.contiguous()
        target = target.contiguous()
        
        # 计算Tversky系数
        tp = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky.mean()


class EdgeAwareLoss(nn.Module):
    """
    边缘感知损失函数 - 强化边缘检测
    
    结合边缘检测来提高分割边界的精度
    """
    def __init__(self, edge_weight: float = 2.0):
        super().__init__()
        self.edge_weight = edge_weight
        
        # Sobel算子
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                    dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                    dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def get_edges(self, x: torch.Tensor) -> torch.Tensor:
        """提取边缘"""
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        edges = torch.sqrt(grad_x**2 + grad_y**2)
        return edges

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 获取预测概率
        pred_sigmoid = torch.sigmoid(pred)
        
        # 提取边缘
        pred_edges = self.get_edges(pred_sigmoid)
        target_edges = self.get_edges(target)
        
        # 边缘损失
        edge_loss = F.mse_loss(pred_edges, target_edges)
        
        # 基础BCE损失
        bce_loss = F.binary_cross_entropy(pred_sigmoid, target)
        
        return bce_loss + self.edge_weight * edge_loss


class HMAUNetLoss(nn.Module):
    """
    HMA-UNet专用综合损失函数
    
    结合多种损失函数，专门为医学图像分割优化：
    1. Focal Loss - 处理类别不平衡
    2. Dice Loss - 优化分割准确性  
    3. Boundary Loss - 提高边界质量
    4. IoU Loss - 直接优化评估指标
    """
    def __init__(
        self,
        focal_weight: float = 0.3,
        dice_weight: float = 0.3,
        boundary_weight: float = 0.2,
        iou_weight: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1e-6,
        adaptive_weights: bool = True
    ):
        super().__init__()
        
        # 损失函数权重
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.iou_weight = iou_weight
        self.adaptive_weights = adaptive_weights
        
        # 初始化各个损失函数
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.boundary_loss = BoundaryLoss()
        self.iou_loss = IoULoss()
        
        # 自适应权重学习
        if adaptive_weights:
            self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: 预测logits (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
        
        Returns:
            包含各个损失值的字典
        """
        # 计算各个损失
        focal_loss = self.focal_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        boundary_loss = self.boundary_loss(pred, target)
        iou_loss = self.iou_loss(pred, target)
        
        if self.adaptive_weights:
            # 使用学习的权重（基于不确定性加权）
            precision_focal = torch.exp(-self.log_vars[0])
            precision_dice = torch.exp(-self.log_vars[1])
            precision_boundary = torch.exp(-self.log_vars[2])
            precision_iou = torch.exp(-self.log_vars[3])
            
            weighted_focal = precision_focal * focal_loss + self.log_vars[0]
            weighted_dice = precision_dice * dice_loss + self.log_vars[1]
            weighted_boundary = precision_boundary * boundary_loss + self.log_vars[2]
            weighted_iou = precision_iou * iou_loss + self.log_vars[3]
            
            total_loss = weighted_focal + weighted_dice + weighted_boundary + weighted_iou
        else:
            # 使用固定权重
            total_loss = (
                self.focal_weight * focal_loss +
                self.dice_weight * dice_loss +
                self.boundary_weight * boundary_loss +
                self.iou_weight * iou_loss
            )
        
        return {
            'total_loss': total_loss,
            'focal_loss': focal_loss,
            'dice_loss': dice_loss,
            'boundary_loss': boundary_loss,
            'iou_loss': iou_loss
        }

    def get_weights(self) -> Dict[str, float]:
        """获取当前权重信息"""
        if self.adaptive_weights:
            weights = {
                'focal_weight': torch.exp(-self.log_vars[0]).item(),
                'dice_weight': torch.exp(-self.log_vars[1]).item(),
                'boundary_weight': torch.exp(-self.log_vars[2]).item(),
                'iou_weight': torch.exp(-self.log_vars[3]).item()
            }
        else:
            weights = {
                'focal_weight': self.focal_weight,
                'dice_weight': self.dice_weight,
                'boundary_weight': self.boundary_weight,
                'iou_weight': self.iou_weight
            }
        return weights


class WeightedCombinedLoss(nn.Module):
    """
    加权组合损失函数 - 灵活配置版本
    
    允许用户自定义各种损失函数的组合和权重
    """
    def __init__(self, loss_config: Dict[str, Any]):
        super().__init__()
        
        self.loss_functions = nn.ModuleDict()
        self.weights = {}
        
        # 根据配置创建损失函数
        for loss_name, config in loss_config.items():
            loss_type = config['type']
            weight = config.get('weight', 1.0)
            params = config.get('params', {})
            
            if loss_type == 'focal':
                self.loss_functions[loss_name] = FocalLoss(**params)
            elif loss_type == 'dice':
                self.loss_functions[loss_name] = DiceLoss(**params)
            elif loss_type == 'boundary':
                self.loss_functions[loss_name] = BoundaryLoss(**params)
            elif loss_type == 'iou':
                self.loss_functions[loss_name] = IoULoss(**params)
            elif loss_type == 'tversky':
                self.loss_functions[loss_name] = TverskyLoss(**params)
            elif loss_type == 'edge_aware':
                self.loss_functions[loss_name] = EdgeAwareLoss(**params)
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
            
            self.weights[loss_name] = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}
        total_loss = 0
        
        for loss_name, loss_fn in self.loss_functions.items():
            loss_value = loss_fn(pred, target)
            losses[loss_name] = loss_value
            total_loss += self.weights[loss_name] * loss_value
        
        losses['total_loss'] = total_loss
        return losses


# 预定义配置
def get_default_loss():
    """获取默认的HMA-UNet损失函数"""
    return HMAUNetLoss()


def get_lightweight_loss():
    """获取轻量级损失函数（适合快速训练）"""
    return HMAUNetLoss(
        focal_weight=0.4,
        dice_weight=0.6,
        boundary_weight=0.0,
        iou_weight=0.0,
        adaptive_weights=False
    )


def get_boundary_focused_loss():
    """获取边界优化损失函数（适合边界质量要求高的任务）"""
    return HMAUNetLoss(
        focal_weight=0.2,
        dice_weight=0.2,
        boundary_weight=0.4,
        iou_weight=0.2,
        adaptive_weights=True
    )


def get_custom_loss(config: Dict[str, Any]):
    """根据配置创建自定义损失函数
    
    Example:
        config = {
            'focal': {'type': 'focal', 'weight': 0.3, 'params': {'alpha': 0.25, 'gamma': 2.0}},
            'dice': {'type': 'dice', 'weight': 0.4, 'params': {'smooth': 1e-6}},
            'boundary': {'type': 'boundary', 'weight': 0.3}
        }
    """
    return WeightedCombinedLoss(config)


# 测试函数
def test_losses():
    """测试所有损失函数"""
    print("🧪 Testing HMA-UNet Loss Functions...")
    
    # 创建测试数据
    batch_size, channels, height, width = 2, 1, 64, 64
    pred = torch.randn(batch_size, channels, height, width)  # logits
    target = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    
    print(f"Input shapes: pred={pred.shape}, target={target.shape}")
    
    # 测试各个损失函数
    loss_functions = {
        'Dice Loss': DiceLoss(),
        'Focal Loss': FocalLoss(),
        'IoU Loss': IoULoss(),
        'Boundary Loss': BoundaryLoss(),
        'Tversky Loss': TverskyLoss(),
        'Edge Aware Loss': EdgeAwareLoss(),
        'HMA-UNet Loss': HMAUNetLoss(),
    }
    
    for name, loss_fn in loss_functions.items():
        try:
            if name == 'HMA-UNet Loss':
                result = loss_fn(pred, target)
                loss_value = result['total_loss']
                print(f"✅ {name}: {loss_value:.4f}")
                print(f"   └─ Focal: {result['focal_loss']:.4f}, Dice: {result['dice_loss']:.4f}")
                print(f"   └─ Boundary: {result['boundary_loss']:.4f}, IoU: {result['iou_loss']:.4f}")
            else:
                loss_value = loss_fn(pred, target)
                print(f"✅ {name}: {loss_value:.4f}")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    print("\n🎉 Loss function tests completed!")


if __name__ == "__main__":
    test_losses()