import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
from typing import Optional, Callable, Tuple, List
import hashlib


class SegmentationDataset(Dataset):
    """
    医学图像分割数据集
    
    支持的数据格式：
    - 图像：RGB格式 (.png, .jpg, .jpeg)
    - 标签：二值化掩码 (.png)
    
    目录结构：
    data/
    ├── train/
    │   ├── images/
    │   │   ├── 001.png
    │   │   └── ...
    │   └── labels/
    │       ├── 001.png
    │       └── ...
    └── test/
        ├── images/
        └── labels/
    """
    
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        is_train: bool = True,
        image_list: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        augment: bool = True
    ):
        """
        Args:
            img_dir: 图像目录路径
            mask_dir: 标签目录路径
            image_size: 图像尺寸 (height, width)
            is_train: 是否为训练模式
            image_list: 指定的图像文件列表，如果为None则使用目录下所有图像
            transform: 自定义变换
            normalize: 是否归一化图像
            augment: 是否进行数据增强
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.is_train = is_train
        self.normalize = normalize
        
        # 获取图像文件列表
        if image_list is not None:
            self.image_files = image_list
        else:
            self.image_files = self._get_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {self.img_dir} 中未找到图像文件")
        
        print(f"在 {'训练' if is_train else '验证/测试'} 集中找到 {len(self.image_files)} 张图像")
        
        # 设置变换
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms(augment and is_train)
    
    def _get_image_files(self):
        """获取所有图像文件"""
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        
        print(f"🔍 正在扫描目录: {self.img_dir}")
        
        for ext in image_extensions:
            pattern = os.path.join(self.img_dir, ext)
            files = glob.glob(pattern)
            print(f"   扫描 {ext}: 找到 {len(files)} 个文件")
            
            # 只保留文件名，不包含路径
            files = [os.path.basename(f) for f in files]
            image_files.extend(files)
        
        # 排序确保一致性
        image_files.sort()
        print(f"总共找到 {len(image_files)} 个图像文件")
        
        return image_files
    
    def _get_default_transforms(self, augment: bool = False):
        """获取默认的数据变换"""
        transforms_list = []
        
        # 基础变换
        transforms_list.extend([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
        ])
        
        # 数据增强 (仅训练时)
        if augment:
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ElasticTransform(p=0.3),
            ])
        
        # 归一化和转换为张量
        if self.normalize:
            transforms_list.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet标准
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                )
            )
        
        transforms_list.append(ToTensorV2())
        
        return A.Compose(transforms_list)
    
    def _get_mask_path(self, image_filename: str) -> str:
        """根据图像文件名获取对应的标签路径"""
        # 移除扩展名并添加.png
        base_name = os.path.splitext(image_filename)[0]
        mask_path = os.path.join(self.mask_dir, base_name + '.png')
        return mask_path
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图像和标签路径
        image_filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, image_filename)
        mask_path = self._get_mask_path(image_filename)
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法加载图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标签
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"无法加载掩码: {mask_path}")
            
            # 二值化标签 (0: 背景, 1: 前景)
            mask = (mask > 127).astype(np.uint8)
        else:
            # 如果没有标签文件，创建空标签
            mask = np.zeros(
                (image.shape[0], image.shape[1]), 
                dtype=np.uint8
            )
            print(f"⚠️  未找到 {image_filename} 的掩码文件，使用空掩码")
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # 确保mask是float类型并且值在[0,1]范围内
        if mask.dtype == torch.uint8:
            mask = mask.float()
        
        # 为分割任务添加通道维度
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
        
        return {
            'image': image,
            'mask': mask,
            'filename': image_filename,
            'image_path': img_path,
            'mask_path': mask_path
        }
    
    def get_sample_names(self):
        """获取所有样本名称"""
        return [os.path.splitext(f)[0] for f in self.image_files]


def get_fixed_split(all_images: List[str], split_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """
    根据文件名创建固定的训练/验证集划分
    使用文件名的哈希值确保相同输入总是产生相同的划分
    
    Args:
        all_images: 所有图像文件名列表
        split_ratio: 训练集比例
        
    Returns:
        (train_images, val_images): 训练集和验证集文件名列表
    """
    # 根据文件名排序确保一致性
    sorted_images = sorted(all_images)
    
    # 使用文件名哈希值创建固定的索引划分
    def get_hash_value(filename):
        """获取文件名的哈希值"""
        return int(hashlib.md5(filename.encode()).hexdigest(), 16)
    
    # 为每个文件创建哈希值并排序
    image_hash_pairs = [(img, get_hash_value(img)) for img in sorted_images]
    image_hash_pairs.sort(key=lambda x: x[1])  # 按哈希值排序
    
    # 计算训练集大小
    train_size = int(len(sorted_images) * split_ratio)
    
    # 基于哈希值排序的结果进行划分
    train_images = [pair[0] for pair in image_hash_pairs[:train_size]]
    val_images = [pair[0] for pair in image_hash_pairs[train_size:]]
    
    # 确保验证集至少有一定数量
    min_val_size = max(1, int(len(sorted_images) * 0.1))  # 至少10%
    if len(val_images) < min_val_size:
        # 重新调整
        val_size = min_val_size
        train_size = len(sorted_images) - val_size
        train_images = [pair[0] for pair in image_hash_pairs[:train_size]]
        val_images = [pair[0] for pair in image_hash_pairs[train_size:]]
    
    return train_images, val_images


def create_dataloaders(
    img_dir: str,
    mask_dir: str,
    batch_size: int = 8,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 4,
    val_img_dir: Optional[str] = None,
    val_mask_dir: Optional[str] = None,
    use_fixed_split: bool = True,
    split_ratio: float = 0.8,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        img_dir: 训练图像目录
        mask_dir: 训练掩码目录
        batch_size: 批次大小
        image_size: 图像尺寸
        num_workers: 数据加载工作进程数
        val_img_dir: 验证图像目录（可选）
        val_mask_dir: 验证掩码目录（可选）
        use_fixed_split: 是否使用固定划分
        split_ratio: 训练集比例
        train_transform: 训练集自定义变换
        val_transform: 验证集自定义变换
        
    Returns:
        (train_loader, val_loader): 训练和验证数据加载器
    """
    
    # 创建验证集数据集
    if val_img_dir is not None and val_mask_dir is not None:
        # 如果提供了验证集路径，使用单独的验证集
        print("🔧 使用独立的验证集目录")
        train_dataset = SegmentationDataset(
            img_dir=img_dir,
            mask_dir=mask_dir,
            image_size=image_size,
            is_train=True,
            transform=train_transform,
            augment=True
        )
        val_dataset = SegmentationDataset(
            img_dir=val_img_dir,
            mask_dir=val_mask_dir,
            image_size=image_size,
            is_train=False,
            transform=val_transform,
            augment=False
        )
    else:
        # 从训练集划分验证集
        print("🔧 从训练数据中划分验证集")
        
        # 获取所有图像文件
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        all_images = []
        
        print(f"🔍 正在扫描目录: {img_dir}")
        for ext in image_extensions:
            pattern = os.path.join(img_dir, ext)
            files = glob.glob(pattern)
            print(f"   扫描 {ext}: 找到 {len(files)} 个文件")
            
            files = [os.path.basename(f) for f in files]
            all_images.extend(files)
        
        all_images = sorted(list(set(all_images)))  # 去重并排序
        print(f"总共扫描到 {len(all_images)} 个图像文件")
        
        if len(all_images) == 0:
            raise ValueError(f"在 {img_dir} 中未找到图像文件")
        
        if use_fixed_split:
            # 使用固定划分
            train_images, val_images = get_fixed_split(all_images, split_ratio)
            print(f"📊 使用固定划分策略:")
            print(f"   训练集: {len(train_images)} 张图像")
            print(f"   验证集: {len(val_images)} 张图像")
            print(f"   验证集比例: {len(val_images)/len(all_images):.1%}")
        else:
            # 使用随机划分
            np.random.seed(42)
            train_idx = np.random.choice(
                len(all_images), int(split_ratio * len(all_images)), replace=False
            )
            val_idx = np.array(list(set(range(len(all_images))) - set(train_idx)))
            
            train_images = [all_images[i] for i in train_idx]
            val_images = [all_images[i] for i in val_idx]
            print(f"📊 使用随机划分策略:")
            print(f"   训练集: {len(train_images)} 张图像")
            print(f"   验证集: {len(val_images)} 张图像")
        
        # 创建训练和验证数据集
        train_dataset = SegmentationDataset(
            img_dir=img_dir,
            mask_dir=mask_dir,
            image_size=image_size,
            is_train=True,
            image_list=train_images,
            transform=train_transform,
            augment=True
        )
        val_dataset = SegmentationDataset(
            img_dir=img_dir,
            mask_dir=mask_dir,
            image_size=image_size,
            is_train=False,
            image_list=val_images,
            transform=val_transform,
            augment=False
        )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_dataloader(
    img_dir: str,
    mask_dir: str,
    batch_size: int = 1,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 4,
    transform: Optional[Callable] = None
) -> DataLoader:
    """
    创建测试数据加载器
    
    Args:
        img_dir: 测试图像目录
        mask_dir: 测试掩码目录
        batch_size: 批次大小（测试时通常为1）
        image_size: 图像尺寸
        num_workers: 数据加载工作进程数
        transform: 自定义变换
        
    Returns:
        test_loader: 测试数据加载器
    """
    test_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        image_size=image_size,
        is_train=False,
        transform=transform,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader


def save_split_info(train_images: List[str], val_images: List[str], save_path: str):
    """保存数据集划分信息到文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=== 数据集划分信息 ===\n")
        f.write(f"训练集数量: {len(train_images)}\n")
        f.write(f"验证集数量: {len(val_images)}\n")
        f.write(f"总数量: {len(train_images) + len(val_images)}\n")
        f.write(f"验证集比例: {len(val_images)/(len(train_images) + len(val_images)):.1%}\n\n")
        
        f.write("训练集图像:\n")
        for img in sorted(train_images):
            f.write(f"  {img}\n")
        
        f.write("\n验证集图像:\n")
        for img in sorted(val_images):
            f.write(f"  {img}\n")
    
    print(f"📝 数据集划分信息已保存至: {save_path}")


def get_dataset_stats(dataset: SegmentationDataset) -> dict:
    """
    计算数据集统计信息
    
    Args:
        dataset: 分割数据集
    
    Returns:
        包含统计信息的字典
    """
    print("📊 正在计算数据集统计信息...")
    
    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    num_pixels = 0
    
    positive_pixels = 0
    total_mask_pixels = 0
    
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample['image']
        mask = sample['mask']
        
        # 图像统计
        if image.dim() == 3:  # (C, H, W)
            pixels = image.view(image.size(0), -1)  # (C, H*W)
            pixel_sum += pixels.sum(dim=1)
            pixel_squared_sum += (pixels ** 2).sum(dim=1)
            num_pixels += pixels.size(1)
        
        # 标签统计
        positive_pixels += (mask > 0.5).sum().item()
        total_mask_pixels += mask.numel()
        
        if (i + 1) % 100 == 0:
            print(f"   已处理 {i + 1}/{len(dataset)} 个样本")
    
    # 计算均值和标准差
    mean = pixel_sum / num_pixels
    var = (pixel_squared_sum / num_pixels) - (mean ** 2)
    std = torch.sqrt(var)
    
    # 计算正负样本比例
    positive_ratio = positive_pixels / total_mask_pixels
    negative_ratio = 1 - positive_ratio
    
    stats = {
        'num_samples': len(dataset),
        'image_mean': mean.tolist(),
        'image_std': std.tolist(),
        'positive_pixel_ratio': positive_ratio,
        'negative_pixel_ratio': negative_ratio,
        'total_pixels': total_mask_pixels,
        'positive_pixels': positive_pixels,
        'negative_pixels': total_mask_pixels - positive_pixels
    }
    
    return stats


def analyze_data_structure():
    """专门分析数据文件夹结构的函数"""
    print("🔍 深度分析数据文件夹结构...")
    
    current_dir = os.getcwd()
    data_root = os.path.join(current_dir, 'data')
    
    if not os.path.exists(data_root):
        print(f"❌ 数据根目录不存在: {data_root}")
        return
    
    print(f"📂 数据根目录: {data_root}")
    
    # 递归分析文件夹结构
    def analyze_directory(path, prefix="", max_files=10):
        """递归分析目录结构"""
        try:
            items = sorted(os.listdir(path))
            for i, item in enumerate(items):
                item_path = os.path.join(path, item)
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                
                if os.path.isdir(item_path):
                    print(f"{prefix}{current_prefix}{item}/")
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    analyze_directory(item_path, next_prefix, max_files)
                else:
                    # 文件
                    if item.lower().endswith(('.png', '.jpg', '.jpeg')):
                        print(f"{prefix}{current_prefix}{item} 📷")
                    elif item.lower().endswith('.py'):
                        print(f"{prefix}{current_prefix}{item} 🐍")
                    else:
                        print(f"{prefix}{current_prefix}{item}")
                    
                    # 如果文件太多，只显示前几个
                    if i >= max_files - 1 and len(items) > max_files:
                        remaining = len(items) - max_files
                        print(f"{prefix}{'    ' if is_last else '│   '}... 还有 {remaining} 个文件")
                        break
                        
        except PermissionError:
            print(f"{prefix}❌ 权限错误，无法读取此目录")
        except Exception as e:
            print(f"{prefix}❌ 错误: {e}")
    
    analyze_directory(data_root)
    
    # 统计信息
    print(f"\n📊 统计信息:")
    
    # 统计各类型文件数量
    for subdir in ['train', 'test']:  # 调整顺序，先检查训练集
        subdir_path = os.path.join(data_root, subdir)
        if os.path.exists(subdir_path):
            print(f"\n   {subdir.upper()} 数据集:")
            
            # 图像文件
            img_dir = os.path.join(subdir_path, 'images')
            if os.path.exists(img_dir):
                image_files = [f for f in os.listdir(img_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"     图像文件: {len(image_files)} 个")
                if image_files:
                    # 分析图像文件名模式
                    numeric_names = [f for f in image_files if f.split('.')[0].isdigit()]
                    if numeric_names:
                        numbers = [int(f.split('.')[0]) for f in numeric_names]
                        print(f"       数字命名文件: {len(numeric_names)} 个")
                        print(f"       编号范围: {min(numbers)} - {max(numbers)}")
            else:
                print(f"     图像目录不存在")
            
            # 标签文件
            mask_dir = os.path.join(subdir_path, 'labels')
            if os.path.exists(mask_dir):
                mask_files = [f for f in os.listdir(mask_dir) 
                            if f.lower().endswith('.png')]
                print(f"     标签文件: {len(mask_files)} 个")
            else:
                print(f"     标签目录不存在")


def create_dummy_labels(img_dir: str, mask_dir: str):
    """为现有图像创建空标签文件"""
    print(f"🏗️  正在为 {img_dir} 中的图像创建空标签文件...")
    
    os.makedirs(mask_dir, exist_ok=True)
    
    if os.path.exists(img_dir):
        image_files = [f for f in os.listdir(img_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        created_count = 0
        for img_file in image_files:
            # 读取图像尺寸
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                
                # 创建空标签（全零图像）
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # 保存标签
                mask_name = os.path.splitext(img_file)[0] + '.png'
                mask_path = os.path.join(mask_dir, mask_name)
                
                if not os.path.exists(mask_path):  # 避免覆盖现有标签
                    cv2.imwrite(mask_path, mask)
                    created_count += 1
                
        print(f"✅ 已为 {created_count} 个图像创建空标签文件")
    else:
        print(f"❌ 图像目录不存在: {img_dir}")


def verify_directory_count(directory: str, description: str) -> int:
    """验证目录中的文件数量"""
    if not os.path.exists(directory):
        print(f"❌ {description}目录不存在: {directory}")
        return 0
    
    try:
        image_files = [f for f in os.listdir(directory) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"✅ {description}: {len(image_files)} 个文件")
        
        if len(image_files) > 0:
            print(f"   示例文件: {image_files[:3]}")
        
        return len(image_files)
    except Exception as e:
        print(f"❌ 读取{description}时出错: {e}")
        return 0


def test_dataset():
    """测试数据集功能 - 修复版本，优先使用训练数据"""
    print("🧪 开始测试数据集功能...")
    
    # 获取当前项目根目录
    current_dir = os.getcwd()
    print(f"📂 当前工作目录: {current_dir}")
    
    # 检查数据文件夹结构
    data_root = os.path.join(current_dir, 'data')
    train_img_dir = os.path.join(data_root, 'train', 'images')
    train_mask_dir = os.path.join(data_root, 'train', 'labels')
    test_img_dir = os.path.join(data_root, 'test', 'images')
    test_mask_dir = os.path.join(data_root, 'test', 'labels')
    
    print(f"\n📁 数据文件夹结构分析:")
    print(f"   数据根目录: {data_root}")
    print(f"   训练图像目录: {train_img_dir}")
    print(f"   训练标签目录: {train_mask_dir}")
    print(f"   测试图像目录: {test_img_dir}")
    print(f"   测试标签目录: {test_mask_dir}")
    
    # 🔥 详细验证每个目录的文件数量
    print(f"\n🔍 详细文件统计:")
    train_img_count = verify_directory_count(train_img_dir, "训练图像")
    train_mask_count = verify_directory_count(train_mask_dir, "训练标签")
    test_img_count = verify_directory_count(test_img_dir, "测试图像")
    test_mask_count = verify_directory_count(test_mask_dir, "测试标签")
    
    # 🔥 优先选择训练数据（修复关键问题）
    print(f"\n🔧 选择数据源:")
    available_data = None
    
    if train_img_count > 0:
        available_data = ('train', train_img_dir, train_mask_dir)
        print(f"✅ 选择训练数据: {train_img_count} 张图像")
    elif test_img_count > 0:
        available_data = ('test', test_img_dir, test_mask_dir)
        print(f"⚠️  回退到测试数据: {test_img_count} 张图像")
    else:
        print("❌ 没有找到任何可用的图像数据")
        print("\n💡 建议:")
        print("   1. 检查 data/train/images/ 目录是否存在且包含图像文件")
        print("   2. 检查 data/test/images/ 目录是否存在且包含图像文件")
        print("   3. 确保图像文件格式为 .png, .jpg, 或 .jpeg")
        print("   4. 检查文件权限")
        
        # 🔥 显示完整的文件夹结构进行诊断
        print(f"\n📁 完整文件夹结构诊断:")
        analyze_data_structure()
        return
    
    data_type, img_dir, mask_dir = available_data
    print(f"📊 使用 {data_type} 数据进行测试")
    print(f"   图像目录: {img_dir}")
    print(f"   标签目录: {mask_dir}")
    
    # 🔥 添加详细的文件扫描日志
    print(f"\n🔍 详细扫描选定目录:")
    final_img_count = verify_directory_count(img_dir, f"{data_type}图像")
    final_mask_count = verify_directory_count(mask_dir, f"{data_type}标签")
    
    # 确保标签目录存在，如果不存在则创建空标签
    if not os.path.exists(mask_dir):
        print(f"⚠️  标签目录不存在，正在创建: {mask_dir}")
        create_dummy_labels(img_dir, mask_dir)
    else:
        # 检查是否需要创建缺失的标签文件
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]
        
        print(f"   当前文件匹配检查:")
        print(f"     图像文件: {len(image_files)} 个")
        print(f"     标签文件: {len(mask_files)} 个")
        
        # 检查哪些图像缺少对应的标签
        missing_masks = []
        for img_file in image_files:
            mask_name = os.path.splitext(img_file)[0] + '.png'
            if mask_name not in mask_files:
                missing_masks.append(img_file)
        
        if missing_masks:
            print(f"⚠️  发现 {len(missing_masks)} 个图像缺少对应的标签文件")
            print(f"     缺失示例: {missing_masks[:3]}")
            print("   正在创建空标签文件...")
            create_dummy_labels(img_dir, mask_dir)
        else:
            print("✅ 所有图像都有对应的标签文件")
    
    try:
        # 创建数据加载器
        print(f"\n🔄 正在创建数据加载器...")
        train_loader, val_loader = create_dataloaders(
            img_dir=img_dir,
            mask_dir=mask_dir,
            batch_size=4,
            image_size=(256, 256),
            num_workers=0,  # 测试时使用0避免多进程问题
            use_fixed_split=True,
            split_ratio=0.8
        )
        
        print(f"\n📈 数据加载器创建成功!")
        print(f"   训练集:")
        print(f"     批次数量: {len(train_loader)}")
        print(f"     样本总数: {len(train_loader.dataset)}")
        
        print(f"   验证集:")
        print(f"     批次数量: {len(val_loader)}")
        print(f"     样本总数: {len(val_loader.dataset)}")
        
        # 🔥 验证总数是否正确
        total_samples = len(train_loader.dataset) + len(val_loader.dataset)
        print(f"   总样本数: {total_samples}")
        
        expected_total = final_img_count
        if total_samples != expected_total:
            print(f"⚠️  样本数量不匹配! 期望: {expected_total}, 实际: {total_samples}")
        else:
            print(f"✅ 样本数量验证通过!")
        
        # 测试加载一个批次的数据
        if len(train_loader) > 0:
            print(f"\n🔍 测试加载批次数据...")
            for batch in train_loader:
                print(f"   批次信息:")
                print(f"     图像张量形状: {batch['image'].shape}")
                print(f"     标签张量形状: {batch['mask'].shape}")
                print(f"     图像数值范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
                print(f"     标签数值范围: [{batch['mask'].min():.3f}, {batch['mask'].max():.3f}]")
                print(f"     标签唯一值: {torch.unique(batch['mask'])}")
                print(f"     文件名示例: {batch['filename'][:2]}...")
                break
        
        # 保存数据集划分信息
        if hasattr(train_loader.dataset, 'image_files') and hasattr(val_loader.dataset, 'image_files'):
            train_files = train_loader.dataset.image_files
            val_files = val_loader.dataset.image_files
            
            split_info_path = os.path.join(data_root, f'{data_type}_dataset_split_info.txt')
            save_split_info(train_files, val_files, split_info_path)
        
        print(f"\n✅ 数据集测试完成!")
        
        # 给出使用建议
        print(f"\n💡 使用建议:")
        print(f"   1. 当前可以使用 {data_type} 数据进行模型训练和测试")
        print(f"   2. 如果没有标签文件，系统会自动创建空标签（全零掩码）")
        print(f"   3. 建议准备真实的标签文件以获得更好的训练效果")
        print(f"   4. 标签文件应为PNG格式的二值化图像（0: 背景, 255: 前景）")
        print(f"   5. 确认训练集有 {final_img_count} 张图像是正确的")
        
        # 显示完整的文件夹结构
        print(f"\n📁 完整文件夹结构:")
        analyze_data_structure()
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        import traceback
        print(f"\n🔍 详细错误信息:")
        traceback.print_exc()
        
        print(f"\n🛠️ 可能的解决方案:")
        print(f"   1. 检查图像文件是否可以正常读取")
        print(f"   2. 确保图像文件格式正确（PNG, JPG, JPEG）")
        print(f"   3. 检查文件权限是否正确")
        print(f"   4. 尝试减少batch_size或image_size")
        print(f"   5. 确保安装了所需的依赖包（cv2, albumentations等）")
        print(f"   6. 检查目录路径是否正确")


def quick_count_images(directory: str) -> int:
    """快速统计目录中的图像文件数量"""
    if not os.path.exists(directory):
        return 0
    
    try:
        count = 0
        for file in os.listdir(directory):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                count += 1
        return count
    except:
        return 0


def main():
    """主函数 - 提供快速检查功能"""
    print("🚀 HMA-UNet 数据集分析工具")
    print("=" * 60)
    
    # 快速统计
    current_dir = os.getcwd()
    train_img_dir = os.path.join(current_dir, 'data', 'train', 'images')
    test_img_dir = os.path.join(current_dir, 'data', 'test', 'images')
    
    train_count = quick_count_images(train_img_dir)
    test_count = quick_count_images(test_img_dir)
    
    print(f"📊 快速统计:")
    print(f"   训练图像: {train_count} 张")
    print(f"   测试图像: {test_count} 张")
    print(f"   总计: {train_count + test_count} 张")
    
    if train_count == 343:
        print("✅ 训练集数量匹配您所说的343张!")
    elif train_count > 0:
        print(f"⚠️  训练集实际有 {train_count} 张，不是343张")
    else:
        print("❌ 训练集为空或不存在")
    
    print("\n" + "=" * 60)
    
    # 运行详细测试
    test_dataset()


if __name__ == "__main__":
    main()