import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
from typing import Optional, Callable, Tuple, List
import hashlib


class IronSpectrumDataset(Dataset):
    """
    铁谱图像分割数据集
    
    支持的数据格式：
    - 图像：PNG格式 (.png)
    - 标签：二值化掩码 (.png)
    """
    
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        img_size: int = 512,
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
            img_size: 图像尺寸
            is_train: 是否为训练模式
            image_list: 指定的图像文件列表，如果为None则使用目录下所有图像
            transform: 自定义变换
            normalize: 是否归一化图像
            augment: 是否进行数据增强
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train
        self.normalize = normalize
        
        # 获取图像文件列表
        if image_list is not None:
            self.image_files = image_list
        else:
            self.image_files = self._get_image_files()
        
        # 确保属性名称一致 - 添加image_list属性以保持兼容性
        self.image_list = self.image_files
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {self.img_dir} 中未找到PNG图像文件")
        
        print(f"在 {'训练' if is_train else '验证/测试'} 集中找到 {len(self.image_files)} 张PNG图像")
        
        # 设置变换
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms(augment and is_train)
    
    def _get_image_files(self):
        """获取所有PNG图像文件"""
        print(f"🔍 正在扫描目录: {self.img_dir}")
        
        # 只扫描PNG格式图片
        pattern = os.path.join(self.img_dir, '*.png')
        files = glob.glob(pattern)
        print(f"   扫描 *.png: 找到 {len(files)} 个文件")
        
        # 只保留文件名，不包含路径
        image_files = [os.path.basename(f) for f in files]
        
        # 排序确保一致性
        image_files.sort()
        print(f"总共找到 {len(image_files)} 个PNG图像文件")
        
        return image_files
    
    def _get_default_transforms(self, augment: bool = False):
        """获取默认的数据变换"""
        transforms_list = []
        
        # 基础变换
        transforms_list.extend([
            A.Resize(height=self.img_size, width=self.img_size),
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
        # PNG图像对应PNG标签
        mask_path = os.path.join(self.mask_dir, image_filename)
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
    基于哈希值的固定数据集划分
    
    Args:
        all_images: 所有图像文件名列表
        split_ratio: 训练集比例
        
    Returns:
        (train_images, val_images): 训练集和验证集图像列表
    """
    def get_hash_value(filename):
        """获取文件名的哈希值"""
        return int(hashlib.md5(filename.encode()).hexdigest(), 16)
    
    # 按文件名排序确保一致性
    sorted_images = sorted(all_images)
    
    # 计算每个文件的哈希值并排序
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
        train_size = len(sorted_images) - min_val_size
        train_images = [pair[0] for pair in image_hash_pairs[:train_size]]
        val_images = [pair[0] for pair in image_hash_pairs[train_size:]]
    
    return train_images, val_images


def create_train_val_dataloaders(
    train_img_dir: str,
    train_mask_dir: str,
    batch_size: int = 16,
    img_size: int = 512,
    num_workers: int = 4,
    split_ratio: float = 0.8,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    从训练数据创建训练和验证数据加载器（固定划分）
    
    Args:
        train_img_dir: 训练图像目录
        train_mask_dir: 训练掩码目录
        batch_size: 批次大小
        img_size: 图像尺寸
        num_workers: 数据加载工作进程数
        split_ratio: 训练集比例
        train_transform: 训练集自定义变换
        val_transform: 验证集自定义变换
        pin_memory: 是否使用固定内存
        
    Returns:
        (train_loader, val_loader): 训练和验证数据加载器
    """
    
    print("🔧 从训练数据中创建固定划分的训练集和验证集")
    
    # 获取所有PNG图像文件
    print(f"🔍 正在扫描训练目录: {train_img_dir}")
    pattern = os.path.join(train_img_dir, '*.png')
    files = glob.glob(pattern)
    print(f"   扫描 *.png: 找到 {len(files)} 个文件")
    
    all_images = [os.path.basename(f) for f in files]
    all_images = sorted(list(set(all_images)))  # 去重并排序
    print(f"总共扫描到 {len(all_images)} 个PNG图像文件")
    
    if len(all_images) == 0:
        raise ValueError(f"在 {train_img_dir} 中未找到PNG图像文件")
    
    # 使用固定划分
    train_images, val_images = get_fixed_split(all_images, split_ratio)
    print(f"📊 使用固定划分策略:")
    print(f"   训练集: {len(train_images)} 张图像")
    print(f"   验证集: {len(val_images)} 张图像")
    print(f"   验证集比例: {len(val_images)/len(all_images):.1%}")
    
    # 创建训练和验证数据集
    train_dataset = IronSpectrumDataset(
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        img_size=img_size,
        is_train=True,
        image_list=train_images,
        transform=train_transform,
        augment=True
    )
    val_dataset = IronSpectrumDataset(
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        img_size=img_size,
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
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def create_test_dataloader(
    test_img_dir: str,
    test_mask_dir: str,
    batch_size: int = 1,
    img_size: int = 512,
    num_workers: int = 4,
    transform: Optional[Callable] = None
) -> DataLoader:
    """
    创建测试数据加载器（独立于训练数据）
    
    Args:
        test_img_dir: 测试图像目录
        test_mask_dir: 测试掩码目录
        batch_size: 批次大小（测试时通常为1）
        img_size: 图像尺寸
        num_workers: 数据加载工作进程数
        transform: 自定义变换
        
    Returns:
        test_loader: 测试数据加载器
    """
    print("🔧 创建测试数据加载器（独立测试集）")
    
    test_dataset = IronSpectrumDataset(
        img_dir=test_img_dir,
        mask_dir=test_mask_dir,
        img_size=img_size,
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
    
    print(f"✅ 测试数据加载器创建完成:")
    print(f"   批次数量: {len(test_loader)}")
    print(f"   样本总数: {len(test_dataset)}")
    print(f"   批次大小: {batch_size}")
    
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


def verify_data_structure(train_img_dir: str, train_mask_dir: str, test_img_dir: str, test_mask_dir: str):
    """验证数据结构是否正确"""
    print("🔍 验证数据结构...")
    
    def check_directory(path, description):
        if os.path.exists(path):
            png_files = [f for f in os.listdir(path) if f.endswith('.png')]
            print(f"   {description}: {len(png_files)} 个PNG文件")
            return len(png_files)
        else:
            print(f"   {description}: 目录不存在")
            return 0
    
    print(f"📁 数据结构验证:")
    train_img_count = check_directory(train_img_dir, "训练图像")
    train_mask_count = check_directory(train_mask_dir, "训练标签")
    test_img_count = check_directory(test_img_dir, "测试图像")
    test_mask_count = check_directory(test_mask_dir, "测试标签")
    
    if train_img_count > 0 and test_img_count > 0:
        print("✅ 数据结构验证通过")
        return True
    else:
        print("❌ 数据结构验证失败")
        return False


def test_dataset(train_img_dir: str = None, train_mask_dir: str = None, 
                test_img_dir: str = None, test_mask_dir: str = None):
    """
    测试数据集功能
    
    Args:
        train_img_dir: 训练图像目录（可选，如果不提供则使用默认路径）
        train_mask_dir: 训练标签目录（可选）
        test_img_dir: 测试图像目录（可选）
        test_mask_dir: 测试标签目录（可选）
    """
    print("🚀 铁谱数据集功能测试")
    print("=" * 60)
    
    # 如果没有提供路径，使用默认路径
    if train_img_dir is None:
        current_dir = os.getcwd()
        train_img_dir = os.path.join(current_dir, 'data', 'trainO', 'images')
        train_mask_dir = os.path.join(current_dir, 'data', 'trainO', 'labels')
        test_img_dir = os.path.join(current_dir, 'data', 'testO', 'images')
        test_mask_dir = os.path.join(current_dir, 'data', 'testO', 'labels')
    
    # 验证数据结构
    if not verify_data_structure(train_img_dir, train_mask_dir, test_img_dir, test_mask_dir):
        return
    
    try:
        # 测试训练/验证数据加载器
        print(f"\n🔄 测试训练/验证数据加载器...")
        train_loader, val_loader = create_train_val_dataloaders(
            train_img_dir=train_img_dir,
            train_mask_dir=train_mask_dir,
            batch_size=4,
            img_size=512,
            num_workers=0,
            split_ratio=0.8
        )
        
        print(f"\n📈 训练/验证数据加载器创建成功!")
        print(f"   训练集批次数量: {len(train_loader)}")
        print(f"   验证集批次数量: {len(val_loader)}")
        
        # 测试加载一个批次
        if len(train_loader) > 0:
            for batch in train_loader:
                print(f"\n🔍 训练批次数据:")
                print(f"   图像张量形状: {batch['image'].shape}")
                print(f"   标签张量形状: {batch['mask'].shape}")
                print(f"   图像数值范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
                print(f"   标签数值范围: [{batch['mask'].min():.3f}, {batch['mask'].max():.3f}]")
                break
        
        # 测试测试数据加载器
        print(f"\n🔄 测试测试数据加载器...")
        test_loader = create_test_dataloader(
            test_img_dir=test_img_dir,
            test_mask_dir=test_mask_dir,
            batch_size=1,
            img_size=512,
            num_workers=0
        )
        
        print(f"\n✅ 数据集功能测试完成!")
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 可以在这里传入自定义路径进行测试
    test_dataset()
# import os
# import numpy as np
# import torch
# import cv2
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import glob
# from typing import Optional, Callable, Tuple, List
# import hashlib
# import random


# class MixupDataset(Dataset):
#     """
#     支持Mixup的数据集包装器
    
#     在数据集层面实现Mixup，自动处理样本混合和标签融合
#     """
    
#     def __init__(
#         self,
#         base_dataset: Dataset,
#         mixup_alpha: float = 1.0,
#         mixup_prob: float = 0.5,
#         mixup_mode: str = 'mixup',
#         enable_mixup: bool = True
#     ):
#         """
#         Args:
#             base_dataset: 基础数据集
#             mixup_alpha: Beta分布参数，控制混合强度
#             mixup_prob: 应用Mixup的概率
#             mixup_mode: 混合模式 ['mixup', 'cutmix', 'segmix']
#             enable_mixup: 是否启用Mixup
#         """
#         self.base_dataset = base_dataset
#         self.mixup_alpha = mixup_alpha
#         self.mixup_prob = mixup_prob
#         self.mixup_mode = mixup_mode
#         self.enable_mixup = enable_mixup
        
#         # 缓存数据集长度
#         self._length = len(base_dataset)
        
#         # 为segmix模式预计算前景比例
#         if mixup_mode == 'segmix':
#             self._compute_foreground_ratios()
    
#     def _compute_foreground_ratios(self):
#         """预计算所有样本的前景比例，用于智能配对"""
#         print("🔍 预计算前景比例用于智能Mixup...")
#         self.fg_ratios = []
        
#         for i in range(min(100, len(self.base_dataset))):  # 采样部分样本
#             try:
#                 sample = self.base_dataset[i]
#                 mask = sample['mask']
#                 if isinstance(mask, torch.Tensor):
#                     fg_ratio = mask.mean().item()
#                 else:
#                     fg_ratio = np.mean(mask)
#                 self.fg_ratios.append(fg_ratio)
#             except:
#                 self.fg_ratios.append(0.1)  # 默认值
        
#         # 扩展到全部样本
#         while len(self.fg_ratios) < len(self.base_dataset):
#             self.fg_ratios.extend(self.fg_ratios[:min(100, len(self.base_dataset))])
        
#         self.fg_ratios = self.fg_ratios[:len(self.base_dataset)]
#         print(f"✅ 前景比例计算完成，平均前景比例: {np.mean(self.fg_ratios):.3f}")
    
#     def __len__(self):
#         return self._length
    
#     def _get_lambda(self):
#         """采样混合参数λ"""
#         if self.mixup_alpha > 0:
#             return np.random.beta(self.mixup_alpha, self.mixup_alpha)
#         else:
#             return 1.0
    
#     def _rand_bbox(self, size, lam):
#         """为CutMix生成随机边界框"""
#         H, W = size[-2:]
#         cut_rat = np.sqrt(1. - lam)
#         cut_w = int(W * cut_rat)
#         cut_h = int(H * cut_rat)

#         # 随机中心点
#         cx = np.random.randint(W)
#         cy = np.random.randint(H)

#         bbx1 = np.clip(cx - cut_w // 2, 0, W)
#         bby1 = np.clip(cy - cut_h // 2, 0, H)
#         bbx2 = np.clip(cx + cut_w // 2, 0, W)
#         bby2 = np.clip(cy + cut_h // 2, 0, H)

#         return bbx1, bby1, bbx2, bby2
    
#     def _segmix_smart_pairing(self, idx):
#         """智能配对：前景少的和前景多的配对"""
#         if not hasattr(self, 'fg_ratios'):
#             return np.random.randint(0, len(self.base_dataset))
        
#         current_fg = self.fg_ratios[idx]
        
#         # 寻找互补的样本
#         if current_fg < 0.2:  # 当前样本前景少，找前景多的
#             candidates = [i for i, fg in enumerate(self.fg_ratios) if fg > 0.3 and i != idx]
#         elif current_fg > 0.4:  # 当前样本前景多，找前景少的
#             candidates = [i for i, fg in enumerate(self.fg_ratios) if fg < 0.3 and i != idx]
#         else:  # 中等前景，随机配对
#             candidates = [i for i in range(len(self.base_dataset)) if i != idx]
        
#         if candidates:
#             return np.random.choice(candidates)
#         else:
#             return np.random.randint(0, len(self.base_dataset))
    
#     def _apply_mixup(self, sample_a, sample_b, lam):
#         """应用标准Mixup"""
#         mixed_image = lam * sample_a['image'] + (1 - lam) * sample_b['image']
#         mixed_mask = lam * sample_a['mask'] + (1 - lam) * sample_b['mask']
        
#         return {
#             'image': mixed_image,
#             'mask': mixed_mask,
#             'filename': f"mixup_{sample_a['filename']}_{sample_b['filename']}",
#             'image_path': sample_a['image_path'],
#             'mask_path': sample_a['mask_path'],
#             'is_mixup': True,
#             'mixup_lam': lam,
#             'mixup_mode': 'mixup'
#         }
    
#     def _apply_cutmix(self, sample_a, sample_b, lam):
#         """应用CutMix"""
#         mixed_image = sample_a['image'].clone()
#         mixed_mask = sample_a['mask'].clone()
        
#         # 生成随机边界框
#         bbx1, bby1, bbx2, bby2 = self._rand_bbox(mixed_image.shape, lam)
        
#         # 应用CutMix
#         mixed_image[:, bby1:bby2, bbx1:bbx2] = sample_b['image'][:, bby1:bby2, bbx1:bbx2]
#         mixed_mask[:, bby1:bby2, bbx1:bbx2] = sample_b['mask'][:, bby1:bby2, bbx1:bbx2]
        
#         # 调整λ基于实际混合区域
#         actual_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (mixed_image.shape[-1] * mixed_image.shape[-2]))
        
#         return {
#             'image': mixed_image,
#             'mask': mixed_mask,
#             'filename': f"cutmix_{sample_a['filename']}_{sample_b['filename']}",
#             'image_path': sample_a['image_path'],
#             'mask_path': sample_a['mask_path'],
#             'is_mixup': True,
#             'mixup_lam': actual_lam,
#             'mixup_mode': 'cutmix'
#         }
    
#     def _apply_segmix(self, sample_a, sample_b, lam):
#         """应用SegMix - 基于分割掩码的智能混合"""
#         mask_a = sample_a['mask']
#         mask_b = sample_b['mask']
        
#         # 确保掩码是2D的 (H, W)，如果是3D则取第一个通道
#         if isinstance(mask_a, torch.Tensor):
#             if len(mask_a.shape) == 3:
#                 mask_a_2d = mask_a[0]  # 取第一个通道
#                 mask_b_2d = mask_b[0]
#             else:
#                 mask_a_2d = mask_a
#                 mask_b_2d = mask_b
            
#             fg_a = (mask_a_2d > 0.5).float()
#             fg_b = (mask_b_2d > 0.5).float()
            
#             # 生成智能混合掩码 (H, W)
#             mix_mask = torch.rand_like(fg_a) < lam
            
#             # 保护重要的前景区域
#             important_fg_a = fg_a * (torch.rand_like(fg_a) < 0.7)
#             important_fg_b = fg_b * (torch.rand_like(fg_b) < 0.7)
            
#             mix_mask = torch.where(important_fg_a > 0.5, torch.ones_like(mix_mask), mix_mask)
#             mix_mask = torch.where(important_fg_b > 0.5, torch.zeros_like(mix_mask), mix_mask)
            
#             mix_mask = mix_mask.float()  # (H, W)
            
#             # 扩展混合掩码到图像通道
#             # 图像: (C, H, W), 掩码mix_mask: (H, W)
#             C = sample_a['image'].shape[0]
#             img_mix_mask = mix_mask.unsqueeze(0).expand(C, -1, -1)  # (H, W) -> (1, H, W) -> (C, H, W)
            
#         else:
#             # NumPy处理
#             if len(mask_a.shape) == 3:
#                 mask_a_2d = mask_a[0]
#                 mask_b_2d = mask_b[0]
#             else:
#                 mask_a_2d = mask_a
#                 mask_b_2d = mask_b
                
#             fg_a = (mask_a_2d > 0.5).astype(np.float32)
#             fg_b = (mask_b_2d > 0.5).astype(np.float32)
            
#             mix_mask = (np.random.rand(*fg_a.shape) < lam).astype(np.float32)
            
#             # 保护重要的前景区域
#             important_fg_a = fg_a * (np.random.rand(*fg_a.shape) < 0.7)
#             important_fg_b = fg_b * (np.random.rand(*fg_b.shape) < 0.7)
            
#             mix_mask = np.where(important_fg_a > 0.5, 1.0, mix_mask)
#             mix_mask = np.where(important_fg_b > 0.5, 0.0, mix_mask)
            
#             # 扩展到图像通道
#             C = sample_a['image'].shape[0]
#             img_mix_mask = np.expand_dims(mix_mask, axis=0)
#             img_mix_mask = np.repeat(img_mix_mask, C, axis=0)
        
#         # 应用混合
#         mixed_image = img_mix_mask * sample_a['image'] + (1 - img_mix_mask) * sample_b['image']
        
#         # 对于掩码混合
#         if isinstance(mask_a, torch.Tensor):
#             if len(mask_a.shape) == 3:
#                 # 3D掩码 (1, H, W)
#                 mask_mix_mask = mix_mask.unsqueeze(0)  # (H, W) -> (1, H, W)
#                 mixed_mask = mask_mix_mask * sample_a['mask'] + (1 - mask_mix_mask) * sample_b['mask']
#             else:
#                 # 2D掩码 (H, W)
#                 mixed_mask = mix_mask * sample_a['mask'] + (1 - mix_mask) * sample_b['mask']
#         else:
#             if len(mask_a.shape) == 3:
#                 mask_mix_mask = np.expand_dims(mix_mask, axis=0)
#                 mixed_mask = mask_mix_mask * sample_a['mask'] + (1 - mask_mix_mask) * sample_b['mask']
#             else:
#                 mixed_mask = mix_mask * sample_a['mask'] + (1 - mix_mask) * sample_b['mask']
        
#         # 计算实际的λ
#         actual_lam = mix_mask.mean().item() if isinstance(mix_mask, torch.Tensor) else mix_mask.mean()
        
#         return {
#             'image': mixed_image,
#             'mask': mixed_mask,
#             'filename': f"segmix_{sample_a['filename']}_{sample_b['filename']}",
#             'image_path': sample_a['image_path'],
#             'mask_path': sample_a['mask_path'],
#             'is_mixup': True,
#             'mixup_lam': actual_lam,
#             'mixup_mode': 'segmix'
#         }
    
#     def __getitem__(self, idx):
#         # 获取主样本
#         sample_a = self.base_dataset[idx]
        
#         # 确保基础样本有所有必需的键
#         if 'is_mixup' not in sample_a:
#             sample_a['is_mixup'] = False
#         if 'mixup_lam' not in sample_a:
#             sample_a['mixup_lam'] = 1.0
#         if 'mixup_mode' not in sample_a:
#             sample_a['mixup_mode'] = 'none'
        
#         # 决定是否应用Mixup
#         if not self.enable_mixup or np.random.random() > self.mixup_prob:
#             # 不应用Mixup，返回带有默认标识的样本
#             return sample_a
        
#         # 选择配对样本
#         if self.mixup_mode == 'segmix':
#             idx_b = self._segmix_smart_pairing(idx)
#         else:
#             idx_b = np.random.randint(0, len(self.base_dataset))
#             while idx_b == idx:  # 确保不是同一个样本
#                 idx_b = np.random.randint(0, len(self.base_dataset))
        
#         sample_b = self.base_dataset[idx_b]
        
#         # 确保配对样本也有所有必需的键
#         if 'is_mixup' not in sample_b:
#             sample_b['is_mixup'] = False
#         if 'mixup_lam' not in sample_b:
#             sample_b['mixup_lam'] = 1.0
#         if 'mixup_mode' not in sample_b:
#             sample_b['mixup_mode'] = 'none'
        
#         # 获取混合参数
#         lam = self._get_lambda()
        
#         # 应用相应的混合策略
#         if self.mixup_mode == 'mixup':
#             mixed_sample = self._apply_mixup(sample_a, sample_b, lam)
#         elif self.mixup_mode == 'cutmix':
#             mixed_sample = self._apply_cutmix(sample_a, sample_b, lam)
#         elif self.mixup_mode == 'segmix':
#             mixed_sample = self._apply_segmix(sample_a, sample_b, lam)
#         else:
#             raise ValueError(f"不支持的混合模式: {self.mixup_mode}")
        
#         return mixed_sample


# class IronSpectrumDataset(Dataset):
#     """
#     铁谱图像分割数据集
    
#     支持的数据格式：
#     - 图像：PNG格式 (.png)
#     - 标签：二值化掩码 (.png)
#     """
    
#     def __init__(
#         self,
#         img_dir: str,
#         mask_dir: str,
#         img_size: int = 512,
#         is_train: bool = True,
#         image_list: Optional[List[str]] = None,
#         transform: Optional[Callable] = None,
#         normalize: bool = True,
#         augment: bool = True,
#         # Mixup参数
#         enable_mixup: bool = False,
#         mixup_alpha: float = 1.0,
#         mixup_prob: float = 0.5,
#         mixup_mode: str = 'mixup'
#     ):
#         """
#         Args:
#             img_dir: 图像目录路径
#             mask_dir: 标签目录路径
#             img_size: 图像尺寸
#             is_train: 是否为训练模式
#             image_list: 指定的图像文件列表，如果为None则使用目录下所有图像
#             transform: 自定义变换
#             normalize: 是否归一化图像
#             augment: 是否进行数据增强
#             enable_mixup: 是否启用Mixup (仅训练时有效)
#             mixup_alpha: Mixup的alpha参数
#             mixup_prob: 应用Mixup的概率
#             mixup_mode: Mixup模式 ['mixup', 'cutmix', 'segmix']
#         """
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.img_size = img_size
#         self.is_train = is_train
#         self.normalize = normalize
        
#         # Mixup参数
#         self.enable_mixup = enable_mixup and is_train  # 只在训练时启用
#         self.mixup_alpha = mixup_alpha
#         self.mixup_prob = mixup_prob
#         self.mixup_mode = mixup_mode
        
#         # 获取图像文件列表
#         if image_list is not None:
#             self.image_files = image_list
#         else:
#             self.image_files = self._get_image_files()
        
#         # 确保属性名称一致 - 添加image_list属性以保持兼容性
#         self.image_list = self.image_files
        
#         if len(self.image_files) == 0:
#             raise ValueError(f"在 {self.img_dir} 中未找到PNG图像文件")
        
#         mixup_info = f" (Mixup: {mixup_mode})" if self.enable_mixup else ""
#         print(f"在 {'训练' if is_train else '验证/测试'} 集中找到 {len(self.image_files)} 张PNG图像{mixup_info}")
        
#         # 设置变换
#         if transform is not None:
#             self.transform = transform
#         else:
#             self.transform = self._get_default_transforms(augment and is_train)
        
#         # 如果启用Mixup，预计算前景比例
#         if self.enable_mixup and self.mixup_mode == 'segmix':
#             self._compute_foreground_ratios()
    
#     def _compute_foreground_ratios(self):
#         """预计算所有样本的前景比例，用于智能配对"""
#         print("🔍 预计算前景比例用于智能Mixup...")
#         self.fg_ratios = []
        
#         # 采样部分样本计算前景比例
#         sample_indices = list(range(0, len(self.image_files), max(1, len(self.image_files) // 50)))
        
#         for idx in sample_indices:
#             try:
#                 # 简单加载掩码
#                 image_filename = self.image_files[idx]
#                 mask_path = self._get_mask_path(image_filename)
                
#                 if os.path.exists(mask_path):
#                     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#                     if mask is not None:
#                         mask = (mask > 127).astype(np.float32)
#                         fg_ratio = mask.mean()
#                     else:
#                         fg_ratio = 0.1
#                 else:
#                     fg_ratio = 0.1
                
#                 self.fg_ratios.append(fg_ratio)
#             except:
#                 self.fg_ratios.append(0.1)
        
#         # 扩展到全部样本
#         avg_fg_ratio = np.mean(self.fg_ratios)
#         self.fg_ratios = [avg_fg_ratio] * len(self.image_files)
        
#         # 为采样的样本设置实际值
#         for i, idx in enumerate(sample_indices):
#             if i < len(self.fg_ratios) and idx < len(self.fg_ratios):
#                 # 平滑处理
#                 pass
        
#         print(f"✅ 前景比例估算完成，平均前景比例: {avg_fg_ratio:.3f}")
    
#     def _get_image_files(self):
#         """获取所有PNG图像文件"""
#         print(f"🔍 正在扫描目录: {self.img_dir}")
        
#         # 只扫描PNG格式图片
#         pattern = os.path.join(self.img_dir, '*.png')
#         files = glob.glob(pattern)
#         print(f"   扫描 *.png: 找到 {len(files)} 个文件")
        
#         # 只保留文件名，不包含路径
#         image_files = [os.path.basename(f) for f in files]
        
#         # 排序确保一致性
#         image_files.sort()
#         print(f"总共找到 {len(image_files)} 个PNG图像文件")
        
#         return image_files
    
#     def _get_default_transforms(self, augment: bool = False):
#         """获取默认的数据变换"""
#         transforms_list = []
        
#         # 基础变换
#         transforms_list.extend([
#             A.Resize(height=self.img_size, width=self.img_size),
#         ])
        
#         # 数据增强 (仅训练时)
#         if augment:
#             transforms_list.extend([
#                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.3),
#                 A.RandomRotate90(p=0.5),
#                 A.ShiftScaleRotate(
#                     shift_limit=0.1,
#                     scale_limit=0.1,
#                     rotate_limit=15,
#                     p=0.5
#                 ),
#                 A.RandomBrightnessContrast(
#                     brightness_limit=0.2,
#                     contrast_limit=0.2,
#                     p=0.5
#                 ),
#                 A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
#                 A.ElasticTransform(p=0.3),
#             ])
        
#         # 归一化和转换为张量
#         if self.normalize:
#             transforms_list.append(
#                 A.Normalize(
#                     mean=[0.485, 0.456, 0.406],  # ImageNet标准
#                     std=[0.229, 0.224, 0.225],
#                     max_pixel_value=255.0
#                 )
#             )
        
#         transforms_list.append(ToTensorV2())
        
#         return A.Compose(transforms_list)
    
#     def _get_mask_path(self, image_filename: str) -> str:
#         """根据图像文件名获取对应的标签路径"""
#         # PNG图像对应PNG标签
#         mask_path = os.path.join(self.mask_dir, image_filename)
#         return mask_path
    
#     def _get_lambda(self):
#         """采样混合参数λ"""
#         if self.mixup_alpha > 0:
#             return np.random.beta(self.mixup_alpha, self.mixup_alpha)
#         else:
#             return 1.0
    
#     def _smart_pairing(self, idx):
#         """智能配对：前景少的和前景多的配对"""
#         if not hasattr(self, 'fg_ratios'):
#             return np.random.randint(0, len(self.image_files))
        
#         current_fg = self.fg_ratios[idx]
        
#         # 寻找互补的样本
#         if current_fg < 0.2:  # 当前样本前景少，找前景多的
#             candidates = [i for i, fg in enumerate(self.fg_ratios) if fg > 0.3 and i != idx]
#         elif current_fg > 0.4:  # 当前样本前景多，找前景少的
#             candidates = [i for i, fg in enumerate(self.fg_ratios) if fg < 0.3 and i != idx]
#         else:  # 中等前景，随机配对
#             candidates = [i for i in range(len(self.image_files)) if i != idx]
        
#         if candidates:
#             return np.random.choice(candidates)
#         else:
#             return np.random.randint(0, len(self.image_files))
    
#     def _apply_mixup_augmentation(self, sample_a, sample_b, lam):
#         """应用Mixup数据增强"""
#         if self.mixup_mode == 'mixup':
#             # 标准Mixup
#             mixed_image = lam * sample_a['image'] + (1 - lam) * sample_b['image']
#             mixed_mask = lam * sample_a['mask'] + (1 - lam) * sample_b['mask']
            
#         elif self.mixup_mode == 'cutmix':
#             # CutMix
#             mixed_image = sample_a['image'].clone()
#             mixed_mask = sample_a['mask'].clone()
            
#             H, W = mixed_image.shape[-2:]
#             cut_rat = np.sqrt(1. - lam)
#             cut_w = int(W * cut_rat)
#             cut_h = int(H * cut_rat)
            
#             cx = np.random.randint(W)
#             cy = np.random.randint(H)
            
#             bbx1 = np.clip(cx - cut_w // 2, 0, W)
#             bby1 = np.clip(cy - cut_h // 2, 0, H)
#             bbx2 = np.clip(cx + cut_w // 2, 0, W)
#             bby2 = np.clip(cy + cut_h // 2, 0, H)
            
#             mixed_image[:, bby1:bby2, bbx1:bbx2] = sample_b['image'][:, bby1:bby2, bbx1:bbx2]
#             mixed_mask[:, bby1:bby2, bbx1:bbx2] = sample_b['mask'][:, bby1:bby2, bbx1:bbx2]
            
#             # 调整λ
#             lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            
#         elif self.mixup_mode == 'segmix':
#             # SegMix - 基于分割掩码的智能混合
#             mask_a = sample_a['mask']
#             mask_b = sample_b['mask']
            
#             # 确保掩码是2D的 (H, W)，如果是3D则取第一个通道
#             if len(mask_a.shape) == 3:
#                 mask_a_2d = mask_a[0]  # 取第一个通道
#                 mask_b_2d = mask_b[0]
#             else:
#                 mask_a_2d = mask_a
#                 mask_b_2d = mask_b
            
#             fg_a = (mask_a_2d > 0.5).float()
#             fg_b = (mask_b_2d > 0.5).float()
            
#             # 生成智能混合掩码 (H, W)
#             mix_mask = torch.rand_like(fg_a) < lam
            
#             # 保护重要的前景区域
#             important_fg_a = fg_a * (torch.rand_like(fg_a) < 0.7)
#             important_fg_b = fg_b * (torch.rand_like(fg_b) < 0.7)
            
#             mix_mask = torch.where(important_fg_a > 0.5, torch.ones_like(mix_mask), mix_mask)
#             mix_mask = torch.where(important_fg_b > 0.5, torch.zeros_like(mix_mask), mix_mask)
            
#             mix_mask = mix_mask.float()  # (H, W)
            
#             # 扩展混合掩码到图像通道
#             # 图像通道: (C, H, W), 我们需要将mix_mask从(H, W)扩展到(C, H, W)
#             C, H, W = sample_a['image'].shape
#             img_mix_mask = mix_mask.unsqueeze(0).expand(C, H, W)  # (H, W) -> (1, H, W) -> (C, H, W)
            
#             # 应用混合
#             mixed_image = img_mix_mask * sample_a['image'] + (1 - img_mix_mask) * sample_b['image']
            
#             # 对于掩码，如果原来是3D，保持3D；如果是2D，保持2D
#             if len(mask_a.shape) == 3:
#                 # 掩码是3D (1, H, W)，需要将mix_mask扩展为相同形状
#                 mask_mix_mask = mix_mask.unsqueeze(0)  # (H, W) -> (1, H, W)
#                 mixed_mask = mask_mix_mask * sample_a['mask'] + (1 - mask_mix_mask) * sample_b['mask']
#             else:
#                 # 掩码是2D (H, W)
#                 mixed_mask = mix_mask * sample_a['mask'] + (1 - mix_mask) * sample_b['mask']
            
#             lam = mix_mask.mean().item()
#         else:
#             raise ValueError(f"不支持的混合模式: {self.mixup_mode}")
        
#         # 返回统一格式的字典，确保所有键都存在
#         return {
#             'image': mixed_image,
#             'mask': mixed_mask,
#             'filename': f"{self.mixup_mode}_{sample_a['filename']}_{sample_b['filename']}",
#             'image_path': sample_a['image_path'],
#             'mask_path': sample_a['mask_path'],
#             'is_mixup': True,
#             'mixup_lam': lam,
#             'mixup_mode': self.mixup_mode
#         }
#     def __len__(self):
#         return len(self.image_files)
    
#     def __getitem__(self, idx):
#         # 获取主样本
#         image_filename = self.image_files[idx]
#         img_path = os.path.join(self.img_dir, image_filename)
#         mask_path = self._get_mask_path(image_filename)
        
#         # 读取图像
#         image = cv2.imread(img_path)
#         if image is None:
#             raise ValueError(f"无法加载图像: {img_path}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # 读取标签
#         if os.path.exists(mask_path):
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#             if mask is None:
#                 raise ValueError(f"无法加载掩码: {mask_path}")
            
#             # 二值化标签 (0: 背景, 1: 前景)
#             mask = (mask > 127).astype(np.uint8)
#         else:
#             # 如果没有标签文件，创建空标签
#             mask = np.zeros(
#                 (image.shape[0], image.shape[1]), 
#                 dtype=np.uint8
#             )
#             print(f"⚠️  未找到 {image_filename} 的掩码文件，使用空掩码")
        
#         # 应用变换
#         if self.transform:
#             transformed = self.transform(image=image, mask=mask)
#             image = transformed['image']
#             mask = transformed['mask']
        
#         # 确保mask是float类型并且值在[0,1]范围内
#         if mask.dtype == torch.uint8:
#             mask = mask.float()
        
#         # 为分割任务添加通道维度
#         if len(mask.shape) == 2:
#             mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
        
#         # 创建基础样本字典 - 确保所有必需键都存在
#         sample_a = {
#             'image': image,
#             'mask': mask,
#             'filename': image_filename,
#             'image_path': img_path,
#             'mask_path': mask_path,
#             'is_mixup': False,  # 默认值
#             'mixup_lam': 1.0,   # 默认值
#             'mixup_mode': 'none'  # 默认值
#         }
        
#         # 决定是否应用Mixup
#         if not self.enable_mixup or np.random.random() > self.mixup_prob:
#             # 不应用Mixup，返回带有默认值的样本
#             return sample_a
        
#         # 选择配对样本
#         if self.mixup_mode == 'segmix':
#             idx_b = self._smart_pairing(idx)
#         else:
#             idx_b = np.random.randint(0, len(self.image_files))
#             while idx_b == idx:
#                 idx_b = np.random.randint(0, len(self.image_files))
        
#         # 递归获取配对样本（禁用其Mixup以避免嵌套）
#         original_enable_mixup = self.enable_mixup
#         self.enable_mixup = False
#         sample_b = self.__getitem__(idx_b)
#         self.enable_mixup = original_enable_mixup
        
#         # 应用Mixup
#         lam = self._get_lambda()
#         mixed_sample = self._apply_mixup_augmentation(sample_a, sample_b, lam)
        
#         return mixed_sample
    
#     def get_sample_names(self):
#         """获取所有样本名称"""
#         return [os.path.splitext(f)[0] for f in self.image_files]


# # 更新工厂函数以支持Mixup
# def create_train_val_dataloaders(
#     train_img_dir: str,
#     train_mask_dir: str,
#     batch_size: int = 16,
#     img_size: int = 512,
#     num_workers: int = 4,
#     split_ratio: float = 0.8,
#     train_transform: Optional[Callable] = None,
#     val_transform: Optional[Callable] = None,
#     pin_memory: bool = True,
#     # Mixup参数
#     enable_mixup: bool = False,
#     mixup_alpha: float = 1.0,
#     mixup_prob: float = 0.5,
#     mixup_mode: str = 'mixup'
# ) -> Tuple[DataLoader, DataLoader]:
#     """
#     从训练数据创建训练和验证数据加载器（固定划分，支持Mixup）
    
#     Args:
#         train_img_dir: 训练图像目录
#         train_mask_dir: 训练掩码目录
#         batch_size: 批次大小
#         img_size: 图像尺寸
#         num_workers: 数据加载工作进程数
#         split_ratio: 训练集比例
#         train_transform: 训练集自定义变换
#         val_transform: 验证集自定义变换
#         pin_memory: 是否使用固定内存
#         enable_mixup: 是否启用Mixup (仅训练集)
#         mixup_alpha: Mixup的alpha参数
#         mixup_prob: 应用Mixup的概率
#         mixup_mode: Mixup模式 ['mixup', 'cutmix', 'segmix']
        
#     Returns:
#         (train_loader, val_loader): 训练和验证数据加载器
#     """
    
#     mixup_info = f" (Mixup: {mixup_mode})" if enable_mixup else ""
#     print(f"🔧 从训练数据中创建固定划分的训练集和验证集{mixup_info}")
    
#     # 获取所有PNG图像文件
#     print(f"🔍 正在扫描训练目录: {train_img_dir}")
#     pattern = os.path.join(train_img_dir, '*.png')
#     files = glob.glob(pattern)
#     print(f"   扫描 *.png: 找到 {len(files)} 个文件")
    
#     all_images = [os.path.basename(f) for f in files]
#     all_images = sorted(list(set(all_images)))  # 去重并排序
#     print(f"总共扫描到 {len(all_images)} 个PNG图像文件")
    
#     if len(all_images) == 0:
#         raise ValueError(f"在 {train_img_dir} 中未找到PNG图像文件")
    
#     # 使用固定划分
#     train_images, val_images = get_fixed_split(all_images, split_ratio)
#     print(f"📊 使用固定划分策略:")
#     print(f"   训练集: {len(train_images)} 张图像")
#     print(f"   验证集: {len(val_images)} 张图像")
#     print(f"   验证集比例: {len(val_images)/len(all_images):.1%}")
    
#     # 创建训练和验证数据集
#     train_dataset = IronSpectrumDataset(
#         img_dir=train_img_dir,
#         mask_dir=train_mask_dir,
#         img_size=img_size,
#         is_train=True,
#         image_list=train_images,
#         transform=train_transform,
#         augment=True,
#         # Mixup参数（仅训练集）
#         enable_mixup=enable_mixup,
#         mixup_alpha=mixup_alpha,
#         mixup_prob=mixup_prob,
#         mixup_mode=mixup_mode
#     )
    
#     val_dataset = IronSpectrumDataset(
#         img_dir=train_img_dir,
#         mask_dir=train_mask_dir,
#         img_size=img_size,
#         is_train=False,
#         image_list=val_images,
#         transform=val_transform,
#         augment=False,
#         # 验证集不使用Mixup
#         enable_mixup=False
#     )
    
#     # 创建数据加载器
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         drop_last=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=pin_memory
#     )
    
#     return train_loader, val_loader


# # 其他函数保持不变
# def get_fixed_split(all_images: List[str], split_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
#     """
#     基于哈希值的固定数据集划分
    
#     Args:
#         all_images: 所有图像文件名列表
#         split_ratio: 训练集比例
        
#     Returns:
#         (train_images, val_images): 训练集和验证集图像列表
#     """
#     def get_hash_value(filename):
#         """获取文件名的哈希值"""
#         return int(hashlib.md5(filename.encode()).hexdigest(), 16)
    
#     # 按文件名排序确保一致性
#     sorted_images = sorted(all_images)
    
#     # 计算每个文件的哈希值并排序
#     image_hash_pairs = [(img, get_hash_value(img)) for img in sorted_images]
#     image_hash_pairs.sort(key=lambda x: x[1])  # 按哈希值排序
    
#     # 计算训练集大小
#     train_size = int(len(sorted_images) * split_ratio)
    
#     # 基于哈希值排序的结果进行划分
#     train_images = [pair[0] for pair in image_hash_pairs[:train_size]]
#     val_images = [pair[0] for pair in image_hash_pairs[train_size:]]
    
#     # 确保验证集至少有一定数量
#     min_val_size = max(1, int(len(sorted_images) * 0.1))  # 至少10%
#     if len(val_images) < min_val_size:
#         # 重新调整
#         train_size = len(sorted_images) - min_val_size
#         train_images = [pair[0] for pair in image_hash_pairs[:train_size]]
#         val_images = [pair[0] for pair in image_hash_pairs[train_size:]]
    
#     return train_images, val_images


# def create_test_dataloader(
#     test_img_dir: str,
#     test_mask_dir: str,
#     batch_size: int = 1,
#     img_size: int = 512,
#     num_workers: int = 4,
#     transform: Optional[Callable] = None
# ) -> DataLoader:
#     """
#     创建测试数据加载器（独立于训练数据，不使用Mixup）
    
#     Args:
#         test_img_dir: 测试图像目录
#         test_mask_dir: 测试掩码目录
#         batch_size: 批次大小（测试时通常为1）
#         img_size: 图像尺寸
#         num_workers: 数据加载工作进程数
#         transform: 自定义变换
        
#     Returns:
#         test_loader: 测试数据加载器
#     """
#     print("🔧 创建测试数据加载器（独立测试集）")
    
#     test_dataset = IronSpectrumDataset(
#         img_dir=test_img_dir,
#         mask_dir=test_mask_dir,
#         img_size=img_size,
#         is_train=False,
#         transform=transform,
#         augment=False,
#         enable_mixup=False  # 测试集不使用Mixup
#     )
    
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )
    
#     print(f"✅ 测试数据加载器创建完成:")
#     print(f"   批次数量: {len(test_loader)}")
#     print(f"   样本总数: {len(test_dataset)}")
#     print(f"   批次大小: {batch_size}")
    
#     return test_loader


# def save_split_info(train_images: List[str], val_images: List[str], save_path: str):
#     """保存数据集划分信息到文件"""
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
#     with open(save_path, 'w', encoding='utf-8') as f:
#         f.write("=== 数据集划分信息 ===\n")
#         f.write(f"训练集数量: {len(train_images)}\n")
#         f.write(f"验证集数量: {len(val_images)}\n")
#         f.write(f"总数量: {len(train_images) + len(val_images)}\n")
#         f.write(f"验证集比例: {len(val_images)/(len(train_images) + len(val_images)):.1%}\n\n")
        
#         f.write("训练集图像:\n")
#         for img in sorted(train_images):
#             f.write(f"  {img}\n")
        
#         f.write("\n验证集图像:\n")
#         for img in sorted(val_images):
#             f.write(f"  {img}\n")
    
#     print(f"📝 数据集划分信息已保存至: {save_path}")


# def verify_data_structure(train_img_dir: str, train_mask_dir: str, test_img_dir: str, test_mask_dir: str):
#     """验证数据结构是否正确"""
#     print("🔍 验证数据结构...")
    
#     def check_directory(path, description):
#         if os.path.exists(path):
#             png_files = [f for f in os.listdir(path) if f.endswith('.png')]
#             print(f"   {description}: {len(png_files)} 个PNG文件")
#             return len(png_files)
#         else:
#             print(f"   {description}: 目录不存在")
#             return 0
    
#     print(f"📁 数据结构验证:")
#     train_img_count = check_directory(train_img_dir, "训练图像")
#     train_mask_count = check_directory(train_mask_dir, "训练标签")
#     test_img_count = check_directory(test_img_dir, "测试图像")
#     test_mask_count = check_directory(test_mask_dir, "测试标签")
    
#     if train_img_count > 0 and test_img_count > 0:
#         print("✅ 数据结构验证通过")
#         return True
#     else:
#         print("❌ 数据结构验证失败")
#         return False


# def test_dataset(train_img_dir: str = None, train_mask_dir: str = None, 
#                 test_img_dir: str = None, test_mask_dir: str = None):
#     """
#     测试数据集功能（包括Mixup）
    
#     Args:
#         train_img_dir: 训练图像目录（可选，如果不提供则使用默认路径）
#         train_mask_dir: 训练标签目录（可选）
#         test_img_dir: 测试图像目录（可选）
#         test_mask_dir: 测试标签目录（可选）
#     """
#     print("🚀 铁谱数据集功能测试 (包含Mixup)")
#     print("=" * 60)
    
#     # 如果没有提供路径，使用默认路径
#     if train_img_dir is None:
#         current_dir = os.getcwd()
#         train_img_dir = os.path.join(current_dir, 'data', 'train', 'images')
#         train_mask_dir = os.path.join(current_dir, 'data', 'train', 'labels')
#         test_img_dir = os.path.join(current_dir, 'data', 'test', 'images')
#         test_mask_dir = os.path.join(current_dir, 'data', 'test', 'labels')
    
#     # 验证数据结构
#     if not verify_data_structure(train_img_dir, train_mask_dir, test_img_dir, test_mask_dir):
#         return
    
#     try:
#         # 测试标准训练/验证数据加载器
#         print(f"\n🔄 测试标准训练/验证数据加载器...")
#         train_loader, val_loader = create_train_val_dataloaders(
#             train_img_dir=train_img_dir,
#             train_mask_dir=train_mask_dir,
#             batch_size=4,
#             img_size=512,
#             num_workers=0,
#             split_ratio=0.8,
#             enable_mixup=False
#         )
        
#         print(f"标准数据加载器创建成功!")
#         print(f"   训练集批次数量: {len(train_loader)}")
#         print(f"   验证集批次数量: {len(val_loader)}")
        
#         # 测试Mixup数据加载器
#         print(f"\n🔄 测试Mixup训练数据加载器...")
#         mixup_modes = ['mixup', 'cutmix', 'segmix']
        
#         for mode in mixup_modes:
#             print(f"\n📊 测试 {mode.upper()} 模式:")
            
#             mixup_train_loader, _ = create_train_val_dataloaders(
#                 train_img_dir=train_img_dir,
#                 train_mask_dir=train_mask_dir,
#                 batch_size=2,
#                 img_size=256,  # 使用较小尺寸加快测试
#                 num_workers=0,
#                 split_ratio=0.8,
#                 enable_mixup=True,
#                 mixup_alpha=1.0,
#                 mixup_prob=0.8,
#                 mixup_mode=mode
#             )
            
#             # 测试加载几个批次
#             mixup_count = 0
#             normal_count = 0
            
#             for i, batch in enumerate(mixup_train_loader):
#                 if i >= 3:  # 只测试前3个批次
#                     break
                    
#                 print(f"   批次 {i+1}:")
#                 print(f"     图像形状: {batch['image'].shape}")
#                 print(f"     掩码形状: {batch['mask'].shape}")
                
#                 if isinstance(batch['is_mixup'], torch.Tensor):
#                     is_mixup_list = batch['is_mixup'].tolist()
#                 else:
#                     is_mixup_list = [batch['is_mixup']] if not isinstance(batch['is_mixup'], list) else batch['is_mixup']
                
#                 for j, is_mixup in enumerate(is_mixup_list):
#                     if is_mixup:
#                         mixup_count += 1
#                         # 安全地获取mixup_lam
#                         lam = None
#                         if 'mixup_lam' in batch:
#                             if isinstance(batch['mixup_lam'], (list, torch.Tensor)) and len(batch['mixup_lam']) > j:
#                                 lam = batch['mixup_lam'][j] if isinstance(batch['mixup_lam'], (list, torch.Tensor)) else batch['mixup_lam']
#                             elif not isinstance(batch['mixup_lam'], (list, torch.Tensor)):
#                                 lam = batch['mixup_lam']
                        
#                         if lam is not None:
#                             print(f"     样本 {j}: Mixup样本 (λ={lam:.3f})")
#                         else:
#                             print(f"     样本 {j}: Mixup样本")
#                     else:
#                         normal_count += 1
#                         print(f"     样本 {j}: 标准样本")
            
#             print(f"   {mode} 测试结果: {mixup_count} 个Mixup样本, {normal_count} 个标准样本")
        
#         print(f"\n✅ Mixup数据集功能测试完成!")
        
#     except Exception as e:
#         print(f"❌ 数据集测试失败: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     # 测试数据集功能，包括Mixup
#     test_dataset()