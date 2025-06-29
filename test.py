import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from models.HMA_UNet import create_hma_unet, load_hma_unet
import argparse
from datetime import datetime


class HMAUNetTester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)

        # 初始化HMA-UNet模型 - 简化为只支持base配置
        print(f"🔧 加载HMA-UNet模型: base (唯一可用配置)")
        
        if hasattr(config, 'model_path') and config.model_path and os.path.exists(config.model_path):
            # 从检查点加载模型
            print(f"📂 从检查点加载: {config.model_path}")
            
            self.model = load_hma_unet(
                filepath=config.model_path,
                config="base",  # 强制使用base配置
                in_channels=3,
                num_classes=1
            ).to(self.device)
            print(f"✅ 模型已从 {config.model_path} 加载")
        else:
            # 创建新模型（用于测试架构）
            print("⚠️ 未指定有效模型路径，创建未训练的base模型")
            self.model = create_hma_unet(
                config="base",  # 强制使用base
                in_channels=3,
                num_classes=1
            ).to(self.device)

        self.model.eval()
        print(f"✅ 模型加载完成 (配置: base)")
        
        # 打印模型信息
        try:
            model_info = self.model.get_model_info()
            print(f"📊 模型参数量: {model_info['total_params']:,}")
            print(f"📊 基础通道数: {model_info['base_channels']}")
        except Exception as e:
            print(f"⚠️ 无法获取模型信息: {e}")

    def preprocess_image(self, image):
        """预处理图像"""
        # 转换为tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image_tensor = image_tensor / 255.0

        # 添加归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor.unsqueeze(0)

    def calculate_hd95(self, pred, target):
        """计算95%豪斯多夫距离"""
        if pred.sum() == 0 and target.sum() == 0:
            return 0.0
        if pred.sum() == 0 or target.sum() == 0:
            return 373.0  # 图像对角线长度作为最大距离

        # 计算边界点
        pred_boundary = pred ^ cv2.erode(pred.astype(np.uint8), np.ones((3, 3)))
        target_boundary = target ^ cv2.erode(target.astype(np.uint8), np.ones((3, 3)))

        pred_points = np.argwhere(pred_boundary)
        target_points = np.argwhere(target_boundary)

        if len(pred_points) == 0 or len(target_points) == 0:
            return 373.0

        # 计算距离
        distances = []
        for point in pred_points:
            min_dist = np.min(np.sqrt(np.sum((target_points - point) ** 2, axis=1)))
            distances.append(min_dist)

        for point in target_points:
            min_dist = np.min(np.sqrt(np.sum((pred_points - point) ** 2, axis=1)))
            distances.append(min_dist)

        return np.percentile(distances, 95)

    def calculate_metrics(self, pred, target):
        """计算准确率、HD95、Dice和IoU四个指标"""
        pred_bin = pred > 0.5
        target_bin = target > 0.5

        # 计算基本指标：TP, TN, FP, FN
        TP = np.logical_and(pred_bin, target_bin).sum()
        TN = np.logical_and(~pred_bin, ~target_bin).sum()
        FP = np.logical_and(pred_bin, ~target_bin).sum()
        FN = np.logical_and(~pred_bin, target_bin).sum()
        
        # 计算总像素数
        N = pred.size
        
        # 1. Dice系数 (DSC)
        dice = 2 * TP / (2 * TP + FP + FN + 1e-6)
        
        # 2. IoU (交并比)
        iou = TP / (TP + FP + FN + 1e-6)
        
        # 3. 准确率 (ACC)
        acc = (TP + TN) / (N + 1e-6)
        
        # 4. 95%豪斯多夫距离 (HD95)
        hd95 = self.calculate_hd95(pred_bin, target_bin)
        
        return {
            'dice': dice,
            'iou': iou,
            'accuracy': acc,
            'hd95': hd95
        }

    def draw_contours(self, image, mask):
        """在原图上绘制分割边缘"""
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        result = image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        return result, binary_mask

    def test(self):
        print(f"🚀 开始测试HMA-UNet模型 (配置: base)...")
        
        # 直接从test目录获取图像列表
        test_images = sorted([
            f for f in os.listdir(self.config.test_img_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        print(f"📂 从 {self.config.test_img_dir} 找到 {len(test_images)} 张测试图像")
        
        if len(test_images) == 0:
            print(f"❌ 在 {self.config.test_img_dir} 中未找到图像文件")
            return

        # 初始化四个指标的累加器
        metrics_sum = {
            'dice': 0, 
            'iou': 0, 
            'accuracy': 0, 
            'hd95': 0
        }
        
        # 记录每张图片的指标
        image_metrics = {}

        # 创建子目录用于保存不同类型的图像
        original_dir = os.path.join(self.config.save_dir, "original")
        binary_dir = os.path.join(self.config.save_dir, "binary")
        contour_dir = os.path.join(self.config.save_dir, "contour")
        groundtruth_dir = os.path.join(self.config.save_dir, "groundtruth")
        
        # 创建子目录
        for dir_path in [original_dir, binary_dir, contour_dir, groundtruth_dir]:
            os.makedirs(dir_path, exist_ok=True)

        with torch.no_grad():
            for image_name in tqdm(test_images, desc="处理测试图像"):
                # 读取图像和标签
                image_path = os.path.join(self.config.test_img_dir, image_name)
                mask_path = os.path.join(self.config.test_mask_dir, image_name)
                
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    print(f"⚠️ 无法读取图像: {image_path}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 读取标签
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"⚠️ 无法读取标签: {mask_path}")
                    # 创建空标签
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                
                # 预处理图像
                image_tensor = self.preprocess_image(image)
                image_tensor = image_tensor.to(self.device)

                # 调整大小为模型输入尺寸
                input_tensor = F.interpolate(
                    image_tensor,
                    size=(self.config.img_size, self.config.img_size),
                    mode="bilinear",
                    align_corners=True,
                )

                # 模型预测
                output = self.model(input_tensor)
                pred = torch.sigmoid(output)

                # 还原到原始尺寸
                pred = F.interpolate(
                    pred,
                    size=(image.shape[0], image.shape[1]),
                    mode="bilinear",
                    align_corners=True,
                )

                # 转换预测结果
                pred_np = pred[0, 0].cpu().numpy()
                mask_np = mask / 255.0

                # 计算评价指标
                metrics = self.calculate_metrics(pred_np, mask_np)
                image_metrics[image_name] = metrics
                
                # 累加指标
                for key in metrics_sum:
                    metrics_sum[key] += metrics[key]

                # 绘制分割边缘
                contour_result, binary_pred = self.draw_contours(image, pred_np)
                
                # 保存图像
                # 1. 原图
                cv2.imwrite(
                    os.path.join(original_dir, image_name),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                )
                
                # 2. 真实标签
                cv2.imwrite(
                    os.path.join(groundtruth_dir, image_name),
                    mask
                )
                
                # 3. 预测结果（二值化）
                cv2.imwrite(os.path.join(binary_dir, image_name), binary_pred)
                
                # 4. 轮廓叠加图
                cv2.imwrite(
                    os.path.join(contour_dir, image_name),
                    cv2.cvtColor(contour_result, cv2.COLOR_RGB2BGR)
                )

        # 计算平均指标
        num_images = len(image_metrics)
        if num_images == 0:
            print("❌ 没有成功处理任何图像")
            return
            
        avg_metrics = {k: v / num_images for k, v in metrics_sum.items()}

        # 找出HD95最高的5张图片
        hd95_sorted = sorted(image_metrics.items(), key=lambda x: x[1]['hd95'], reverse=True)[:5]

        # 打印结果
        print("\n===== HMA-UNet模型分割性能评估结果 =====")
        print(f"模型配置: base (唯一可用配置)")
        print(f"测试图像数量: {num_images}")
        print(f"Dice系数 (DSC): {avg_metrics['dice']:.4f}")
        print(f"交并比 (IoU): {avg_metrics['iou']:.4f}")
        print(f"准确率 (ACC): {avg_metrics['accuracy']:.4f}")
        print(f"95%豪斯多夫距离 (HD95): {avg_metrics['hd95']:.4f} 像素")
        print("==========================")

        # 打印HD95最高的图片
        print("\n===== HD95最高的5张图片 =====")
        for i, (img_name, metrics) in enumerate(hd95_sorted):
            print(f"{i+1}. {img_name}: HD95={metrics['hd95']:.4f}, Dice={metrics['dice']:.4f}")
        print("==========================")

        # 保存评价指标
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = os.path.join(self.config.save_dir, f"metrics_{timestamp}.txt")
        
        with open(metrics_file, "w", encoding='utf-8') as f:
            f.write("===== HMA-UNet模型分割性能评估结果 =====\n")
            f.write(f"模型配置: base (唯一可用配置)\n")
            if hasattr(self.config, 'model_path') and self.config.model_path:
                f.write(f"模型路径: {self.config.model_path}\n")
            f.write(f"测试图像数量: {num_images}\n")
            f.write(f"Dice系数 (DSC): {avg_metrics['dice']:.4f}\n")
            f.write(f"交并比 (IoU): {avg_metrics['iou']:.4f}\n")
            f.write(f"准确率 (ACC): {avg_metrics['accuracy']:.4f}\n")
            f.write(f"95%豪斯多夫距离 (HD95): {avg_metrics['hd95']:.4f} 像素\n")
            f.write("==========================\n\n")
            
            # 保存HD95最高的图片信息
            f.write("===== HD95最高的5张图片 =====\n")
            for i, (img_name, metrics) in enumerate(hd95_sorted):
                f.write(f"{i+1}. {img_name}: HD95={metrics['hd95']:.4f}, Dice={metrics['dice']:.4f}\n")
            f.write("==========================\n\n")
            
            # 保存详细指标（每张图片）
            f.write("===== 详细指标（每张图片） =====\n")
            for img_name, metrics in image_metrics.items():
                f.write(f"{img_name}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
                       f"ACC={metrics['accuracy']:.4f}, HD95={metrics['hd95']:.4f}\n")
        
        print(f"\n📁 结果已保存至 {self.config.save_dir}")
        print(f"📄 指标文件: {metrics_file}")
        return avg_metrics


def get_available_configs():
    """获取可用的模型配置 - 简化为只支持base"""
    return ['base']


def parse_args():
    """解析命令行参数 - 简化版本"""
    parser = argparse.ArgumentParser(description='HMA-UNet模型测试 (仅支持base配置)')
    
    # 模型相关参数 - 简化
    parser.add_argument('--model_path', type=str, default=None,
                       help='训练好的模型路径（.pth文件）')
    
    # 数据相关参数
    parser.add_argument('--test_img_dir', type=str, default='./data/test/images',
                       help='测试图像目录')
    parser.add_argument('--test_mask_dir', type=str, default='./data/test/labels',
                       help='测试标签目录')
    parser.add_argument('--img_size', type=int, default=512,
                       help='输入图像尺寸')
    
    # 输出相关参数
    parser.add_argument('--save_dir', type=str, default='./results/test/',
                       help='结果保存目录')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (auto/cuda/cpu)')
    
    return parser.parse_args()


class Config:
    """配置类 - 简化版本"""
    def __init__(self, args):
        # 从命令行参数初始化
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        # 固定使用base配置
        self.model_config = 'base'
        
        # 自动设备选择
        if self.device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 确保目录存在
        os.makedirs(self.save_dir, exist_ok=True)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    config = Config(args)
    
    # 打印配置信息
    print("=" * 60)
    print("HMA-UNet 模型测试配置 (仅支持base)")
    print("=" * 60)
    print(f"模型配置: base (唯一可用)")
    if config.model_path:
        print(f"模型路径: {config.model_path}")
        print(f"模型文件存在: {os.path.exists(config.model_path) if config.model_path else False}")
    else:
        print("模型路径: 未指定 (将创建未训练模型)")
    print(f"测试图像目录: {config.test_img_dir}")
    print(f"测试标签目录: {config.test_mask_dir}")
    print(f"图像尺寸: {config.img_size}")
    print(f"设备: {config.device}")
    print(f"结果保存目录: {config.save_dir}")
    print("=" * 60)
    
    # 验证数据目录
    if not os.path.exists(config.test_img_dir):
        print(f"❌ 测试图像目录不存在: {config.test_img_dir}")
        print("请检查路径或创建测试数据")
        return
    
    if not os.path.exists(config.test_mask_dir):
        print(f"❌ 测试标签目录不存在: {config.test_mask_dir}")
        print("请检查路径或创建测试标签")
        return
    
    # 检查CUDA可用性
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，切换到CPU")
        config.device = 'cpu'
    
    # 开始测试
    try:
        tester = HMAUNetTester(config)
        print("\n🎯 开始模型测试...")
        metrics = tester.test()
        
        if metrics:
            print("\n🎉 测试完成！")
            print(f"📊 最终结果: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
                  f"ACC={metrics['accuracy']:.4f}, HD95={metrics['hd95']:.4f}")
        else:
            print("❌ 测试失败")
            
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 可以直接运行的简化版本
    class SimpleConfig:
        img_size = 512
        model_path = "./checkpoints/HMA_UNet/HMA_UNet_base_20250627_101313/HMA_UNet_base_best_model.pth"  # 根据您的工作空间调整
        test_img_dir = "./data/test/images/"  # 使用您工作空间中的test目录
        test_mask_dir = "./data/test/labels/"  # 使用您工作空间中的test目录
        save_dir = "./results/HMA_UNet/"
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 检查是否有命令行参数
    import sys
    if len(sys.argv) > 1:
        # 有命令行参数，使用argparse
        main()
    else:
        # 没有命令行参数，使用简化配置直接运行
        print("🚀 使用默认配置运行HMA-UNet测试")
        tester = HMAUNetTester(SimpleConfig())
        metrics = tester.test()
        
        if metrics:
            print("\n🎉 测试完成！")
            print(f"📊 最终结果: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
                  f"ACC={metrics['accuracy']:.4f}, HD95={metrics['hd95']:.4f}")
        else:
            print("❌ 测试失败")