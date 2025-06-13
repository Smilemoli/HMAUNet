import torch
import torch.nn as nn
import traceback
import time
import sys
import os

# 确保从正确的目录运行
if os.getcwd().endswith("models"):
    os.chdir("..")  # 切换到项目根目录

# 添加项目根目录到Python路径
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Current working directory:", os.getcwd())
print("Python path:", sys.path[:3])


def test_imports():
    """测试导入是否正常工作"""
    print("=" * 80)
    print("Import Test")
    print("=" * 80)

    try:
        # 测试各个组件的导入
        print("🧩 Testing component imports...")

        # 首先测试基础模块
        try:
            from models.backbones.ResNet_blocks import (
                ResidualBlock,
                ConvBlock,
                TransposeConvBlock,
            )
            print("   ✅ ResNet blocks imported successfully")
        except Exception as e:
            print(f"   ❌ ResNet blocks import failed: {e}")
            return False

        try:
            from models.backbones.convnext_blocks import ConvNeXtV2Block
            print("   ✅ ConvNeXt blocks imported successfully")
        except Exception as e:
            print(f"   ❌ ConvNeXt blocks import failed: {e}")
            return False

        try:
            from models.backbones.vss_blocks import VSSBlock
            print("   ✅ VSS blocks imported successfully")
        except Exception as e:
            print(f"   ❌ VSS blocks import failed: {e}")
            return False

        try:
            from models.components.csfg_module import CSFGModule
            print("   ✅ CSFGModule imported successfully")
        except Exception as e:
            print(f"   ❌ CSFGModule import failed: {e}")
            return False

        try:
            from models.components.encoder import HybridEncoder
            print("   ✅ HybridEncoder imported successfully")
        except Exception as e:
            print(f"   ❌ HybridEncoder import failed: {e}")
            return False

        try:
            from models.components.hma_module import HMABottleneck
            print("   ✅ HMABottleneck imported successfully")
        except Exception as e:
            print(f"   ❌ HMABottleneck import failed: {e}")
            return False

        try:
            from models.components.decoder import ResNetDecoder, OutputHead
            print("   ✅ ResNetDecoder imported successfully")
        except Exception as e:
            print(f"   ❌ ResNetDecoder import failed: {e}")
            return False

        print("   ✅ All component imports test PASSED")
        return True

    except Exception as e:
        print(f"   ❌ Components test FAILED: {e}")
        traceback.print_exc()
        return False


def create_hma_unet_local(config="tiny", in_channels=3, num_classes=1, **kwargs):
    """本地创建HMA-UNet模型的函数"""
    from models.components.encoder import HybridEncoder
    from models.components.hma_module import HMABottleneck
    from models.components.decoder import ResNetDecoder, OutputHead

    # 根据配置设置参数
    if config == "tiny":
        base_channels = 32
        encoder_depths = [2, 2, 2, 2]
        encoder_drop_path_rate = 0.1
        bottleneck_num_levels = 3
        bottleneck_drop_path_rate = 0.15
        csfg_reduction_ratio = 8
        decoder_num_res_blocks = 2
        d_state = 16
        dropout = 0.1
    elif config == "small":
        base_channels = 48
        encoder_depths = [2, 2, 4, 2]
        encoder_drop_path_rate = 0.15
        bottleneck_num_levels = 3
        bottleneck_drop_path_rate = 0.2
        csfg_reduction_ratio = 6
        decoder_num_res_blocks = 3
        d_state = 16
        dropout = 0.1
    elif config == "base":
        base_channels = 64
        encoder_depths = [3, 3, 6, 3]
        encoder_drop_path_rate = 0.2
        bottleneck_num_levels = 4
        bottleneck_drop_path_rate = 0.25
        csfg_reduction_ratio = 4
        decoder_num_res_blocks = 3
        d_state = 24
        dropout = 0.1
    else:
        raise ValueError(f"Unknown config: {config}")

    # 应用额外参数
    for key, value in kwargs.items():
        locals()[key] = value

    class HMAUNetLocal(nn.Module):
        def __init__(self):
            super().__init__()

            self.in_channels = in_channels
            self.num_classes = num_classes
            self.base_channels = base_channels

            # 计算各阶段通道数
            self.encoder_channels = [
                2 * base_channels,  # Stage 1: 2C
                4 * base_channels,  # Stage 2: 4C
                8 * base_channels,  # Stage 3: 8C
                8 * base_channels,  # Stage 4: 8C
            ]

            # 1. 混合式编码器
            self.encoder = HybridEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                depths=encoder_depths,
                drop_path_rate=encoder_drop_path_rate,
                d_state=d_state,
            )

            # 2. HMA瓶颈层
            self.bottleneck = HMABottleneck(
                in_channels=self.encoder_channels[3],
                out_channels=self.encoder_channels[3],
                d_state=d_state,
                num_levels=bottleneck_num_levels,
                drop_path_rate=bottleneck_drop_path_rate,
                use_checkpoint=False,
            )

            # 3. ResNet解码器
            self.decoder = ResNetDecoder(
                base_channels=base_channels,
                encoder_channels=self.encoder_channels,
                reduction_ratio=csfg_reduction_ratio,
                use_transpose_conv=True,
                num_res_blocks=decoder_num_res_blocks,
            )

            # 4. 输出头
            self.output_head = OutputHead(
                in_channels=base_channels, num_classes=num_classes, dropout=dropout
            )

        def forward(self, x):
            input_size = x.shape[2:]

            # 编码器
            encoder_features = self.encoder(x)
            x_enc1, x_enc2, x_enc3, x_enc4 = encoder_features

            # Stem特征
            x_stem = self.encoder.stem(x)

            # 瓶颈层
            bottleneck_features = self.bottleneck(x_enc4)

            # 解码器
            decoder_features = [x_enc1, x_enc2, x_enc3, x_stem]
            decoded_features = self.decoder(bottleneck_features, decoder_features)

            # 输出头
            output = self.output_head(decoded_features)

            # 确保输出尺寸匹配
            if output.shape[2:] != input_size:
                output = torch.nn.functional.interpolate(
                    output, size=input_size, mode="bilinear", align_corners=True
                )

            return output

        def get_model_info(self):
            return {
                "model_name": "HMA-UNet",
                "input_channels": self.in_channels,
                "num_classes": self.num_classes,
                "base_channels": self.base_channels,
                "encoder_channels": self.encoder_channels,
                "total_params": sum(p.numel() for p in self.parameters()),
                "trainable_params": sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                ),
            }

    return HMAUNetLocal()


def test_basic_components():
    """测试基础组件"""
    print("\n" + "=" * 80)
    print("Basic Components Test")
    print("=" * 80)

    try:
        # 测试ResNet块
        print("🔧 Testing ResNet blocks...")
        from models.backbones.ResNet_blocks import ResidualBlock

        res_block = ResidualBlock(64, 64)
        x = torch.randn(1, 64, 32, 32)
        with torch.no_grad():
            y = res_block(x)
        print(f"   ResidualBlock: {x.shape} -> {y.shape}")

        # 测试ConvNeXt块
        print("🔧 Testing ConvNeXt blocks...")
        from models.backbones.convnext_blocks import ConvNeXtV2Block

        convnext_block = ConvNeXtV2Block(64)
        x = torch.randn(1, 64, 32, 32)  # ConvNeXt接受(N,C,H,W)格式
        with torch.no_grad():
            y = convnext_block(x)
        print(f"   ConvNeXtV2Block: {x.shape} -> {y.shape}")

        # 测试VSS块（使用简化版本）
        print("🔧 Testing VSS blocks...")
        from models.backbones.vss_blocks import VSSBlock

        vss_block = VSSBlock(hidden_dim=64)
        x = torch.randn(1, 32, 32, 64)  # VSS期望(N,H,W,C)格式
        with torch.no_grad():
            y = vss_block(x)
        print(f"   VSSBlock: {x.shape} -> {y.shape}")

        print("   ✅ Basic components test PASSED")

    except Exception as e:
        print(f"   ❌ Basic components test FAILED: {e}")
        traceback.print_exc()


def test_encoder_features():
    """测试编码器特征输出"""
    print("\n" + "=" * 80)
    print("Encoder Feature Maps Test")
    print("=" * 80)

    try:
        from models.components.encoder import HybridEncoder

        # 创建编码器
        encoder = HybridEncoder(in_channels=3, base_channels=32)
        encoder.eval()  # 设置为评估模式

        # 测试输入
        x = torch.randn(1, 3, 128, 128)
        print(f"Input shape: {x.shape}")

        # 获取编码器特征
        print("\n🔍 Testing encoder feature extraction...")
        with torch.no_grad():
            encoder_features = encoder(x)

        print("\nEncoder feature shapes:")
        for i, feature in enumerate(encoder_features):
            print(f"   Stage {i+1}: {feature.shape}")

        # 测试stem输出
        with torch.no_grad():
            stem_output = encoder.stem(x)
        print(f"   Stem output: {stem_output.shape}")

        print("   ✅ Encoder features test PASSED")

    except Exception as e:
        print(f"   ❌ Encoder features test FAILED: {e}")
        traceback.print_exc()


def test_bottleneck_module():
    """测试HMA瓶颈层"""
    print("\n" + "=" * 80)
    print("HMA Bottleneck Module Test")
    print("=" * 80)

    try:
        from models.components.hma_module import HMABottleneck

        # 创建瓶颈层
        bottleneck = HMABottleneck(in_channels=256, d_state=16)
        bottleneck.eval()  # 设置为评估模式

        # 模拟编码器输出 - 修正为H/16
        x_enc4 = torch.randn(1, 256, 8, 8)  # H/16 = 128/16 = 8
        print(f"Bottleneck input shape: {x_enc4.shape}")

        # 测试瓶颈层
        print("\n🔄 Testing HMA bottleneck...")
        with torch.no_grad():
            bottleneck_output = bottleneck(x_enc4)

        print(f"Bottleneck output shape: {bottleneck_output.shape}")

        # 验证输出形状
        assert (
            bottleneck_output.shape == x_enc4.shape
        ), f"Expected {x_enc4.shape}, got {bottleneck_output.shape}"

        # 获取特征信息
        feature_info = bottleneck.get_feature_info()
        print("\nBottleneck feature info:")
        for key, value in feature_info.items():
            print(f"   {key}: {value}")

        print("   ✅ HMA bottleneck test PASSED")

    except Exception as e:
        print(f"   ❌ HMA bottleneck test FAILED: {e}")
        traceback.print_exc()


def test_csfg_module():
    """测试CSFG智能跳跃连接模块"""
    print("\n" + "=" * 80)
    print("CSFG Module Test")
    print("=" * 80)

    try:
        from models.components.csfg_module import CSFGModule, CSFGSkipConnection

        # 测试基础CSFG模块
        print("🔧 Testing basic CSFG module...")
        csfg = CSFGModule(enc_channels=128, dec_channels=64)
        csfg.eval()  # 设置为评估模式

        x_enc = torch.randn(1, 128, 32, 32)
        g_up = torch.randn(1, 64, 32, 32)

        with torch.no_grad():
            x_fused = csfg(x_enc, g_up)
        print(f"   CSFG fusion: enc{x_enc.shape} + dec{g_up.shape} -> {x_fused.shape}")

        # 测试完整的CSFG跳跃连接
        print("🔧 Testing CSFG skip connection...")
        csfg_skip = CSFGSkipConnection(
            enc_channels=128, dec_channels=64, out_channels=96
        )
        csfg_skip.eval()  # 设置为评估模式

        with torch.no_grad():
            skip_output = csfg_skip(x_enc, g_up)
        print(
            f"   CSFG skip: enc{x_enc.shape} + dec{g_up.shape} -> {skip_output.shape}"
        )

        # 测试注意力权重获取
        print("🔧 Testing attention weights...")
        attention_weights = csfg.get_attention_weights(x_enc, g_up)
        print(f"   Detail weight: {attention_weights['detail_weight'][0]:.3f}")
        print(f"   Local weight: {attention_weights['local_weight'][0]:.3f}")
        print(f"   Context weight: {attention_weights['context_weight'][0]:.3f}")

        print("   ✅ CSFG module test PASSED")

    except Exception as e:
        print(f"   ❌ CSFG module test FAILED: {e}")
        traceback.print_exc()


def test_decoder_module():
    """测试解码器模块"""
    print("\n" + "=" * 80)
    print("Decoder Module Test")
    print("=" * 80)

    try:
        from models.components.decoder import ResNetDecoder

        # 创建解码器
        decoder = ResNetDecoder(base_channels=32)
        decoder.eval()  # 设置为评估模式

        # 修正输入尺寸 - 基于编码器的实际输出
        bottleneck_features = torch.randn(1, 256, 8, 8)   # HMA输出 (H/16)
        x_enc1 = torch.randn(1, 64, 32, 32)   # Stage 1 output (H/4)  
        x_enc2 = torch.randn(1, 128, 16, 16)  # Stage 2 output (H/8)
        x_enc3 = torch.randn(1, 256, 8, 8)    # Stage 3 output (H/16)
        x_stem = torch.randn(1, 32, 64, 64)   # Stem output (H/2)

        encoder_features = [x_enc1, x_enc2, x_enc3, x_stem]

        print("Decoder input shapes:")
        print(f"   Bottleneck: {bottleneck_features.shape}")
        for i, feat in enumerate(encoder_features):
            stage_name = ["enc1", "enc2", "enc3", "stem"][i]
            print(f"   {stage_name}: {feat.shape}")

        # 测试解码器
        print("\n🔄 Testing decoder...")
        with torch.no_grad():
            decoder_output = decoder(bottleneck_features, encoder_features)

        print(f"Decoder output shape: {decoder_output.shape}")

        # 获取解码器各阶段通道信息
        decoder_channels = decoder.get_feature_channels()
        print("\nDecoder stage channels:")
        for stage, channels in decoder_channels.items():
            print(f"   {stage}: {channels}")

        print("   ✅ Decoder test PASSED")

    except Exception as e:
        print(f"   ❌ Decoder test FAILED: {e}")
        traceback.print_exc()


def test_model_architecture():
    """测试模型架构和前向传播"""
    print("\n" + "=" * 80)
    print("HMA-UNet Model Architecture Test")
    print("=" * 80)

    # 测试配置 - 使用更小的尺寸和简化配置
    configs = {
        "tiny": {
            "input_size": (128, 128),  # 使用128x128避免尺寸问题
            "batch_size": 1,
            "expected_params": "~2-3M",
        },
    }

    for config_name, config in configs.items():
        print(f"\n🧪 Testing {config_name.upper()} configuration...")
        try:
            # 创建模型
            model = create_hma_unet_local(
                config=config_name, in_channels=3, num_classes=1
            )
            model.eval()  # 设置为评估模式

            # 创建测试输入
            batch_size = config["batch_size"]
            input_size = config["input_size"]
            x = torch.randn(batch_size, 3, *input_size)

            print(f"   Input shape: {x.shape}")

            # 模型信息
            model_info = model.get_model_info()
            total_params = model_info["total_params"]
            print(f"   Total parameters: {total_params:,}")
            print(f"   Expected: {config['expected_params']}")
            print(f"   Base channels: {model_info['base_channels']}")

            # 前向传播测试
            print("   🔄 Testing forward pass...")
            with torch.no_grad():
                start_time = time.time()
                output = model(x)
                forward_time = time.time() - start_time

            print(f"   Output shape: {output.shape}")
            print(f"   Forward time: {forward_time:.4f}s")

            # 验证输出形状
            expected_shape = (batch_size, 1, *input_size)
            assert (
                output.shape == expected_shape
            ), f"Expected {expected_shape}, got {output.shape}"

            # 验证输出值范围
            output_min, output_max = output.min().item(), output.max().item()
            print(f"   Output range: [{output_min:.4f}, {output_max:.4f}]")

            print(f"   ✅ {config_name.upper()} configuration test PASSED")

        except Exception as e:
            print(f"   ❌ {config_name.upper()} configuration test FAILED: {e}")
            traceback.print_exc()


def test_model_gradient_flow():
    """测试模型梯度流动"""
    print("\n" + "=" * 80)
    print("Gradient Flow Test")
    print("=" * 80)

    try:
        # 创建模型
        model = create_hma_unet_local(config="tiny", in_channels=3, num_classes=1)
        model.train()

        # 创建测试数据
        x = torch.randn(1, 3, 128, 128, requires_grad=True)
        y_true = torch.randint(0, 2, (1, 1, 128, 128)).float()

        # 前向传播
        y_pred = model(x)

        # 计算简单损失
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)

        # 反向传播
        loss.backward()

        # 检查梯度
        has_grad = 0
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad += 1

        grad_ratio = has_grad / total_params if total_params > 0 else 0

        print(f"   Loss value: {loss.item():.6f}")
        print(
            f"   Parameters with gradients: {has_grad}/{total_params} ({grad_ratio:.2%})"
        )

        if grad_ratio > 0.8:  # 80%以上的参数有梯度
            print("   ✅ Gradient flow test PASSED")
        else:
            print("   ⚠️  Warning: Low gradient coverage")

    except Exception as e:
        print(f"   ❌ Gradient flow test FAILED: {e}")
        traceback.print_exc()


def test_data_flow_consistency():
    """测试数据流一致性"""
    print("\n" + "=" * 80)
    print("Data Flow Consistency Test")
    print("=" * 80)

    try:
        # 创建模型
        model = create_hma_unet_local(config="tiny", in_channels=3, num_classes=1)
        model.eval()

        # 测试输入
        x = torch.randn(1, 3, 128, 128)
        print(f"Input: {x.shape}")

        # 逐步测试数据流
        print("\n🔍 Testing step-by-step data flow...")

        with torch.no_grad():
            # 1. Stem层
            stem_out = model.encoder.stem(x)
            print(f"   Stem output: {stem_out.shape}")

            # 2. 编码器各阶段
            enc_features = model.encoder(x)
            for i, feat in enumerate(enc_features):
                print(f"   Encoder stage {i+1}: {feat.shape}")

            # 3. 瓶颈层
            bottleneck_out = model.bottleneck(enc_features[-1])
            print(f"   Bottleneck output: {bottleneck_out.shape}")

            # 4. 解码器
            decoder_features = [
                enc_features[0],
                enc_features[1],
                enc_features[2],
                stem_out,
            ]
            decoded_out = model.decoder(bottleneck_out, decoder_features)
            print(f"   Decoder output: {decoded_out.shape}")

            # 5. 输出头
            final_out = model.output_head(decoded_out)
            print(f"   Final output: {final_out.shape}")

            # 6. 完整前向传播
            complete_out = model(x)
            print(f"   Complete forward: {complete_out.shape}")

            # 验证一致性
            assert torch.allclose(
                final_out, complete_out, atol=1e-6
            ), "Output mismatch!"

        print("   ✅ Data flow consistency test PASSED")

    except Exception as e:
        print(f"   ❌ Data flow consistency test FAILED: {e}")
        traceback.print_exc()


def test_memory_usage():
    """测试内存使用情况"""
    print("\n" + "=" * 80)
    print("Memory Usage Test")
    print("=" * 80)

    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # 不同配置的内存测试
        configs = ["tiny"]

        for config in configs:
            print(f"\n💾 Testing {config} configuration memory usage...")

            # 记录初始内存
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # 创建模型
            model = create_hma_unet_local(config=config, in_channels=3, num_classes=1)
            model.eval()  # 设置为评估模式

            # 记录模型加载后内存
            model_memory = process.memory_info().rss / 1024 / 1024  # MB

            # 前向传播
            x = torch.randn(1, 3, 128, 128)
            with torch.no_grad():
                output = model(x)

            # 记录前向传播后内存
            forward_memory = process.memory_info().rss / 1024 / 1024  # MB

            print(f"   Initial memory: {initial_memory:.1f} MB")
            print(
                f"   After model loading: {model_memory:.1f} MB (+{model_memory-initial_memory:.1f} MB)"
            )
            print(
                f"   After forward pass: {forward_memory:.1f} MB (+{forward_memory-model_memory:.1f} MB)"
            )

            # 清理
            del model, x, output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("   ✅ Memory usage test PASSED")

    except ImportError:
        print("   ⚠️  psutil not available, skipping memory test")
    except Exception as e:
        print(f"   ❌ Memory usage test FAILED: {e}")


def main():
    """主测试函数"""
    print("🚀 Starting HMA-UNet Model Tests...\n")

    # 设置随机种子
    torch.manual_seed(42)

    # 首先测试导入
    if not test_imports():
        print("❌ Import test failed. Cannot proceed with other tests.")
        return

    # 运行其他测试
    test_basic_components()
    test_encoder_features()
    test_csfg_module()
    test_bottleneck_module()
    test_decoder_module()
    test_model_architecture()
    test_model_gradient_flow()
    test_data_flow_consistency()
    test_memory_usage()

    print("\n" + "=" * 80)
    print("🎉 HMA-UNet tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()