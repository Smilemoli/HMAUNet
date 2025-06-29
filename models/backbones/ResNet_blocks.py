import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18/34."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, 
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block for ResNet-50/101/152."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation,
                              groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet34Encoder(nn.Module):
    """ResNet-34 encoder for feature extraction."""
    
    def __init__(self, in_channels=3, norm_layer=None, pretrained=False):
        super(ResNet34Encoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.norm_layer = norm_layer
        self.in_channels = 64
        
        # Initial layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-34 layers: [3, 4, 6, 3] blocks
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        
        # Initialize weights
        self._initialize_weights()
        
        if pretrained:
            self._load_pretrained_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, 
                           norm_layer=norm_layer))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, norm_layer=norm_layer))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_weights(self):
        """Load pretrained ResNet-34 weights from torchvision."""
        try:
            import torchvision.models as models
            pretrained_model = models.resnet34(pretrained=True)
            
            # Load state dict excluding the final classification layer
            model_dict = self.state_dict()
            pretrained_dict = pretrained_model.state_dict()
            
            # Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and 'fc' not in k}
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print("Loaded pretrained ResNet-34 weights")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)  # /2
        x = self.bn1(x)
        x = self.relu(x)
        x0 = x  # 1/2 resolution, 64 channels
        
        x = self.maxpool(x)  # /4
        
        # ResNet layers
        x1 = self.layer1(x)  # 1/4 resolution, 64 channels
        x2 = self.layer2(x1)  # 1/8 resolution, 128 channels
        x3 = self.layer3(x2)  # 1/16 resolution, 256 channels
        x4 = self.layer4(x3)  # 1/32 resolution, 512 channels
        
        return [x0, x1, x2, x3, x4]


class ResidualBlock(nn.Module):
    """Simple residual block for feature refinement in HMA and CSFG modules."""
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, 
                 padding=1, norm_layer=None, activation=None, dropout=0.0):
        super(ResidualBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.norm1 = norm_layer(out_channels)
        self.activation = activation
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                              stride=1, padding=padding, bias=False)
        self.norm2 = norm_layer(out_channels)
        
        # Identity mapping or projection
        self.projection = None
        if in_channels != out_channels or stride != 1:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.projection is not None:
            identity = self.projection(x)
        
        out += identity
        out = self.activation(out)
        
        return out


class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, bias=False, norm_layer=None, activation=None, dropout=0.0):
        super(ConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=bias)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.activation = activation if activation else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DoubleConvBlock(nn.Module):
    """Double convolution block commonly used in U-Net architectures."""
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_layer=None, 
                 activation=None, dropout=0.0):
        super(DoubleConvBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            activation,
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            activation
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UpConvBlock(nn.Module):
    """Upsampling convolution block for decoder."""
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='bilinear'):
        super(UpConvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class TransposeConvBlock(nn.Module):
    """Transpose convolution block for upsampling."""
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                 norm_layer=None, activation=None):
        super(TransposeConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
        
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, 
                                               kernel_size=kernel_size, stride=stride, 
                                               padding=padding, bias=False)
        self.norm = norm_layer(out_channels)
        self.activation = activation
    
    def forward(self, x):
        x = self.transpose_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


# ResNet-34 configuration
def get_resnet34_config():
    """Get ResNet-34 configuration."""
    return {
        'block': BasicBlock,
        'layers': [3, 4, 6, 3],  # ResNet-34 layer configuration
        'channels': [64, 128, 256, 512],  # Output channels for each layer
        'strides': [1, 2, 2, 2],  # Strides for each layer
    }


def resnet34_encoder(in_channels=3, pretrained=False):
    """Create ResNet-34 encoder."""
    return ResNet34Encoder(in_channels=in_channels, pretrained=pretrained)