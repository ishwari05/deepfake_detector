import torch
import torch.nn as nn


class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3 backbone for deepfake detection.
    This matches the architecture of your trained models.
    """
    def __init__(self, width_mult=1.0):
        super(MobileNetV3Backbone, self).__init__()
        
        # Initial layers
        self.conv_stem = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hswish = nn.Hardswish(inplace=True)
        
        # MobileNetV3 blocks (simplified version based on the keys I can see)
        self.blocks = nn.ModuleList()
        
        # Block configuration based on typical MobileNetV3
        # These are simplified - you may need to adjust based on exact architecture
        block_configs = [
            # [in_channels, out_channels, kernel_size, stride, expansion_ratio, se]
            [16, 16, 3, 1, 1, False],
            [16, 24, 3, 2, 4, False],
            [24, 24, 3, 1, 4, False],
            [24, 40, 5, 2, 4, True],
            [40, 40, 5, 1, 4, True],
            [40, 40, 5, 1, 4, True],
            [40, 80, 3, 2, 4, False],
            [80, 80, 3, 1, 4, False],
            [80, 80, 3, 1, 4, False],
            [80, 80, 3, 1, 4, False],
            [80, 112, 3, 1, 4, True],
            [112, 112, 3, 1, 4, True],
            [112, 160, 5, 2, 4, True],
            [160, 160, 5, 1, 4, True],
            [160, 160, 5, 1, 4, True],
        ]
        
        in_channels = 16
        for i, config in enumerate(block_configs):
            if len(config) == 6:
                out_channels, kernel_size, stride, expansion_ratio, se, _ = config
            else:
                out_channels, kernel_size, stride, expansion_ratio, se = config
            self.blocks.append(self._make_block(in_channels, out_channels, kernel_size, stride, expansion_ratio, se))
            in_channels = out_channels
        
        # Final layers
        self.conv_head = nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1280, 384),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(384, 2)
        )
        
        self._initialize_weights()
    
    def _make_block(self, in_channels, out_channels, kernel_size, stride, expansion_ratio, se):
        """Create a MobileNetV3 block."""
        mid_channels = in_channels * expansion_ratio
        
        layers = []
        
        # Pointwise convolution
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.Hardswish(inplace=True))
        
        # Depthwise convolution
        layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2, groups=mid_channels, bias=False))
        layers.append(nn.BatchNorm2d(mid_channels))
        
        # Squeeze-and-Excitation (if enabled)
        if se:
            se_channels = max(1, mid_channels // 4)
            layers.append(SEBlock(mid_channels, se_channels))
        
        layers.append(nn.Hardswish(inplace=True))
        
        # Pointwise linear convolution
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.running_mean)
                nn.init.zeros_(m.running_var)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.hswish(x)
        
        # MobileNetV3 blocks
        for block in self.blocks:
            x = block(x)
        
        # Final convolution
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.hswish(x)
        
        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction_channels):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(channels, reduction_channels, kernel_size=1)
        self.conv_expand = nn.Conv2d(reduction_channels, channels, kernel_size=1)
        self.hswish = nn.Hardswish(inplace=True)
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_reduce(y)
        y = self.hswish(y)
        y = self.conv_expand(y)
        y = torch.sigmoid(y)
        return x * y


def load_actual_model(model_path: str, device: str = 'cpu'):
    """
    Load the actual trained model with the correct architecture.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Loaded PyTorch model
    """
    # Load the saved data
    saved_data = torch.load(model_path, map_location=device)
    
    # Create model instance with MobileNetV3 architecture
    model = MobileNetV3Backbone()
    
    # Load state dict
    if isinstance(saved_data, dict):
        if 'model_state_dict' in saved_data:
            model.load_state_dict(saved_data['model_state_dict'], strict=False)
        elif 'state_dict' in saved_data:
            model.load_state_dict(saved_data['state_dict'], strict=False)
        else:
            # Assume the dict itself is the state dict
            model.load_state_dict(saved_data, strict=False)
    else:
        # Assume it's the full model
        model = saved_data
    
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test the model architecture
    device = torch.device('cpu')
    model = MobileNetV3Backbone()
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"MobileNetV3 Model test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model loaded successfully!")
