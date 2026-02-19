import torch
import torch.nn as nn


class DeepfakeCNN(nn.Module):
    """
    Standard CNN architecture for deepfake detection.
    This should match the architecture used for training your models.
    """
    def __init__(self, num_classes=2):
        super(DeepfakeCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = self.adaptive_pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x


def load_trained_model(model_path: str, device: str = 'cpu'):
    """
    Load a trained model from the given path.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Loaded PyTorch model
    """
    # Load the saved data
    saved_data = torch.load(model_path, map_location=device)
    
    # Create model instance
    model = DeepfakeCNN(num_classes=2)
    
    # Load state dict
    if isinstance(saved_data, dict):
        if 'model_state_dict' in saved_data:
            model.load_state_dict(saved_data['model_state_dict'])
        elif 'state_dict' in saved_data:
            model.load_state_dict(saved_data['state_dict'])
        else:
            # Assume the dict itself is the state dict
            model.load_state_dict(saved_data)
    else:
        # Assume it's the full model
        model = saved_data
    
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test the model architecture
    device = torch.device('cpu')
    model = DeepfakeCNN(num_classes=2)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Model architecture test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model loaded successfully!")
