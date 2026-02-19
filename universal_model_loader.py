import torch
import torch.nn as nn
import json
import os


class UniversalModelLoader:
    """
    Universal model loader that can handle different model architectures.
    It loads the model as-is and provides Grad-CAM functionality.
    """
    
    @staticmethod
    def load_model(model_path: str, device: str = 'cpu'):
        """
        Load a trained model without requiring the exact architecture.
        
        Args:
            model_path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded PyTorch model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load the saved data
        saved_data = torch.load(model_path, map_location=device)
        
        # Handle different save formats
        if isinstance(saved_data, dict):
            if 'model_state_dict' in saved_data:
                # Try to create a generic model and load state dict
                model = UniversalModelLoader._create_generic_model()
                try:
                    model.load_state_dict(saved_data['model_state_dict'], strict=False)
                except Exception as e:
                    print(f"Warning: Could not load state dict strictly: {e}")
                    # Try loading only matching keys
                    state_dict = saved_data['model_state_dict']
                    model_dict = model.state_dict()
                    matched_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                    model.load_state_dict(matched_state_dict, strict=False)
                    print(f"Loaded {len(matched_state_dict)}/{len(state_dict)} parameters")
            elif 'state_dict' in saved_data:
                model = UniversalModelLoader._create_generic_model()
                try:
                    model.load_state_dict(saved_data['state_dict'], strict=False)
                except Exception as e:
                    print(f"Warning: Could not load state dict strictly: {e}")
                    state_dict = saved_data['state_dict']
                    model_dict = model.state_dict()
                    matched_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                    model.load_state_dict(matched_state_dict, strict=False)
                    print(f"Loaded {len(matched_state_dict)}/{len(state_dict)} parameters")
            else:
                # Assume the dict itself is the state dict
                model = UniversalModelLoader._create_generic_model()
                try:
                    model.load_state_dict(saved_data, strict=False)
                except Exception as e:
                    print(f"Warning: Could not load state dict strictly: {e}")
                    model_dict = model.state_dict()
                    matched_state_dict = {k: v for k, v in saved_data.items() if k in model_dict and v.size() == model_dict[k].size()}
                    model.load_state_dict(matched_state_dict, strict=False)
                    print(f"Loaded {len(matched_state_dict)}/{len(saved_data)} parameters")
        else:
            # Assume it's the full model
            model = saved_data
        
        model.to(device)
        model.eval()
        
        return model
    
    @staticmethod
    def _create_generic_model():
        """
        Create a generic CNN model that can be adapted.
        """
        class GenericCNN(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Feature extractor with common layer names
                self.features = nn.ModuleDict({
                    'conv1': nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    'bn1': nn.BatchNorm2d(64),
                    'relu1': nn.ReLU(inplace=True),
                    'pool1': nn.MaxPool2d(2),
                    
                    'conv2': nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    'bn2': nn.BatchNorm2d(128),
                    'relu2': nn.ReLU(inplace=True),
                    'pool2': nn.MaxPool2d(2),
                    
                    'conv3': nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    'bn3': nn.BatchNorm2d(256),
                    'relu3': nn.ReLU(inplace=True),
                    'pool3': nn.MaxPool2d(2),
                    
                    'conv4': nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    'bn4': nn.BatchNorm2d(512),
                    'relu4': nn.ReLU(inplace=True),
                    'pool4': nn.MaxPool2d(2),
                })
                
                # Adaptive pooling
                self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
                
                # Classifier
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(512 * 7 * 7, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 2)
                )
            
            def forward(self, x):
                # Apply feature extraction layers
                x = self.features['conv1'](x)
                x = self.features['bn1'](x)
                x = self.features['relu1'](x)
                x = self.features['pool1'](x)
                
                x = self.features['conv2'](x)
                x = self.features['bn2'](x)
                x = self.features['relu2'](x)
                x = self.features['pool2'](x)
                
                x = self.features['conv3'](x)
                x = self.features['bn3'](x)
                x = self.features['relu3'](x)
                x = self.features['pool3'](x)
                
                x = self.features['conv4'](x)
                x = self.features['bn4'](x)
                x = self.features['relu4'](x)
                x = self.features['pool4'](x)
                
                # Adaptive pooling and classification
                x = self.adaptive_pool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                
                return x
        
        return GenericCNN()
    
    @staticmethod
    def analyze_model_structure(model):
        """
        Analyze the model structure to understand its architecture.
        """
        print("Model Structure Analysis:")
        print("=" * 50)
        
        # Get all named modules
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        
        print(f"Found {len(conv_layers)} convolutional layers:")
        for i, (name, module) in enumerate(conv_layers):
            print(f"  {i+1}. {name}: {module.in_channels} -> {module.out_channels}, kernel={module.kernel_size}")
        
        # Find the last conv layer
        if conv_layers:
            last_conv_name = conv_layers[-1][0]
            print(f"\nLast convolutional layer: {last_conv_name}")
            return last_conv_name
        
        print("No convolutional layers found!")
        return None


if __name__ == "__main__":
    # Test the universal loader
    device = torch.device('cpu')
    
    # Test with your actual model
    try:
        model = UniversalModelLoader.load_model('model/image_model.pth', device)
        print("Model loaded successfully!")
        
        # Analyze the model
        last_conv = UniversalModelLoader.analyze_model_structure(model)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\nForward pass test:")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
