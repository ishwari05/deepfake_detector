import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import sys
from typing import Dict, Any, Optional
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.explainability.gradcam import GradCAM, find_last_conv_layer, preprocess_image
from src.explainability.explanation_builder import ExplanationBuilder
from src.utils.blackbox_model_wrapper import load_blackbox_model


class ImageDeepfakeDetector:
    """
    Image deepfake detector with Grad-CAM explainability.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to trained PyTorch model
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.explanation_builder = ExplanationBuilder()
        
        # Initialize Grad-CAM (automatically find last conv layer)
        target_layer = find_last_conv_layer(self.model)
        self.gradcam = GradCAM(self.model, target_layer)
        
        print(f"Initialized image detector with Grad-CAM on layer: {target_layer}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained PyTorch model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model using black box wrapper
        model = load_blackbox_model(model_path, self.device)
        
        return model
    
    def detect(self, image_path: str, output_dir: str = "outputs") -> Dict[str, Any]:
        """
        Detect deepfake in image with explainability.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing detection results and explanations
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess image
        try:
            input_tensor = preprocess_image(image_path).to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {e}")
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            fake_prob = probabilities[0, 1].item()  # Assuming class 1 = FAKE
            real_prob = probabilities[0, 0].item()   # Assuming class 0 = REAL
        
        # Determine classification using OPTIMIZED threshold
        classification = "FAKE" if fake_prob > 0.6 else "REAL"
        fake_probability = fake_prob
        
        # Generate Grad-CAM
        try:
            # Use predicted class for Grad-CAM
            class_idx = 1 if classification == "FAKE" else 0
            cam = self.gradcam.generate_cam(input_tensor, class_idx)
            
            # Get activation regions
            activation_regions = self.gradcam.get_activation_regions(cam)
            
            # Generate and save heatmap overlay
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            heatmap_path = os.path.join(output_dir, f"{image_name}_heatmap.jpg")
            self.gradcam.generate_heatmap_overlay(image_path, cam, heatmap_path)
            
        except Exception as e:
            print(f"Warning: Grad-CAM generation failed: {e}")
            cam = None
            activation_regions = {'bbox': None, 'centroid': None, 'area_ratio': 0.0}
            heatmap_path = None
        
        # Generate explanation
        image_shape = cv2.imread(image_path).shape[:2]  # (height, width)
        explanation_text = self.explanation_builder.build_image_explanation(
            fake_probability, activation_regions, image_shape
        )
        
        # Prepare results
        results = {
            'fake_probability': fake_probability,
            'real_probability': real_prob,
            'classification': classification,
            'heatmap_overlay_path': heatmap_path,
            'explanation_text': explanation_text,
            'activation_regions': activation_regions,
            'input_image_path': image_path
        }
        
        # Save results as JSON
        results_path = os.path.join(output_dir, f"{image_name}_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = self._convert_numpy_types(results.copy())
            json.dump(json_results, f, indent=2)
        
        results['results_json_path'] = results_path
        
        return results
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def batch_detect(self, image_paths: list, output_dir: str = "outputs") -> list:
        """
        Detect deepfakes in multiple images.
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save outputs
            
        Returns:
            List of detection results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.detect(image_path, output_dir)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'input_image_path': image_path,
                    'error': str(e),
                    'classification': 'ERROR'
                })
        
        return results


class DummyDeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Last conv layer
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 2 classes: REAL, FAKE
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_dummy_model():
    """
    Create a dummy CNN model for testing purposes.
    In practice, you would load your actual trained model.
    """
    return DummyDeepfakeModel()


if __name__ == "__main__":
    # Example usage
    # Note: This is for demonstration. Replace with your actual model path.
    
    # Create dummy model for testing (remove this and load your actual model)
    dummy_model = create_dummy_model()
    dummy_model_path = "dummy_image_model.pth"
    torch.save(dummy_model.state_dict(), dummy_model_path)
    
    # Initialize detector
    detector = ImageDeepfakeDetector(dummy_model_path)
    
    # Example detection (you would provide actual image path)
    # results = detector.detect("path/to/your/image.jpg")
    # print("Detection Results:")
    # print(f"Classification: {results['classification']}")
    # print(f"Fake Probability: {results['fake_probability']:.3f}")
    # print(f"Explanation: {results['explanation_text']}")
    # print(f"Heatmap saved to: {results['heatmap_overlay_path']}")
    
    print("Image detector initialized successfully!")
    print("Replace dummy model with your actual trained model to use for real detection.")
