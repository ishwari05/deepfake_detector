#!/usr/bin/env python3
"""
Fixed Deepfake Detector with Optimized Threshold
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from typing import Dict, Any, Optional
import json

from explainability.gradcam import GradCAM, find_last_conv_layer, preprocess_image
from explainability.explanation_builder import ExplanationBuilder
from blackbox_model_wrapper import load_blackbox_model


class FixedImageDeepfakeDetector:
    """
    Fixed image deepfake detector with optimized threshold.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', threshold: float = 0.6):
        """
        Initialize the detector with optimized threshold.
        
        Args:
            model_path: Path to trained PyTorch model
            device: Device to run inference on ('cpu' or 'cuda')
            threshold: Classification threshold (optimized to 0.6)
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.explanation_builder = ExplanationBuilder()
        self.threshold = threshold  # OPTIMIZED THRESHOLD: 0.6       
        # Initialize Grad-CAM
        target_layer = find_last_conv_layer(self.model)
        self.gradcam = GradCAM(self.model, target_layer)
        
        print(f"Initialized FIXED detector with threshold {threshold}")
        print(f"Grad-CAM on layer: {target_layer}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained PyTorch model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = load_blackbox_model(model_path, self.device)
        return model
    
    def detect(self, image_path: str, output_dir: str = "outputs") -> Dict[str, Any]:
        """
        Detect deepfake in image with explainability using FIXED threshold.
        
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
            fake_prob = probabilities[0, 1].item()
            real_prob = probabilities[0, 0].item()
        
        # FIXED: Use optimized threshold
        classification = "FAKE" if fake_prob > self.threshold else "REAL"
        fake_probability = fake_prob
        
        # Generate Grad-CAM
        try:
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
        image_shape = cv2.imread(image_path).shape[:2]
        explanation_text = self.explanation_builder.build_image_explanation(
            fake_probability, activation_regions, image_shape
        )
        
        # Prepare results
        results = {
            'fake_probability': fake_probability,
            'real_probability': real_prob,
            'classification': classification,
            'threshold_used': self.threshold,
            'heatmap_overlay_path': heatmap_path,
            'explanation_text': explanation_text,
            'activation_regions': activation_regions,
            'input_image_path': image_path
        }
        
        # Save results as JSON
        results_path = os.path.join(output_dir, f"{image_name}_results.json")
        with open(results_path, 'w') as f:
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


if __name__ == "__main__":
    # Test the fixed detector
    detector = FixedImageDeepfakeDetector("model/image_model.pth", threshold=0.6)
    
    # Test on sample images
    import glob
    real_images = glob.glob("dataset/real_vs_fake/real-vs-fake/test/real/*")[:10]
    fake_images = glob.glob("dataset/real_vs_fake/real-vs-fake/train/fake/*")[:10]
    
    print("Testing FIXED detector:")
    print("=" * 40)
    
    correct = 0
    total = 0
    
    for img_path in real_images + fake_images:
        try:
            result = detector.detect(img_path, "fixed_test_outputs")
            pred = result['classification']
            gt = 'REAL' if 'real' in img_path else 'FAKE'
            
            if pred == gt:
                correct += 1
            total += 1
            
            print(f"{os.path.basename(img_path)}: {pred} (GT: {gt}) ✓" if pred == gt else f"{os.path.basename(img_path)}: {pred} (GT: {gt}) ✗")
            
        except Exception as e:
            print(f"Error: {e}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nFIXED Detector Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Correct: {correct}/{total}")
