import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import os


class GradCAM:
    """
    Grad-CAM implementation for PyTorch CNN models.
    Generates class activation maps to explain model predictions.
    """
    
    def __init__(self, model: nn.Module, target_layer_name: str):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer_name: Name of the target convolutional layer
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
        else:
            raise ValueError(f"Layer '{self.target_layer_name}' not found in model")
    
    def generate_cam(self, 
                     input_tensor: torch.Tensor, 
                     class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W)
            class_idx: Target class index. If None, uses predicted class.
            
        Returns:
            CAM as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # (H, W)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = torch.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.detach().cpu().numpy()
    
    def generate_heatmap_overlay(self, 
                                image_path: str, 
                                cam: np.ndarray, 
                                save_path: str,
                                alpha: float = 0.4) -> str:
        """
        Generate and save heatmap overlay on original image.
        
        Args:
            image_path: Path to original image
            cam: Class activation map
            save_path: Path to save overlay image
            alpha: Transparency factor for heatmap
            
        Returns:
            Path to saved overlay image
        """
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        # Save overlay
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, overlay)
        
        return save_path
    
    def get_activation_regions(self, cam: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Extract regions of high activation from CAM.
        
        Args:
            cam: Class activation map
            threshold: Threshold for high activation
            
        Returns:
            Dictionary containing bounding box and centroid of high activation region
        """
        # Apply threshold
        binary_cam = (cam > threshold).astype(np.uint8)
        
        if binary_cam.sum() == 0:
            return {
                'bbox': None,
                'centroid': None,
                'area_ratio': 0.0
            }
        
        # Find contours
        contours, _ = cv2.findContours(binary_cam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'bbox': None,
                'centroid': None,
                'area_ratio': 0.0
            }
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Get centroid
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # Calculate area ratio
        activation_area = cv2.contourArea(largest_contour)
        total_area = cam.shape[0] * cam.shape[1]
        area_ratio = activation_area / total_area
        
        return {
            'bbox': (x, y, w, h),
            'centroid': (cx, cy),
            'area_ratio': area_ratio
        }


def find_last_conv_layer(model: nn.Module) -> str:
    """
    Automatically find the last convolutional layer in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Name of the last convolutional layer
    """
    conv_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)
    
    if not conv_layers:
        raise ValueError("No convolutional layers found in model")
    
    return conv_layers[-1]


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image_path: Path to image
        target_size: Target size for resizing
        
    Returns:
        Preprocessed tensor ready for model input
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize to [0, 1] and convert to float32
    image = image.astype(np.float32) / 255.0
    
    # Standardize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    
    return tensor
