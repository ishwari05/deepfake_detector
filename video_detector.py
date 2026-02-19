import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from typing import Dict, Any, List, Optional
import json
import tempfile

from explainability.gradcam import GradCAM, find_last_conv_layer, preprocess_image
from explainability.explanation_builder import ExplanationBuilder
from blackbox_model_wrapper import load_blackbox_model


class VideoDeepfakeDetector:
    """
    Video deepfake detector with Grad-CAM explainability.
    Analyzes video frames and provides explanations for suspicious regions.
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
        
        print(f"Initialized video detector with Grad-CAM on layer: {target_layer}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained PyTorch model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model using black box wrapper
        model = load_blackbox_model(model_path, self.device)
        
        return model
    
    def extract_frames(self, video_path: str, max_frames: int = 30, sample_rate: int = 1) -> List[np.ndarray]:
        """
        Extract frames from video for analysis.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            sample_rate: Sample every nth frame
            
        Returns:
            List of extracted frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_count % sample_rate == 0:
                    frames.append(frame)
                    
                    # Limit number of frames
                    if len(frames) >= max_frames:
                        break
                
                frame_count += 1
                
        finally:
            cap.release()
        
        return frames
    
    def save_frame(self, frame: np.ndarray, output_path: str):
        """Save frame to disk."""
        cv2.imwrite(output_path, frame)
    
    def analyze_frame(self, frame: np.ndarray, frame_index: int, output_dir: str) -> Dict[str, Any]:
        """
        Analyze a single frame for deepfake detection.
        
        Args:
            frame: Frame as numpy array
            frame_index: Index of the frame in the video
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing frame analysis results
        """
        # Save frame temporarily for preprocessing
        temp_frame_path = os.path.join(output_dir, f"temp_frame_{frame_index}.jpg")
        self.save_frame(frame, temp_frame_path)
        
        try:
            # Preprocess frame
            input_tensor = preprocess_image(temp_frame_path).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                fake_prob = probabilities[0, 1].item()  # Assuming class 1 = FAKE
                real_prob = probabilities[0, 0].item()   # Assuming class 0 = REAL
            
            # Determine classification using OPTIMIZED threshold
            classification = "FAKE" if fake_prob > 0.6 else "REAL"
            
            # Generate Grad-CAM
            class_idx = 1 if classification == "FAKE" else 0
            cam = self.gradcam.generate_cam(input_tensor, class_idx)
            
            # Get activation regions
            activation_regions = self.gradcam.get_activation_regions(cam)
            
            # Determine if frame is suspicious (high fake probability or significant activation)
            is_suspicious = (fake_prob > 0.6) or (activation_regions['area_ratio'] > 0.05)
            
            # Generate and save heatmap overlay
            frame_name = f"frame_{frame_index:04d}"
            heatmap_path = os.path.join(output_dir, f"{frame_name}_heatmap.jpg")
            self.gradcam.generate_heatmap_overlay(temp_frame_path, cam, heatmap_path)
            
            return {
                'frame_index': frame_index,
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'classification': classification,
                'is_suspicious': is_suspicious,
                'activation_regions': activation_regions,
                'heatmap_path': heatmap_path,
                'cam': cam  # Store CAM for later analysis
            }
            
        except Exception as e:
            print(f"Error analyzing frame {frame_index}: {e}")
            return {
                'frame_index': frame_index,
                'error': str(e),
                'classification': 'ERROR'
            }
            
        finally:
            # Clean up temporary frame
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
    
    def detect(self, video_path: str, output_dir: str = "video_outputs") -> Dict[str, Any]:
        """
        Detect deepfake in video with explainability.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing detection results and explanations
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract frames
        print(f"Extracting frames from {video_path}...")
        frames = self.extract_frames(video_path, max_frames=30, sample_rate=2)
        
        if not frames:
            raise ValueError("No frames could be extracted from video")
        
        print(f"Analyzing {len(frames)} frames...")
        
        # Analyze each frame
        frame_results = []
        for i, frame in enumerate(frames):
            result = self.analyze_frame(frame, i, output_dir)
            frame_results.append(result)
            
            if i % 5 == 0:
                print(f"Processed frame {i+1}/{len(frames)}")
        
        # Aggregate results
        valid_results = [r for r in frame_results if 'error' not in r]
        
        if not valid_results:
            raise ValueError("No valid frame results obtained")
        
        # Calculate overall fake probability (average across frames)
        fake_probabilities = [r['fake_probability'] for r in valid_results]
        overall_fake_prob = np.mean(fake_probabilities)
        overall_real_prob = 1 - overall_fake_prob
        
        # Determine overall classification
        classification = "FAKE" if overall_fake_prob > 0.5 else "REAL"
        
        # Get top 3 most suspicious frames
        suspicious_frames = [r for r in valid_results if r.get('is_suspicious', False)]
        suspicious_frames.sort(key=lambda x: x['fake_probability'], reverse=True)
        top_suspicious = suspicious_frames[:3]
        
        # Prepare frame results for explanation
        frame_explanation_data = []
        for r in valid_results:
            frame_explanation_data.append({
                'frame_index': r['frame_index'],
                'is_suspicious': r.get('is_suspicious', False),
                'activation_regions': r.get('activation_regions', {}),
                'fake_probability': r['fake_probability']
            })
        
        # Generate explanation
        video_shape = frames[0].shape[:2]  # (height, width)
        explanation_text = self.explanation_builder.build_video_explanation(
            overall_fake_prob, frame_explanation_data, video_shape
        )
        
        # Get frame summary
        frame_summary = self.explanation_builder.build_frame_summary(valid_results)
        
        # Prepare heatmap paths for top suspicious frames
        heatmap_overlay_paths = []
        for frame_result in top_suspicious:
            if 'heatmap_path' in frame_result:
                heatmap_overlay_paths.append(frame_result['heatmap_path'])
        
        # Prepare results
        results = {
            'fake_probability': overall_fake_prob,
            'real_probability': overall_real_prob,
            'classification': classification,
            'top_suspicious_frames': [r['frame_index'] for r in top_suspicious],
            'heatmap_overlay_paths': heatmap_overlay_paths,
            'explanation_text': explanation_text,
            'frame_summary': frame_summary,
            'total_frames_analyzed': len(valid_results),
            'suspicious_frames_count': len(suspicious_frames),
            'input_video_path': video_path
        }
        
        # Save results as JSON
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        results_path = os.path.join(output_dir, f"{video_name}_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = self._convert_numpy_types(results.copy())
            json.dump(json_results, f, indent=2)
        
        results['results_json_path'] = results_path
        
        print(f"Video analysis complete. Classification: {classification}")
        print(f"Found {len(suspicious_frames)} suspicious frames out of {len(valid_results)} analyzed")
        
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


class DummyVideoDeepfakeModel(nn.Module):
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


def create_dummy_video_model():
    """
    Create a dummy CNN model for testing purposes.
    In practice, you would load your actual trained model.
    """
    return DummyVideoDeepfakeModel()


if __name__ == "__main__":
    # Example usage
    # Note: This is for demonstration. Replace with your actual model path.
    
    # Create dummy model for testing (remove this and load your actual model)
    dummy_model = create_dummy_video_model()
    dummy_model_path = "dummy_video_model.pth"
    torch.save(dummy_model.state_dict(), dummy_model_path)
    
    # Initialize detector
    detector = VideoDeepfakeDetector(dummy_model_path)
    
    # Example detection (you would provide actual video path)
    # results = detector.detect("path/to/your/video.mp4")
    # print("Video Detection Results:")
    # print(f"Classification: {results['classification']}")
    # print(f"Fake Probability: {results['fake_probability']:.3f}")
    # print(f"Suspicious Frames: {results['top_suspicious_frames']}")
    # print(f"Explanation: {results['explanation_text']}")
    # print(f"Heatmaps saved for frames: {results['heatmap_overlay_paths']}")
    
    print("Video detector initialized successfully!")
    print("Replace dummy model with your actual trained model to use for real detection.")
