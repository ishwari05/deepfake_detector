# Intelligent Deepfake Detection System

An MVP-level deepfake detection system with Grad-CAM explainability for both images and videos.

## Features

- **Image Deepfake Detection**: Detect manipulated images with confidence scores
- **Video Deepfake Detection**: Analyze video frames to detect temporal inconsistencies
- **Grad-CAM Explainability**: Visualize which regions contribute to the detection decision
- **Human-Readable Explanations**: Get detailed explanations of why content is classified as fake/real
- **Batch Processing**: Analyze multiple images at once
- **Modular Design**: Clean, extensible codebase

## Project Structure

```
deepfake/
├── explainability/
│   ├── gradcam.py              # Grad-CAM implementation
│   └── explanation_builder.py  # Explanation generation
├── model/
│   ├── image_model.pth         # Trained image detection model
│   └── video_model.pth         # Trained video detection model
├── image_detector.py           # Image detection pipeline
├── video_detector.py           # Video detection pipeline
├── demo.py                     # Main demo script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your trained models are in the `model/` directory:
   - `model/image_model.pth` for image detection
   - `model/video_model.pth` for video detection

## Usage

### Image Detection

```bash
# Detect deepfake in a single image
python demo.py --image path/to/image.jpg

# Batch detection on a directory of images
python demo.py --batch path/to/images/
```

### Video Detection

```bash
# Detect deepfake in a video
python demo.py --video path/to/video.mp4
```

### Advanced Options

```bash
# Specify custom model paths
python demo.py --image test.jpg --image-model custom_model.pth

# Use GPU acceleration
python demo.py --image test.jpg --device cuda

# Custom output directory
python demo.py --image test.jpg --output-dir my_results
```

## Output Format

### Image Detection Results

```json
{
  "fake_probability": 0.85,
  "real_probability": 0.15,
  "classification": "FAKE",
  "heatmap_overlay_path": "outputs/test_heatmap.jpg",
  "explanation_text": "High activation detected in facial region suggesting manipulation artifacts.",
  "activation_regions": {
    "bbox": [120, 80, 150, 100],
    "centroid": [195, 130],
    "area_ratio": 0.12
  }
}
```

### Video Detection Results

```json
{
  "fake_probability": 0.78,
  "real_probability": 0.22,
  "classification": "FAKE",
  "top_suspicious_frames": [5, 12, 23],
  "heatmap_overlay_paths": [
    "video_outputs/frame_0005_heatmap.jpg",
    "video_outputs/frame_0012_heatmap.jpg",
    "video_outputs/frame_0023_heatmap.jpg"
  ],
  "explanation_text": "Suspicious activity detected in 3 out of 15 analyzed frames...",
  "frame_summary": {
    "total_frames": 15,
    "suspicious_frames": [5, 12, 23],
    "avg_activation_area": 0.08
  }
}
```

## Explainability Features

### Grad-CAM Visualization
- Generates heatmaps showing which image regions influenced the classification
- Overlays heatmaps on original images for easy interpretation
- Automatically detects the last convolutional layer in your model

### Region Analysis
- Identifies specific facial regions (eyes, mouth, nose, etc.) with high activation
- Provides bounding box coordinates for suspicious areas
- Calculates activation area as percentage of total image

### Human-Readable Explanations
- Simple, intuitive explanations of detection results
- Identifies manipulation artifacts and their locations
- Provides confidence levels and temporal patterns for videos

## Model Requirements

Your PyTorch models should:
- Accept input tensors of shape `(batch_size, 3, height, width)`
- Output logits for 2 classes: REAL (0) and FAKE (1)
- Include at least one convolutional layer for Grad-CAM

### Example Model Architecture

```python
class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ... more conv layers
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
```

## API Reference

### ImageDeepfakeDetector

```python
detector = ImageDeepfakeDetector(model_path, device='cpu')
results = detector.detect(image_path, output_dir="outputs")
```

### VideoDeepfakeDetector

```python
detector = VideoDeepfakeDetector(model_path, device='cpu')
results = detector.detect(video_path, output_dir="video_outputs")
```

### GradCAM

```python
gradcam = GradCAM(model, target_layer_name)
cam = gradcam.generate_cam(input_tensor, class_idx)
heatmap_path = gradcam.generate_heatmap_overlay(image_path, cam, save_path)
```

## Limitations

- MVP-level implementation with basic Grad-CAM (no advanced segmentation)
- Assumes standard CNN architecture
- Explanation quality depends on model architecture and training data
- Video analysis uses frame-by-frame approach (no temporal modeling)

## Future Enhancements

- Integration of SHAP or LIME for additional explainability
- Attention map visualization
- Temporal consistency modeling for videos
- Advanced facial region segmentation
- Real-time processing capabilities

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure model files exist and are compatible with PyTorch
2. **CUDA Errors**: Use `--device cpu` if GPU issues occur
3. **Memory Issues**: Reduce batch size or use CPU for large videos
4. **Grad-CAM Failures**: Check that your model has convolutional layers

### Debug Mode

Add print statements or use Python debugger to trace issues:

```python
import pdb; pdb.set_trace()
```

## License

This project is for educational and research purposes. Please ensure compliance with applicable laws and regulations when using deepfake detection technology.
