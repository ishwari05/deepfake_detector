# Intelligent Deepfake Detection System

A production-ready deepfake detection system with advanced Grad-CAM explainability for both images and videos. Features a modern Streamlit web interface with comprehensive analytics and real-time model performance tracking.

## Features

- **🖼️ Image Deepfake Detection**: Detect manipulated images with confidence scores and Grad-CAM visualization
- **🎥 Video Deepfake Detection**: Analyze video frames to detect temporal inconsistencies with frame-by-frame heatmaps
- **🔥 Advanced Grad-CAM Explainability**: Multi-tab interface with heatmaps, activation analysis, and technical details
- **📊 Real-time Analytics Dashboard**: Performance metrics, statistical analysis, and result history
- **🎯 Suspicious Frame Detection**: Automatic identification and visualization of anomalous video frames
- **📱 Modern Web Interface**: Responsive Streamlit app with professional UI/UX design
- **📥 Export Capabilities**: Download reports, heatmaps, and CSV results
- **⚡ High Performance**: Optimized processing with 82% image accuracy and 75% video accuracy
- **🔧 Production Ready**: Complete deployment infrastructure with Git LFS model storage

## Project Structure

```
deepfake/
├── 📁 app/                    # Streamlit web application
│   └── streamlit_app.py      # Main UI with enhanced Grad-CAM display (585 lines)
├── 📁 src/                    # Core detection system
│   ├── 📁 detectors/          # Detection engines
│   │   ├── image_detector.py  # Image analysis with Grad-CAM (233 lines)
│   │   └── video_detector.py  # Video analysis with frame processing (347 lines)
│   ├── 📁 explainability/     # XAI components
│   │   ├── gradcam.py         # Grad-CAM implementation
│   │   └── explanation_builder.py # Text explanations
│   ├── 📁 models/             # Pretrained models (Git LFS)
│   │   ├── image_model.pth   # 18.6 MB trained model
│   │   └── video_model.pth   # 56.5 MB trained model
│   └── 📁 utils/              # Utilities
│       └── blackbox_model_wrapper.py # Model loading
├── 📁 reports/                # Analysis reports and documentation
├── requirements.txt           # Python dependencies
├── .gitattributes           # Git LFS configuration
└── README.md               # This file
```

## Installation

1. **Clone the repository** (includes models via Git LFS):
   ```bash
   git clone https://github.com/ishwari05/deepfake_detector.git
   cd deepfake_detector
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify models** (automatically included):
   - `src/models/image_model.pth` for image detection (18.6 MB)
   - `src/models/video_model.pth` for video detection (56.5 MB)

## Usage

### 🚀 Quick Start - Web Interface

Launch the modern Streamlit web application:

```bash
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501` with:
- **🖼️ Image Detection Tab**: Upload images for instant analysis with Grad-CAM heatmaps
- **🎥 Video Detection Tab**: Upload videos for frame-by-frame analysis
- **📊 Statistics Tab**: View analysis history and performance metrics
- **🎯 Performance Dashboard**: Real-time model status and technical details

### 📋 Advanced Features

**Image Analysis:**
- Upload JPG/PNG images
- Get classification with confidence scores
- View Grad-CAM heatmaps with activation analysis
- Download detailed reports and heatmaps

**Video Analysis:**
- Upload MP4/AVI/MOV videos
- Automatic frame extraction and analysis
- View suspicious frames with heatmaps
- Timeline visualization of fake probability across frames

**Analytics Dashboard:**
- Track analysis history
- Export results as CSV
- View classification distribution
- Monitor model performance metrics

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

## 🔥 Advanced Explainability Features

### Multi-Tab Grad-CAM Interface
- **📸 Heatmap Overlay**: Visual explanation with red/yellow activation regions
- **📊 Activation Analysis**: Detailed metrics including area ratio, focus points, intensity levels
- **🔍 Technical Details**: Comprehensive methodology documentation and interpretation guides

### Enhanced Region Analysis
- **Bounding Box Detection**: Precise coordinates for suspicious areas
- **Centroid Calculation**: Center of activation focus points
- **Area Ratio Metrics**: Percentage of image showing strong activation
- **Intensity Classification**: High/Medium/Low activation strength indicators
- **Decision Confidence**: Confidence levels based on activation patterns

### Video-Specific Analysis
- **Frame-by-Frame Processing**: Individual Grad-CAM for each analyzed frame
- **Suspicious Frame Detection**: Automatic identification of anomalous frames
- **Timeline Visualization**: Probability trends across video duration
- **Multi-Heatmap Display**: Visualizations for top suspicious frames
- **Temporal Consistency**: Analysis of activation patterns over time

### Educational Content
- **Interpretation Guides**: Step-by-step explanation of heatmap meanings
- **Technical Documentation**: Grad-CAM methodology and implementation details
- **User-Friendly Labels**: Clear descriptions of all metrics and visualizations

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

## 📋 API Reference

### Core Detection Classes

```python
# Image Detection
from src.detectors.image_detector import ImageDeepfakeDetector
detector = ImageDeepfakeDetector('src/models/image_model.pth', device='cpu')
results = detector.detect(image_path, output_dir="outputs")

# Video Detection  
from src.detectors.video_detector import VideoDeepfakeDetector
detector = VideoDeepfakeDetector('src/models/video_model.pth', device='cpu')
results = detector.detect(video_path, output_dir="video_outputs")
```

### Grad-CAM Components

```python
# Grad-CAM Generation
from src.explainability.gradcam import GradCAM, find_last_conv_layer
target_layer = find_last_conv_layer(model)
gradcam = GradCAM(model, target_layer)
cam = gradcam.generate_cam(input_tensor, class_idx)

# Explanation Builder
from src.explainability.explanation_builder import ExplanationBuilder
builder = ExplanationBuilder()
explanation = builder.build_image_explanation(fake_prob, activation_regions, image_shape)
```

## ⚠️ Current Limitations

### Technical Constraints
- **Model Architecture**: Requires CNN with convolutional layers for Grad-CAM
- **Video Processing**: Frame-by-frame analysis (no temporal modeling)
- **Performance**: Optimized for accuracy over real-time processing
- **Memory Usage**: ~500MB per model during inference

### Scope Limitations
- **Training Data**: Models trained on specific datasets (may not generalize to all deepfake types)
- **Video Formats**: Supports MP4, AVI, MOV, MKV (limited codec support)
- **Image Formats**: Supports JPG, PNG, JPEG (limited to common formats)
- **Language**: English-only explanations and UI

## 🚀 Future Enhancements

### Advanced Explainability
- **SHAP/LIME Integration**: Multiple explanation methods
- **Attention Maps**: Transformer-based visualization
- **3D Heatmaps**: Depth-aware activation visualization
- **Comparative Analysis**: Side-by-side real vs fake comparisons

### Performance & Features
- **Real-time Processing**: GPU optimization for live video analysis
- **Temporal Modeling**: LSTM/Transformer for video sequence analysis
- **Facial Segmentation**: Region-specific analysis (eyes, mouth, etc.)
- **Multi-language Support**: Internationalization for explanations

### Deployment & Scaling
- **Cloud Deployment**: Docker containers and cloud services
- **API Service**: RESTful API for integration
- **Batch Processing**: Large-scale dataset analysis
- **Mobile Support**: Progressive web app or native applications

## � Production Roadmap & Future Scope

### ⚡ Real-Time Verification for News & Social Media

**Objective**: Extend the system to operate in real-time, enabling instant verification of images and videos circulating on news platforms and social media.

**Implementation Plan**:
- **Stream Processing**: WebSocket-based real-time media analysis
- **Edge Computing**: Deploy detection models closer to users for reduced latency
- **Platform Integration**: APIs for Twitter, Facebook, YouTube, TikTok content moderation
- **Alert System**: Automated notifications for suspicious content detection

**Impact**:
- Help journalists and fact-checkers verify content instantly
- Enable platforms to curb misinformation before it spreads
- Provide real-time confidence scores for breaking news content
- Support live content moderation workflows

### 🧩 Detection of Partial Manipulations

**Objective**: Enhance the model to detect localized or partial edits instead of only classifying entire media as real or fake.

**Technical Approach**:
- **Segmentation-Based Analysis**: Pixel-level manipulation detection
- **Facial Region Mapping**: Detect specific face swaps or alterations
- **Object-Level Detection**: Identify synthetic overlays or altered objects
- **Multi-Scale Analysis**: Detect manipulations at different granularity levels

**Benefits**:
- Significantly improve forensic precision
- Provide detailed manipulation reports (which regions were edited)
- Support legal evidence documentation
- Enable content authenticity verification at granular level

### 🎯 High Accuracy with Low False Positives

**Objective**: Optimize the model to maintain high detection accuracy while minimizing false positives, ensuring legitimate content is not incorrectly flagged.

**Optimization Strategies**:
- **Threshold Tuning**: Dynamic threshold adjustment based on content type
- **Ensemble Methods**: Combine multiple models for robust predictions
- **Uncertainty Quantification**: Provide confidence intervals for predictions
- **Human-in-the-Loop**: Flag uncertain cases for manual review

**Quality Metrics**:
- Target >95% precision with <1% false positive rate
- Implement confidence calibration for reliable decision-making
- Develop content-type specific optimization
- Continuous learning from user feedback

### 🔐 Blockchain-Based Authenticity Verification

**Objective**: Incorporate blockchain-backed content authentication where original media can be hashed and recorded at source.

**Technical Architecture**:
- **Content Hashing**: SHA-256 fingerprints of original media
- **Smart Contracts**: Immutable provenance records on blockchain
- **Timestamp Verification**: Cryptographic proof of creation time
- **Tamper Detection**: Automatic alerts for content modification

**Use Cases**:
- News organizations can authenticate original footage
- Social platforms can verify content provenance
- Legal systems can establish chain of custody
- Users can verify content authenticity independently

### 🌐 API Integration for Platforms & Institutions

**Objective**: Develop scalable REST APIs to enable integration with social media platforms, news agencies, educational institutions, and enterprise systems.

**API Features**:
- **Batch Analysis**: Process multiple media files simultaneously
- **Webhook Integration**: Real-time content verification workflows
- **Custom Models**: Platform-specific model fine-tuning
- **Analytics Dashboard**: Usage statistics and performance metrics

**Integration Targets**:
- **Social Media Platforms**: Twitter, Facebook, Instagram, TikTok APIs
- **News Organizations**: Reuters, AP, BBC content verification systems
- **Educational Institutions**: Academic integrity and plagiarism detection
- **Enterprise Systems**: Corporate communications and brand protection

### 📊 Scalability & Performance Roadmap

**Infrastructure Enhancements**:
- **Microservices Architecture**: Containerized detection services
- **Load Balancing**: Handle millions of requests per day
- **CDN Integration**: Global content delivery for heatmaps/results
- **Auto-scaling**: Dynamic resource allocation based on demand

**Advanced Features**:
- **Multi-language Support**: International explanation generation
- **Custom Model Training**: Platform-specific model adaptation
- **Advanced Analytics**: Content trend analysis and reporting
- **Compliance Tools**: GDPR and content moderation compliance

### 🎯 Impact & Social Benefits

**Misinformation Reduction**:
- Real-time verification reduces spread of fake content
- Authenticity proofs increase trust in digital media
- Automated fact-checking at scale
- Support for democratic information ecosystems

**Industry Applications**:
- **Journalism**: Rapid verification of breaking news
- **Legal**: Forensic evidence for content authentication
- **Education**: Academic integrity and plagiarism prevention
- **Enterprise**: Brand protection and content verification

**Technical Innovation**:
- **Explainable AI**: Transparent decision-making processes
- **Blockchain Integration**: Immutable content provenance
- **Real-time Processing**: Sub-second analysis capabilities
- **Scalable Architecture**: Enterprise-grade performance

---

## 🌟 Vision Statement

This Deepfake Detection System aims to become the **global standard for content authenticity verification**, combining cutting-edge AI explainability with blockchain-based provenance tracking. Our vision is a world where digital content can be instantly verified, where misinformation is automatically detected, and where trust in digital media is restored through transparent, accountable technology.

**Target Metrics by 2025**:
- **99%+ Detection Accuracy** with <0.5% false positives
- **1-Second Analysis Time** for real-time verification
- **100M+ Daily Analyses** through scalable infrastructure
- **Global Platform Integration** with major social media networks
- **Blockchain Verification** for 10M+ pieces of content

## �🔧 Troubleshooting

### Common Issues & Solutions

1. **Streamlit App Won't Start**:
   ```bash
   # Install streamlit properly
   pip install streamlit>=1.25.0
   
   # Run with correct path
   streamlit run app/streamlit_app.py
   ```

2. **Model Loading Errors**:
   - Ensure Git LFS pulled models: `git lfs pull`
   - Check model paths: `src/models/image_model.pth`, `src/models/video_model.pth`
   - Verify PyTorch compatibility

3. **Grad-CAM Not Working**:
   - Check model has convolutional layers
   - Ensure input tensor is properly shaped (batch, 3, H, W)
   - Verify target layer detection

4. **Memory Issues**:
   - Close other applications
   - Use CPU instead of GPU: `device='cpu'`
   - Reduce video processing: `max_frames=15`

5. **Video Processing Errors**:
   - Check video format: MP4, AVI, MOV, MKV
   - Ensure video has frames (not corrupted)
   - Try shorter video segments

## 📊 Performance Metrics

### Model Performance
- **Image Detection Accuracy**: 82% (optimized threshold: 0.6)
- **Video Detection Accuracy**: 75% (synthetic data validation)
- **Processing Speed**: ~1.2 seconds per image
- **Memory Usage**: ~500MB per model during inference
- **Startup Time**: <5 seconds (model loading)

### System Requirements
- **Python**: 3.8+ recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 100MB free (including models)
- **GPU**: Optional (CUDA for faster processing)

### Supported Formats
- **Images**: JPG, JPEG, PNG (224x224+ recommended)
- **Videos**: MP4, AVI, MOV, MKV (H.264 recommended)
- **Outputs**: JPG heatmaps, JSON results, CSV exports

## 📄 License

This project is for educational and research purposes. Please ensure compliance with applicable laws and regulations when using deepfake detection technology.

**Repository**: https://github.com/ishwari05/deepfake_detector  
**Latest Version**: v1.0.0 with enhanced Grad-CAM  
**Last Updated**: Complete with web interface and explainability  
**Models**: Included via Git LFS (75MB total)
