#!/usr/bin/env python3
"""
Streamlit Dashboard Using Your Actual Trained Models
Connects to image_detector.py and video_detector.py for real predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your actual detectors
try:
    from image_detector import ImageDeepfakeDetector
    from video_detector import VideoDeepfakeDetector
    print("✅ Successfully imported actual detectors")
except ImportError as e:
    print(f"❌ Import error: {e}")
    st.error(f"Failed to import detectors: {e}")

# Configure page
st.set_page_config(
    page_title="Real Deepfake Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .confidence-high { color: #ef4444; font-weight: bold; }
    .confidence-medium { color: #f59e0b; font-weight: bold; }
    .confidence-low { color: #10b981; font-weight: bold; }
    .explanation-box {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .heatmap-box {
        border: 2px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'detectors_loaded' not in st.session_state:
    st.session_state.detectors_loaded = False

# Load models instantly at startup
@st.cache_resource
def load_models_instantly():
    """Load models once and cache them for instant access."""
    try:
        # Import your actual detectors
        from image_detector import ImageDeepfakeDetector
        from video_detector import VideoDeepfakeDetector
        
        # Model paths
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        image_model_path = os.path.join(model_dir, 'image_model.pth')
        video_model_path = os.path.join(model_dir, 'video_model.pth')
        
        # Load models
        image_detector = None
        video_detector = None
        
        if os.path.exists(image_model_path):
            print("🚀 Loading image model...")
            image_detector = ImageDeepfakeDetector(image_model_path)
            print("✅ Image model loaded!")
        
        if os.path.exists(video_model_path):
            print("🚀 Loading video model...")
            video_detector = VideoDeepfakeDetector(video_model_path)
            print("✅ Video model loaded!")
        
        return image_detector, video_detector
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return None, None

# Load models immediately when app starts
image_detector, video_detector = load_models_instantly()

def load_actual_models():
    """Load your actual trained models."""
    try:
        # Model paths
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        image_model_path = os.path.join(model_dir, 'image_model.pth')
        video_model_path = os.path.join(model_dir, 'video_model.pth')
        
        # Check if models exist
        if not os.path.exists(image_model_path):
            st.error(f"❌ Image model not found: {image_model_path}")
            return None, None
        
        if not os.path.exists(video_model_path):
            st.warning(f"⚠️ Video model not found: {video_model_path}")
        
        # Load image detector
        with st.spinner("Loading image detector..."):
            image_detector = ImageDeepfakeDetector(image_model_path)
            print("✅ Image detector loaded successfully")
        
        # Load video detector (if available)
        video_detector = None
        if os.path.exists(video_model_path):
            with st.spinner("Loading video detector..."):
                video_detector = VideoDeepfakeDetector(video_model_path)
                print("✅ Video detector loaded successfully")
        
        return image_detector, video_detector
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        print(f"Model loading error: {e}")
        return None, None

def analyze_with_actual_model(detector, uploaded_file, file_type="image"):
    """Analyze file using your actual trained model."""
    try:
        # Save uploaded file temporarily
        temp_dir = "temp_analysis"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Run actual detection
        with st.spinner(f"Analyzing {file_type} with trained model..."):
            result = detector.detect(temp_path, temp_dir)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return result
        
    except Exception as e:
        st.error(f"❌ Analysis error: {e}")
        return None

def display_actual_results(result, uploaded_file):
    """Display results from your actual model."""
    if not result:
        return
    
    classification = result['classification']
    fake_prob = result['fake_probability']
    real_prob = result['real_probability']
    confidence = max(fake_prob, real_prob)
    
    # Determine confidence class
    if confidence > 0.7:
        confidence_class = "confidence-high"
    elif confidence > 0.5:
        confidence_class = "confidence-medium"
    else:
        confidence_class = "confidence-low"
    
    # Main result card
    st.markdown(f"""
    <div class="result-card">
        <h3 style="color: {'#ef4444' if classification == 'FAKE' else '#10b981'}">
            🎭 {classification.upper()} DETECTED
        </h3>
        <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
            <div>
                <strong>Confidence:</strong>
                <span class="{confidence_class}">{confidence*100:.1f}%</span>
            </div>
            <div>
                <strong>Fake Probability:</strong> {fake_prob:.3f}
            </div>
            <div>
                <strong>Real Probability:</strong> {real_prob:.3f}
            </div>
        </div>
        <div style="background: #f0f9ff; padding: 0.5rem; border-radius: 0.25rem; margin-top: 0.5rem;">
            <strong>🔍 Threshold Logic:</strong> Fake Prob &lt; 0.6 = REAL | Fake Prob ≥ 0.6 = FAKE
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Heatmap and visualization
    st.markdown("### 🌡️ Grad-CAM Heatmap Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Original Image")
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
    
    with col2:
        st.markdown("#### 🔥 Grad-CAM Heatmap")
        # Check if heatmap overlay exists
        heatmap_path = result.get('heatmap_overlay_path')
        if heatmap_path and os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="Grad-CAM heatmap overlay", use_container_width=True)
        else:
            st.info("Grad-CAM heatmap not available")
    
    # Detailed analysis
    st.markdown("### 📈 Detailed Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 Classification Breakdown")
        fig, ax = plt.subplots(figsize=(6, 4))
        categories = ['Fake', 'Real']
        values = [fake_prob, real_prob]
        colors = ['#ef4444', '#10b981']
        ax.bar(categories, values, color=colors)
        ax.set_ylabel('Probability')
        ax.set_title('Model Output')
        ax.set_ylim(0, 1)
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### 📊 Confidence Analysis")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Confidence'], [confidence], color='blue')
        ax.set_ylabel('Confidence')
        ax.set_title('Overall Confidence')
        ax.set_ylim(0, 1)
        st.pyplot(fig)
    
    with col3:
        st.markdown("#### 🔍 Threshold Analysis")
        threshold = 0.6  # Your optimized threshold
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
        ax.axhline(y=fake_prob, color='blue', linewidth=2, label=f'Fake Prob: {fake_prob:.3f}')
        ax.axhline(y=real_prob, color='green', linewidth=2, label=f'Real Prob: {real_prob:.3f}')
        ax.set_ylabel('Probability')
        ax.set_title('Threshold vs Predictions')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add classification text
        ax.text(0.5, 0.95, f'CLASSIFICATION: {classification}', 
                ha='center', va='top', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue' if classification == 'REAL' else 'lightcoral'))
        
        st.pyplot(fig)
    
    # Activation regions
    if result.get('activation_regions'):
        st.markdown("### 📍 Activation Regions")
        regions = result['activation_regions']
        
        if regions.get('bbox'):
            x, y, w, h = regions['bbox']
            st.markdown(f"""
            <div class="explanation-box">
                <h4>🔍 Region Analysis</h4>
                <p><strong>Activation Bounding Box:</strong> ({x}, {y}, {w}, {h})</p>
                <p><strong>Activation Area:</strong> {regions['area_ratio']*100:.1f}% of image</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Explanation
    explanation = result.get('explanation_text', 'No explanation available.')
    st.markdown(f"""
    <div class="explanation-box">
        <h4>🔍 Model Explanation</h4>
        <p>{explanation}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download results
    st.markdown("### 📄 Download Results")
    
    # Create report
    report = f"""
DEEPFAKE DETECTION REPORT
========================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
File: {uploaded_file.name}
Classification: {classification}
Confidence: {confidence*100:.1f}%
Fake Probability: {fake_prob:.3f}
Real Probability: {real_prob:.3f}

EXPLANATION:
{explanation}

Model: Your Trained CNN with Grad-CAM
Threshold: 0.6
"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📥 Report (TXT)",
            data=report,
            file_name=f"deepfake_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col2:
        csv_data = pd.DataFrame([{
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Filename': uploaded_file.name,
            'Classification': classification,
            'Confidence': f"{confidence*100:.1f}%",
            'Fake_Probability': f"{fake_prob:.3f}",
            'Real_Probability': f"{real_prob:.3f}"
        }]).to_csv(index=False)
        
        st.download_button(
            label="📥 Data (CSV)",
            data=csv_data,
            file_name=f"deepfake_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    """Main Streamlit application with instant model loading."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        🎭 Real Deepfake Detection System
    </div>
    <p style="text-align: center; color: #6b7280; margin-bottom: 2rem;">
        Models Pre-Loaded • Instant Analysis • Grad-CAM Explainability • 82% Accuracy
    </p>
    """, unsafe_allow_html=True)
    
    # Model status display
    st.markdown("## 🚀 Model Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if image_detector:
            st.success("✅ Image Model: LOADED & READY")
        else:
            st.error("❌ Image Model: NOT LOADED")
    
    with col2:
        if video_detector:
            st.success("✅ Video Model: LOADED & READY")
        else:
            st.warning("⚠️ Video Model: NOT AVAILABLE")
    
    # Main navigation
    tab1, tab2, tab3 = st.tabs(["🖼️ Image Detection", "🎥 Video Detection", "📊 Statistics"])
    
    # Image Detection Tab
    with tab1:
        st.markdown("## 🖼️ Image Deepfake Detection")
        st.markdown("Upload an image for instant deepfake detection - models are already loaded!")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Analyze button - no loading needed!
            if st.button("� Analyze Instantly", type="primary"):
                if image_detector:
                    result = analyze_with_actual_model(image_detector, uploaded_file, "image")
                    
                    if result:
                        # Add to history
                        result_entry = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'filename': uploaded_file.name,
                            'type': 'image',
                            'classification': result['classification'],
                            'confidence': max(result['fake_probability'], result['real_probability']),
                            'fake_probability': result['fake_probability'],
                            'real_probability': result['real_probability']
                        }
                        st.session_state.results_history.append(result_entry)
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("## 🎯 Instant Prediction Results")
                        display_actual_results(result, uploaded_file)
                else:
                    st.error("❌ Image model not loaded")
    
    # Video Detection Tab
    with tab2:
        st.markdown("## 🎥 Video Deepfake Detection")
        st.markdown("Upload a video for frame-by-frame deepfake analysis.")
        
        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video_uploader"
        )
        
        if uploaded_video is not None:
            st.info("📹 Video uploaded. Click 'Analyze Video' to start detection.")
            
            if st.button("🔍 Analyze Video", type="primary"):
                if video_detector:
                    result = analyze_with_actual_model(video_detector, uploaded_video, "video")
                    
                    if result:
                        # Add to history
                        result_entry = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'filename': uploaded_video.name,
                            'type': 'video',
                            'classification': result['classification'],
                            'confidence': max(result['fake_probability'], result['real_probability']),
                            'fake_probability': result['fake_probability'],
                            'real_probability': result['real_probability']
                        }
                        st.session_state.results_history.append(result_entry)
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("## 🎯 Video Analysis Results")
                        display_actual_results(result, uploaded_video)
                else:
                    st.error("❌ Video model not loaded")
    
    # Statistics Tab
    with tab3:
        st.markdown("## 📊 Analytics & Statistics")
        
        if st.session_state.results_history:
            df = pd.DataFrame(st.session_state.results_history)
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyzed", len(df))
            
            with col2:
                fake_count = len(df[df['classification'] == 'FAKE'])
                st.metric("Fake Detected", fake_count)
            
            with col3:
                real_count = len(df[df['classification'] == 'REAL'])
                st.metric("Real Detected", real_count)
            
            with col4:
                avg_confidence = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Classification Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                df['classification'].value_counts().plot(kind='bar', ax=ax, color=['#10b981', '#ef4444'])
                ax.set_title("Real vs Fake Distribution")
                ax.set_ylabel("Count")
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### 📈 Confidence Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(df['confidence'], bins=20, alpha=0.7, color='#3b82f6')
                ax.set_title("Confidence Distribution")
                ax.set_xlabel("Confidence")
                ax.set_ylabel("Count")
                st.pyplot(fig)
            
            # Results table
            st.markdown("#### 📋 Results History")
            st.dataframe(df[['timestamp', 'filename', 'type', 'classification', 'confidence', 'fake_probability']], 
                        use_container_width=True)
            
            # Download all results
            csv_all = df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Results (CSV)",
                data=csv_all,
                file_name=f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No results yet. Upload and analyze some images to see statistics.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; margin-top: 2rem;">
    <p>🎭 Real Deepfake Detection System • Your Actual Trained Models • 82% Accuracy</p>
    <p>Using image_detector.py and video_detector.py • Real Predictions • Grad-CAM Explainability</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
