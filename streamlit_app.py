#!/usr/bin/env python3
"""
Streamlit Web Interface for Deepfake Detection System

Professional web app with Grad-CAM explainability, real-time analysis,
and interactive visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import json
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Import your detection modules
from image_detector import ImageDeepfakeDetector
from video_detector import VideoDeepfakeDetector
from audio_detector import AudioDeepfakeDetector

# Configure Streamlit page
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        border-left: 4px solid #3b82f6;
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .confidence-high {
        color: #ef4444;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f59e0b;
        font-weight: bold;
    }
    .confidence-low {
        color: #10b981;
        font-weight: bold;
    }
    .explanation-box {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'detectors' not in st.session_state:
    st.session_state.detectors = {}

def load_detectors():
    """Load detection models."""
    with st.spinner("Loading detection models..."):
        try:
            if 'image' not in st.session_state.detectors:
                st.session_state.detectors['image'] = ImageDeepfakeDetector("model/image_model.pth")
            
            if 'video' not in st.session_state.detectors:
                st.session_state.detectors['video'] = VideoDeepfakeDetector("model/video_model.pth")
            
            if 'audio' not in st.session_state.detectors:
                st.session_state.detectors['audio'] = AudioDeepfakeDetector("audio_model.pth")
                
            st.success("✅ All models loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            return False
    
    return True

def analyze_image(uploaded_file, detector):
    """Analyze uploaded image."""
    with st.spinner("Analyzing image..."):
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Run detection
            result = detector.detect(temp_path, "streamlit_outputs")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
            
        except Exception as e:
            st.error(f"❌ Error analyzing image: {e}")
            return None

def display_result(result, media_type="image"):
    """Display detection result with visualizations."""
    if not result:
        return
    
    # Classification and confidence
    classification = result['classification']
    confidence = max(result['fake_probability'], result['real_probability'])
    confidence_class = 'confidence-high' if confidence > 0.7 else 'confidence-medium' if confidence > 0.5 else 'confidence-low'
    
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
                <strong>Fake Probability:</strong> {result['fake_probability']:.3f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Explanation
    st.markdown(f"""
    <div class="explanation-box">
        <h4>🔍 Explainable Analysis</h4>
        <p>{result['explanation_text']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Heatmap visualization
    if result.get('heatmap_overlay_path') and os.path.exists(result['heatmap_overlay_path']):
        st.markdown("### 🌡️ Grad-CAM Heatmap")
        heatmap_img = Image.open(result['heatmap_overlay_path'])
        st.image(heatmap_img, caption="Grad-CAM visualization showing suspicious regions", use_column_width=True)
    
    # Activation regions
    if result.get('activation_regions'):
        regions = result['activation_regions']
        if regions.get('bbox'):
            x, y, w, h = regions['bbox']
            st.markdown(f"""
            <div class="explanation-box">
                <h4>📍 Region Analysis</h4>
                <p><strong>Activation Bounding Box:</strong> ({x}, {y}, {w}, {h})</p>
                <p><strong>Activation Area:</strong> {regions['area_ratio']*100:.1f}% of image</p>
            </div>
            """, unsafe_allow_html=True)

def display_statistics():
    """Display overall statistics."""
    if not st.session_state.results_history:
        st.info("No results yet. Upload and analyze some media to see statistics.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.results_history)
    
    # Statistics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_analyzed = len(df)
        st.metric("Total Analyzed", total_analyzed)
    
    with col2:
        fake_count = len(df[df['classification'] == 'FAKE'])
        st.metric("Fake Detected", fake_count)
    
    with col3:
        real_count = len(df[df['classification'] == 'REAL'])
        st.metric("Real Detected", real_count)
    
    with col4:
        avg_confidence = df['fake_probability'].mean()
        st.metric("Avg Fake Probability", f"{avg_confidence:.3f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Classification Distribution")
        fig, ax = plt.subplots()
        df['classification'].value_counts().plot(kind='bar', ax=ax, color=['#10b981', '#ef4444'])
        ax.set_title("Real vs Fake Distribution")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 📈 Confidence Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['fake_probability'], bins=20, alpha=0.7, color='#3b82f6')
        ax.set_title("Fake Probability Distribution")
        ax.set_xlabel("Fake Probability")
        ax.set_ylabel("Count")
        st.pyplot(fig)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        🎭 Deepfake Detection System
    </div>
    <p style="text-align: center; color: #6b7280; margin-bottom: 2rem;">
        Grad-CAM Explainability • Real-time Analysis • 82% Accuracy
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Settings")
    
    # Load models
    if st.sidebar.button("🚀 Load Models", type="primary"):
        load_detectors()
    
    # Check if models are loaded
    models_loaded = all(key in st.session_state.detectors for key in ['image', 'video', 'audio'])
    
    if not models_loaded:
        st.warning("⚠️ Please load models first using the button in the sidebar.")
        return
    
    # Main navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Detection", "🎥 Video Detection", "🎵 Audio Detection", "📊 Statistics"])
    
    # Image Detection Tab
    with tab1:
        st.markdown("## 🖼️ Image Deepfake Detection")
        st.markdown("Upload an image to detect if it's a deepfake with Grad-CAM explainability.")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary"):
                detector = st.session_state.detectors['image']
                result = analyze_image(uploaded_file, detector)
                
                if result:
                    # Add to history
                    st.session_state.results_history.append({
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'type': 'image',
                        'filename': uploaded_file.name,
                        'classification': result['classification'],
                        'fake_probability': result['fake_probability'],
                        'explanation': result['explanation_text']
                    })
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 🎯 Detection Results")
                    display_result(result, "image")
    
    # Video Detection Tab
    with tab2:
        st.markdown("## 🎥 Video Deepfake Detection")
        st.markdown("Upload a video to detect deepfakes with frame-by-frame analysis.")
        
        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video_uploader"
        )
        
        if uploaded_video is not None:
            st.info("📹 Video uploaded. Click 'Analyze Video' to start detection.")
            
            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("Analyzing video frames..."):
                    # Save video temporarily
                    temp_path = f"temp_{uploaded_video.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_video.getbuffer())
                    
                    try:
                        detector = st.session_state.detectors['video']
                        result = detector.detect(temp_path, "streamlit_outputs")
                        
                        # Add to history
                        st.session_state.results_history.append({
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'type': 'video',
                            'filename': uploaded_video.name,
                            'classification': result['classification'],
                            'fake_probability': result['fake_probability'],
                            'explanation': result['explanation_text']
                        })
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("## 🎯 Detection Results")
                        display_result(result, "video")
                        
                    except Exception as e:
                        st.error(f"❌ Error analyzing video: {e}")
                    
                    finally:
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
    
    # Audio Detection Tab
    with tab3:
        st.markdown("## 🎵 Audio Deepfake Detection")
        st.markdown("Upload audio to detect AI-generated voice synthesis.")
        
        uploaded_audio = st.file_uploader(
            "Choose audio file...",
            type=['wav', 'mp3', 'm4a', 'flac'],
            key="audio_uploader"
        )
        
        if uploaded_audio is not None:
            st.info("🎵 Audio uploaded. Click 'Analyze Audio' to start detection.")
            
            if st.button("🔍 Analyze Audio", type="primary"):
                with st.spinner("Analyzing audio features..."):
                    try:
                        detector = st.session_state.detectors['audio']
                        
                        # Save audio temporarily
                        temp_path = f"temp_{uploaded_audio.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_audio.getbuffer())
                        
                        result = detector.detect(temp_path, "streamlit_outputs")
                        
                        # Add to history
                        st.session_state.results_history.append({
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'type': 'audio',
                            'filename': uploaded_audio.name,
                            'classification': result['classification'],
                            'fake_probability': result['fake_probability'],
                            'explanation': result['explanation_text']
                        })
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("## 🎯 Detection Results")
                        display_result(result, "audio")
                        
                    except Exception as e:
                        st.error(f"❌ Error analyzing audio: {e}")
                    
                    finally:
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
    
    # Statistics Tab
    with tab4:
        st.markdown("## 📊 Analytics & Statistics")
        display_statistics()
        
        # Results history
        if st.session_state.results_history:
            st.markdown("### 📋 Detection History")
            df = pd.DataFrame(st.session_state.results_history)
            st.dataframe(df[['timestamp', 'type', 'filename', 'classification', 'fake_probability']], 
                        use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"deepfake_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; margin-top: 2rem;">
    <p>🎭 Deepfake Detection System • Grad-CAM Explainability • 82% Accuracy</p>
    <p>Built with Streamlit • PyTorch • OpenCV • Grad-CAM</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
