#!/usr/bin/env python3
"""
Quick Fix for Deepfake Detection Threshold
"""

import os
import glob
from image_detector import ImageDeepfakeDetector

def test_different_thresholds(model_path, test_images, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Test different classification thresholds."""
    detector = ImageDeepfakeDetector(model_path)
    
    print("Testing Different Classification Thresholds")
    print("=" * 50)
    
    results = {}
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        
        correct = 0
        total = 0
        
        for img_path in test_images[:100]:  # Test subset for speed
            try:
                result = detector.detect(img_path, "threshold_test")
                
                # Apply threshold
                fake_prob = result['fake_probability']
                prediction = 'FAKE' if fake_prob > threshold else 'REAL'
                
                # Simple ground truth (assuming first half are real, second half fake)
                gt = 'REAL' if 'real' in img_path.lower() else 'FAKE'
                
                if prediction == gt:
                    correct += 1
                total += 1
                
            except Exception as e:
                continue
        
        accuracy = correct / total if total > 0 else 0
        results[threshold] = accuracy
        
        print(f"Threshold {threshold}: {accuracy:.3f} accuracy ({correct}/{total})")
    
    # Find best threshold
    best_threshold = max(results.keys(), key=lambda k: results[k])
    best_accuracy = results[best_threshold]
    
    print(f"\n" + "="*50)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Best threshold: {best_threshold}")
    print(f"Best accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    
    return best_threshold, best_accuracy

if __name__ == "__main__":
    # Test on your dataset
    model_path = "model/image_model.pth"
    
    # Get sample images
    real_images = glob.glob("dataset/real_vs_fake/real-vs-fake/test/real/*")[:50]
    fake_images = glob.glob("dataset/real_vs_fake/real-vs-fake/train/fake/*")[:50]
    test_images = real_images + fake_images
    
    best_threshold, best_accuracy = test_different_thresholds(model_path, test_images)
    
    print(f"\nRECOMMENDATION: Use threshold {best_threshold} for {best_accuracy*100:.1f}% accuracy")
