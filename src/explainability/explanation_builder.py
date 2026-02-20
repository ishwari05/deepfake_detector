from typing import Dict, Any, List, Tuple
import numpy as np


class ExplanationBuilder:
    """
    Builds human-readable explanations for deepfake detection results.
    """
    
    def __init__(self):
        """Initialize the explanation builder."""
        self.region_descriptions = {
            'facial': 'facial region',
            'eye': 'eye region',
            'mouth': 'mouth region',
            'nose': 'nose region',
            'forehead': 'forehead region',
            'cheek': 'cheek region',
            'jaw': 'jaw region',
            'background': 'background region',
            'unknown': 'image region'
        }
    
    def build_image_explanation(self, 
                              fake_probability: float,
                              activation_regions: Dict[str, Any],
                              image_shape: Tuple[int, int]) -> str:
        """
        Build explanation for image deepfake detection.
        
        Args:
            fake_probability: Probability of being fake (0-1)
            activation_regions: Dictionary with bbox, centroid, area_ratio
            image_shape: (height, width) of the image
            
        Returns:
            Human-readable explanation text
        """
        # Determine classification
        classification = "FAKE" if fake_probability > 0.5 else "REAL"
        confidence = max(fake_probability, 1 - fake_probability) * 100
        
        # Get region information
        bbox = activation_regions.get('bbox')
        centroid = activation_regions.get('centroid')
        area_ratio = activation_regions.get('area_ratio', 0.0)
        
        # Determine which region is activated
        region_type = self._classify_region(centroid, image_shape) if centroid else 'unknown'
        region_desc = self.region_descriptions[region_type]
        
        # Build explanation
        if classification == "FAKE":
            explanation = f"Deepfake detected with {confidence:.1f}% confidence. "
            
            if bbox and area_ratio > 0.01:  # Significant activation
                x, y, w, h = bbox
                explanation += f"High activation detected in the {region_desc} "
                explanation += f"(approximate location: {x}-{x+w}, {y}-{y+h}) "
                explanation += f"covering {area_ratio*100:.1f}% of the image area. "
                explanation += "This suggests manipulation artifacts in this region."
            else:
                explanation += f"Subtle anomalies detected across the {region_desc}. "
                explanation += "These patterns are consistent with AI-generated manipulation."
        else:
            explanation = f"Image classified as REAL with {confidence:.1f}% confidence. "
            
            if bbox and area_ratio > 0.01:
                explanation += f"Low activation in the {region_type} region. "
                explanation += "No significant manipulation artifacts detected."
            else:
                explanation += "No suspicious patterns or manipulation artifacts found."
        
        return explanation
    
    def build_video_explanation(self,
                              fake_probability: float,
                              frame_results: List[Dict[str, Any]],
                              video_shape: Tuple[int, int]) -> str:
        """
        Build explanation for video deepfake detection.
        
        Args:
            fake_probability: Probability of being fake (0-1)
            frame_results: List of frame analysis results
            video_shape: (height, width) of video frames
            
        Returns:
            Human-readable explanation text
        """
        # Determine classification
        classification = "FAKE" if fake_probability > 0.5 else "REAL"
        confidence = max(fake_probability, 1 - fake_probability) * 100
        
        # Analyze frame patterns
        suspicious_frames = [f for f in frame_results if f.get('is_suspicious', False)]
        num_suspicious = len(suspicious_frames)
        total_frames = len(frame_results)
        
        # Get most common activated regions
        activated_regions = []
        for frame_result in suspicious_frames:
            if frame_result.get('activation_regions'):
                centroid = frame_result['activation_regions'].get('centroid')
                if centroid:
                    region_type = self._classify_region(centroid, video_shape)
                    activated_regions.append(region_type)
        
        # Count region frequencies
        region_counts = {}
        for region in activated_regions:
            region_counts[region] = region_counts.get(region, 0) + 1
        
        most_common_region = max(region_counts.keys(), key=lambda k: region_counts[k]) if region_counts else 'unknown'
        region_desc = self.region_descriptions[most_common_region]
        
        # Build explanation
        if classification == "FAKE":
            explanation = f"Deepfake detected in video with {confidence:.1f}% confidence. "
            
            if num_suspicious > 0:
                explanation += f"Suspicious activity detected in {num_suspicious} out of {total_frames} analyzed frames. "
                
                if region_counts:
                    explanation += f"Primary manipulation appears in the {region_desc} "
                    explanation += f"({region_counts[most_common_region]} frames). "
                
                # Temporal pattern analysis
                if num_suspicious > 1:
                    frame_indices = [f['frame_index'] for f in suspicious_frames]
                    explanation += f"Affected frames: {frame_indices[:5]}"
                    if len(frame_indices) > 5:
                        explanation += f" ... (and {len(frame_indices)-5} more)"
                    explanation += ". "
                
                explanation += "This pattern suggests temporal inconsistencies typical of deepfake manipulation."
            else:
                explanation += "Subtle temporal inconsistencies detected across multiple frames. "
                explanation += "These patterns are consistent with AI-generated video manipulation."
        else:
            explanation = f"Video classified as REAL with {confidence:.1f}% confidence. "
            
            if num_suspicious == 0:
                explanation += "No significant manipulation artifacts detected across analyzed frames. "
                explanation += "Temporal patterns appear consistent with authentic video."
            else:
                explanation += f"Minor anomalies detected in {num_suspicious} frames, but insufficient for deepfake classification. "
                explanation += "Overall video appears authentic."
        
        return explanation
    
    def _classify_region(self, centroid: Tuple[int, int], image_shape: Tuple[int, int]) -> str:
        """
        Classify which facial region the centroid belongs to.
        
        Args:
            centroid: (x, y) coordinates
            image_shape: (height, width) of image
            
        Returns:
            Region type string
        """
        if not centroid:
            return 'unknown'
        
        x, y = centroid
        height, width = image_shape
        
        # Normalize coordinates
        x_norm = x / width
        y_norm = y / height
        
        # Simple region classification based on normalized coordinates
        # These are rough estimates for typical face proportions
        if y_norm < 0.3:
            return 'forehead'
        elif y_norm < 0.5:
            if x_norm < 0.3 or x_norm > 0.7:
                return 'eye'
            elif 0.4 <= x_norm <= 0.6:
                return 'nose'
            else:
                return 'cheek'
        elif y_norm < 0.7:
            if 0.3 <= x_norm <= 0.7:
                return 'mouth'
            else:
                return 'cheek'
        elif y_norm < 0.85:
            return 'jaw'
        else:
            return 'background'
    
    def build_frame_summary(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build summary of frame-level analysis.
        
        Args:
            frame_results: List of frame analysis results
            
        Returns:
            Summary dictionary
        """
        if not frame_results:
            return {
                'total_frames': 0,
                'suspicious_frames': [],
                'most_suspicious_frame': None,
                'avg_activation_area': 0.0
            }
        
        # Sort frames by activation area (most suspicious first)
        sorted_frames = sorted(frame_results, 
                             key=lambda f: f.get('activation_regions', {}).get('area_ratio', 0), 
                             reverse=True)
        
        suspicious_frames = [f for f in frame_results if f.get('is_suspicious', False)]
        
        # Calculate average activation area
        activation_areas = [f.get('activation_regions', {}).get('area_ratio', 0) 
                          for f in frame_results]
        avg_activation_area = np.mean(activation_areas) if activation_areas else 0.0
        
        return {
            'total_frames': len(frame_results),
            'suspicious_frames': [f['frame_index'] for f in suspicious_frames],
            'most_suspicious_frame': sorted_frames[0] if sorted_frames else None,
            'avg_activation_area': avg_activation_area,
            'top_3_suspicious': sorted_frames[:3]
        }
