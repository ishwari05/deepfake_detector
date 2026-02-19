import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def create_fixed_heatmap(width=300, height=200, regions=None):
    """Create consistent, realistic heatmap."""
    # Create base heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Fixed activation centers (consistent)
    centers = [
        (width//4, height//3, 0.8),      # Eyes
        (3*width//4, height//3, 0.8),   # Other eye
        (width//2, height//2, 0.9),     # Nose/mouth
        (width//2, height//4, 0.6),     # Forehead
        (width//2, 3*height//4, 0.7)  # Jaw
    ]
    
    # Generate consistent heatmap
    for cx, cy, intensity in centers:
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                activation = intensity * np.exp(-distance**2 / (2 * 25**2))
                heatmap[y, x] = max(heatmap[y, x], activation)
    
    # Apply consistent colormap
    colormap = plt.get_cmap('jet')
    heatmap_colored = colormap(heatmap)
    return (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

# Test the fixed heatmap
if __name__ == "__main__":
    heatmap = create_fixed_heatmap()
    img = Image.fromarray(heatmap)
    img.save("fixed_heatmap_test.png")
    print("Fixed heatmap saved!")
