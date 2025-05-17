from PIL import Image
import numpy as np
import os

def create_sample_image(output_path, width=800, height=600):
    """Create a sample image for testing the steganography"""
    # Create a gradient background
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient fill
    for y in range(height):
        for x in range(width):
            r = int(255 * (x / width))
            g = int(255 * (y / height))
            b = int(128 + 127 * np.sin(x * y / 10000))
            image_array[y, x] = [r, g, b]
    
    # Convert to PIL Image and save
    image = Image.fromarray(image_array)
    image.save(output_path)
    print(f"Sample image created at {output_path}")
    
    return output_path

if __name__ == "__main__":
    output_path = "sample_cover.jpg"
    create_sample_image(output_path)
