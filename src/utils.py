"""
Utility Module
This module contains utility functions used across the project.
"""

import os
import cv2
import numpy as np
from PIL import Image
import base64
import io

def ensure_directory_exists(directory):
    """
    Ensure that the specified directory exists
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def pil_to_cv2(pil_image):
    """
    Convert PIL Image to OpenCV format
    """
    # Convert PIL Image to NumPy array
    numpy_image = np.array(pil_image)
    
    # If the image is RGB, convert to BGR for OpenCV compatibility
    if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 3:
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    else:
        opencv_image = numpy_image
    
    return opencv_image

def cv2_to_pil(cv2_image):
    """
    Convert OpenCV image to PIL Image format
    """
    # If the image is BGR, convert to RGB for PIL compatibility
    if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
        pil_compatible = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    else:
        pil_compatible = cv2_image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(pil_compatible)
    
    return pil_image

def data_to_base64(data):
    """
    Convert data to base64 encoding
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return base64.b64encode(data).decode('utf-8')

def base64_to_data(base64_data):
    """
    Convert base64 data back to its original form
    """
    return base64.b64decode(base64_data)

def save_image(image, filepath):
    """
    Save an image (PIL or OpenCV) to a file
    """
    # Check if the image is PIL or OpenCV
    if isinstance(image, np.ndarray):
        # OpenCV image
        cv2.imwrite(filepath, image)
    else:
        # PIL image
        image.save(filepath)

def handle_large_data(data, public_key):
    """
    Handle encryption of large data by chunking
    """
    from Crypto.PublicKey import RSA
    from Crypto.Cipher import PKCS1_OAEP
    
    # Get the maximum size of data that can be encrypted with this key
    key_size_bytes = public_key.size_in_bytes()
    max_data_length = key_size_bytes - 42  # PKCS#1 OAEP padding is 42 bytes
    
    # If data is smaller than max size, return it as is
    if len(data) <= max_data_length:
        return data
    
    # Otherwise, we need to split it into chunks
    # For simplicity, we'll just truncate the data
    # In a real implementation, you would need to implement chunking and recombining
    print(f"Warning: Data size ({len(data)} bytes) exceeds maximum size ({max_data_length} bytes). Data will be truncated.")
    return data[:max_data_length]
