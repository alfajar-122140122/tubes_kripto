"""
QR Code Module
This module handles the generation and reading of QR codes with encrypted data.
"""

import qrcode
from PIL import Image
import cv2
import numpy as np
import os

class QRCodeProcessor:
    def __init__(self):
        """
        Initialize QR Code processor
        """
        pass
    
    def generate_qr_code(self, data, filename="qrcode.png", size=10, border=4):
        """
        Generate a QR code from data
        
        Args:
            data (str): The data to encode in the QR code
            filename (str): Output filename for the QR code image
            size (int): Size of the QR code (box size)
            border (int): Border size
            
        Returns:
            PIL.Image: The generated QR code image
        """
        # Generate QR code instance
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=size,
            border=border,
        )
        
        # Add data to the QR code
        qr.add_data(data)
        qr.make(fit=True)
        
        # Create an image from the QR Code instance
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Save the QR code if filename is specified
        if filename:
            qr_img.save(filename)
        
        return qr_img
    
    def read_qr_code(self, image_path):
        """
        Read a QR code from an image
        
        Args:
            image_path (str): Path to the image containing the QR code
            
        Returns:
            str: The decoded data from the QR code
        """
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        
        # Check if image was loaded successfully
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Initialize QR code detector
        detector = cv2.QRCodeDetector()
        
        # Detect and decode QR code
        data, bbox, _ = detector.detectAndDecode(img)
        
        # If no QR code was found, try using a different approach
        if not data and bbox is None:
            try:
                # Try using pyzbar for QR detection if available
                try:
                    from pyzbar.pyzbar import decode
                    decoded_objects = decode(img)
                    if decoded_objects:
                        data = decoded_objects[0].data.decode('utf-8')
                except ImportError:
                    print("pyzbar not available, using alternative detection method")
                    # If pyzbar is not available, try converting the image and using OpenCV again
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    data, bbox, _ = detector.detectAndDecode(gray)
            except Exception as e:
                print(f"Error in alternative QR detection: {e}")
        
        return data
