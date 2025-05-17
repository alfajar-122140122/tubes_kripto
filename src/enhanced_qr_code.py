"""
Enhanced QR Code Module
This module handles the generation and reading of QR codes with enhanced detection capabilities.
"""

import qrcode
from PIL import Image
import cv2
import numpy as np
import os
from pyzbar.pyzbar import decode
import tempfile

class EnhancedQRCodeProcessor:
    def __init__(self):
        """
        Initialize QR Code processor with enhanced detection
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
    
    def preprocess_image(self, image):
        """
        Preprocess the image to enhance QR code detection
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            list: List of preprocessed images
        """
        processed_images = []
        
        # Original grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        processed_images.append(("Original", gray))
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(("Adaptive", adaptive))
        
        # Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("Otsu", otsu))
        
        # Equalize histogram
        equalized = cv2.equalizeHist(gray)
        processed_images.append(("Equalized", equalized))
        
        # Equalize followed by Otsu
        _, eq_otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("Equalized+Otsu", eq_otsu))
        
        # Median blur followed by Otsu
        median = cv2.medianBlur(gray, 3)
        _, median_otsu = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("Median+Otsu", median_otsu))
        
        # Bilateral filter for edge preservation and noise removal
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        _, bilateral_otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("Bilateral+Otsu", bilateral_otsu))
        
        # Sharpen filter
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(gray, -1, kernel_sharpen)
        _, sharpen_otsu = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("Sharpen+Otsu", sharpen_otsu))
        
        return processed_images
    
    def read_qr_code(self, image_path, save_processed=False, output_dir=None):
        """
        Read a QR code from an image with enhanced detection
        
        Args:
            image_path (str): Path to the image containing the QR code
            save_processed (bool): Whether to save processed images
            output_dir (str): Directory to save processed images
            
        Returns:
            str: The decoded data from the QR code
        """
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        
        # Check if image was loaded successfully
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Create output directory if specified
        if save_processed and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess the image to enhance QR code detection
        processed_images = self.preprocess_image(img)
        
        # Try to detect QR code in each processed image
        for name, processed_img in processed_images:
            # Save processed image if requested
            if save_processed and output_dir:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                processed_path = os.path.join(output_dir, f"{base_name}_{name}.png")
                cv2.imwrite(processed_path, processed_img)
            
            # Try OpenCV QR detector
            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(processed_img)
            
            if data:
                print(f"QR code detected using OpenCV ({name}): {data}")
                return data
            
            # Try zbar if available
            try:
                # Save to temporary file for pyzbar
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_filename = tmp.name
                    cv2.imwrite(tmp_filename, processed_img)
                
                # Try to decode with pyzbar
                decoded_objects = decode(Image.open(tmp_filename))
                
                # Clean up temp file
                try:
                    os.unlink(tmp_filename)
                except:
                    pass
                
                if decoded_objects:
                    data = decoded_objects[0].data.decode('utf-8')
                    print(f"QR code detected using pyzbar ({name}): {data}")
                    return data
            except Exception as e:
                print(f"Error with pyzbar ({name}): {e}")
        
        # If all else fails, try to use standard libraries through CLI
        try:
            print("Attempting to use standard libraries through CLI...")
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_filename = tmp.name
                # Try with the original image
                cv2.imwrite(tmp_filename, img)
                
                # Try zbarcam through CLI if installed
                import subprocess
                try:
                    result = subprocess.run(['zbarimg', '--quiet', '--raw', tmp_filename], 
                                           capture_output=True, text=True, check=True)
                    data = result.stdout.strip()
                    if data:
                        print(f"QR code detected using zbarimg: {data}")
                        return data
                except Exception as e:
                    print(f"Error with zbarimg: {e}")
                
                # Clean up
                try:
                    os.unlink(tmp_filename)
                except:
                    pass
        except Exception as e:
            print(f"Error trying CLI tools: {e}")
        
        print("Failed to detect QR code in image")
        return None
