"""
DCT Steganography Module
This module implements steganography using Discrete Cosine Transform (DCT) to hide
a QR code image inside a cover image.
"""

import cv2
import numpy as np
from PIL import Image
import os
from scipy.fftpack import dct, idct

class DCTSteganography:
    def __init__(self):
        """
        Initialize DCT steganography
        """
        self.block_size = 8
        self.alpha = 0.1  # Embedding strength factor
    
    def _split_into_blocks(self, image):
        """
        Split the image into 8x8 blocks
        """
        height, width = image.shape[:2]
        blocks = []
        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                # Handle boundary cases
                h = min(self.block_size, height - i)
                w = min(self.block_size, width - j)
                
                if h == self.block_size and w == self.block_size:
                    blocks.append((i, j, image[i:i+self.block_size, j:j+self.block_size]))
        
        return blocks
    
    def _apply_dct(self, block):
        """
        Apply DCT to a block
        """
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _apply_idct(self, block):
        """
        Apply inverse DCT to a block
        """
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def _prepare_images(self, cover_path, qr_path):
        """
        Prepare cover and QR images for processing
        """
        # Load cover image and convert to grayscale if it's not
        cover = cv2.imread(cover_path)
        if cover is None:
            raise ValueError(f"Failed to load cover image from {cover_path}")
        
        # If color image, convert to grayscale
        if len(cover.shape) == 3:
            cover_gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
        else:
            cover_gray = cover
        
        # Load QR code image and convert to grayscale
        qr = cv2.imread(qr_path)
        if qr is None:
            raise ValueError(f"Failed to load QR code image from {qr_path}")
        
        if len(qr.shape) == 3:
            qr_gray = cv2.cvtColor(qr, cv2.COLOR_BGR2GRAY)
        else:
            qr_gray = qr
        
        # Resize QR code to match cover dimensions
        qr_resized = cv2.resize(qr_gray, (cover_gray.shape[1], cover_gray.shape[0]))
        
        # Normalize pixel values to range 0-1
        cover_norm = cover_gray.astype(np.float32) / 255.0
        qr_norm = qr_resized.astype(np.float32) / 255.0
        
        # Threshold QR code to binary (0 and 1)
        _, qr_binary = cv2.threshold(qr_norm, 0.5, 1.0, cv2.THRESH_BINARY)
        
        return cover, cover_norm, qr_binary
    
    def hide(self, cover_path, qr_path, output_path="stego_image.png"):
        """
        Hide a QR code inside a cover image using DCT
        
        Args:
            cover_path (str): Path to the cover image
            qr_path (str): Path to the QR code image
            output_path (str): Path to save the output image with embedded QR code
            
        Returns:
            str: Path to the steganographic image
        """
        # Prepare images
        original_cover, cover_norm, qr_binary = self._prepare_images(cover_path, qr_path)
        
        # Create a copy of the normalized cover image for embedding
        stego_img = cover_norm.copy()
        
        # Split cover image into blocks
        blocks = self._split_into_blocks(cover_norm)
        
        # Process each block
        for i, j, block in blocks:
            # Get the corresponding QR code block
            qr_block = qr_binary[i:i+self.block_size, j:j+self.block_size]
            
            # Apply DCT to the block
            dct_block = self._apply_dct(block)
            
            # Modify the mid-frequency coefficients based on the QR code
            # Use mid-frequency components which are less perceptible to human eye
            for m in range(1, 4):
                for n in range(1, 4):
                    if m + n >= 3:  # Mid-frequency selection
                        if qr_block[m, n] > 0.5:  # QR code pixel is white
                            dct_block[m, n] += self.alpha
                        else:  # QR code pixel is black
                            dct_block[m, n] -= self.alpha
            
            # Apply inverse DCT
            idct_block = self._apply_idct(dct_block)
            
            # Replace the block in the stego image
            stego_img[i:i+self.block_size, j:j+self.block_size] = idct_block
        
        # Clip values to ensure they are in the valid range [0, 1]
        stego_img = np.clip(stego_img, 0.0, 1.0)
        
        # Convert back to 8-bit image
        stego_img_8bit = (stego_img * 255).astype(np.uint8)
        
        # If original cover was color, merge the modified grayscale channel back
        if len(original_cover.shape) == 3:
            # Create a colored output with the modified grayscale channel
            output_img = original_cover.copy()
            # Modify the luminance without affecting chrominance significantly
            # A simple approach is to modify all channels equally
            for c in range(3):
                output_img[:, :, c] = stego_img_8bit
        else:
            output_img = stego_img_8bit
        
        # Save the output image
        cv2.imwrite(output_path, output_img)
        
        return output_path
    
    def extract(self, stego_path, output_path="extracted_qrcode.png"):
        """
        Extract the hidden QR code from a steganographic image
        
        Args:
            stego_path (str): Path to the steganographic image
            output_path (str): Path to save the extracted QR code
            
        Returns:
            str: Path to the extracted QR code image
        """
        # Load the steganographic image
        stego_img = cv2.imread(stego_path)
        if stego_img is None:
            raise ValueError(f"Failed to load steganographic image from {stego_path}")
        
        # Convert to grayscale if it's not already
        if len(stego_img.shape) == 3:
            stego_gray = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)
        else:
            stego_gray = stego_img
        
        # Create an empty image to hold the extracted QR code
        extracted_qr = np.zeros_like(stego_gray, dtype=np.float32)
        
        # Normalize stego image to range 0-1
        stego_norm = stego_gray.astype(np.float32) / 255.0
        
        # Split into blocks
        blocks = self._split_into_blocks(stego_norm)
        
        # Process each block
        for i, j, block in blocks:
            # Apply DCT to the block
            dct_block = self._apply_dct(block)
            
            # Extract information from the mid-frequency coefficients
            for m in range(1, 4):
                for n in range(1, 4):
                    if m + n >= 3:  # Mid-frequency selection
                        # Determine if coefficient was modified positively or negatively
                        if dct_block[m, n] > 0:
                            extracted_qr[i+m, j+n] = 255
                        else:
                            extracted_qr[i+m, j+n] = 0
          # Apply post-processing to improve QR code visibility
        # Apply Gaussian blur to reduce noise
        extracted_qr = cv2.GaussianBlur(extracted_qr, (5, 5), 0)
        
        # Apply adaptive thresholding to enhance contrast
        extracted_qr_8bit = extracted_qr.astype(np.uint8)
        # Use adaptive thresholding
        binary_qr = cv2.adaptiveThreshold(
            extracted_qr_8bit, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((5, 5), np.uint8)  # Increased kernel size
        morph_qr = cv2.morphologyEx(binary_qr, cv2.MORPH_OPEN, kernel)
        morph_qr = cv2.morphologyEx(morph_qr, cv2.MORPH_CLOSE, kernel)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        morph_qr = clahe.apply(morph_qr)
        
        # Save the extracted QR code
        cv2.imwrite(output_path, morph_qr)
        
        return output_path
