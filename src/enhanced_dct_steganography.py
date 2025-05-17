"""
Enhanced DCT Steganography Module
This module implements steganography using Discrete Cosine Transform (DCT) to hide
a QR code image inside a cover image with improved extraction algorithm.
"""

import cv2
import numpy as np
from PIL import Image
import os
from scipy.fftpack import dct, idct

class EnhancedDCTSteganography:
    def __init__(self):
        """
        Initialize DCT steganography with enhanced extraction capabilities
        """
        self.block_size = 8
        self.alpha = 0.1  # Embedding strength factor
        self.mid_freq_bands = [(4,1), (3,2), (2,3), (1,4)]  # Mid frequency bands for better embedding
    
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
        # Load cover image
        cover = cv2.imread(cover_path)
        if cover is None:
            raise ValueError(f"Failed to load cover image from {cover_path}")
        
        # Keep the original color cover image for later
        original_cover = cover.copy()
        
        # Convert to grayscale if not already
        if len(cover.shape) == 3:
            cover_gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
        else:
            cover_gray = cover
        
        # Load QR code image
        qr = cv2.imread(qr_path)
        if qr is None:
            raise ValueError(f"Failed to load QR code image from {qr_path}")
        
        # Convert to grayscale if not already
        if len(qr.shape) == 3:
            qr_gray = cv2.cvtColor(qr, cv2.COLOR_BGR2GRAY)
        else:
            qr_gray = qr
        
        # Resize QR code to match cover dimensions (maintaining aspect ratio)
        h, w = cover_gray.shape
        qr_resized = cv2.resize(qr_gray, (w, h))
        
        # Normalize pixel values to range 0-1
        cover_norm = cover_gray.astype(np.float32) / 255.0
        qr_norm = qr_resized.astype(np.float32) / 255.0
        
        # Threshold QR code to binary (0 and 1)
        _, qr_binary = cv2.threshold(qr_norm, 0.5, 1.0, cv2.THRESH_BINARY)
        
        return original_cover, cover_norm, qr_binary
    
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
        
        # Create an embedding map to help with extraction
        embed_map = np.zeros_like(cover_norm)
        
        # Process each block
        for i, j, block in blocks:
            # Get the corresponding QR code block
            qr_block = qr_binary[i:i+self.block_size, j:j+self.block_size]
            
            # Apply DCT to the block
            dct_block = self._apply_dct(block)
            
            # Embedding in specific mid-frequency bands
            for m, n in self.mid_freq_bands:
                if qr_block[m, n] > 0.5:  # QR code pixel is white
                    dct_block[m, n] += self.alpha
                    embed_map[i+m, j+n] = 1
                else:  # QR code pixel is black
                    dct_block[m, n] -= self.alpha
                    embed_map[i+m, j+n] = -1
            
            # Apply inverse DCT
            idct_block = self._apply_idct(dct_block)
            
            # Replace the block in the stego image
            stego_img[i:i+self.block_size, j:j+self.block_size] = idct_block
        
        # Clip values to ensure they are in the valid range [0, 1]
        stego_img = np.clip(stego_img, 0.0, 1.0)
        
        # Convert back to 8-bit image
        stego_img_8bit = (stego_img * 255).astype(np.uint8)
        
        # If original cover was color, create a colored output
        if len(original_cover.shape) == 3:
            # Create a colored output with the modified luminance channel
            output_img = original_cover.copy()
            
            # Convert to YCrCb color space
            ycrcb = cv2.cvtColor(original_cover, cv2.COLOR_BGR2YCrCb)
            
            # Replace Y channel with our steganographic image
            ycrcb[:,:,0] = stego_img_8bit
            
            # Convert back to BGR
            output_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            output_img = stego_img_8bit
        
        # Save the output image
        cv2.imwrite(output_path, output_img)
        
        # Save embedding map for later use in extraction
        embed_map_path = os.path.splitext(output_path)[0] + "_map.npy"
        np.save(embed_map_path, embed_map)
        
        return output_path
    
    def extract(self, stego_path, output_path="extracted_qrcode.png", use_map=False):
        """
        Extract the hidden QR code from a steganographic image
        
        Args:
            stego_path (str): Path to the steganographic image
            output_path (str): Path to save the extracted QR code
            use_map (bool): Whether to use the embedding map for extraction
            
        Returns:
            str: Path to the extracted QR code image
        """
        # Load the steganographic image
        stego_img = cv2.imread(stego_path)
        if stego_img is None:
            raise ValueError(f"Failed to load steganographic image from {stego_path}")
        
        # Convert to YCrCb if it's a color image and extract the Y channel
        if len(stego_img.shape) == 3:
            ycrcb = cv2.cvtColor(stego_img, cv2.COLOR_BGR2YCrCb)
            stego_gray = ycrcb[:,:,0]  # Y channel contains the hidden data
        else:
            stego_gray = stego_img
        
        # Create an empty image for the extracted QR code
        h, w = stego_gray.shape
        extracted_qr = np.zeros((h, w), dtype=np.float32)
        
        # Normalize stego image
        stego_norm = stego_gray.astype(np.float32) / 255.0
        
        # Check if embedding map exists and should be used
        map_path = os.path.splitext(stego_path)[0] + "_map.npy"
        if use_map and os.path.exists(map_path):
            print("Using embedding map for extraction")
            embed_map = np.load(map_path)
            
            # Extract QR code using the map
            extracted_qr = (embed_map > 0).astype(np.float32) * 255.0
        else:
            # Split into blocks
            blocks = self._split_into_blocks(stego_norm)
            
            # Process each block
            for i, j, block in blocks:
                # Apply DCT to the block
                dct_block = self._apply_dct(block)
                
                # Extract information from the mid-frequency bands
                for m, n in self.mid_freq_bands:
                    # Use a threshold to determine if the coefficient was modified
                    if dct_block[m, n] > 0.01:  # Positive modification
                        extracted_qr[i+m, j+n] = 255
                    else:  # Negative modification
                        extracted_qr[i+m, j+n] = 0
        
        # Apply post-processing to improve QR code visibility
        extracted_qr_8bit = extracted_qr.astype(np.uint8)
        
        # Apply adaptive thresholding
        binary_qr = cv2.adaptiveThreshold(
            extracted_qr_8bit, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Apply noise removal
        kernel = np.ones((3, 3), np.uint8)
        denoised_qr = cv2.morphologyEx(binary_qr, cv2.MORPH_OPEN, kernel)
        denoised_qr = cv2.morphologyEx(denoised_qr, cv2.MORPH_CLOSE, kernel)
        
        # Apply median blur to reduce salt-and-pepper noise
        denoised_qr = cv2.medianBlur(denoised_qr, 3)
        
        # Enhance the QR code with sharpen filter
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened_qr = cv2.filter2D(denoised_qr, -1, kernel_sharpen)
        
        # Final thresholding to ensure binary image
        _, binary_qr_final = cv2.threshold(sharpened_qr, 127, 255, cv2.THRESH_BINARY)
        
        # Save the extracted QR code
        cv2.imwrite(output_path, binary_qr_final)
        
        # Also save intermediate processed versions for analysis
        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # Save different processed versions
        cv2.imwrite(os.path.join(base_dir, f"{base_name}_denoised.png"), denoised_qr)
        cv2.imwrite(os.path.join(base_dir, f"{base_name}_sharpened.png"), sharpened_qr)
        
        return output_path
