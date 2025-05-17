import numpy as np
import cv2
from scipy.fft import dct, idct
from skimage.metrics import structural_similarity as ssim

def dct_2d(image_block):
    """Apply 2D DCT to an image block"""
    return dct(dct(image_block.T, norm='ortho').T, norm='ortho')

def idct_2d(dct_block):
    """Apply 2D inverse DCT to a DCT block"""
    return idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

def embed_qr_in_image(cover_image_path, qr_image, alpha=0.1, output_path="stego_image.png"):
    """
    Embed QR code into a cover image using DCT
    
    Parameters:
    - cover_image_path: path to the cover image
    - qr_image: numpy array of the QR code
    - alpha: embedding strength (lower value = more imperceptible but less robust)
    - output_path: path to save the output steganography image
    
    Returns:
    - stego_image: image with embedded QR code
    """
    # Load cover image
    cover_image = cv2.imread(cover_image_path)
    if cover_image is None:
        raise ValueError(f"Could not load cover image from {cover_image_path}")
    
    # Convert to grayscale if needed
    if len(cover_image.shape) > 2:
        cover_gray = cv2.cvtColor(cover_image, cv2.COLOR_BGR2GRAY)
    else:
        cover_gray = cover_image.copy()
    
    # Resize QR code to appropriate dimensions (must be smaller than cover image)
    qr_size = min(cover_gray.shape[0] // 2, cover_gray.shape[1] // 2)
    qr_resized = cv2.resize(qr_image, (qr_size, qr_size))
    
    # Enhance QR code contrast to improve extraction later
    _, qr_enhanced = cv2.threshold(qr_resized, 127, 255, cv2.THRESH_BINARY)
    
    # Add a border around the QR code for better detection
    border_width = 10
    qr_with_border = cv2.copyMakeBorder(
        qr_enhanced, 
        border_width, border_width, border_width, border_width, 
        cv2.BORDER_CONSTANT, 
        value=255
    )
    
    # Adjust size after adding border
    qr_size_with_border = qr_with_border.shape[0]
    
    # Normalize QR code to [0,1]
    qr_normalized = qr_with_border / 255.0
    
    # Create stego image as a copy of cover image
    stego_image = cover_image.copy()
    stego_gray = cover_gray.copy()
    
    # Get top-left block of the image for embedding
    block_size = qr_size_with_border
    cover_block = stego_gray[:block_size, :block_size].astype(float)
    
    # Apply DCT to the cover block
    cover_dct = dct_2d(cover_block)
    
    # Save original QR code for reference
    cv2.imwrite("original_qr_to_embed.png", qr_with_border)
    
    # Embed QR code into DCT coefficients
    # We'll use mid-frequency coefficients for better imperceptibility and robustness
    # Use a mask to focus on mid-frequency coefficients
    mask = np.ones_like(cover_dct)
    mask[:5, :5] = 0  # Protect the DC and very low frequency components
    
    # Weighted embedding to preserve more information
    cover_dct += alpha * qr_normalized * np.abs(cover_dct) * mask
    
    # Apply inverse DCT to get the modified block
    stego_block = idct_2d(cover_dct)
    
    # Clip values to valid range
    stego_block = np.clip(stego_block, 0, 255).astype(np.uint8)
    
    # Replace the block in the stego image
    if len(stego_image.shape) > 2:
        stego_image[:block_size, :block_size, 0] = stego_block
        stego_image[:block_size, :block_size, 1] = stego_block
        stego_image[:block_size, :block_size, 2] = stego_block
    else:
        stego_image[:block_size, :block_size] = stego_block
    
    # Save the stego image
    cv2.imwrite(output_path, stego_image)
    
    # Save metadata for extraction
    metadata = {
        'qr_size': qr_size,
        'qr_size_with_border': qr_size_with_border,
        'alpha': alpha,
        'border_width': border_width
    }
    np.save('stego_metadata.npy', metadata)
    
    print(f"Stego image saved as {output_path}")
    return stego_image

def extract_qr_from_image(stego_image_path, metadata_path="stego_metadata.npy"):
    """
    Extract QR code from stego image using DCT
    
    Parameters:
    - stego_image_path: path to the stego image
    - metadata_path: path to metadata file
    
    Returns:
    - extracted_qr: extracted QR code image
    """
    # Load stego image
    stego_image = cv2.imread(stego_image_path)
    if stego_image is None:
        raise ValueError(f"Could not load stego image from {stego_image_path}")
    
    # Convert to grayscale if needed
    if len(stego_image.shape) > 2:
        stego_gray = cv2.cvtColor(stego_image, cv2.COLOR_BGR2GRAY)
    else:
        stego_gray = stego_image.copy()
    
    # Load metadata
    try:
        metadata = np.load(metadata_path, allow_pickle=True).item()
        qr_size = metadata.get('qr_size', None)
        qr_size_with_border = metadata.get('qr_size_with_border', None)
        alpha = metadata.get('alpha', 0.1)
        border_width = metadata.get('border_width', 10)
    except (FileNotFoundError, KeyError):
        print("Metadata file not found or invalid. Using default values.")
        qr_size_with_border = min(stego_gray.shape[0] // 2, stego_gray.shape[1] // 2)
        qr_size = qr_size_with_border - 20  # Approximate size without border
        alpha = 0.1
        border_width = 10
    
    # If we don't have qr_size_with_border in older metadata files
    if qr_size_with_border is None and qr_size is not None:
        qr_size_with_border = qr_size
    
    # Get the block with the embedded QR code
    block_size = qr_size_with_border
    stego_block = stego_gray[:block_size, :block_size].astype(float)
    
    # Calculate DCT of the block
    stego_dct = dct_2d(stego_block)
    
    # Create mask for extraction (similar to the one used for embedding)
    mask = np.ones_like(stego_dct)
    mask[:5, :5] = 0
    
    # Extract QR code from DCT coefficients using the mask
    # We need to handle division by zero or very small values
    denominator = alpha * np.abs(stego_dct) * mask
    denominator[denominator < 0.01] = 1  # Prevent division by very small numbers
    
    extracted_qr_normalized = stego_dct / denominator
    
    # Scale the values to the typical image range [0, 255]
    min_val = np.percentile(extracted_qr_normalized, 5)
    max_val = np.percentile(extracted_qr_normalized, 95)
    
    extracted_qr = np.clip((extracted_qr_normalized - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
    
    # Apply adaptive thresholding for better binarization
    extracted_qr_binary = cv2.adaptiveThreshold(
        extracted_qr, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
    # Enhanced processing for better QR code recognition
    # Apply morphological operations to clean up the QR code
    kernel = np.ones((3, 3), np.uint8)
    extracted_qr_binary = cv2.morphologyEx(extracted_qr_binary, cv2.MORPH_CLOSE, kernel)
    extracted_qr_binary = cv2.morphologyEx(extracted_qr_binary, cv2.MORPH_OPEN, kernel)
    
    # Remove border if it was added during embedding
    if border_width > 0 and qr_size is not None:
        inner_size = qr_size
        start_idx = border_width
        end_idx = start_idx + inner_size
        
        if end_idx <= extracted_qr_binary.shape[0] and end_idx <= extracted_qr_binary.shape[1]:
            extracted_qr_binary = extracted_qr_binary[start_idx:end_idx, start_idx:end_idx]
    
    # Try local area normalization to improve contrast
    normalized = np.zeros_like(extracted_qr_binary)
    cell_size = 8
    
    for i in range(0, extracted_qr_binary.shape[0], cell_size):
        for j in range(0, extracted_qr_binary.shape[1], cell_size):
            i_end = min(i + cell_size, extracted_qr_binary.shape[0])
            j_end = min(j + cell_size, extracted_qr_binary.shape[1])
            
            block = extracted_qr_binary[i:i_end, j:j_end]
            if np.std(block) > 10:  # Only normalize blocks with some variance
                _, normalized_block = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                normalized[i:i_end, j:j_end] = normalized_block
            else:
                normalized[i:i_end, j:j_end] = block
    
    # Save both versions to see which one works better
    cv2.imwrite('extracted_qr.png', extracted_qr_binary)
    cv2.imwrite('extracted_qr_normalized.png', normalized)
    
    print("QR code extracted and saved as extracted_qr.png")
    return extracted_qr_binary

def evaluate_steganography(original_image_path, stego_image_path):
    """
    Evaluate the quality of steganography using metrics like PSNR and SSIM
    """
    original = cv2.imread(original_image_path)
    stego = cv2.imread(stego_image_path)
    
    if original is None or stego is None:
        print("Error loading images for evaluation")
        return
    
    # Calculate PSNR
    mse = np.mean((original - stego) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # Calculate SSIM
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(original_gray, stego_gray)
    
    print(f"PSNR: {psnr} dB")
    print(f"SSIM: {ssim_score}")
    
    return psnr, ssim_score
