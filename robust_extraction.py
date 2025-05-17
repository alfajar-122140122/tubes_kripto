import os
import cv2
import numpy as np
from scipy.fft import dct, idct
import qrcode
from pyzbar.pyzbar import decode
from skimage.restoration import denoise_wavelet
from skimage.morphology import erosion, dilation, disk

def preprocess_extracted_qr(qr_image):
    """
    Apply various preprocessing techniques to improve QR code readability
    
    Returns multiple versions of the processed image for QR decoding attempts
    """
    # Ensure we have a grayscale image
    if len(qr_image.shape) > 2:
        gray = cv2.cvtColor(qr_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = qr_image.copy()
    
    # Create a list to store all processed versions
    processed_versions = []
    
    # Pre-processing before applying various techniques
    # 1. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. Apply sharpening filter to enhance edges
    kernel_sharpen = np.array([[-1, -1, -1], 
                              [-1, 9, -1], 
                              [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
    
    # Now apply various binarization techniques on original, blurred and sharpened images
    image_versions = [gray, blurred, sharpened]
    
    # Apply each technique to each image version
    for img_version in image_versions:
        # 1. Basic binary thresholding with different thresholds
        for thresh in [127, 100, 150, 80, 170]:
            _, binary = cv2.threshold(img_version, thresh, 255, cv2.THRESH_BINARY)
            processed_versions.append(binary)
        
        # 2. Adaptive thresholding with different parameters
        for block_size in [7, 11, 15]:
            for c in [2, 4, 6]:
                adaptive = cv2.adaptiveThreshold(
                    img_version, 255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 
                    block_size, c
                )
                processed_versions.append(adaptive)
                
                # Also try inverted version
                processed_versions.append(255 - adaptive)
        
        # 3. Otsu's thresholding
        _, otsu = cv2.threshold(img_version, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_versions.append(otsu)
        
        # 4. Triangle thresholding (useful for bimodal images)
        _, triangle = cv2.threshold(img_version, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        processed_versions.append(triangle)
        
        # 5. Contrast-based techniques for each version
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_version)
        _, enhanced_binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
        processed_versions.append(enhanced_binary)
    
    # Morphological operations on some of the binary images
    for base_img in [processed_versions[0], processed_versions[2], processed_versions[5]]:
        # Different kernel sizes
        for kernel_size in [(3, 3), (5, 5)]:
            kernel = np.ones(kernel_size, np.uint8)
            
            # Dilation
            dilated = cv2.dilate(base_img, kernel, iterations=1)
            processed_versions.append(dilated)
            
            # Erosion
            eroded = cv2.erode(base_img, kernel, iterations=1)
            processed_versions.append(eroded)
            
            # Opening (erosion followed by dilation)
            opened = cv2.morphologyEx(base_img, cv2.MORPH_OPEN, kernel)
            processed_versions.append(opened)
            
            # Closing (dilation followed by erosion)
            closed = cv2.morphologyEx(base_img, cv2.MORPH_CLOSE, kernel)
            processed_versions.append(closed)
            
            # Combination of opening and closing
            combo = cv2.morphologyEx(cv2.morphologyEx(base_img, cv2.MORPH_OPEN, kernel), 
                                   cv2.MORPH_CLOSE, kernel)
            processed_versions.append(combo)
    
    # 6. Denoising with wavelet
    try:
        denoised = denoise_wavelet(gray, method='BayesShrink', mode='soft', 
                                wavelet='db1', rescale_sigma=True)
        denoised = (denoised * 255).astype(np.uint8)
        _, denoised_binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
        processed_versions.append(denoised_binary)
    except Exception as e:
        print(f"Wavelet denoising failed: {e}")
    
    # 7. Edge-based techniques
    edges = cv2.Canny(gray, 100, 200)
    processed_versions.append(edges)
    
    # 8. Invert some versions (QR code might be inverted)
    for i in range(min(10, len(processed_versions))):
        processed_versions.append(255 - processed_versions[i])
    
    # 9. Perspective correction attempts
    # First try to find corners using Harris corner detection
    corner_img = cv2.cornerHarris(gray, 2, 3, 0.04)
    corner_img = cv2.dilate(corner_img, None)
    
    # Get coordinates of potential corners
    coords = np.where(corner_img > 0.01 * corner_img.max())
    
    # If we find enough corners, try perspective transform
    if len(coords[0]) >= 4:
        try:
            # Convert to list of points
            points = np.column_stack((coords[1], coords[0]))
            
            # Find the extremes (approximating the corners)
            rect = np.zeros((4, 2), dtype="float32")
            
            # Top-left: smallest sum of coordinates
            s = points.sum(axis=1)
            rect[0] = points[np.argmin(s)]
            
            # Bottom-right: largest sum of coordinates
            rect[2] = points[np.argmax(s)]
            
            # Top-right: smallest difference of coordinates
            diff = np.diff(points, axis=1)
            rect[1] = points[np.argmin(diff)]
            
            # Bottom-left: largest difference of coordinates
            rect[3] = points[np.argmax(diff)]
            
            # Compute the width/height for the perspective transform
            widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
            widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
            heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # Compute the perspective transform matrix
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            
            # Apply perspective transform
            warped = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))
            
            # Apply thresholding to the warped image
            _, warped_binary = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
            processed_versions.append(warped_binary)
            
            # Also try OTSU on the warped image
            _, warped_otsu = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_versions.append(warped_otsu)
        except Exception as e:
            print(f"Perspective transform failed: {e}")
    
    return processed_versions

def robust_qr_extraction(stego_image_path, metadata_path="stego_metadata.npy"):
    """
    Robust QR code extraction with multiple preprocessing techniques
    """
    # Import the regular extraction function
    from dct_steganography import extract_qr_from_image
    
    # First try the standard extraction
    extracted_qr = extract_qr_from_image(stego_image_path, metadata_path)
    
    # Also try the normalized version
    normalized_qr_path = "extracted_qr_normalized.png"
    if os.path.exists(normalized_qr_path):
        try:
            normalized_qr = cv2.imread(normalized_qr_path, cv2.IMREAD_GRAYSCALE)
            if normalized_qr is not None:
                # Try to decode the normalized version first
                detector = cv2.QRCodeDetector()
                data, bbox, _ = detector.detectAndDecode(normalized_qr)
                
                if data:
                    print("Successfully decoded QR code from normalized version")
                    return data
                
                # Try with pyzbar
                try:
                    decoded = decode(normalized_qr)
                    if decoded:
                        data = decoded[0].data.decode('utf-8')
                        print("Successfully decoded QR code from normalized version with pyzbar")
                        return data
                except Exception as e:
                    print(f"pyzbar decoding failed on normalized version: {e}")
        except Exception as e:
            print(f"Error processing normalized QR: {e}")
    
    # Apply preprocessing to get multiple versions
    processed_versions = preprocess_extracted_qr(extracted_qr)
    
    # Try to decode each version
    for i, version in enumerate(processed_versions):
        # Save each version for debugging
        cv2.imwrite(f"qr_processed_{i}.png", version)
        
        # Try to decode with OpenCV QR detector
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(version)
        
        if data:
            print(f"Successfully decoded QR code with preprocessing method {i}")
            return data
        
        # Try with pyzbar
        try:
            decoded = decode(version)
            if decoded:
                data = decoded[0].data.decode('utf-8')
                print(f"Successfully decoded QR code with pyzbar, method {i}")
                return data
        except Exception as e:
            print(f"pyzbar decoding failed on method {i}: {e}")
    
    # If all fails, try to use direct content-based recovery
    # This is a last resort attempt to reconstruct the QR code manually
    try:
        print("Attempting manual QR code reconstruction...")
        reconstructed = reconstruct_qr_code(extracted_qr)
        if reconstructed is not None:
            cv2.imwrite("qr_reconstructed.png", reconstructed)
            
            # Try to decode the reconstructed QR
            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(reconstructed)
            
            if data:
                print("Successfully decoded reconstructed QR code")
                return data
            
            try:
                decoded = decode(reconstructed)
                if decoded:
                    data = decoded[0].data.decode('utf-8')
                    print("Successfully decoded reconstructed QR code with pyzbar")
                    return data
            except:
                pass
    except Exception as e:
        print(f"Reconstruction attempt failed: {e}")
    
    print("Warning: Could not decode QR with any preprocessing method")
    return None

def reconstruct_qr_code(qr_image):
    """
    Attempt to manually reconstruct a QR code from a damaged image
    """
    # This is a simple implementation - in a real-world scenario, 
    # you would need a more sophisticated algorithm
    
    # Ensure we have a binary image
    _, binary = cv2.threshold(qr_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find all contours
    contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Create a blank canvas
    reconstructed = np.ones_like(binary) * 255
    
    # Sort contours by area (biggest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Take the largest contours
    for i, contour in enumerate(contours[:min(100, len(contours))]):
        # Check if the contour is square-like
        x, y, w, h = cv2.boundingRect(contour)
        if 0.7 <= w / h <= 1.3 and w > 5:  # Square-like and not too small
            cv2.drawContours(reconstructed, [contour], -1, 0, -1)  # Fill the contour
    
    # Apply morphological operations to clean up the result
    kernel = np.ones((3, 3), np.uint8)
    reconstructed = cv2.morphologyEx(reconstructed, cv2.MORPH_CLOSE, kernel)
    reconstructed = cv2.morphologyEx(reconstructed, cv2.MORPH_OPEN, kernel)
    
    return reconstructed

def extract_and_decrypt_robust(stego_image_path):
    """
    Extract QR code with robust methods and decrypt
    """
    from rsa_qrcode import load_rsa_keys, rsa_decrypt
    
    # Try robust extraction
    encoded_data = robust_qr_extraction(stego_image_path)
    
    if not encoded_data:
        print("Error: Could not decode QR code despite robust extraction attempts")
        return None
    
    print(f"Robustly decoded data: {encoded_data}")
    
    # Load RSA keys
    private_key, public_key = load_rsa_keys()
    
    # Decrypt the data
    try:
        decrypted_message = rsa_decrypt(encoded_data, private_key)
        print(f"Decrypted message: {decrypted_message}")
        return decrypted_message
    except Exception as e:
        print(f"Error decrypting message: {e}")
        return None
