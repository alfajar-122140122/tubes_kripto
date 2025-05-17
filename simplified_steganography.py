import cv2
import numpy as np
from scipy.fft import dct, idct
import base64
from pyzbar.pyzbar import decode
from rsa_qrcode import load_rsa_keys, rsa_encrypt, rsa_decrypt, generate_qr_code

def dct_2d(image_block):
    """Apply 2D DCT to an image block"""
    return dct(dct(image_block.T, norm='ortho').T, norm='ortho')

def idct_2d(dct_block):
    """Apply 2D inverse DCT to a DCT block"""
    return idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

def embed_in_image(cover_image_path, qr_image, output_path="simplified_stego.png", alpha=0.5):
    """
    Direct embedding with higher strength for better extraction
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
    
    # Resize QR code to appropriate dimensions
    qr_size = min(cover_gray.shape[0] // 4, cover_gray.shape[1] // 4)
    qr_resized = cv2.resize(qr_image, (qr_size, qr_size))
    
    # Add padding around QR code
    padding = 20
    qr_with_padding = cv2.copyMakeBorder(
        qr_resized,
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=255
    )
    
    # Get the size after padding
    padded_size = qr_with_padding.shape[0]
    
    # Create a composite image where the QR code is embedded at the top-left
    stego_image = cover_image.copy()
    
    # For direct embedding in spatial domain
    if padded_size <= min(cover_gray.shape[0], cover_gray.shape[1]):
        # Simple spatial domain embedding at top left
        blend_factor = 0.7  # Higher value for more visible QR code
        
        # Convert QR to 3-channel if needed
        if len(cover_image.shape) > 2:
            qr_with_padding_3ch = cv2.merge([qr_with_padding, qr_with_padding, qr_with_padding])
            
            # Blend the QR code with the cover image
            roi = cover_image[:padded_size, :padded_size]
            blended = cv2.addWeighted(roi, 1 - blend_factor, qr_with_padding_3ch, blend_factor, 0)
            
            # Place the blended result back in the cover image
            stego_image[:padded_size, :padded_size] = blended
        else:
            # For grayscale images
            roi = cover_gray[:padded_size, :padded_size]
            blended = cv2.addWeighted(roi, 1 - blend_factor, qr_with_padding, blend_factor, 0)
            stego_image[:padded_size, :padded_size] = blended
    
    # Save the result
    cv2.imwrite(output_path, stego_image)
    
    # Save QR separately for reference
    cv2.imwrite("qr_embedded_simplified.png", qr_with_padding)
    
    print(f"Simplified stego image saved as {output_path}")
    return stego_image

def extract_from_image(stego_image_path):
    """
    Simplified extraction to recover the QR code
    """
    # Load the stego image
    stego_image = cv2.imread(stego_image_path)
    if stego_image is None:
        raise ValueError(f"Could not load stego image from {stego_image_path}")
    
    # Convert to grayscale if needed
    if len(stego_image.shape) > 2:
        stego_gray = cv2.cvtColor(stego_image, cv2.COLOR_BGR2GRAY)
    else:
        stego_gray = stego_image.copy()
    
    # Estimate size based on image dimensions
    qr_size = min(stego_gray.shape[0] // 3, stego_gray.shape[1] // 3)
    
    # Extract the top-left region where the QR code should be
    extracted_qr = stego_gray[:qr_size, :qr_size]
    
    # Apply adaptive thresholding
    extracted_qr_binary = cv2.adaptiveThreshold(
        extracted_qr, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Save the extracted QR
    cv2.imwrite("extracted_qr_simplified.png", extracted_qr_binary)
    
    # Try to decode the QR code
    try:
        decoded = decode(extracted_qr_binary)
        if decoded:
            data = decoded[0].data.decode('utf-8')
            return data
    except Exception as e:
        print(f"Error decoding simplified QR: {e}")
    
    # If first attempt fails, try additional processing
    try:
        # Invert the image
        inverted = 255 - extracted_qr_binary
        cv2.imwrite("extracted_qr_simplified_inverted.png", inverted)
        
        decoded = decode(inverted)
        if decoded:
            data = decoded[0].data.decode('utf-8')
            return data
    except Exception as e:
        print(f"Error decoding inverted simplified QR: {e}")
    
    # Try OpenCV QR detector as a last resort
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(extracted_qr_binary)
    if data:
        return data
    
    data, _, _ = detector.detectAndDecode(inverted)
    if data:
        return data
    
    return None

def encrypt_embed_simplified(message, cover_image, output_path="simplified_stego.png"):
    """
    Simplified encryption + embedding process
    """
    # Load or generate RSA keys
    private_key, public_key = load_rsa_keys()
    
    # Encrypt the message
    print(f"Encrypting message: {message}")
    encrypted_data = rsa_encrypt(message, public_key)
    print(f"Encrypted data length: {len(encrypted_data)}")
    
    # Generate QR code with encrypted data (version 1, error correction level L)
    qr_img = generate_qr_code(encrypted_data, version=1, error_correction=0)
    print("QR code generated")
    
    # Save QR code
    qr_path = "encrypted_qr_simplified.png"
    cv2.imwrite(qr_path, qr_img)
    
    # Embed QR code in cover image
    stego_img = embed_in_image(cover_image, qr_img, output_path=output_path)
    
    return output_path

def extract_decrypt_simplified(stego_image):
    """
    Simplified extraction + decryption process
    """
    # Extract QR code data
    encoded_data = extract_from_image(stego_image)
    
    if not encoded_data:
        print("Error: Could not decode QR code from simplified extraction")
        return None
    
    print(f"Decoded data from QR: {encoded_data[:30]}...")
    
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

if __name__ == "__main__":
    # Example usage
    message = "This is a secret message hidden with simplified steganography"
    cover_image = "natural_cover.jpg"
    stego_path = encrypt_embed_simplified(message, cover_image)
    
    # Extract and decrypt
    decrypted = extract_decrypt_simplified(stego_path)
    print(f"Recovered message: {decrypted}")
