import os
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import qrcode
import cv2
import numpy as np
from io import BytesIO
import base64

def generate_rsa_keys(key_size=2048):
    """
    Generate RSA key pair with specified key size
    """
    key = RSA.generate(key_size)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    
    # Save keys to files
    with open('private_key.pem', 'wb') as f:
        f.write(private_key)
    with open('public_key.pem', 'wb') as f:
        f.write(public_key)
    
    return private_key, public_key

def load_rsa_keys():
    """
    Load RSA keys from files
    """
    try:
        with open('private_key.pem', 'rb') as f:
            private_key = RSA.import_key(f.read())
        with open('public_key.pem', 'rb') as f:
            public_key = RSA.import_key(f.read())
        return private_key, public_key
    except FileNotFoundError:
        print("Keys not found. Generating new keys...")
        private_key_data, public_key_data = generate_rsa_keys()
        private_key = RSA.import_key(private_key_data)
        public_key = RSA.import_key(public_key_data)
        return private_key, public_key

def rsa_encrypt(message, public_key):
    """
    Encrypt a message using RSA public key
    """
    cipher = PKCS1_OAEP.new(public_key)
    
    # Convert string to bytes if it's a string
    if isinstance(message, str):
        message = message.encode('utf-8')
        
    # Encrypt data
    encrypted_data = cipher.encrypt(message)
    
    # Encode as base64 for better handling
    encoded_data = base64.b64encode(encrypted_data).decode('utf-8')
    
    return encoded_data

def rsa_decrypt(encrypted_data, private_key):
    """
    Decrypt a message using RSA private key
    """
    # Decode from base64
    encrypted_bytes = base64.b64decode(encrypted_data)
    
    cipher = PKCS1_OAEP.new(private_key)
    decrypted_data = cipher.decrypt(encrypted_bytes)
    
    # Try to decode as utf-8 if possible
    try:
        return decrypted_data.decode('utf-8')
    except UnicodeDecodeError:
        return decrypted_data

def generate_qr_code(data, version=1, error_correction=qrcode.constants.ERROR_CORRECT_L):
    """
    Generate a QR code with specified version and error correction level
    """
    qr = qrcode.QRCode(
        version=version,
        error_correction=error_correction,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert PIL image to numpy array
    img_array = np.array(img)
    
    # Ensure it's a binary image (0 and 255)
    img_array = (img_array * 255).astype(np.uint8)
    
    return img_array

def decode_qr_code(img_array):
    """
    Decode QR code from numpy array image
    """
    # Convert to grayscale if needed
    if len(img_array.shape) > 2 and img_array.shape[2] > 1:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_array
    
    # Threshold the image
    _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find QR codes in the image
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img_thresh)
    
    if not data:
        # If standard detection fails, try with ZBar (if available)
        try:
            import pyzbar.pyzbar as pyzbar
            decoded = pyzbar.decode(img_thresh)
            if decoded:
                data = decoded[0].data.decode('utf-8')
        except ImportError:
            pass
    
    return data

# Example usage
if __name__ == "__main__":
    # Generate or load RSA keys
    private_key, public_key = load_rsa_keys()
    
    # Example message
    message = "Hello, this is a secret message!"
    
    # Encrypt the message
    encrypted_data = rsa_encrypt(message, public_key)
    print(f"Encrypted data: {encrypted_data}")
    
    # Generate QR code with encrypted data
    qr_img = generate_qr_code(encrypted_data)
    
    # Save QR code
    cv2.imwrite("encrypted_qr.png", qr_img)
    
    # Read QR code
    decoded_data = decode_qr_code(qr_img)
    print(f"Decoded QR data: {decoded_data}")
    
    # Decrypt the data
    decrypted_message = rsa_decrypt(decoded_data, private_key)
    print(f"Decrypted message: {decrypted_message}")
