import os
import argparse
from rsa_qrcode import load_rsa_keys, rsa_encrypt, rsa_decrypt, generate_qr_code, decode_qr_code
from dct_steganography import embed_qr_in_image, extract_qr_from_image, evaluate_steganography
from robust_extraction import extract_and_decrypt_robust

def encrypt_and_embed(message, cover_image, output_image="stego_image.png", alpha=0.1):
    """
    Encrypt a message using RSA, convert to QR code, and embed it in a cover image
    """
    # Load or generate RSA keys
    private_key, public_key = load_rsa_keys()
    
    # Encrypt the message
    print(f"Encrypting message: {message}")
    encrypted_data = rsa_encrypt(message, public_key)
    print(f"Encrypted data (base64): {encrypted_data}")
    
    # Generate QR code with encrypted data (version 1, error correction level L)
    qr_img = generate_qr_code(encrypted_data, version=1, error_correction=0)  # 0 = ERROR_CORRECT_L
    print("QR code generated")
    
    # Save QR code
    qr_path = "encrypted_qr.png"
    import cv2
    cv2.imwrite(qr_path, qr_img)
    print(f"QR code saved as {qr_path}")
    
    # Embed QR code in cover image
    stego_img = embed_qr_in_image(cover_image, qr_img, alpha=alpha, output_path=output_image)
    print(f"QR code embedded in cover image and saved as {output_image}")
    
    # Evaluate steganography quality
    evaluate_steganography(cover_image, output_image)
    
    return output_image

def extract_and_decrypt(stego_image):
    """
    Extract QR code from stego image and decrypt the message
    """
    # Extract QR code from stego image
    extracted_qr = extract_qr_from_image(stego_image)
    print("QR code extracted from stego image")
    
    # Decode QR code
    encoded_data = decode_qr_code(extracted_qr)
    if not encoded_data:
        print("Standard QR decoding failed. Trying robust extraction...")
        return extract_and_decrypt_robust(stego_image)
    
    print(f"Decoded data from QR: {encoded_data}")
    
    # Load RSA keys
    private_key, public_key = load_rsa_keys()
    
    # Decrypt the data
    try:
        decrypted_message = rsa_decrypt(encoded_data, private_key)
        print(f"Decrypted message: {decrypted_message}")
        return decrypted_message
    except Exception as e:
        print(f"Standard decryption failed: {e}. Trying robust extraction...")
        return extract_and_decrypt_robust(stego_image)

def main():
    parser = argparse.ArgumentParser(description='RSA + QR Code + DCT Steganography')
    parser.add_argument('--mode', choices=['encrypt', 'decrypt'], required=True, 
                        help='Mode: encrypt or decrypt')
    parser.add_argument('--message', help='Message to encrypt (required for encrypt mode)')
    parser.add_argument('--cover', help='Cover image path (required for encrypt mode)')
    parser.add_argument('--stego', help='Stego image path (required for decrypt mode)')
    parser.add_argument('--output', help='Output image path for encrypt mode', default="stego_image.png")
    parser.add_argument('--alpha', type=float, help='Embedding strength (0.01-0.2)', default=0.1)
    
    args = parser.parse_args()
    
    if args.mode == 'encrypt':
        if not args.message or not args.cover:
            parser.error("Encrypt mode requires --message and --cover")
        
        encrypt_and_embed(args.message, args.cover, args.output, args.alpha)
    
    elif args.mode == 'decrypt':
        if not args.stego:
            parser.error("Decrypt mode requires --stego")
        
        extract_and_decrypt(args.stego)

if __name__ == "__main__":
    main()
