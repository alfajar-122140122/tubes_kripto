"""
Main Application
This is the main entry point for the QR Code encryption and steganography application.
It integrates RSA encryption, QR code generation, and DCT-based steganography.
"""

import os
import argparse
import time
from src.rsa_encryption import RSAEncryption
from src.qr_code import QRCodeProcessor
from src.dct_steganography import DCTSteganography
from src.utils import ensure_directory_exists, handle_large_data

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='QR Code Encryption and Steganography Using RSA and DCT'
    )
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
      # Generate keys command
    gen_keys_parser = subparsers.add_parser('generate-keys',
                                          help='Generate RSA key pair')
    gen_keys_parser.add_argument('--key-size', type=int, default=2048,
                               help='RSA key size in bits')
    gen_keys_parser.add_argument('--private-key', type=str, default='private_key.pem',
                               help='Path to save the private key')
    gen_keys_parser.add_argument('--public-key', type=str, default='public_key.pem',
                               help='Path to save the public key')
    gen_keys_parser.add_argument('--output-dir', type=str, default='output',
                              help='Directory to save output files')
      # Encrypt and generate QR code command
    encrypt_parser = subparsers.add_parser('encrypt',
                                          help='Encrypt data and generate QR code')
    encrypt_parser.add_argument('--data', type=str, required=True,
                              help='Data to encrypt')
    encrypt_parser.add_argument('--public-key', type=str, default='public_key.pem',
                              help='Path to the public key')
    encrypt_parser.add_argument('--qr-size', type=int, default=10,
                              help='Size of QR code boxes')
    encrypt_parser.add_argument('--qr-output', type=str, default='qrcode.png',
                              help='Output path for the QR code image')
    encrypt_parser.add_argument('--output-dir', type=str, default='output',
                             help='Directory to save output files')
      # Hide QR code using steganography command
    hide_parser = subparsers.add_parser('hide',
                                        help='Hide QR code in a cover image')
    hide_parser.add_argument('--qr-path', type=str, required=True,
                           help='Path to the QR code image')
    hide_parser.add_argument('--cover-path', type=str, required=True,
                           help='Path to the cover image')
    hide_parser.add_argument('--stego-output', type=str, default='stego_image.png',
                           help='Output path for the steganographic image')
    hide_parser.add_argument('--output-dir', type=str, default='output',
                          help='Directory to save output files')
      # Extract QR code from stego image command
    extract_parser = subparsers.add_parser('extract',
                                         help='Extract QR code from stego image')
    extract_parser.add_argument('--stego-path', type=str, required=True,
                              help='Path to the steganographic image')
    extract_parser.add_argument('--extracted-output', type=str, default='extracted_qrcode.png',
                              help='Output path for the extracted QR code')
    extract_parser.add_argument('--output-dir', type=str, default='output',
                             help='Directory to save output files')
      # Decrypt QR code command
    decrypt_parser = subparsers.add_parser('decrypt',
                                         help='Read QR code and decrypt data')
    decrypt_parser.add_argument('--qr-path', type=str, required=True,
                              help='Path to the QR code image')
    decrypt_parser.add_argument('--private-key', type=str, default='private_key.pem',
                              help='Path to the private key')
    decrypt_parser.add_argument('--output-dir', type=str, default='output',
                              help='Directory to save output files')
      # All-in-one command
    pipeline_parser = subparsers.add_parser('pipeline',
                                         help='Execute the full encryption and hiding pipeline')
    pipeline_parser.add_argument('--data', type=str, required=True,
                              help='Data to encrypt')
    pipeline_parser.add_argument('--cover-path', type=str, required=True,
                              help='Path to the cover image')
    pipeline_parser.add_argument('--key-size', type=int, default=2048,
                               help='RSA key size in bits')
    pipeline_parser.add_argument('--output-dir', type=str, default='output',
                              help='Directory to save output files')
      # Reverse pipeline command
    reverse_parser = subparsers.add_parser('reverse-pipeline',
                                         help='Execute the full extraction and decryption pipeline')
    reverse_parser.add_argument('--stego-path', type=str, required=True,
                              help='Path to the steganographic image')
    reverse_parser.add_argument('--private-key', type=str, default='private_key.pem',
                              help='Path to the private key')
    reverse_parser.add_argument('--output-dir', type=str, default='output',
                              help='Directory to save output files')
    
    return parser.parse_args()

def generate_keys(args):
    """
    Generate RSA key pair
    """
    print("Generating RSA keys...")
    
    # Ensure the output directory exists
    output_dir = args.output_dir
    ensure_directory_exists(output_dir)
    
    # Create paths to save the keys
    private_key_path = os.path.join(output_dir, args.private_key)
    public_key_path = os.path.join(output_dir, args.public_key)
    
    # Generate keys
    rsa = RSAEncryption(key_size=args.key_size)
    private_key, public_key = rsa.generate_keys()
    
    # Save keys to files
    rsa.save_keys_to_file(
        private_key_file=private_key_path,
        public_key_file=public_key_path
    )
    
    print(f"Private key saved to: {private_key_path}")
    print(f"Public key saved to: {public_key_path}")
    
    return private_key, public_key

def encrypt_and_generate_qr(args):
    """
    Encrypt data and generate QR code
    """
    print("Encrypting data and generating QR code...")
    
    # Ensure the output directory exists
    output_dir = args.output_dir
    ensure_directory_exists(output_dir)
    
    # Create path for the QR code
    qr_output_path = os.path.join(output_dir, args.qr_output)
    
    # Load RSA public key
    rsa = RSAEncryption()
    rsa.load_keys_from_file(public_key_file=args.public_key)
    
    # Process data to ensure it's not too large for RSA encryption
    data = handle_large_data(args.data.encode('utf-8'), rsa.public_key)
    
    # Encrypt the data
    encrypted_data = rsa.encrypt(data)
    
    # Generate QR code
    qr_processor = QRCodeProcessor()
    qr_img = qr_processor.generate_qr_code(
        encrypted_data,
        filename=qr_output_path,
        size=args.qr_size
    )
    
    print(f"Encrypted data: {encrypted_data}")
    print(f"QR code saved to: {qr_output_path}")
    
    return encrypted_data, qr_output_path

def hide_qr_in_image(args):
    """
    Hide QR code in a cover image using DCT steganography
    """
    print("Hiding QR code in cover image...")
    
    # Ensure the output directory exists
    output_dir = args.output_dir
    ensure_directory_exists(output_dir)
    
    # Create path for the stego image
    stego_output_path = os.path.join(output_dir, args.stego_output)
    
    # Hide QR code in cover image
    dct_steg = DCTSteganography()
    output_path = dct_steg.hide(
        cover_path=args.cover_path,
        qr_path=args.qr_path,
        output_path=stego_output_path
    )
    
    print(f"Steganographic image saved to: {output_path}")
    
    return output_path

def extract_qr_from_image(args):
    """
    Extract QR code from steganographic image
    """
    print("Extracting QR code from steganographic image...")
    
    # Ensure the output directory exists
    output_dir = args.output_dir
    ensure_directory_exists(output_dir)
    
    # Create path for the extracted QR code
    extracted_output_path = os.path.join(output_dir, args.extracted_output)
    
    # Extract QR code from stego image
    dct_steg = DCTSteganography()
    output_path = dct_steg.extract(
        stego_path=args.stego_path,
        output_path=extracted_output_path
    )
    
    print(f"Extracted QR code saved to: {output_path}")
    
    return output_path

def decrypt_qr_code(args):
    """
    Read QR code and decrypt data
    """
    print("Reading QR code and decrypting data...")
    
    # Ensure the output directory exists if provided
    if hasattr(args, 'output_dir'):
        ensure_directory_exists(args.output_dir)
        # If private_key is a relative path, join it with output_dir
        if not os.path.isabs(args.private_key):
            args.private_key = os.path.join(args.output_dir, args.private_key)
    
    # Read QR code
    qr_processor = QRCodeProcessor()
    encrypted_data = qr_processor.read_qr_code(args.qr_path)
    
    if not encrypted_data:
        print("Error: Could not read data from QR code.")
        return None
    
    # Load RSA private key
    rsa = RSAEncryption()
    rsa.load_keys_from_file(private_key_file=args.private_key)
    
    # Decrypt the data
    decrypted_data = rsa.decrypt(encrypted_data)
    
    print(f"Decrypted data: {decrypted_data}")
    
    return decrypted_data

def execute_pipeline(args):
    """
    Execute the full encryption and hiding pipeline
    """
    print("Executing full encryption and hiding pipeline...")
    
    # Ensure the output directory exists
    output_dir = args.output_dir
    ensure_directory_exists(output_dir)
    
    # Generate keys
    key_args = argparse.Namespace(
        output_dir=args.output_dir,
        key_size=args.key_size,
        private_key='private_key.pem',
        public_key='public_key.pem'
    )
    generate_keys(key_args)
    
    # Encrypt and generate QR code
    encrypt_args = argparse.Namespace(
        output_dir=args.output_dir,
        data=args.data,
        public_key=os.path.join(args.output_dir, 'public_key.pem'),
        qr_size=10,
        qr_output='qrcode.png'
    )
    encrypted_data, qr_path = encrypt_and_generate_qr(encrypt_args)
    
    # Hide QR code in cover image
    hide_args = argparse.Namespace(
        output_dir=args.output_dir,
        qr_path=qr_path,
        cover_path=args.cover_path,
        stego_output='stego_image.png'
    )
    stego_path = hide_qr_in_image(hide_args)
    
    print("Pipeline completed successfully!")
    print(f"Encrypted data: {encrypted_data}")
    print(f"QR code: {qr_path}")
    print(f"Steganographic image: {stego_path}")
    
    return stego_path

def execute_reverse_pipeline(args):
    """
    Execute the full extraction and decryption pipeline
    """
    print("Executing full extraction and decryption pipeline...")
    
    # Ensure the output directory exists
    output_dir = args.output_dir
    ensure_directory_exists(output_dir)
    
    # Extract QR code from stego image
    extract_args = argparse.Namespace(
        output_dir=args.output_dir,
        stego_path=args.stego_path,
        extracted_output='extracted_qrcode.png'
    )
    extracted_qr_path = extract_qr_from_image(extract_args)
      # Decrypt QR code
    decrypt_args = argparse.Namespace(
        qr_path=extracted_qr_path,
        private_key=args.private_key,
        output_dir=args.output_dir
    )
    decrypted_data = decrypt_qr_code(decrypt_args)
    
    print("Reverse pipeline completed successfully!")
    print(f"Extracted QR code: {extracted_qr_path}")
    print(f"Decrypted data: {decrypted_data}")
    
    return decrypted_data

def main():
    """
    Main function
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Execute the requested command
    if args.command == 'generate-keys':
        generate_keys(args)
    elif args.command == 'encrypt':
        encrypt_and_generate_qr(args)
    elif args.command == 'hide':
        hide_qr_in_image(args)
    elif args.command == 'extract':
        extract_qr_from_image(args)
    elif args.command == 'decrypt':
        decrypt_qr_code(args)
    elif args.command == 'pipeline':
        execute_pipeline(args)
    elif args.command == 'reverse-pipeline':
        execute_reverse_pipeline(args)
    else:
        print("Error: Invalid command. Use --help to see available commands.")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
