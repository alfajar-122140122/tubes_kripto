# QR Code Encryption and Steganography Using RSA and DCT

This project implements a layered security solution for QR codes using two security techniques: RSA encryption and DCT-based steganography. This dual-layer approach provides enhanced security for sensitive information shared via QR codes.

## Features

- **RSA Encryption**: Generate RSA key pairs, encrypt data using public keys, and decrypt data using private keys
- **QR Code Generation**: Convert encrypted data into QR codes
- **DCT Steganography**: Hide QR codes within cover images using Discrete Cosine Transform technique
- **Full Pipeline**: Complete end-to-end encryption, QR code generation, and steganographic hiding process
- **Reverse Pipeline**: Extract QR codes from steganographic images and decrypt the data

## Requirements

- Python 3.7+
- Required Python packages:
  - pycryptodome (for RSA encryption)
  - qrcode (for QR code generation)
  - pillow (for image processing)
  - numpy (for matrix operations)
  - opencv-python (for image processing)
  - scipy (for DCT transformation)

## Installation

1. Clone or download this repository
2. Install required packages:

```bash
pip install pycryptodome qrcode pillow numpy opencv-python scipy scikit-image
```

## Usage

The application provides several commands for different operations:

### Generate RSA Key Pair

```bash
python main.py generate-keys --output-dir output --key-size 2048
```

This will generate a private key (`private_key.pem`) and a public key (`public_key.pem`) in the specified output directory.

### Encrypt Data and Generate QR Code

```bash
python main.py encrypt --data "Your secret message" --public-key output/public_key.pem --qr-output encrypted_qrcode.png --output-dir output
```

This encrypts the provided data using the specified public key and generates a QR code containing the encrypted data.

### Hide QR Code in an Image

```bash
python main.py hide --qr-path output/encrypted_qrcode.png --cover-path cover_image.jpg --stego-output hidden_qrcode.png --output-dir output
```

This hides the QR code within the cover image using DCT-based steganography.

### Extract QR Code from Steganographic Image

```bash
python main.py extract --stego-path output/hidden_qrcode.png --extracted-output extracted_qrcode.png --output-dir output
```

This extracts the hidden QR code from the steganographic image.

### Decrypt Data from QR Code

```bash
python main.py decrypt --qr-path output/extracted_qrcode.png --private-key output/private_key.pem
```

This reads the QR code and decrypts the data using the specified private key.

### Execute Full Pipeline

```bash
python main.py pipeline --data "Your secret message" --cover-path cover_image.jpg --key-size 2048 --output-dir output
```

This executes the full pipeline: generating keys, encrypting data, creating a QR code, and hiding it in the cover image.

### Execute Reverse Pipeline

```bash
python main.py reverse-pipeline --stego-path output/stego_image.png --private-key output/private_key.pem --output-dir output
```

This executes the reverse pipeline: extracting the QR code from the steganographic image and decrypting the data.

## Project Structure

- `main.py`: Main entry point for the application
- `src/rsa_encryption.py`: RSA encryption module
- `src/qr_code.py`: QR code generation and reading module
- `src/dct_steganography.py`: DCT-based steganography module
- `src/utils.py`: Utility functions

## How It Works

1. **RSA Encryption**: Uses asymmetric cryptography for secure data encryption
2. **QR Code Generation**: Encrypted data is encoded into a QR code
3. **DCT Steganography**: The QR code is hidden within a cover image using the Discrete Cosine Transform technique, which modifies frequency coefficients in a way that is resilient to common image processing operations while being imperceptible to the human eye

## Limitations

- RSA encryption has a maximum data size limit based on the key size
- The quality of QR code extraction depends on the steganographic image quality
- Large modifications to the steganographic image may corrupt the hidden QR code

## Security Considerations

- Keep the private key secure; anyone with access to it can decrypt the data
- The security of the system depends on both the strength of the RSA encryption and the imperceptibility of the steganographic technique
- This implementation is for educational purposes and may need additional security measures for production use
