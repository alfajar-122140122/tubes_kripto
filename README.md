# RSA + QR Code + DCT Steganography

This project implements a secure steganography system combining RSA encryption, QR Codes, and DCT-based image steganography.

## System Architecture

The main components of the system are:

1. **Input Data**: User provides the text or information to be secured.
2. **RSA Encryption**: Information is encrypted using the RSA algorithm with a public key.
3. **QR Code Generation**: Encrypted data is encoded into a QR Code (Version 1, Error Correction Level L, 152-bit).
4. **DCT-based Steganography**: The QR Code is embedded into a cover image using the Discrete Cosine Transform algorithm.
5. **Extraction and Decryption**: The system can extract the hidden QR Code, decode it, and decrypt the message using the RSA private key.

## Requirements

- Python 3.6+
- Required libraries:
  - numpy
  - pillow
  - pycryptodome
  - qrcode
  - opencv-python
  - scikit-image

## Installation

```bash
pip install numpy pillow pycryptodome qrcode opencv-python scikit-image
```

For the GUI interface, the standard tkinter library is used (included with Python).

## Usage

### Command-Line Interface

To encrypt and embed a message:

```bash
python main.py --mode encrypt --message "Your secret message" --cover path/to/cover_image.jpg --output path/to/output_image.png --alpha 0.1
```

To extract and decrypt a message:

```bash
python main.py --mode decrypt --stego path/to/stego_image.png
```

### GUI Interface

For a more user-friendly experience, you can use the GUI:

```bash
python gui.py
```

## Modules

1. **rsa_qrcode.py**: Handles RSA encryption/decryption and QR code generation/reading.
2. **dct_steganography.py**: Implements the DCT-based steganography for embedding and extraction.
3. **robust_extraction.py**: Contains enhanced QR code extraction algorithms to improve resilience.
4. **main.py**: Command-line interface for the system.
5. **gui.py**: Graphical user interface for easier interaction.

## Key Features

- **Security**: RSA encryption ensures that even if the steganography is detected, the message remains secure.
- **Robust QR Extraction**: Multiple preprocessing techniques are used to ensure successful extraction even if the stego image is slightly distorted.
- **User-friendly Interface**: Simple GUI for easy use without requiring technical knowledge.
- **Quality Assessment**: Measures PSNR and SSIM to evaluate the quality of steganography.

## Notes

- The default RSA key size is 2048 bits (can be changed in the GUI).
- The embedding strength (alpha) controls the tradeoff between imperceptibility and robustness. Higher values make the hidden data more robust but potentially more visible.
- QR Code uses Version 1 with Low (L) error correction level, which can store up to 152 bits of data.
