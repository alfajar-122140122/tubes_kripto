"""
RSA Encryption Module
This module handles the generation of RSA key pairs, encryption, and decryption of data.
"""

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
import os

class RSAEncryption:
    def __init__(self, key_size=2048):
        """
        Initialize RSA encryption with a specified key size (default: 2048 bits)
        """
        self.key_size = key_size
        self.public_key = None
        self.private_key = None

    def generate_keys(self):
        """
        Generate a new RSA key pair
        """
        # Generate a new RSA key pair
        key = RSA.generate(self.key_size)
        
        # Extract the private and public keys
        self.private_key = key
        self.public_key = key.publickey()
        
        return self.private_key, self.public_key
    
    def save_keys_to_file(self, private_key_file="private_key.pem", public_key_file="public_key.pem"):
        """
        Save the RSA keys to files
        """
        if self.private_key is None or self.public_key is None:
            raise ValueError("Keys have not been generated yet. Call generate_keys first.")
        
        # Save the private key
        with open(private_key_file, 'wb') as f:
            f.write(self.private_key.export_key('PEM'))
        
        # Save the public key
        with open(public_key_file, 'wb') as f:
            f.write(self.public_key.export_key('PEM'))
    
    def load_keys_from_file(self, private_key_file="private_key.pem", public_key_file="public_key.pem"):
        """
        Load the RSA keys from files
        """
        # Load the private key if the file exists
        if os.path.exists(private_key_file):
            with open(private_key_file, 'rb') as f:
                self.private_key = RSA.import_key(f.read())
        else:
            self.private_key = None
        
        # Load the public key if the file exists
        if os.path.exists(public_key_file):
            with open(public_key_file, 'rb') as f:
                self.public_key = RSA.import_key(f.read())
        else:
            self.public_key = None
        
        return self.private_key, self.public_key
    
    def encrypt(self, data):
        """
        Encrypt data using the public key
        
        Args:
            data (str): The data to encrypt
            
        Returns:
            str: Base64 encoded encrypted data
        """
        if self.public_key is None:
            raise ValueError("Public key not available. Generate or load keys first.")
        
        # Convert string to bytes if needed
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Create a cipher using the public key
        cipher = PKCS1_OAEP.new(self.public_key)
        
        # Encrypt the data
        encrypted_data = cipher.encrypt(data)
        
        # Return base64 encoded encrypted data
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt(self, encrypted_data):
        """
        Decrypt data using the private key
        
        Args:
            encrypted_data (str): Base64 encoded encrypted data
            
        Returns:
            str: Decrypted data
        """
        if self.private_key is None:
            raise ValueError("Private key not available. Generate or load keys first.")
        
        # Decode base64 data
        if isinstance(encrypted_data, str):
            encrypted_data = base64.b64decode(encrypted_data)
        
        # Create a cipher using the private key
        cipher = PKCS1_OAEP.new(self.private_key)
        
        # Decrypt the data
        try:
            decrypted_data = cipher.decrypt(encrypted_data)
            return decrypted_data.decode('utf-8')
        except Exception as e:
            print(f"Error during decryption: {e}")
            return None
