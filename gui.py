import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
from PIL import Image, ImageTk
import cv2
import numpy as np

from rsa_qrcode import load_rsa_keys, generate_rsa_keys
from main import encrypt_and_embed, extract_and_decrypt

class SteganographyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RSA + QR Code + DCT Steganography")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variables
        self.cover_image_path = tk.StringVar()
        self.stego_image_path = tk.StringVar()
        self.secret_message = tk.StringVar()
        self.alpha_value = tk.DoubleVar(value=0.1)
        self.processing = False
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.encrypt_tab = ttk.Frame(self.notebook)
        self.decrypt_tab = ttk.Frame(self.notebook)
        self.keys_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.encrypt_tab, text="Hide Message")
        self.notebook.add(self.decrypt_tab, text="Extract Message")
        self.notebook.add(self.keys_tab, text="RSA Keys")
        
        # Setup tabs
        self.setup_encrypt_tab()
        self.setup_decrypt_tab()
        self.setup_keys_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Check for existing RSA keys
        try:
            if not os.path.exists('public_key.pem') or not os.path.exists('private_key.pem'):
                self.status_var.set("Generating new RSA keys...")
                self.root.update()
                generate_rsa_keys()
                self.status_var.set("RSA keys generated successfully")
            else:
                self.status_var.set("Loaded existing RSA keys")
        except Exception as e:
            self.status_var.set(f"Error with RSA keys: {e}")
    
    def setup_encrypt_tab(self):
        # Left panel - Controls
        left_panel = ttk.Frame(self.encrypt_tab)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Message input
        ttk.Label(left_panel, text="Secret Message:").pack(anchor=tk.W, pady=(0, 5))
        message_entry = ttk.Entry(left_panel, textvariable=self.secret_message, width=40)
        message_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Cover image selection
        ttk.Label(left_panel, text="Cover Image:").pack(anchor=tk.W, pady=(0, 5))
        cover_frame = ttk.Frame(left_panel)
        cover_frame.pack(fill=tk.X, pady=(0, 10))
        
        cover_entry = ttk.Entry(cover_frame, textvariable=self.cover_image_path)
        cover_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        cover_button = ttk.Button(cover_frame, text="Browse...", command=self.browse_cover_image)
        cover_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Alpha strength
        ttk.Label(left_panel, text=f"Embedding Strength (0.01-0.2):").pack(anchor=tk.W, pady=(0, 5))
        alpha_scale = ttk.Scale(left_panel, from_=0.01, to=0.2, variable=self.alpha_value, orient=tk.HORIZONTAL)
        alpha_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Value display for alpha
        alpha_frame = ttk.Frame(left_panel)
        alpha_frame.pack(fill=tk.X, pady=(0, 10))
        
        def update_alpha_label(*args):
            alpha_value_label.config(text=f"{self.alpha_value.get():.2f}")
        
        ttk.Label(alpha_frame, text="Value:").pack(side=tk.LEFT)
        alpha_value_label = ttk.Label(alpha_frame, text=f"{self.alpha_value.get():.2f}")
        alpha_value_label.pack(side=tk.LEFT)
        
        self.alpha_value.trace_add("write", update_alpha_label)
        
        # Encrypt button
        encrypt_button = ttk.Button(left_panel, text="Hide Message", command=self.encrypt_message)
        encrypt_button.pack(fill=tk.X, pady=10)
        
        # Progress bar
        self.encrypt_progress = ttk.Progressbar(left_panel, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.encrypt_progress.pack(fill=tk.X, pady=(10, 0))
        
        # Right panel - Image preview
        right_panel = ttk.Frame(self.encrypt_tab)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(right_panel, text="Image Preview:").pack(anchor=tk.W)
        
        self.encrypt_preview_frame = ttk.LabelFrame(right_panel, text="Cover Image")
        self.encrypt_preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.encrypt_preview_label = ttk.Label(self.encrypt_preview_frame)
        self.encrypt_preview_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def setup_decrypt_tab(self):
        # Left panel - Controls
        left_panel = ttk.Frame(self.decrypt_tab)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Stego image selection
        ttk.Label(left_panel, text="Stego Image:").pack(anchor=tk.W, pady=(0, 5))
        stego_frame = ttk.Frame(left_panel)
        stego_frame.pack(fill=tk.X, pady=(0, 10))
        
        stego_entry = ttk.Entry(stego_frame, textvariable=self.stego_image_path)
        stego_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        stego_button = ttk.Button(stego_frame, text="Browse...", command=self.browse_stego_image)
        stego_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Decrypt button
        decrypt_button = ttk.Button(left_panel, text="Extract Message", command=self.decrypt_message)
        decrypt_button.pack(fill=tk.X, pady=10)
        
        # Progress bar
        self.decrypt_progress = ttk.Progressbar(left_panel, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.decrypt_progress.pack(fill=tk.X, pady=(10, 0))
        
        # Extracted message
        ttk.Label(left_panel, text="Extracted Message:").pack(anchor=tk.W, pady=(10, 5))
        self.extracted_message = tk.Text(left_panel, height=5, width=40, wrap=tk.WORD)
        self.extracted_message.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Right panel - Image preview
        right_panel = ttk.Frame(self.decrypt_tab)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(right_panel, text="Image Preview:").pack(anchor=tk.W)
        
        self.decrypt_preview_frame = ttk.LabelFrame(right_panel, text="Stego Image")
        self.decrypt_preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.decrypt_preview_label = ttk.Label(self.decrypt_preview_frame)
        self.decrypt_preview_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # QR code frame
        self.qr_preview_frame = ttk.LabelFrame(right_panel, text="Extracted QR Code")
        self.qr_preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.qr_preview_label = ttk.Label(self.qr_preview_frame)
        self.qr_preview_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_keys_tab(self):
        # RSA key info and regeneration
        frame = ttk.Frame(self.keys_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="RSA Key Information", font=("TkDefaultFont", 12, "bold")).pack(pady=(0, 10))
        
        info_text = (
            "The system uses RSA encryption for securing your message before embedding it into the QR code. "
            "The RSA algorithm requires a pair of keys:\n\n"
            "- Public Key: Used for encrypting your message\n"
            "- Private Key: Used for decrypting your message\n\n"
            "The keys are automatically generated and stored in the application folder."
        )
        
        info_label = ttk.Label(frame, text=info_text, wraplength=600, justify=tk.LEFT)
        info_label.pack(fill=tk.X, pady=10)
        
        # Key status
        key_frame = ttk.Frame(frame)
        key_frame.pack(fill=tk.X, pady=10)
        
        self.key_status = ttk.Label(key_frame, text="")
        self.key_status.pack(side=tk.LEFT)
        
        self.check_key_status()
        
        # Regenerate button
        regen_frame = ttk.Frame(frame)
        regen_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(regen_frame, text="Generate new RSA keys:").pack(side=tk.LEFT)
        
        key_size_var = tk.IntVar(value=2048)
        key_size_combo = ttk.Combobox(regen_frame, textvariable=key_size_var, width=10)
        key_size_combo['values'] = (1024, 2048, 3072, 4096)
        key_size_combo.pack(side=tk.LEFT, padx=5)
        
        regen_button = ttk.Button(regen_frame, text="Generate New Keys", 
                                 command=lambda: self.regenerate_keys(key_size_var.get()))
        regen_button.pack(side=tk.LEFT, padx=5)
    
    def check_key_status(self):
        if os.path.exists('public_key.pem') and os.path.exists('private_key.pem'):
            self.key_status.config(text="Status: RSA keys are present and ready to use", foreground="green")
        else:
            self.key_status.config(text="Status: RSA keys not found", foreground="red")
    
    def regenerate_keys(self, key_size):
        if self.processing:
            return
        
        if messagebox.askyesno("Confirm Key Regeneration", 
                             "Generating new keys will invalidate any previously encrypted messages. Continue?"):
            self.processing = True
            self.status_var.set("Generating new RSA keys...")
            
            def key_gen_thread():
                try:
                    generate_rsa_keys(key_size)
                    self.status_var.set(f"RSA keys generated successfully (size: {key_size})")
                    self.check_key_status()
                except Exception as e:
                    self.status_var.set(f"Error generating keys: {e}")
                finally:
                    self.processing = False
            
            threading.Thread(target=key_gen_thread).start()
    
    def browse_cover_image(self):
        filepath = filedialog.askopenfilename(
            title="Select Cover Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if filepath:
            self.cover_image_path.set(filepath)
            self.load_preview_image(filepath, self.encrypt_preview_label)
    
    def browse_stego_image(self):
        filepath = filedialog.askopenfilename(
            title="Select Stego Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if filepath:
            self.stego_image_path.set(filepath)
            self.load_preview_image(filepath, self.decrypt_preview_label)
            # Clear QR preview and extracted message
            self.qr_preview_label.config(image="")
            self.extracted_message.delete(1.0, tk.END)
    
    def load_preview_image(self, filepath, label_widget):
        try:
            # Open image and resize for preview
            image = Image.open(filepath)
            image.thumbnail((300, 300))  # Resize for display
            photo = ImageTk.PhotoImage(image)
            
            # Update preview
            label_widget.config(image=photo)
            label_widget.image = photo  # Keep reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def encrypt_message(self):
        if self.processing:
            return
        
        # Validate inputs
        message = self.secret_message.get()
        cover_path = self.cover_image_path.get()
        alpha = self.alpha_value.get()
        
        if not message:
            messagebox.showerror("Error", "Please enter a message")
            return
        
        if not cover_path:
            messagebox.showerror("Error", "Please select a cover image")
            return
        
        if not 0.01 <= alpha <= 0.2:
            messagebox.showerror("Error", "Embedding strength must be between 0.01 and 0.2")
            return
        
        # Start processing
        self.processing = True
        self.encrypt_progress.start()
        self.status_var.set("Processing...")
        
        def encrypt_thread():
            try:
                # Create output filename
                base_path, ext = os.path.splitext(cover_path)
                output_path = f"{base_path}_stego.png"
                
                # Run encryption
                stego_path = encrypt_and_embed(message, cover_path, output_path, alpha)
                
                # Update GUI in main thread
                self.root.after(0, lambda: self.encryption_complete(stego_path))
            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Encryption failed: {e}"))
            finally:
                self.processing = False
                self.root.after(0, self.encrypt_progress.stop)
        
        # Run in background thread
        threading.Thread(target=encrypt_thread).start()
    
    def encryption_complete(self, stego_path):
        self.status_var.set(f"Message hidden successfully. Saved to: {os.path.basename(stego_path)}")
        messagebox.showinfo("Success", f"Message hidden successfully.\nSaved to: {stego_path}")
        
        # Show preview of stego image
        self.load_preview_image(stego_path, self.encrypt_preview_label)
        
        # Also show QR code if available
        qr_path = "encrypted_qr.png"
        if os.path.exists(qr_path):
            # Switch to decrypt tab and load stego image there too
            self.notebook.select(1)  # Switch to decrypt tab
            self.stego_image_path.set(stego_path)
            self.load_preview_image(stego_path, self.decrypt_preview_label)
            self.load_preview_image(qr_path, self.qr_preview_label)
    
    def decrypt_message(self):
        if self.processing:
            return
        
        # Validate input
        stego_path = self.stego_image_path.get()
        
        if not stego_path:
            messagebox.showerror("Error", "Please select a stego image")
            return
        
        # Start processing
        self.processing = True
        self.decrypt_progress.start()
        self.status_var.set("Extracting message...")
        
        # Clear previous result
        self.extracted_message.delete(1.0, tk.END)
        
        def decrypt_thread():
            try:
                # Run extraction and decryption
                decrypted_message = extract_and_decrypt(stego_path)
                
                # Show QR code if extracted
                extracted_qr_path = "extracted_qr.png"
                
                # Update GUI in main thread
                self.root.after(0, lambda: self.decryption_complete(decrypted_message, extracted_qr_path))
            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Extraction failed: {e}"))
            finally:
                self.processing = False
                self.root.after(0, self.decrypt_progress.stop)
        
        # Run in background thread
        threading.Thread(target=decrypt_thread).start()
    
    def decryption_complete(self, message, qr_path):
        if message:
            self.status_var.set("Message extracted successfully")
            self.extracted_message.insert(1.0, message)
        else:
            self.status_var.set("Failed to extract message")
            self.extracted_message.insert(1.0, "Failed to extract message")
        
        # Show extracted QR code if available
        if os.path.exists(qr_path):
            self.load_preview_image(qr_path, self.qr_preview_label)
    
    def show_error(self, message):
        self.status_var.set(message)
        messagebox.showerror("Error", message)

if __name__ == "__main__":
    root = tk.Tk()
    app = SteganographyApp(root)
    root.mainloop()
