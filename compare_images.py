"""
Script untuk menampilkan perbandingan gambar cover dan stego
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def compare_images(cover_path, stego_path):
    """
    Membuat perbandingan visual antara gambar cover dan gambar stego
    """
    # Membaca gambar
    cover_img = Image.open(cover_path)
    stego_img = Image.open(stego_path)
    
    # Menampilkan kedua gambar
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(np.array(cover_img))
    plt.title('Gambar Cover Asli')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(np.array(stego_img))
    plt.title('Gambar Stego (dengan QR tersembunyi)')
    plt.axis('off')
    
    # Menyimpan gambar perbandingan
    comparison_path = "comparison.png"
    plt.savefig(comparison_path, bbox_inches='tight')
    plt.close()
    
    print(f"Perbandingan gambar disimpan ke '{comparison_path}'")
    
    # Membuat analisis perbedaan piksel
    cover_array = np.array(cover_img)
    stego_array = np.array(stego_img)
    
    # Menghitung perbedaan absolut
    diff = np.abs(cover_array.astype(np.float32) - stego_array.astype(np.float32))
    
    # Menyimpan gambar diferensial
    plt.figure(figsize=(10, 8))
    plt.imshow(diff * 5)  # dikali 5 untuk memperjelas perbedaan
    plt.colorbar(label='Perbedaan Piksel * 5')
    plt.title('Perbedaan Piksel antara Gambar Cover dan Stego')
    plt.savefig('pixel_difference.png', bbox_inches='tight')
    plt.close()
    
    print(f"Visualisasi perbedaan piksel disimpan ke 'pixel_difference.png'")
    
    # Menganalisis perbedaan statistik
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    print(f"Perbedaan rata-rata: {mean_diff:.2f} per piksel")
    print(f"Perbedaan maksimum: {max_diff:.2f}")
    
    return True

if __name__ == "__main__":
    # Jalur gambar
    cover_path = "natural_cover.jpg"
    stego_path = "output/stego_image.png"
    
    # Bandingkan gambar
    compare_images(cover_path, stego_path)
