"""
Script untuk mengunduh gambar cover yang lebih alami
"""

import requests
from PIL import Image
import io
import os

def download_natural_cover_image(save_path="natural_cover.jpg"):
    """
    Mengunduh gambar alami dari picsum.photos untuk digunakan sebagai cover image
    """
    try:
        # URL untuk gambar dengan resolusi 800x600 dari picsum.photos
        url = "https://picsum.photos/800/600"
        
        print(f"Mengunduh gambar dari {url}...")
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            # Membuka gambar sebagai objek PIL Image
            img = Image.open(io.BytesIO(response.content))
            
            # Menyimpan gambar
            img.save(save_path)
            print(f"Gambar berhasil diunduh dan disimpan ke '{save_path}'")
            
            # Menampilkan informasi gambar
            width, height = img.size
            print(f"Ukuran gambar: {width}x{height} piksel")
            print(f"Format gambar: {img.format}")
            
            return save_path
        else:
            print(f"Gagal mengunduh gambar. Status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        return None

if __name__ == "__main__":
    # Unduh dan simpan gambar
    download_natural_cover_image()
