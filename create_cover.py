from PIL import Image
import numpy as np

def create_cover_image(output_path="cover_image.jpg", width=512, height=512):
    """
    Membuat gambar cover sederhana untuk steganografi
    """
    # Buat gambar dengan pattern
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Buat gradasi warna
    for y in range(height):
        for x in range(width):
            r = int(255 * (0.5 + 0.5 * np.sin(x/30)))
            g = int(255 * (0.5 + 0.5 * np.sin(y/30)))
            b = int(255 * (0.5 + 0.5 * np.sin((x+y)/60)))
            img[y, x] = [r, g, b]
    
    # Tambahkan pola kotak-kotak
    box_size = 32
    for y in range(0, height, box_size):
        for x in range(0, width, box_size):
            if (x//box_size + y//box_size) % 2 == 0:
                y_end = min(y + box_size, height)
                x_end = min(x + box_size, width)
                img[y:y_end, x:x_end] = img[y:y_end, x:x_end] * 0.8  # Sedikit lebih gelap
    
    # Konversi ke PIL Image dan simpan
    pil_img = Image.fromarray(img)
    pil_img.save(output_path)
    print(f"Gambar cover berhasil dibuat di: {output_path}")
    return output_path

if __name__ == "__main__":
    create_cover_image()
