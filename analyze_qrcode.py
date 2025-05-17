"""
Script untuk menampilkan QR code yang diekstrak
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def view_extracted_qr_code():
    """
    Menampilkan dan menganalisis QR code yang diekstrak
    """
    # Baca QR code yang diekstrak
    qr_path = "output/extracted_qrcode.png"
    img = cv2.imread(qr_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Tidak dapat membaca gambar: {qr_path}")
        return
    
    # Tampilkan QR code asli
    plt.figure(figsize=(12, 12))
    
    plt.subplot(221)
    plt.imshow(img, cmap='gray')
    plt.title("QR Code yang Diekstrak (Original)")
    plt.axis('off')
    
    # Tingkatkan kontras dengan equalize histogram
    equ_img = cv2.equalizeHist(img)
    plt.subplot(222)
    plt.imshow(equ_img, cmap='gray')
    plt.title("QR Code dengan Histogram Equalization")
    plt.axis('off')
    
    # Binerisasi dengan threshold adaptif
    thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    plt.subplot(223)
    plt.imshow(thresh1, cmap='gray')
    plt.title("QR Code dengan Adaptive Threshold")
    plt.axis('off')
    
    # Binerisasi dengan threshold Otsu
    _, thresh2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.subplot(224)
    plt.imshow(thresh2, cmap='gray')
    plt.title("QR Code dengan Otsu Threshold")
    plt.axis('off')
    
    # Simpan gambar perbandingan
    plt.tight_layout()
    plt.savefig("qrcode_analysis.png")
    plt.close()
    
    print("Analisis QR code disimpan ke 'qrcode_analysis.png'")
    
    # Coba baca QR code dari berbagai versi yang diproses
    process_qr_code(img, "Original")
    process_qr_code(equ_img, "Equalized")
    process_qr_code(thresh1, "Adaptive")
    process_qr_code(thresh2, "Otsu")
    
    # Simpan versi yang diproses
    cv2.imwrite("output/qr_enhanced_equ.png", equ_img)
    cv2.imwrite("output/qr_enhanced_adaptive.png", thresh1)
    cv2.imwrite("output/qr_enhanced_otsu.png", thresh2)
    
    print("Versi QR code yang ditingkatkan kualitasnya disimpan di folder output/")

def process_qr_code(img, desc):
    """
    Mencoba membaca QR code dan melaporkan hasilnya
    """
    try:
        # Coba dengan detector OpenCV
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)
        
        if data:
            print(f"QR Code ({desc}) berhasil dibaca dengan OpenCV: {data}")
            return True
        
        # Coba dengan pyzbar jika tersedia
        try:
            from pyzbar.pyzbar import decode
            decoded_objects = decode(img)
            if decoded_objects:
                data = decoded_objects[0].data.decode('utf-8')
                print(f"QR Code ({desc}) berhasil dibaca dengan pyzbar: {data}")
                return True
        except Exception as e:
            print(f"Error saat mencoba menggunakan pyzbar untuk QR Code ({desc}): {e}")
        
        print(f"QR Code ({desc}) tidak dapat dibaca")
        return False
        
    except Exception as e:
        print(f"Error saat memproses QR Code ({desc}): {e}")
        return False

if __name__ == "__main__":
    view_extracted_qr_code()
