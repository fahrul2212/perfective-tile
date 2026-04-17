import cv2
import numpy as np
import os

def extract_shadow(image_path, output_path):
    # 1. Baca gambar
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Gambar {image_path} tidak ditemukan!")
        return

    # 2. Konversi BGR ke ruang warna LAB
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Ambil channel L (Lightness [0])
    l_channel = lab_img[:, :, 0]
    
    # 3. Metode Thresholding pada L channel
    # Invert L channel agar area bayangan (gelap/nilai L rendah) menjadi terang
    l_channel_inv = cv2.bitwise_not(l_channel) 
    
    # Menggunakan Otsu's binarization secara otomatis mencari threshold terbaik.
    # Output berupa hitam putih (0-255), 255 untuk shadow
    _, shadow_mask = cv2.threshold(l_channel_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 4. Operasi Morfologi (Morphology) untuk membersihkan bintik noise yang tidak dibutuhkan
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_mask_clean = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    
    # Memperhalus outline pixel dari bayangannya
    shadow_mask_clean = cv2.GaussianBlur(shadow_mask_clean, (5, 5), 0)

    # 5. Simpan Hasilnya (Hitam Putih 0-255)
    cv2.imwrite(output_path, shadow_mask_clean)
    print(f"Berhasil! Shadow Mask (0-255) disimpan di: {output_path}")

if __name__ == "__main__":
    assets_dir = r"C:\Project\simpel\assets"
    output_dir = r"C:\Project\simpel\tests\shadow"
    
    # Pastikan direktori output ada (walau sudah pasti ada)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Mengekstraksi bayangan 1.jpg...")
    extract_shadow(
        os.path.join(assets_dir, "1.jpg"), 
        os.path.join(output_dir, "mask_1.png")
    )
    
    print("Mengekstraksi bayangan 2.jpg...")
    extract_shadow(
        os.path.join(assets_dir, "2.jpg"), 
        os.path.join(output_dir, "mask_2.png")
    )
