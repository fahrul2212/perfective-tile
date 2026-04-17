# Riset Ekstraksi Bayangan (Shadow Extraction) dari Citra Tunggal

Dokumen ini berisi riset, dasar keilmuan, dan sampel implementasi untuk mengekstraksi bayangan dari sebuah gambar tunggal (seperti `1.jpg` dan `2.jpg` di folder `assets`), lalu merubahnya menjadi format mask hitam putih (0-255), di mana 255 merepresentasikan area bayangan dan 0 adalah area non-bayangan.

---

## 1. Dasar Keilmuan (Theoretical Background)

Mengekstraksi bayangan dari satu gambar statis (single-image shadow detection) merupakan masalah klasik dalam **Computer Vision**. Tantangan utamanya adalah membedakan apakah sebuah piksel berwarna gelap karena memang warnanya gelap (objek berwarna hitam) atau karena objek tersebut tertutup bayangan (kurangnya iluminasi atau pencahayaan).

Berdasarkan berbagai literatur pengolahan citra digital, metode yang paling efektif tanpa menggunakan Machine Learning adalah dengan **memisahkan informasi warna (chrominance) dan informasi pencahayaan (luminance/lightness)**. 

### A. Mengapa Skema Warna RGB Tidak Cocok?
Dalam format RGB (Red, Green, Blue), informasi warna dan pencahayaan bercampur di ketiga channel. Jika gambar sedikit gelap, ketiga nilai R, G, dan B akan turun secara bersamaan. Hal ini mempersulit pembuatan formula matematis untuk mengisolasi "sedikit cahaya" tanpa merusak warna asli.

### B. Solusi: Ruang Warna LAB dan HSV
Untuk mendeteksi bayangan, kita harus mengkonversi citra dari BGR (standar OpenCV) ke ruang warna yang merepresentasikan cahaya secara independen:

1. **LAB (CIELAB) Color Space:** 
   - **L (Lightness):** Menyimpan nilai terang/gelap (0-255 di OpenCV).
   - **A (Green-Red) dan B (Blue-Yellow):** Menyimpan informasi warna (chrominance).
   - **Fakta:** Di dalam ruang warna LAB, area bayangan memiliki ciri khas penurunan yang sangat drastis pada nilai channel **L**. 

2. **HSV Color Space:** 
   - **H (Hue), S (Saturation), V (Value):** Bayangan biasanya menurunkan **V** dan menaikkan **S**. Beberapa rasio seperi $(S - V) / (S + V)$ sering dipakai pada metode deteksi bayangan tingkat lanjut.

---

## 2. Pendekatan Ekstraksi (Metodologi)

Berdasarkan riset, kita akan menggunakan pendekatan ruang warna **LAB** karena lebih stabil di bermacam-macam gambar produk dan lantai.

**Langkah-langkah Ekstraksinya:**
1. **Grayscale / Channel Isolation:** Konversi gambar asli (BGR) ke LAB. Ambil hanya channel **L**.
2. **Thresholding (Ambang Batas):** Ekstrak piksel-piksel yang nilai **L** -nya berada di bawah batas intensitas tertentu (artinya sangat gelap). Kita bisa menggunakan metode *Otsu's Binarization* yang dapat mencari batas *threshold* paling optimal secara otomatis.
3. **Inversion:** Area bayangan yang aslinya gelap (nilai mendekati 0), dibalik (invert) menjadi putih murni (255), sedangkan background menjadi hitam (0).
4. **Morphological Operations:** Membersihkan *noise* piksel (bercak-bercak kecil) dengan operasi `cv2.morphologyEx` agar bayangan terlihat *smooth* dan utuh.

---

## 3. Sampel Kode Python (Eksekusi 0-255)

Berikut adalah kode Python murni (`sample_shadow.py`) yang dapat berjalan untuk file `1.jpg` dan `2.jpg`.

```python
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
    # Bayangan memiliki nilai lightness yang rendah. Parameter 100 bisa disesuaikan
    # berdasarkan kondisi pencahayaan. Area gelap (L <= threshold) menjadi 255 (putih),
    # dan area terang (L > threshold) menjadi 0 (hitam).
    
    # Cara 1: Global Thresholding manual
    # _, shadow_mask = cv2.threshold(l_channel, 90, 255, cv2.THRESH_BINARY_INV)
    
    # Cara 2: Otsu's Thresholding untuk pencarian threshold otomatis dari histogram 
    # (Sangat baik jika warna latar membedakan dirinya dari objek gelap)
    # L kita invert dulu agar area gelap menjadi terang sebelum Otsu
    l_channel_inv = cv2.bitwise_not(l_channel) 
    _, shadow_mask = cv2.threshold(l_channel_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 4. Operasi Morfologi (Morphology) untuk membersihkan detail bintik noise
    # Menggunakan Morphological Open (Erosi lalu Dilasi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_mask_clean = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    
    # Opsional: Jika kita butuh memperhalus (blur border)
    shadow_mask_clean = cv2.GaussianBlur(shadow_mask_clean, (5, 5), 0)

    # 5. Simpan Hasilnya (Hitam Putih 0-255)
    cv2.imwrite(output_path, shadow_mask_clean)
    print(f"Shadow Mask (0-255) berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    assets_dir = r"C:\Project\simpel\assets"
    
    # Jalankan untuk 1.jpg
    extract_shadow(
        os.path.join(assets_dir, "1.jpg"), 
        r"C:\Project\simpel\tests\shadow\mask_1.png"
    )
    
    # Jalankan untuk 2.jpg
    extract_shadow(
        os.path.join(assets_dir, "2.jpg"), 
        r"C:\Project\simpel\tests\shadow\mask_2.png"
    )
```

## Kesimpulan
Pendekatan **LAB Color Space** dipadukan dengan nilai konversi Lightness (L) dan ambang batas (Thresholding) adalah metode komputasi terbaik dan tercepat (tanpa AI) untuk mengisolasi bayangan dari `0-255` yang dapat dieksport menjadi *Alpha Mask*. Untuk integrasi selanjutnya, teknik Morphological (Dilation/Erosion) sangat krusial agar sisa potongannya tidak terputus-putus.
