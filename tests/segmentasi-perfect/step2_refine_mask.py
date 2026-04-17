import os
import sys
import numpy as np
from PIL import Image, ImageFilter

# Masukkan root path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

def main():
    print("=== Step 2: Mask Refinement & Smoothing ===")
    
    input_dir = os.path.join(ROOT, "tests", "segmentasi-perfect", "outputs")
    mask_file = os.path.join(input_dir, "mask_instance_0.png")
    output_file = os.path.join(input_dir, "mask_instance_0_refined.png")
    
    if not os.path.exists(mask_file):
        print(f"[-] File mask tidak ditemukan: {mask_file}")
        return

    # 1. Load Mask Asli
    print(f"[*] Loading mask: {mask_file}")
    mask = Image.open(mask_file).convert("L")
    
    # 2. Hapus Noise Aggressive (Median Filter)
    # Untuk gambar 4K, kita butuh size yang besar (misal 25) agar bintik-bintik noise hilang
    print("[*] Menghapus noise dengan Median Filter (Aggressive 25px)...")
    refined = mask.filter(ImageFilter.MedianFilter(size=25)) 
    
    # 3. Smoothing & Gap Filling (Blur + Threshold)
    # Radius besar (15px) untuk menjamin kelengkungan yang mulus dan menghilangkan gerigi
    print("[*] Menghaluskan tepian dan menutup lubang (Gaussian 15px)...")
    refined = refined.filter(ImageFilter.GaussianBlur(radius=15))
    
    # Thresholding
    # Kita gunakan 128 sebagai standar untuk menjaga volume area tetap stabil
    refined = refined.point(lambda p: 255 if p > 128 else 0)
    
    # 4. Save Hasil
    refined.save(output_file)
    print(f"[+] Mask yang sudah halus disimpan ke: {output_file}")
    
    # 5. Visualisasi Perbandingan (Opsional)
    # Membuat gambar berdampingan untuk melihat perbedaan
    comparison = Image.new("L", (mask.width * 2, mask.height))
    comparison.paste(mask, (0, 0))
    comparison.paste(refined, (mask.width, 0))
    
    comp_path = os.path.join(input_dir, "refinement_comparison.png")
    comparison.save(comp_path)
    print(f"[+] File perbandingan (Before vs After) disimpan ke: {comp_path}")
    
    print("=== Selesai ===")

if __name__ == "__main__":
    main()
