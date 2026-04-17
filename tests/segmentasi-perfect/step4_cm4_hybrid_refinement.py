import os
import sys
import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Masukkan root path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

def main():
    print("=== Step 4: Hybrid Refinement (Inspired by CM4 Matting) ===")
    
    output_dir = os.path.join(ROOT, "tests", "segmentasi-perfect", "outputs")
    img_path = os.path.join(ROOT, "assets", "1.jpg")
    mask_path = os.path.join(output_dir, "mask_instance_0_refined.png")
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print("[-] File input tidak lengkap.")
        return

    # 1. Load Data
    img_orig = Image.open(img_path).convert("RGB")
    mask_bin = Image.open(mask_path).convert("L")
    
    # 2. Simulasi TRIMAP (Kunci dari Matting seperti ViTMatte dalam CM4)
    # Trimap membagi area menjadi 3: Pasti Depan, Pasti Belakang, dan Unknown (Tepian)
    print("[*] Generating Trimap-like Soft Boundary...")
    
    mask_np = np.array(mask_bin).astype(np.float32) / 255.0
    
    # Erosi (Mengecilkan mask untuk mendapatkan area 'Pasti Lantai')
    # Kita gunakan filter Max/Min untuk simulasi morfologi di PIL
    inner_mask = mask_bin.filter(ImageFilter.MinFilter(size=15)) # Erosi
    outer_mask = mask_bin.filter(ImageFilter.MaxFilter(size=15)) # Dilasi
    
    # Area Unknown (Tepian/Boundary) adalah selisih antara Dilasi dan Erosi
    boundary_zone = np.array(outer_mask).astype(np.float32) - np.array(inner_mask).astype(np.float32)
    boundary_zone = np.clip(boundary_zone, 0, 255).astype(np.uint8)
    
    # 3. Alpha Matting Refinement
    # Di area boundary, kita buat gradasi halus (soft edge)
    # Ini mensimulasikan presisi tinggi agar tidak ada aliasing (gerigi)
    soft_mask = mask_bin.filter(ImageFilter.GaussianBlur(radius=8))
    
    # 4. Gabungkan: Pasti Lantai (1.0) + Boundary (Soft)
    # Logika: Jika di inner_mask => 255, Jika di boundary => gunakan soft_mask
    final_np = np.array(inner_mask).astype(np.float32)
    final_np[boundary_zone > 0] = np.array(soft_mask).astype(np.float32)[boundary_zone > 0]
    
    # 5. Save Perfect Alpha Mask
    # Hasil ini bukan lagi hitam-putih keras, tapi memiliki nilai alpha (0-255) di tepian
    final_mask = Image.fromarray(final_np.astype(np.uint8), mode="L")
    output_path = os.path.join(output_dir, "1_perfect_alpha_mask.png")
    final_mask.save(output_path)
    print(f"[+] Perfect Alpha Mask (Matting style) disimpan ke: {output_path}")

    # 6. Visualisasi Hasil Akhir (Composite)
    # Coba tempelkan warna merah di area mask untuk melihat kehalusan tepian
    red_layer = Image.new("RGB", img_orig.size, (255, 0, 0))
    # Gunakan alpha mask kita untuk blending
    composite = Image.composite(red_layer, img_orig, final_mask)
    
    comp_path = os.path.join(output_dir, "1_cm4_style_composite.jpg")
    composite.save(comp_path)
    print(f"[+] Visualisasi Composite CM4-Style disimpan ke: {comp_path}")
    
    print("=== Selesai ===")

if __name__ == "__main__":
    main()
