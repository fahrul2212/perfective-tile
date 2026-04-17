import os
import sys
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

# Masukkan root path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

def main():
    print("=== Step 3: Shadow Extraction on Segmented Floor ===")
    
    output_dir = os.path.join(ROOT, "tests", "segmentasi-perfect", "outputs")
    img_path = os.path.join(ROOT, "assets", "1.jpg")
    mask_path = os.path.join(output_dir, "mask_instance_0_refined.png")
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print("[-] File input tidak lengkap.")
        return

    # 1. Load Data
    img_orig = Image.open(img_path).convert("RGB")
    mask_refined = Image.open(mask_path).convert("L")
    
    # 2. Konversi ke Grayscale untuk analisis intensitas cahaya
    # Kita menggunakan Luminance (L) agar lebih akurat terhadap persepsi mata
    img_gray = img_orig.convert("L")
    
    # 3. Estimasi "Warna Dasar Lantai" (Lantai tanpa bayangan)
    # Kita ambil nilai rata-rata atau median dari area yang terang di dalam mask
    img_np = np.array(img_gray).astype(np.float32)
    mask_np = np.array(mask_refined).astype(np.float32) / 255.0
    
    # Ambil pixel yang hanya ada di dalam mask
    floor_pixels = img_np[mask_np > 0.5]
    
    # Ambil persentil ke-80 sebagai estimasi cahaya lantai tanpa bayangan (diffuse floor color)
    # Kita tidak ambil Max agar tidak terganggu oleh 'highlight' atau pantulan lampu
    base_floor_lightness = np.percentile(floor_pixels, 80)
    print(f"[*] Estimated Base Floor Lightness: {base_floor_lightness:.2f}")

    # 4. Kalkulasi Shadow Map (Dengan pembersihan noise tekstur lantai)
    # Selisih kegelapan antara lantai bersih dan kondisi saat ini
    shadow_map = base_floor_lightness - img_np
    
    # --- MEMBERSIHKAN TEKSTUR ---
    # Jika selisih kegelapan kecil (< 45), itu kemungkinan besar hanya tekstur kayu/nat, bukan bayangan asli.
    # Kita set pixel tersebut ke 0 (tidak ada bayangan).
    shadow_map[shadow_map < 45] = 0
    
    shadow_map = np.clip(shadow_map, 0, 255)
    
    # --- MEMPERKUAT BAYANGAN ---
    if shadow_map.max() > 0:
        # Gunakan normalisasi dan eksponen (gamma) untuk menekan abu-abu samar
        # dan menonjolkan area yang benar-benar gelap.
        shadow_map = (shadow_map / shadow_map.max()) ** 2.0 * 255.0
    
    # 5. Masking Shadow Map
    # Kita hanya butuh bayangan yang ada DI ATAS lantai
    shadow_map = shadow_map * mask_np
    
    # 6. Final Smoothing untuk Shadow Map
    # Bayangan biasanya bersifat soft (umbram/penumbra)
    shadow_img = Image.fromarray(shadow_map.astype(np.uint8), mode="L")
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=5))
    
    # 7. Simpan Hasil
    shadow_out_path = os.path.join(output_dir, "1_shadow_map.png")
    shadow_img.save(shadow_out_path)
    print(f"[+] Shadow Map berhasil disimpan ke: {shadow_out_path}")

    # 8. Visualisasi (Shadow Mask di atas putih)
    # Membuat visualisasi bayangan di atas background putih
    # Agar user bisa melihat bentuk bayangannya saja
    vis_shadow = ImageOps.invert(shadow_img)
    vis_path = os.path.join(output_dir, "1_shadow_visualization.jpg")
    vis_shadow.save(vis_path)
    print(f"[+] Visualisasi Shadow (Bayangan saja) disimpan ke: {vis_path}")
    
    print("=== Selesai ===")

if __name__ == "__main__":
    main()
