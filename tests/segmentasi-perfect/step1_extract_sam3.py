import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from huggingface_hub import login
import time

# Masukkan root path agar bisa import dari core atau baca assets
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

def main():
    print("=== Step 1: Ekstraksi Mask Baseline SAM3 ===")
    
    # 1. Pastikan output folder ada
    output_dir = os.path.join(ROOT, "tests", "segmentasi-perfect", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Setup & Load Model
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN tidak ditemukan di environment. Jika repo privat, login mungkin gagal.")
    else:
        login(token=hf_token)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Pilih dtype terbaik yang didukung hardware
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    print(f"[*] Loading SAM3 Model (facebook/sam3) on {device} with dtype={dtype}...")
    start_load = time.time()
    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
    model = Sam3Model.from_pretrained(
        "facebook/sam3", 
        token=hf_token, 
        torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"[+] Model loaded in {time.time() - start_load:.2f} detik")

    # 3. Load Image
    img_path = os.path.join(ROOT, "assets", "1.jpg")
    if not os.path.exists(img_path):
        print(f"[-] File tidak ditemukan: {img_path}")
        return

    print(f"[*] Processing image: {img_path}")
    image = Image.open(img_path).convert("RGB")
    original_size = image.size # (W, H)
    print(f"    Original Size: {original_size}")

    # 4. Preprocess
    # Resize untuk mengurangi beban VRAM (maks 1024), tapi output tetap di-scale ke resolusi asli
    MAX_SIZE = 1024
    w, h = image.size
    if max(w, h) > MAX_SIZE:
        scale = MAX_SIZE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = image.resize((new_w, new_h), Image.LANCZOS)
        print(f"    Resized for inference: {(new_w, new_h)}")
    else:
        image_resized = image

    text_prompt = "floor"
    print(f"[*] Text Prompt: '{text_prompt}'")
    
    inputs = processor(images=image_resized, text=text_prompt, return_tensors="pt").to(device)

    # 5. Inference
    print("[*] Running inference...")
    start_inf = time.time()
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device, dtype=dtype):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
    print(f"[+] Inference done in {time.time() - start_inf:.2f} detik")

    # 6. Post-processing untuk mendapatkan mask presisi tinggi
    print("[*] Post-processing mask...")
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.1,
        mask_threshold=0.5,
        target_sizes=[original_size[::-1]] # (H, W) untuk PIL target sizes
    )[0]

    masks = results.get("masks", [])
    if len(masks) == 0:
        print("[-] Tidak ada mask 'floor' yang terdeteksi.")
        return
    
    print(f"[+] Terdeteksi {len(masks)} instance mask.")
    
    # 7. Simpan mask individual dan siapkan overlay multi-warna
    overlay = image.copy()
    # Daftar warna BGR/RGB sederhana untuk membedakan instance
    colors = [
        (0, 255, 0),   # Hijau
        (255, 0, 0),   # Merah
        (0, 0, 255),   # Biru
        (255, 255, 0), # Cyan
        (255, 0, 255), # Magenta
        (0, 255, 255), # Kuning
    ]

    for i, mask_tensor in enumerate(masks):
        # Konversi ke numpy
        mask_np = (mask_tensor.to(torch.uint8) * 255).cpu().numpy()
        
        # Simpan mask individual hitam-putih
        mask_img = Image.fromarray(mask_np, mode="L")
        inst_path = os.path.join(output_dir, f"mask_instance_{i}.png")
        mask_img.save(inst_path)
        print(f"    [>] Instance {i} disimpan ke: {inst_path}")

        # Tambahkan ke overlay dengan warna berbeda
        color = colors[i % len(colors)]
        colored_layer = Image.new("RGB", overlay.size, color)
        
        # Alpha mask untuk blending (transparansi 50% untuk area yang terdeteksi)
        alpha_mask = Image.fromarray((mask_np * 0.5).astype(np.uint8), mode="L")
        overlay.paste(colored_layer, mask=alpha_mask)

    # Simpan hasil visualisasi gabungan
    out_overlay_path = os.path.join(output_dir, "1_sam3_overlay.jpg")
    overlay.save(out_overlay_path, format="JPEG")
    print(f"[+] Visualisasi gabungan (Multi-Color Overlay) disimpan ke: {out_overlay_path}")
    print("=== Selesai ===")

if __name__ == "__main__":
    main()
