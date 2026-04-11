import torch
import cv2
import numpy as np
import time
from pathlib import Path
from core.model import ST_RoomNet
import torch.nn.functional as F

# Configuration
ASSETS_DIR = Path("assets")
OUTPUT_DIR = Path("api_outputs")
WEIGHT_PATH = "weights/persfective.pth"
INPUT_SIZE = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_largest_cc(mask):
    """Keep only the largest connected component of a binary mask."""
    if mask is None or np.sum(mask) == 0:
        return mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, 4])
    mask_cleaned = np.zeros_like(mask)
    mask_cleaned[labels == largest_label] = 255
    return mask_cleaned

def load_model():
    print(f"Using device: {DEVICE}")
    model = ST_RoomNet(ref_path="assets/ref_img2.png", out_size=(INPUT_SIZE, INPUT_SIZE))
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    if DEVICE.type == 'cuda':
        model.half() # Max Speed
    model.eval()
    
    # Warm up
    print("Warming up...")
    with torch.no_grad():
        d = torch.zeros((1, 3, 400, 400)).to(DEVICE)
        if DEVICE.type == 'cuda':
             d = d.half()
        for _ in range(3): _ = model(d)
        
    return model

def process_image(model, image_path):
    # 1. Load Original
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    orig_h, orig_w = img_bgr.shape[:2]
    
    # 2. Preprocess for Model (Full GPU Acceleration)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    with torch.no_grad():
        full_img_gpu = torch.from_numpy(img_rgb).to(DEVICE).float().permute(2, 0, 1) / 255.0
        # Resize on GPU
        inp_batch = F.interpolate(full_img_gpu.unsqueeze(0), size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False)
        if DEVICE.type == 'cuda':
            inp_batch = inp_batch.half()
        
        # 3. Inference
        start_time = time.perf_counter()
        out = model(inp_batch)
        inference_time = time.perf_counter() - start_time
        
        # 4. Post-process (SMOOTH & NOISE-FREE)
        dist = torch.abs(out - 4.0)
        mask_soft = torch.clamp(1.0 - dist, min=0.0)
        mask_soft = F.avg_pool2d(mask_soft, kernel_size=3, stride=1, padding=1)
        
        mask_upscaled = F.interpolate(mask_soft, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        mask_final_gpu = (mask_upscaled[0, 0] > 0.5).byte() * 255
        mask_cpu = mask_final_gpu.cpu().numpy()
        
    # 5. Noise Removal (LCC)
    mask_cleaned = get_largest_cc(mask_cpu)
    
    # 6. Final Visualization
    mask_bgr = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, mask_bgr, 0.4, 0)
    
    return mask_cleaned, overlay, inference_time

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model = load_model()
    
    # Get all images
    image_paths = sorted(list(ASSETS_DIR.glob("*.jpg")) + list(ASSETS_DIR.glob("*.png")))
    image_paths = [p for p in image_paths if "ref_img2" not in p.name]
    
    if not image_paths:
        print("No images found in assets/")
        return

    print(f"Processing {len(image_paths)} images in bulk...")
    
    total_start = time.time()
    for i, path in enumerate(image_paths):
        # results = process_image(model, path)
        try:
            results = process_image(model, path)
        except Exception as e:
            print(f"  [!] Error processing {path.name}: {e}")
            continue

        if results:
            mask, overlay, inf_time = results
            
            # Save files (JPG for speed)
            stem = path.stem
            cv2.imwrite(str(OUTPUT_DIR / f"{stem}_pred.jpg"), mask, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(OUTPUT_DIR / f"{stem}_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  ✓ {path.name} done in {inf_time*1000:.2f}ms")
            
    total_time = time.time() - total_start
    print("-" * 30)
    print(f"All images processed in {total_time:.2f}s")
    print(f"Average: {total_time/len(image_paths):.2f}s per image")
    print(f"Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
