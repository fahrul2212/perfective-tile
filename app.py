import torch
import cv2
import numpy as np
import io
import time
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path
import torch.nn.functional as F
from core.model import ST_RoomNet

# --- Configuration ---
ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / "api_outputs"
WEIGHT_PATH = "weights/persfective.pth"
INPUT_SIZE = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global model container
model_container = {}

def get_largest_cc(mask):
    """Keep only the largest connected component of a binary mask."""
    if mask is None or np.sum(mask) == 0:
        return mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1: # Only background
        return mask
    # stats[:, 4] is the 'area' of each component. Skip label 0 (background).
    largest_label = 1 + np.argmax(stats[1:, 4])
    mask_cleaned = np.zeros_like(mask)
    mask_cleaned[labels == largest_label] = 255
    return mask_cleaned

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    print(f"[*] Starting server on device: {DEVICE}")
    try:
        model = ST_RoomNet(ref_path="assets/ref_img2.png", out_size=(INPUT_SIZE, INPUT_SIZE))
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        if DEVICE.type == 'cuda':
            model.half() # Convert to FP16 for max speed on GPU
        model.eval()
        
        # Warm-up phase: Run a dummy inference to prime the GPU
        print("Warming up GPU...")
        with torch.no_grad():
            dummy_input = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE)).to(DEVICE)
            if DEVICE.type == 'cuda':
                dummy_input = dummy_input.half()
            for _ in range(3):
                _ = model(dummy_input)
        
        model_container['model'] = model
        print("[*] Model loaded successfully.")
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
    
    yield
    
    # Shutdown: Clean up
    model_container.clear()
    print("[*] Server shutting down.")

app = FastAPI(lifespan=lifespan, title="Room Layout API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_container.get('model'):
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 1. Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        orig_h, orig_w = img_bgr.shape[:2]
        
        # 2. Preprocess for Model (Full GPU Acceleration)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Move to GPU as soon as possible
        with torch.no_grad():
            full_img_gpu = torch.from_numpy(img_rgb).to(DEVICE).float().permute(2, 0, 1) / 255.0
            # Resize on GPU
            inp_batch = F.interpolate(full_img_gpu.unsqueeze(0), size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False)
            if DEVICE.type == 'cuda':
                inp_batch = inp_batch.half()
            
            # 3. Inference
            start_time = time.perf_counter()
            out = model_container['model'](inp_batch) # [1, 1, 400, 400]
            inference_time = (time.perf_counter() - start_time) * 1000
            
            # 4. Post-process (SMOOTH & NOISE-FREE)
            # Create a soft probability mask for Class 4 (Floor)
            # This replaces (out.round() == 4) to ensure smooth anti-aliased edges
            dist = torch.abs(out - 4.0)
            mask_soft = torch.clamp(1.0 - dist, min=0.0) # 1.0 at center 4.0, drops to 0.0 at 3.0/5.0
            
            # Apply subtle GPU Gaussian blurring for extra smoothness
            # Using a simple 3x3 box blur as proxy for Gaussian if not using dedicated lib, 
            # Or just rely on bilinear upscaling
            mask_soft = F.avg_pool2d(mask_soft, kernel_size=3, stride=1, padding=1)
            
            # Upscale to original resolution with bilinear interpolation
            mask_upscaled = F.interpolate(mask_soft, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            
            # Final threshold and move to CPU
            mask_final_gpu = (mask_upscaled[0, 0] > 0.5).byte() * 255
            mask_cpu = mask_final_gpu.cpu().numpy()
            
        # 5. Noise Removal (Largest Connected Component)
        # This removes isolated dots, "dashed lines" at edges, etc.
        mask_cleaned = get_largest_cc(mask_cpu)
        
        # 6. Final Visualization
        mask_bgr = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, mask_bgr, 0.4, 0)
        
        # 7. Save Results (Fast JPG)
        timestamp = int(time.time() * 1000)
        mask_filename = f"res_{timestamp}_mask.jpg"
        overlay_filename = f"res_{timestamp}_overlay.jpg"
        
        mask_path = OUTPUT_DIR / mask_filename
        overlay_path = OUTPUT_DIR / overlay_filename
        
        cv2.imwrite(str(mask_path), mask_cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return {
            "status": "success",
            "inference_time_ms": round(inference_time, 2),
            "mask_url": f"/outputs/{mask_filename}",
            "overlay_url": f"/outputs/{overlay_filename}",
            "resolution": f"{orig_w}x{orig_h}"
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Static routes for outputs and frontend
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
