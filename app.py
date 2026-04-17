import cv2
import numpy as np
import time
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from core.config import Config
from core.postprocess import (
    get_largest_cc, fill_floor_bottom,
    refine_mask_smooth, extract_shadow_map, generate_alpha_mask
)
from core.inference import roomnet_service
from core.sam3_client import sam3_client
from utils.perspective import render_ceramic_perspective

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifecycle Server (Startup/Teardown)
    roomnet_service.initialize()
    yield
    print("[*] Server shutting down.")

app = FastAPI(lifespan=lifespan, title="RoomVision AI — Virtual Try-On")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files
app.mount("/outputs", StaticFiles(directory=str(Config.OUTPUT_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(Config.ROOT_DIR / "static")), name="static")

@app.get("/")
async def read_index():
    # Load frontend UI
    return FileResponse(Config.ROOT_DIR / "static/index.html")

@app.post("/predict")
async def predict_layout(file: UploadFile = File(...)):
    if roomnet_service.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        # 1. Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        orig_h, orig_w = img_bgr.shape[:2]
        
        # 2. RoomNet inference → raw mask
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            img_gpu = torch.from_numpy(img_rgb).to(Config.DEVICE).float().permute(2, 0, 1) / 255.0
            mask_cpu, inference_time = roomnet_service.predict(img_gpu, orig_h, orig_w)
            
        # 3. Postprocess basic (existing — TIDAK diubah)
        mask_cleaned = get_largest_cc(mask_cpu)
        mask_cleaned = fill_floor_bottom(mask_cleaned)
        
        # 4. NEW: Refine mask smooth (Median + Gaussian)
        mask_refined = refine_mask_smooth(mask_cleaned)
        
        # 5. Visualizer — overlay mask halus
        mask_bgr = cv2.cvtColor(mask_refined, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, mask_bgr, 0.4, 0)
        
        # 6. SAM3 Mask — decode langsung ke numpy untuk VTO
        _, img_encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_bytes = img_encoded.tobytes()
        
        mask_sam3 = await sam3_client.get_floor_mask_decoded(
            img_bytes, target_size=(orig_w, orig_h)
        )
        
        # 7. NEW: VTO render dengan SAM3 mask clipping + Smart Trapezoid Fitting
        vto_bgr = render_ceramic_perspective(
            img_bgr, mask_cleaned, mask_sam3=mask_sam3
        )
        
        # 8. NEW: Shadow extraction pada area lantai
        # Gunakan SAM3 mask jika tersedia, fallback ke mask_refined
        shadow_mask_source = mask_sam3 if mask_sam3 is not None else mask_refined
        shadow_map = extract_shadow_map(img_bgr, shadow_mask_source)
        
        # 9. Save semua output
        timestamp = int(time.time() * 1000)
        
        mask_filename = f"res_{timestamp}_mask.jpg"
        overlay_filename = f"res_{timestamp}_overlay.jpg"
        vto_filename = f"res_{timestamp}_vto.jpg"
        shadow_filename = f"res_{timestamp}_shadow.png"
        
        cv2.imwrite(str(Config.OUTPUT_DIR / mask_filename), mask_refined, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(Config.OUTPUT_DIR / overlay_filename), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(Config.OUTPUT_DIR / vto_filename), vto_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(Config.OUTPUT_DIR / shadow_filename), shadow_map)
        
        # SAM3 mask raw (untuk display di frontend)
        sam3_mask_url = None
        if mask_sam3 is not None:
            sam3_filename = f"res_{timestamp}_sam3_mask.png"
            cv2.imwrite(str(Config.OUTPUT_DIR / sam3_filename), mask_sam3)
            sam3_mask_url = f"/outputs/{sam3_filename}"

        return {
            "status": "success",
            "inference_time_ms": round(inference_time, 2),
            "mask_url": f"/outputs/{mask_filename}",
            "overlay_url": f"/outputs/{overlay_filename}",
            "vto_url": f"/outputs/{vto_filename}",
            "shadow_url": f"/outputs/{shadow_filename}",
            "sam3_mask_url": sam3_mask_url,
            "resolution": f"{orig_w}x{orig_h}"
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
