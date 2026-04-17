"""
app.py — Application Layer (Thin Router)
─────────────────────────────────────────────────────────────────
Hanya bertugas: routing HTTP, file I/O, dan return JSON.
Semua logika processing ada di services/vto_pipeline.py.

Arsitektur:
  app.py → services/ → core/ + utils/
"""
import cv2
import numpy as np
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from core.config import Config
from core.inference import roomnet_service
from services.vto_pipeline import VTOPipeline


# ── Lifecycle ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
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

# Pipeline instance
pipeline = VTOPipeline()


# ── Routes ──────────────────────────────────
@app.get("/")
async def read_index():
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
        
        # 2. Run VTO pipeline (semua logika ada di sini)
        result = await pipeline.process(img_bgr)
        
        # 3. Save outputs
        timestamp = int(time.time() * 1000)
        
        files_to_save = {
            "mask": (f"res_{timestamp}_mask.jpg", result.mask_refined, [cv2.IMWRITE_JPEG_QUALITY, 95]),
            "overlay": (f"res_{timestamp}_overlay.jpg", result.overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]),
            "vto": (f"res_{timestamp}_vto.jpg", result.vto_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]),
            "shadow": (f"res_{timestamp}_shadow.png", result.shadow_map, None),
        }
        
        urls = {}
        for key, (filename, img, params) in files_to_save.items():
            if params:
                cv2.imwrite(str(Config.OUTPUT_DIR / filename), img, params)
            else:
                cv2.imwrite(str(Config.OUTPUT_DIR / filename), img)
            urls[f"{key}_url"] = f"/outputs/{filename}"
        
        # SAM3 mask (optional)
        sam3_mask_url = None
        if result.mask_sam3 is not None:
            sam3_filename = f"res_{timestamp}_sam3_mask.png"
            cv2.imwrite(str(Config.OUTPUT_DIR / sam3_filename), result.mask_sam3)
            sam3_mask_url = f"/outputs/{sam3_filename}"

        return {
            "status": "success",
            "inference_time_ms": round(result.inference_time_ms, 2),
            **urls,
            "sam3_mask_url": sam3_mask_url,
            "resolution": result.resolution,
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
