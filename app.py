"""
app.py — Application Layer (Thin Router)
─────────────────────────────────────────────────────────────────
Split Pipeline Architecture:
  /analyze     → AI berat 1x per foto → return scene_id + blueprint
  /render-tile → Load blueprint + warp tile baru → ~100ms
  /tile-catalog → Daftar tile yang tersedia
  /predict     → Legacy endpoint (analyze + render dalam 1 panggilan)

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
from pydantic import BaseModel

from core.config import Config
from core.inference import roomnet_service
from services.vto_pipeline import VTOPipeline
from services.scene_cache import scene_cache
from utils.tile_catalog import get_tile_catalog


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
app.mount("/tiles", StaticFiles(directory=str(Config.ROOT_DIR / "assets" / "tile")), name="tiles")
app.mount("/assets", StaticFiles(directory=str(Config.ROOT_DIR / "assets")), name="assets")


# Pipeline instance
pipeline = VTOPipeline()


# ── Request Models ──────────────────────────────────
class RenderTileRequest(BaseModel):
    scene_id: str
    tile_id: str


# ── Routes ──────────────────────────────────

@app.get("/")
async def read_index():
    return FileResponse(Config.ROOT_DIR / "static/index.html")


# ═══════════════════════════════════════════════════
# NEW: /analyze — AI berat 1x per foto
# ═══════════════════════════════════════════════════

@app.post("/analyze")
async def analyze_room(file: UploadFile = File(...)):
    """
    Fase A — Analisis ruangan (1x per foto, berat ~3-5 detik).
    Hasilkan Scene Blueprint yang di-cache untuk tile swap cepat.
    """
    if roomnet_service.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Run analyze pipeline
        result = await pipeline.analyze(img_bgr)
        bp = result.blueprint
        prefix = scene_cache.get_scene_url_prefix(bp.scene_id)
        
        # Save preview VTO
        preview_path = scene_cache.base_dir / bp.scene_id / "preview_vto.jpg"
        cv2.imwrite(str(preview_path), result.vto_preview, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return {
            "status": "success",
            "scene_id": bp.scene_id,
            "inference_time_ms": round(result.inference_time_ms, 2),
            "resolution": result.resolution,
            "preview_vto_url": f"{prefix}/preview_vto.jpg",
            "overlay_url": f"{prefix}/overlay.jpg",
            "mask_url": f"{prefix}/mask_refined.png",
            "shadow_url": f"{prefix}/shadow_map.png",
            "sam3_mask_url": f"{prefix}/mask_sam3.png" if bp.mask_sam3_path else None,
            "original_url": f"{prefix}/original.jpg",
        }
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════
# NEW: /render-tile — Tile swap ringan ~100ms
# ═══════════════════════════════════════════════════

@app.post("/render-tile")
async def render_tile(req: RenderTileRequest):
    """
    Fase B — Re-render tile tanpa AI (~100ms).
    Menggunakan cached Scene Blueprint dari /analyze.
    """
    try:
        result = pipeline.render_tile(req.scene_id, req.tile_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Scene not found: {req.scene_id}")
        
        # Save rendered VTO
        timestamp = int(time.time() * 1000)
        filename = f"vto_{req.scene_id}_{req.tile_id}_{timestamp}.jpg"
        filepath = Config.OUTPUT_DIR / filename
        cv2.imwrite(str(filepath), result.vto_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return {
            "status": "success",
            "vto_url": f"/outputs/{filename}",
            "render_time_ms": round(result.render_time_ms, 2),
            "tile_id": result.tile_id,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════
# NEW: /tile-catalog — Daftar tile yang tersedia
# ═══════════════════════════════════════════════════

@app.get("/tile-catalog")
async def tile_catalog():
    """Return daftar tile keramik yang tersedia."""
    tiles = get_tile_catalog()
    return {
        "status": "success",
        "count": len(tiles),
        "tiles": tiles,
    }


# ═══════════════════════════════════════════════════
# LEGACY: /predict — tetap ada untuk backward compat
# ═══════════════════════════════════════════════════

@app.post("/predict")
async def predict_layout(file: UploadFile = File(...)):
    """Legacy endpoint — analyze + render dalam 1 panggilan."""
    if roomnet_service.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        result = await pipeline.process(img_bgr)
        
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
