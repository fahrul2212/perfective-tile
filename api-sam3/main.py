from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager  # Ganti @on_event (deprecated)
from PIL import Image
import io
import time
import os

from sam3_service import Sam3Service

# Config dari environment (bukan hardcode)
SAM3_PORT = int(os.getenv("SAM3_PORT", "8001"))  # Port 8001, terpisah dari main app (8000)

service_container = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager menggantikan @on_event yang deprecated."""
    # Startup
    print(f"[SAM3] Loading model... (port {SAM3_PORT})")
    try:
        service_container["sam3"] = Sam3Service()
        print("[SAM3] Model loaded successfully!")
    except Exception as e:
        print(f"[SAM3] FAILED to load model: {e}")
    yield
    # Shutdown
    service_container.clear()
    print("[SAM3] Service shutting down.")



app = FastAPI(
    title="SAM3 Floor Mask API",
    description="API untuk mendeteksi mask lantai menggunakan SAM3 dari Hugging Face.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "SAM3 Floor Mask API v2.0. POST image ke /predict/floor"}

@app.get("/health")
def health_check():
    """Endpoint untuk dicek oleh main app sebelum kirim request."""
    return {
        "status": "ok",
        "model_loaded": "sam3" in service_container,
        "port": SAM3_PORT
    }

@app.post("/predict/floor", summary="SAM3 floor segmentation",
          responses={200: {"content": {"image/png": {}}}})
async def predict_floor(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    if "sam3" not in service_container:
        raise HTTPException(status_code=503, detail="SAM3 model belum siap atau gagal load.")

    try:
        start_time = time.perf_counter()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        mask_bytes = service_container["sam3"].predict_floor_mask(image)
        elapsed = time.perf_counter() - start_time

        return Response(
            content=mask_bytes,
            media_type="image/png",
            headers={"X-Process-Time": f"{elapsed:.4f}s"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat memproses: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=SAM3_PORT)  # reload=True dihapus (hanya dev via CLI)
