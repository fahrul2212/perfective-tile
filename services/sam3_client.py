"""
services/sam3_client.py
─────────────────────────────────────────────────────────────────
Adapter Layer: HTTP client untuk SAM3 microservice.

Dipindahkan dari core/ ke services/ karena ini adalah adapter (ring luar),
bukan domain logic. Core layer seharusnya tidak tahu tentang HTTP.

Referensi arsitektur:
  - Robert C. Martin, "Clean Architecture" (2017), Ch.22: The Clean Architecture
  - Alistair Cockburn, "Hexagonal Architecture" — Adapters & Ports
"""
import httpx
import numpy as np
import cv2
from typing import Optional, Tuple
from core.config import Config


class SAM3Client:
    async def get_floor_mask(self, image_bytes: bytes, filename: str = "image.jpg") -> Optional[bytes]:
        """Ambil raw bytes mask dari SAM3 microservice."""
        try:
            async with httpx.AsyncClient(timeout=Config.SAM3_TIMEOUT) as client:
                sam3_resp = await client.post(
                    f"{Config.SAM3_BASE_URL}/predict/floor",
                    files={"file": (filename, image_bytes, "image/jpeg")}
                )
                if sam3_resp.status_code == 200:
                    return sam3_resp.content
                else:
                    print(f"[SAM3] Service error {sam3_resp.status_code}, skip image ke-4.")
                    return None
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            print(f"[SAM3] Service tidak tersedia ({e.__class__.__name__}), skip image ke-4.")
            return None
        except Exception as e:
            print(f"[SAM3] Unexpected error: {e}, skip image ke-4.")
            return None

    async def get_floor_mask_decoded(
        self, image_bytes: bytes, target_size: Tuple[int, int] = None, filename: str = "image.jpg"
    ) -> Optional[np.ndarray]:
        """
        Ambil SAM3 mask dan langsung decode ke np.ndarray grayscale.
        
        Args:
            image_bytes: JPEG bytes dari gambar input
            target_size: (width, height) untuk resize mask agar match gambar asli
            filename: nama file untuk multipart upload
            
        Returns:
            np.ndarray (H, W) uint8 binary mask, atau None jika gagal
        """
        raw_bytes = await self.get_floor_mask(image_bytes, filename)
        if raw_bytes is None:
            return None
        
        # Decode bytes → numpy array
        nparr = np.frombuffer(raw_bytes, np.uint8)
        mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("[SAM3] Gagal decode response mask.")
            return None
        
        # Resize jika target_size diberikan
        if target_size is not None:
            w, h = target_size
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Binarize
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        print(f"[SAM3] Mask decoded: {mask.shape}  pixel lantai: {np.sum(mask > 0):,}")
        return mask


sam3_client = SAM3Client()
