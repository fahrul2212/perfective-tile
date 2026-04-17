"""
services/vto_pipeline.py
─────────────────────────────────────────────────────────────────
Orchestration Layer: VTO (Virtual Try-On) Pipeline.

Mengatur alur lengkap dari gambar input → semua output (VTO, mask, shadow).
app.py cukup memanggil pipeline.process() tanpa perlu tahu detail step-nya.

Arsitektur:
  - Memanggil core/inference (RoomNet)
  - Memanggil core/postprocess (mask cleanup, refinement, shadow)
  - Memanggil services/sam3_client (SAM3 adapter)
  - Memanggil utils/perspective (VTO rendering)

Referensi:
  - Martin Fowler, "Patterns of Enterprise Application Architecture" — Service Layer
"""
import cv2
import numpy as np
import time
import torch
from dataclasses import dataclass
from typing import Optional

from core.config import Config
from core.inference import roomnet_service
from core.postprocess import (
    get_largest_cc, fill_floor_bottom,
    refine_mask_smooth, extract_shadow_map
)
from services.sam3_client import sam3_client
from utils.perspective import render_ceramic_perspective


@dataclass
class VTOResult:
    """Data class untuk hasil pipeline VTO."""
    vto_bgr: np.ndarray              # Hasil VTO tile perspektif
    mask_refined: np.ndarray          # Mask yang sudah dihaluskan
    overlay_bgr: np.ndarray           # Overlay mask di atas foto
    shadow_map: np.ndarray            # Peta bayangan
    mask_sam3: Optional[np.ndarray]   # SAM3 mask (bisa None)
    inference_time_ms: float          # Waktu inference RoomNet
    resolution: str                   # "WxH"


class VTOPipeline:
    """
    Orchestrator untuk pipeline Virtual Try-On end-to-end.
    
    Alur:
    1. RoomNet inference → raw mask
    2. Postprocess: get_largest_cc + fill_floor_bottom
    3. Refine mask smooth (Median + Gaussian)
    4. SAM3 mask (async HTTP ke microservice)
    5. VTO render (perspective tile + SAM3 clipping)
    6. Shadow extraction
    """
    
    async def process(self, img_bgr: np.ndarray) -> VTOResult:
        """
        Jalankan pipeline VTO lengkap.
        
        Args:
            img_bgr: Foto ruangan BGR (dari upload user)
            
        Returns:
            VTOResult berisi semua output siap disimpan
        """
        orig_h, orig_w = img_bgr.shape[:2]
        
        # ── Step 1: RoomNet inference ──────────────────
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            img_gpu = (
                torch.from_numpy(img_rgb)
                .to(Config.DEVICE).float()
                .permute(2, 0, 1) / 255.0
            )
            mask_cpu, inference_time = roomnet_service.predict(img_gpu, orig_h, orig_w)
        
        # ── Step 2: Basic postprocess ──────────────────
        mask_cleaned = get_largest_cc(mask_cpu)
        mask_cleaned = fill_floor_bottom(mask_cleaned)
        
        # ── Step 3: Smooth refinement ──────────────────
        mask_refined = refine_mask_smooth(mask_cleaned)
        
        # ── Step 4: Overlay visualization ──────────────
        mask_bgr = cv2.cvtColor(mask_refined, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, mask_bgr, 0.4, 0)
        
        # ── Step 5: SAM3 mask ──────────────────────────
        _, img_encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_bytes = img_encoded.tobytes()
        
        mask_sam3 = await sam3_client.get_floor_mask_decoded(
            img_bytes, target_size=(orig_w, orig_h)
        )
        
        # ── Step 6: VTO rendering ──────────────────────
        vto_bgr = render_ceramic_perspective(
            img_bgr, mask_cleaned, mask_sam3=mask_sam3
        )
        
        # ── Step 7: Shadow extraction ──────────────────
        shadow_source = mask_sam3 if mask_sam3 is not None else mask_refined
        shadow_map = extract_shadow_map(img_bgr, shadow_source)
        
        return VTOResult(
            vto_bgr=vto_bgr,
            mask_refined=mask_refined,
            overlay_bgr=overlay,
            shadow_map=shadow_map,
            mask_sam3=mask_sam3,
            inference_time_ms=inference_time,
            resolution=f"{orig_w}x{orig_h}",
        )
