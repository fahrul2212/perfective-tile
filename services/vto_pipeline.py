"""
services/vto_pipeline.py
─────────────────────────────────────────────────────────────────
Orchestration Layer: VTO (Virtual Try-On) Pipeline.

Split Pipeline Architecture (Phase 1):
  - analyze()     → BERAT: AI inference + postprocess → simpan Blueprint (1x per foto)
  - render_tile()  → RINGAN: load blueprint + warp tile baru (~100ms, Nx per tile swap)

Referensi:
  - Blueprint Architecture (upgrade/phase-1_split-pipeline.md)
  - Martin Fowler, "PEAA" — Service Layer + Caching Patterns
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
from services.scene_cache import scene_cache, SceneBlueprint
from utils.perspective import (
    detect_4_points, smart_trapezoid_fitting,
    calc_cols_rows, render_ceramic_perspective,
    render_tile_fast
)
from utils.perspective.renderer import _resolve_tile_path
from utils.tile_catalog import get_tile_path


# ── Data Classes ──────────────────

@dataclass
class AnalyzeResult:
    """Hasil dari analyze() — blueprint + preview."""
    blueprint: SceneBlueprint
    vto_preview: np.ndarray            # Preview VTO dengan default tile
    mask_refined: np.ndarray
    overlay_bgr: np.ndarray
    shadow_map: np.ndarray
    mask_sam3: Optional[np.ndarray]
    inference_time_ms: float
    resolution: str


@dataclass
class RenderResult:
    """Hasil dari render_tile() — hanya VTO image baru."""
    vto_bgr: np.ndarray
    render_time_ms: float
    tile_id: str


# ── Backward compat ──────────────────
@dataclass
class VTOResult:
    """Legacy data class — tetap dipertahankan untuk backward compatibility."""
    vto_bgr: np.ndarray
    mask_refined: np.ndarray
    overlay_bgr: np.ndarray
    shadow_map: np.ndarray
    mask_sam3: Optional[np.ndarray]
    inference_time_ms: float
    resolution: str


class VTOPipeline:
    """
    Orchestrator untuk pipeline Virtual Try-On.
    
    Split Pipeline:
      analyze()      → AI berat 1x → simpan Scene Blueprint
      render_tile()   → Load blueprint + warp tile → ~100ms
      process()      → Legacy: analyze + render dalam 1 panggilan
    """
    
    # ═══════════════════════════════════════════════════
    # FASE A — ANALYZE (1x per foto, BERAT ~3-5 detik)
    # ═══════════════════════════════════════════════════
    
    async def analyze(self, img_bgr: np.ndarray) -> AnalyzeResult:
        """
        Jalankan AI inference + postprocess + simpan blueprint.
        Dipanggil HANYA 1x per foto.
        """
        orig_h, orig_w = img_bgr.shape[:2]
        
        # ── Step 1: RoomNet inference ──
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            img_gpu = (
                torch.from_numpy(img_rgb)
                .to(Config.DEVICE).float()
                .permute(2, 0, 1) / 255.0
            )
            mask_cpu, inference_time = roomnet_service.predict(img_gpu, orig_h, orig_w)
        
        # ── Step 2: Postprocess ──
        mask_cleaned = get_largest_cc(mask_cpu)
        mask_cleaned = fill_floor_bottom(mask_cleaned)
        mask_refined = refine_mask_smooth(mask_cleaned)
        
        # ── Step 3: Overlay ──
        mask_bgr = cv2.cvtColor(mask_refined, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, mask_bgr, 0.4, 0)
        
        # ── Step 4: SAM3 mask ──
        _, img_encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_bytes = img_encoded.tobytes()
        mask_sam3 = await sam3_client.get_floor_mask_decoded(
            img_bytes, target_size=(orig_w, orig_h)
        )
        
        # ── Step 5: Perspective detection + trapezoid fitting ──
        mask_for_sam = mask_sam3 if mask_sam3 is not None else mask_cleaned
        if mask_for_sam.shape[:2] != (orig_h, orig_w):
            mask_for_sam = cv2.resize(mask_for_sam, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        _, mask_for_sam = cv2.threshold(mask_for_sam, 127, 255, cv2.THRESH_BINARY)
        
        pts = detect_4_points(mask_cleaned)
        if pts is not None:
            pts = smart_trapezoid_fitting(mask_for_sam, pts)
        
        cols, rows = calc_cols_rows(pts) if pts else (3, 5)
        
        # ── Step 6: Shadow extraction ──
        shadow_source = mask_sam3 if mask_sam3 is not None else mask_refined
        shadow_map = extract_shadow_map(img_bgr, shadow_source)
        
        # ── Step 7: Generate preview VTO ──
        vto_preview = render_ceramic_perspective(
            img_bgr, mask_cleaned, mask_sam3=mask_sam3
        )
        
        # ── Step 8: Save to Scene Cache ──
        scene_id = scene_cache.generate_id()
        blueprint = scene_cache.save(
            scene_id=scene_id,
            img_bgr=img_bgr,
            mask_cleaned=mask_cleaned,
            mask_sam3=mask_sam3,
            mask_refined=mask_refined,
            shadow_map=shadow_map,
            overlay_bgr=overlay,
            perspective_pts=pts if pts else {},
            grid_cols=cols,
            grid_rows=rows,
            inference_time_ms=inference_time,
        )
        
        return AnalyzeResult(
            blueprint=blueprint,
            vto_preview=vto_preview,
            mask_refined=mask_refined,
            overlay_bgr=overlay,
            shadow_map=shadow_map,
            mask_sam3=mask_sam3,
            inference_time_ms=inference_time,
            resolution=f"{orig_w}x{orig_h}",
        )
    
    # ═══════════════════════════════════════════════════
    # FASE B — RENDER TILE (Nx per swap, FAST ~50ms)
    # ═══════════════════════════════════════════════════
    
    # In-memory cache untuk warp mask per scene
    _warp_mask_cache: dict = {}
    
    def render_tile(self, scene_id: str, tile_id: str) -> Optional[RenderResult]:
        """
        FAST re-render tile — skip detect+fitting, pakai cached pts.
        
        Optimasi:
        - Pakai render_tile_fast() bukan render_ceramic_perspective()
        - Skip detect_4_points() + smart_trapezoid_fitting()
        - Cache warp_mask antar swap
        - INTER_LINEAR bukan INTER_CUBIC
        """
        start = time.perf_counter()
        
        # Load blueprint + arrays dari cache
        blueprint = scene_cache.load_blueprint(scene_id)
        if blueprint is None:
            print(f"[pipeline] Scene not found: {scene_id}")
            return None
        
        arrays = scene_cache.load_arrays(scene_id)
        if arrays is None or arrays["img_bgr"] is None:
            print(f"[pipeline] Arrays not found for: {scene_id}")
            return None
        
        # Resolve tile path
        tile_path = get_tile_path(tile_id)
        
        # Get cached warp mask (if available)
        cached_warp = self._warp_mask_cache.get(scene_id)
        
        # Use pre-computed perspective points from blueprint
        mask_sam3 = arrays["mask_sam3"] if arrays["mask_sam3"] is not None else arrays["mask_cleaned"]
        
        vto_bgr, warp_mask = render_tile_fast(
            img_bgr=arrays["img_bgr"],
            mask_sam3=mask_sam3,
            pts=blueprint.perspective_pts,
            tile_path=tile_path,
            cached_warp_mask=cached_warp,
        )
        
        # Cache warp mask untuk swap berikutnya
        if warp_mask is not None and scene_id not in self._warp_mask_cache:
            self._warp_mask_cache[scene_id] = warp_mask
        
        render_time = (time.perf_counter() - start) * 1000
        print(f"[pipeline] FAST tile: {tile_id}  ({render_time:.0f}ms)")
        
        return RenderResult(
            vto_bgr=vto_bgr,
            render_time_ms=render_time,
            tile_id=tile_id,
        )
    
    # ═══════════════════════════════════════════════════
    # LEGACY — process() tetap ada untuk backward compat
    # ═══════════════════════════════════════════════════
    
    async def process(self, img_bgr: np.ndarray) -> VTOResult:
        """Legacy: analyze + render dalam 1 panggilan."""
        result = await self.analyze(img_bgr)
        return VTOResult(
            vto_bgr=result.vto_preview,
            mask_refined=result.mask_refined,
            overlay_bgr=result.overlay_bgr,
            shadow_map=result.shadow_map,
            mask_sam3=result.mask_sam3,
            inference_time_ms=result.inference_time_ms,
            resolution=result.resolution,
        )
