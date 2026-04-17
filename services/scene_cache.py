"""
services/scene_cache.py
─────────────────────────────────────────────────────────────────
Scene Blueprint Cache — menyimpan hasil analisis AI ke disk.

Saat user upload foto, AI (RoomNet + SAM3) jalan 1x → hasilnya disimpan
sebagai "Scene Blueprint". Ketika user ganti tile, hanya perlu load
blueprint + render tile baru (~100ms) tanpa AI lagi.

Arsitektur:
  - Disk-based storage (folder per scene_id)
  - JSON metadata + binary mask/image files
  - In-memory LRU untuk akses cepat

Referensi:
  - Blueprint Architecture (Phase 1 riset)
  - Martin Fowler, "PEAA" — Caching Patterns
"""
import json
import uuid
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from collections import OrderedDict

from core.config import Config


# ── Blueprint Data Classes ──────────────────

@dataclass
class SceneBlueprint:
    """Semua data yang dibutuhkan untuk re-render tile tanpa AI."""
    scene_id: str
    created_at: float                          # timestamp
    resolution_w: int
    resolution_h: int
    
    # Perspective points (serializable)
    perspective_pts: Dict[str, Any]            # dict dari detect_4_points + smart_fit
    grid_cols: int
    grid_rows: int
    
    # File paths (relative to scene dir)
    original_path: str = ""
    mask_cleaned_path: str = ""
    mask_sam3_path: str = ""
    mask_refined_path: str = ""
    shadow_map_path: str = ""
    overlay_path: str = ""
    
    # Metadata
    inference_time_ms: float = 0.0


# ── Scene Cache Manager ──────────────────

class SceneCache:
    """
    Disk-based scene blueprint cache dengan in-memory LRU.
    
    Storage layout:
        outputs/scenes/{scene_id}/
        ├── blueprint.json           ← metadata + perspective points
        ├── original.jpg             ← foto asli
        ├── mask_cleaned.png         ← RoomNet mask (bersih)
        ├── mask_sam3.png            ← SAM3 mask
        ├── mask_refined.png         ← Refined mask
        ├── shadow_map.png           ← Shadow map
        └── overlay.jpg              ← Overlay visualization
    """
    
    def __init__(self, base_dir: Path = None, max_memory_cache: int = 10):
        self.base_dir = base_dir or (Config.OUTPUT_DIR / "scenes")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory LRU cache untuk numpy arrays (avoid disk reads)
        self._memory: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()
        self._max_memory = max_memory_cache
    
    def _scene_dir(self, scene_id: str) -> Path:
        d = self.base_dir / scene_id
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def generate_id(self) -> str:
        """Generate unique scene ID."""
        ts = int(time.time() * 1000)
        short = uuid.uuid4().hex[:6]
        return f"scene_{ts}_{short}"
    
    def save(
        self,
        scene_id: str,
        img_bgr: np.ndarray,
        mask_cleaned: np.ndarray,
        mask_sam3: Optional[np.ndarray],
        mask_refined: np.ndarray,
        shadow_map: np.ndarray,
        overlay_bgr: np.ndarray,
        perspective_pts: dict,
        grid_cols: int,
        grid_rows: int,
        inference_time_ms: float,
    ) -> SceneBlueprint:
        """Simpan semua data scene ke disk + memory cache."""
        d = self._scene_dir(scene_id)
        h, w = img_bgr.shape[:2]
        
        # Save binary files
        cv2.imwrite(str(d / "original.jpg"), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(d / "mask_cleaned.png"), mask_cleaned)
        cv2.imwrite(str(d / "mask_refined.png"), mask_refined)
        cv2.imwrite(str(d / "shadow_map.png"), shadow_map)
        cv2.imwrite(str(d / "overlay.jpg"), overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        sam3_path = ""
        if mask_sam3 is not None:
            cv2.imwrite(str(d / "mask_sam3.png"), mask_sam3)
            sam3_path = "mask_sam3.png"
        
        # Serializable perspective points (convert tuples)
        pts_serial = {}
        for k, v in perspective_pts.items():
            if isinstance(v, (tuple, list)):
                pts_serial[k] = list(v)
            else:
                pts_serial[k] = v
        
        # Build blueprint
        bp = SceneBlueprint(
            scene_id=scene_id,
            created_at=time.time(),
            resolution_w=w,
            resolution_h=h,
            perspective_pts=pts_serial,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            original_path="original.jpg",
            mask_cleaned_path="mask_cleaned.png",
            mask_sam3_path=sam3_path,
            mask_refined_path="mask_refined.png",
            shadow_map_path="shadow_map.png",
            overlay_path="overlay.jpg",
            inference_time_ms=inference_time_ms,
        )
        
        # Save JSON metadata
        with open(d / "blueprint.json", "w") as f:
            json.dump(asdict(bp), f, indent=2)
        
        # Memory cache
        self._memory[scene_id] = {
            "img_bgr": img_bgr,
            "mask_cleaned": mask_cleaned,
            "mask_sam3": mask_sam3,
            "mask_refined": mask_refined,
            "shadow_map": shadow_map,
        }
        self._evict_lru()
        
        print(f"[cache] Scene saved: {scene_id}  ({w}x{h})")
        return bp
    
    def load_blueprint(self, scene_id: str) -> Optional[SceneBlueprint]:
        """Load blueprint metadata dari disk."""
        bp_path = self.base_dir / scene_id / "blueprint.json"
        if not bp_path.exists():
            return None
        with open(bp_path, "r") as f:
            data = json.load(f)
        return SceneBlueprint(**data)
    
    def load_arrays(self, scene_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Load numpy arrays — dari memory cache atau disk."""
        # Memory cache hit
        if scene_id in self._memory:
            self._memory.move_to_end(scene_id)
            return self._memory[scene_id]
        
        # Disk fallback
        d = self.base_dir / scene_id
        if not d.exists():
            return None
        
        arrays = {
            "img_bgr": cv2.imread(str(d / "original.jpg")),
            "mask_cleaned": cv2.imread(str(d / "mask_cleaned.png"), cv2.IMREAD_GRAYSCALE),
            "mask_refined": cv2.imread(str(d / "mask_refined.png"), cv2.IMREAD_GRAYSCALE),
            "shadow_map": cv2.imread(str(d / "shadow_map.png"), cv2.IMREAD_GRAYSCALE),
        }
        
        sam3_path = d / "mask_sam3.png"
        arrays["mask_sam3"] = cv2.imread(str(sam3_path), cv2.IMREAD_GRAYSCALE) if sam3_path.exists() else None
        
        # Populate memory cache
        self._memory[scene_id] = arrays
        self._evict_lru()
        
        return arrays
    
    def _evict_lru(self):
        """Evict oldest entries jika memory cache penuh."""
        while len(self._memory) > self._max_memory:
            evicted_id, _ = self._memory.popitem(last=False)
            print(f"[cache] Evicted from memory: {evicted_id}")
    
    def get_scene_url_prefix(self, scene_id: str) -> str:
        """Return URL prefix untuk akses static files."""
        return f"/outputs/scenes/{scene_id}"


# Singleton instance
scene_cache = SceneCache()
