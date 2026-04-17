"""
utils/perspective/renderer.py
─────────────────────────────────────────────────────────────────
Rendering tile keramik perspektif (VTO) dengan SAM3 mask clipping.

Referensi:
  - OpenCV getPerspectiveTransform — 4-point homography estimation
  - Hartley & Zisserman, "Multiple View Geometry in CV" (2004)
"""
import cv2
import numpy as np

from utils.perspective.detect_points import detect_4_points
from utils.perspective.trapezoid_fitting import smart_trapezoid_fitting
from utils.perspective.grid import calc_cols_rows


def _resolve_tile_path(tile_path: str):
    """Resolve tile asset path: try as-is, then relative to project root."""
    from pathlib import Path as _Path
    ROOT = _Path(__file__).parent.parent.parent

    p = _Path(tile_path)
    if p.exists():
        return str(p)
    p2 = ROOT / tile_path
    if p2.exists():
        return str(p2)
    return tile_path


def render_ceramic_perspective(
    img_bgr:   np.ndarray,
    mask_st:   np.ndarray,
    mask_sam3: np.ndarray = None,
    tile_path: str = r"assets/tile/Concord-60x60-PGC66K001-ALASKA WHITE.jpeg",
) -> np.ndarray:
    """
    Render tile keramik perspektif di atas foto ruangan.
    
    Pipeline:
    1. Deteksi 4 titik perspektif dari mask ST_RoomNet
    2. Smart Trapezoid Fitting menggunakan SAM3 mask (jika tersedia)
    3. Dynamic grid sizing
    4. Warp texture dengan perspective transform
    5. Clip ke SAM3 mask boundary (tile hanya muncul di area lantai)
    
    Args:
        img_bgr: Foto ruangan BGR
        mask_st: Mask dari ST_RoomNet (untuk deteksi titik)
        mask_sam3: Mask dari SAM3 (untuk clipping). Jika None, gunakan mask_st.
        tile_path: Path ke gambar tile keramik
        
    Returns:
        BGR image dengan tile ditempel di area lantai
    """
    h_orig, w_orig = img_bgr.shape[:2]
    
    # Fallback: jika SAM3 tidak tersedia, gunakan mask_st
    if mask_sam3 is None:
        mask_sam3 = mask_st.copy()
    
    # Pastikan ukuran mask sesuai
    if mask_sam3.shape[:2] != (h_orig, w_orig):
        mask_sam3 = cv2.resize(mask_sam3, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    _, mask_sam3 = cv2.threshold(mask_sam3, 127, 255, cv2.THRESH_BINARY)
    
    # Step 1: Deteksi 4 titik dari mask ST_RoomNet
    pts = detect_4_points(mask_st)
    if pts is None:
        print("[perspective] GAGAL mendeteksi 4 titik. Return gambar asli.")
        return img_bgr.copy()
    
    # Step 2: Smart Trapezoid Fitting
    pts = smart_trapezoid_fitting(mask_sam3, pts)
    
    # Step 3: Dynamic grid
    cols, rows = calc_cols_rows(pts)
    
    canvas_h = pts["canvas_h"]
    canvas_w = pts["canvas_w"]
    shift_y  = pts["shift_y"]
    shift_x  = pts["shift_x"]
    
    # Step 4: Build texture sheet
    TILE_PX = 500
    tile_w, tile_h = TILE_PX, TILE_PX
    tex_w = cols * tile_w
    tex_h = rows * tile_h
    
    resolved = _resolve_tile_path(tile_path)
    tile_img = cv2.imread(resolved) if resolved else None
    
    if tile_img is None:
        print(f"[perspective] WARN: tile tidak ditemukan: {tile_path}")
        tile_img = np.full((tile_h, tile_w, 3), (220, 220, 220), dtype=np.uint8)
    else:
        tile_img = cv2.resize(tile_img, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
    
    # Build texture sheet
    texture = np.zeros((tex_h, tex_w, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            texture[r * tile_h:(r + 1) * tile_h,
                    c * tile_w:(c + 1) * tile_w] = tile_img
    
    # Grout (nat) — digambar SEBELUM warp agar ikut terdistorsi perspektif
    GROUT_COLOR     = (80, 78, 75)
    GROUT_THICKNESS = 4
    for c in range(cols + 1):
        x = min(c * tile_w, tex_w - 1)
        cv2.line(texture, (x, 0), (x, tex_h - 1), GROUT_COLOR, GROUT_THICKNESS)
    for r in range(rows + 1):
        y = min(r * tile_h, tex_h - 1)
        cv2.line(texture, (0, y), (tex_w - 1, y), GROUT_COLOR, GROUT_THICKNESS)
    
    # Step 5: Perspective warp
    src_pts = np.float32([
        [0,         0        ],
        [tex_w - 1, 0        ],
        [0,         tex_h - 1],
        [tex_w - 1, tex_h - 1],
    ])
    dst_pts = np.float32([
        pts["C_TL"], pts["C_TR"],
        pts["C_BL"], pts["C_BR"],
    ])
    
    M_warp = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    warped_ceramic = cv2.warpPerspective(
        texture, M_warp, (canvas_w, canvas_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    ones = np.ones((tex_h, tex_w), dtype=np.uint8) * 255
    warp_mask = cv2.warpPerspective(
        ones, M_warp, (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    
    # Step 6: Composite dengan SAM3 mask clipping
    img_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    img_canvas[shift_y:shift_y + h_orig, shift_x:shift_x + w_orig] = img_bgr
    
    mask_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask_canvas[shift_y:shift_y + h_orig, shift_x:shift_x + w_orig] = mask_sam3
    
    # Intersection: tile hanya muncul di area SAM3 mask
    combined_mask = cv2.bitwise_and(warp_mask, mask_canvas)
    
    result = img_canvas.copy()
    result[combined_mask > 0] = warped_ceramic[combined_mask > 0]
    
    # Crop ke resolusi asli
    result_cropped = result[shift_y:shift_y + h_orig, shift_x:shift_x + w_orig]
    
    print(f"[perspective] VTO selesai — tiles: {cols}×{rows} pixel terisi: {np.count_nonzero(combined_mask):,}")
    return result_cropped
