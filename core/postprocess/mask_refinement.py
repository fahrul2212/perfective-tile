"""
core/postprocess/mask_refinement.py
─────────────────────────────────────────────────────────────────
Fungsi-fungsi refinement mask lantai: smoothing dan alpha matting.
Porting dari riset:
  - tests/segmentasi-perfect/step2_refine_mask.py
  - tests/segmentasi-perfect/step4_cm4_hybrid_refinement.py

Referensi:
  - Gonzalez & Woods, "Digital Image Processing" (4th Ed.), Ch.5
  - Li et al., "Bridging Composite and Real: Matting", CVPR 2024 (CM4)
"""
import numpy as np
import cv2


def refine_mask_smooth(mask: np.ndarray) -> np.ndarray:
    """
    Menghaluskan mask lantai menggunakan Median Filter + Gaussian Blur.
    Riset: step2_refine_mask.py
    - Median Filter (25px) → menghapus noise/bintik tanpa merusak edge shape
    - Gaussian Blur (15px) → menghaluskan tepian agar tidak gerigi
    - Threshold 128 → menjaga volume area stabil
    
    Referensi:
    - Gonzalez & Woods, "Digital Image Processing" (4th Ed.), Ch.5: Spatial Filtering
    - Median filter unggul untuk salt-and-pepper noise pada binary mask
    """
    from PIL import Image, ImageFilter
    
    if mask is None or np.sum(mask) == 0:
        return mask
    
    pil_mask = Image.fromarray(mask, mode="L")
    
    # 1. Median Filter — hapus noise agresif (bintik-bintik kecil)
    refined = pil_mask.filter(ImageFilter.MedianFilter(size=25))
    
    # 2. Gaussian Blur — haluskan tepian (smooth boundary)
    refined = refined.filter(ImageFilter.GaussianBlur(radius=15))
    
    # 3. Threshold binary — kembalikan ke mask hitam-putih
    refined = refined.point(lambda p: 255 if p > 128 else 0)
    
    return np.array(refined)


def generate_alpha_mask(mask: np.ndarray) -> np.ndarray:
    """
    Membuat alpha mask dengan soft edges (terinspirasi CM4 Matting / ViTMatte).
    Riset: step4_cm4_hybrid_refinement.py
    
    Teknik Trimap-like:
    1. Erosi (MinFilter 15) → area 'Pasti Lantai' (core)
    2. Dilasi (MaxFilter 15) → area 'Pasti Background' (outer)
    3. Boundary zone = dilasi - erosi → area ambiguous
    4. Soft mask (GaussianBlur 8) di boundary zone
    5. Gabung: core=255, boundary=soft, outer=0
    
    Referensi:
    - Li et al., "Bridging Composite and Real: Matting", CVPR 2024 (CM4)
    - Yao et al., "ViTMatte: Boosting Image Matting with Pretrained ViTs", 2023
    """
    from PIL import Image, ImageFilter
    
    if mask is None or np.sum(mask) == 0:
        return mask
    
    pil_mask = Image.fromarray(mask, mode="L")
    
    # Erosi & dilasi
    inner_mask = pil_mask.filter(ImageFilter.MinFilter(size=15))
    outer_mask = pil_mask.filter(ImageFilter.MaxFilter(size=15))
    
    # Boundary zone
    boundary_zone = np.array(outer_mask).astype(np.float32) - np.array(inner_mask).astype(np.float32)
    boundary_zone = np.clip(boundary_zone, 0, 255).astype(np.uint8)
    
    # Soft mask di area boundary
    soft_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=8))
    
    # Gabungkan: inner = 255, boundary = soft
    final_np = np.array(inner_mask).astype(np.float32)
    final_np[boundary_zone > 0] = np.array(soft_mask).astype(np.float32)[boundary_zone > 0]
    
    return final_np.astype(np.uint8)
