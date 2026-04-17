"""
core/postprocess/shadow.py
─────────────────────────────────────────────────────────────────
Shadow extraction dari area lantai.
Porting dari riset: tests/segmentasi-perfect/step3_extract_shadow_on_floor.py

Referensi:
  - Finlayson et al., "Removing Shadows from Images", ECCV 2002
"""
import numpy as np
import cv2


def extract_shadow_map(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Ekstraksi peta bayangan (shadow map) pada area lantai.
    
    Teknik:
    1. Estimasi base floor lightness dari percentile 80 (menghindari highlight)
    2. Shadow map = selisih kegelapan (base - pixel)
    3. Threshold < 45 untuk membersihkan tekstur lantai
    4. Gamma correction (power 2.0) untuk memperkuat bayangan nyata
    5. Gaussian blur untuk soft shadow (simulasi penumbra)
    
    Referensi:
    - Finlayson et al., "Removing Shadows from Images", ECCV 2002
    - Shadow = deviation dari estimasi diffuse floor color
    """
    from PIL import Image, ImageFilter
    
    if mask is None or np.sum(mask) == 0:
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    
    # Konversi ke grayscale (luminance)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask_f = mask.astype(np.float32) / 255.0
    
    # Pixel lantai saja
    floor_pixels = img_gray[mask_f > 0.5]
    if len(floor_pixels) == 0:
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    
    # Estimasi cahaya lantai tanpa bayangan (percentile 80)
    base_floor_lightness = np.percentile(floor_pixels, 80)
    
    # Shadow map
    shadow_map = base_floor_lightness - img_gray
    shadow_map[shadow_map < 45] = 0  # Bersihkan tekstur
    shadow_map = np.clip(shadow_map, 0, 255)
    
    # Gamma correction — perkuat bayangan nyata
    if shadow_map.max() > 0:
        shadow_map = (shadow_map / shadow_map.max()) ** 2.0 * 255.0
    
    # Masking — hanya di area lantai
    shadow_map = shadow_map * mask_f
    
    # Gaussian blur untuk soft shadow
    shadow_pil = Image.fromarray(shadow_map.astype(np.uint8), mode="L")
    shadow_pil = shadow_pil.filter(ImageFilter.GaussianBlur(radius=5))
    
    return np.array(shadow_pil)
