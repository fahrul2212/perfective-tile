import numpy as np
import cv2

def get_largest_cc(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""
    if mask is None or np.sum(mask) == 0:
        return mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1: # Only background
        return mask
    # stats[:, 4] is the 'area' of each component. Skip label 0 (background).
    largest_label = 1 + np.argmax(stats[1:, 4])
    mask_cleaned = np.zeros_like(mask)
    mask_cleaned[labels == largest_label] = 255
    return mask_cleaned

def fill_floor_bottom(mask: np.ndarray) -> np.ndarray:
    """
    Mengisi seluruh area di bawah mask lantai agar menutupi bagian bawah gambar sepenuhnya.
    """
    if mask is None or np.sum(mask) == 0:
        return mask
    h, w = mask.shape
    refined = mask.copy()
    # 1. Row-wise fill: Isi celah horizontal
    for y in range(h):
        row = mask[y, :]
        nz = np.where(row > 0)[0]
        if len(nz) > 1:
            refined[y, nz[0]:nz[-1]+1] = 255
    # 2. Column-wise fill: Tarik ke bawah hingga ujung gambar
    for x in range(w):
        col = refined[:, x]
        nz = np.where(col > 0)[0]
        if len(nz) > 0:
            refined[nz[0]:, x] = 255
    return refined


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


def extract_shadow_map(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Ekstraksi peta bayangan (shadow map) pada area lantai.
    Riset: step3_extract_shadow_on_floor.py
    
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
