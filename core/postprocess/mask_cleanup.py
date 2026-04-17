"""
core/postprocess/mask_cleanup.py
─────────────────────────────────────────────────────────────────
Fungsi-fungsi pembersihan dasar mask lantai.
Bertanggung jawab atas: noise removal (connected component) dan fill-bottom.

Ini adalah fungsi ASLI yang sudah stabil — TIDAK diubah logikanya.
"""
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
