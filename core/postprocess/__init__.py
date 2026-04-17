"""
core/postprocess/ — Domain Layer: Mask Processing
Re-export semua fungsi publik agar backward compatible.
Import dari core.postprocess tetap bekerja.
"""
from core.postprocess.mask_cleanup import get_largest_cc, fill_floor_bottom
from core.postprocess.mask_refinement import refine_mask_smooth, generate_alpha_mask
from core.postprocess.shadow import extract_shadow_map

__all__ = [
    "get_largest_cc",
    "fill_floor_bottom",
    "refine_mask_smooth",
    "generate_alpha_mask",
    "extract_shadow_map",
]
