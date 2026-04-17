"""
utils/perspective/ — Infrastructure Layer: VTO Rendering Tools
Re-export fungsi utama untuk backward compatibility.
"""
from utils.perspective.detect_points import detect_4_points
from utils.perspective.trapezoid_fitting import smart_trapezoid_fitting
from utils.perspective.grid import calc_cols_rows
from utils.perspective.renderer import render_ceramic_perspective, render_tile_fast, _resolve_tile_path

__all__ = [
    "detect_4_points",
    "smart_trapezoid_fitting",
    "calc_cols_rows",
    "render_ceramic_perspective",
]
