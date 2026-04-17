"""
utils/perspective/grid.py
─────────────────────────────────────────────────────────────────
Dynamic grid sizing untuk tile keramik berdasarkan ukuran trapezoid.
"""
import numpy as np


def calc_cols_rows(pts):
    """
    Hitung jumlah kolom dan baris tile berdasarkan ukuran trapezoid.
    Menggunakan rasio perspektif (w_bottom / w_top) untuk foreshortening yang realistis.
    """
    C_BL = np.array(pts["C_BL"])
    C_BR = np.array(pts["C_BR"])
    C_TL = np.array(pts["C_TL"])
    C_TR = np.array(pts["C_TR"])

    w_b = np.linalg.norm(C_BR - C_BL)
    w_t = np.linalg.norm(C_TR - C_TL)

    w_orig = pts["canvas_w"] / 3.0

    # Base 2.4 → tile besar, nuansa mewah ubin 60x60
    base_cols_for_full_width = 2.4

    cols = max(3, int(round((w_b / w_orig) * base_cols_for_full_width)))
    if cols > 20: cols = 20

    depth_ratio = w_b / max(1.0, w_t)
    rows = max(3, int(round(cols * depth_ratio * 0.55)))
    if rows > 35: rows = 35

    return cols, rows
