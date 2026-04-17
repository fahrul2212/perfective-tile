"""
utils/perspective/trapezoid_fitting.py
─────────────────────────────────────────────────────────────────
Smart Trapezoid Fitting: Mask-Weighted Asymmetric Shift.
Menyesuaikan 4 titik trapezoid agar menutupi seluruh SAM3 mask.

Referensi:
  - Adaptive bounding polygon fitting, geometric optimization
"""
import cv2
import numpy as np
import math


def smart_trapezoid_fitting(mask_sam3: np.ndarray, pts: dict) -> dict:
    """
    Menyesuaikan 4 titik trapezoid agar menutupi seluruh area SAM3 mask.
    Menggunakan Mask-Weighted Asymmetric Shift:
    - Garis atas dinaikkan sampai semua mask tercover
    - Sisi kiri/kanan diperlebar dengan bobot proporsional (atas vs bawah)
    """
    canvas_h = pts["canvas_h"]
    canvas_w = pts["canvas_w"]
    shift_x = pts["shift_x"]
    shift_y = pts["shift_y"]

    h_orig, w_orig = mask_sam3.shape[:2]
    mask_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask_canvas[shift_y:shift_y+h_orig, shift_x:shift_x+w_orig] = mask_sam3

    P_TL = list(map(float, pts["C_TL"]))
    P_TR = list(map(float, pts["C_TR"]))
    P_BL = list(map(float, pts["C_BL"]))
    P_BR = list(map(float, pts["C_BR"]))

    PIXEL_TOLERANCE = 10 * max(canvas_w, canvas_h)
    MAX_OUTER_LOOP = 5
    step_top = 5
    step_side = 3
    alpha_shift = 2.0

    def normalize(v):
        norm = math.hypot(*v)
        return (v[0]/norm, v[1]/norm) if norm > 0 else (0, 0)

    for outer in range(MAX_OUTER_LOOP):
        poly = np.array([P_TL, P_TR, P_BR, P_BL], dtype=np.int32)
        mask_trap = np.zeros_like(mask_canvas)
        cv2.fillPoly(mask_trap, [poly], 255)

        uncovered = cv2.subtract(mask_canvas, mask_trap)
        uncovered_count = np.count_nonzero(uncovered)

        if uncovered_count <= PIXEL_TOLERANCE:
            break

        # Naikkan garis atas
        max_iter_top = (canvas_h // step_top) + 1
        for _ in range(max_iter_top):
            d_top = (P_TR[0] - P_TL[0], P_TR[1] - P_TL[1])
            d_top_norm = normalize(d_top)
            n_up = (d_top_norm[1], -d_top_norm[0])

            ys, xs = np.where(mask_canvas > 0)
            if len(xs) == 0: break

            dist_up = (xs - P_TL[0]) * n_up[0] + (ys - P_TL[1]) * n_up[1]
            if not np.any(dist_up > 0):
                break

            new_TL_y = P_TL[1] + step_top * n_up[1]
            new_TR_y = P_TR[1] + step_top * n_up[1]
            if new_TL_y < 0 or new_TR_y < 0:
                break

            P_TL[0] += step_top * n_up[0]; P_TL[1] += step_top * n_up[1]
            P_TR[0] += step_top * n_up[0]; P_TR[1] += step_top * n_up[1]

        # Perlebar sisi kiri & kanan
        for _ in range(2000):
            d_L = (P_BL[0] - P_TL[0], P_BL[1] - P_TL[1])
            d_L_norm = normalize(d_L)
            n_left = (-d_L_norm[1], d_L_norm[0])

            d_R = (P_BR[0] - P_TR[0], P_BR[1] - P_TR[1])
            d_R_norm = normalize(d_R)
            n_right = (d_R_norm[1], -d_R_norm[0])

            poly = np.array([P_TL, P_TR, P_BR, P_BL], dtype=np.int32)
            mask_trap = np.zeros_like(mask_canvas)
            cv2.fillPoly(mask_trap, [poly], 255)

            uncovered = cv2.subtract(mask_canvas, mask_trap)
            ys_unc, xs_unc = np.where(uncovered > 0)
            if len(xs_unc) == 0:
                break

            dist_left = (xs_unc - P_TL[0]) * n_left[0] + (ys_unc - P_TL[1]) * n_left[1]
            ada_kiri = np.sum(dist_left > 0) > 10

            dist_right = (xs_unc - P_TR[0]) * n_right[0] + (ys_unc - P_TR[1]) * n_right[1]
            ada_kanan = np.sum(dist_right > 0) > 10

            if not ada_kiri and not ada_kanan:
                break

            midline_y = (P_TL[1] + P_TR[1] + P_BL[1] + P_BR[1]) / 4.0
            ys_mask, _ = np.where(mask_canvas > 0)
            count_T = np.sum(ys_mask < midline_y)
            count_B = np.sum(ys_mask >= midline_y)
            total_mask = count_T + count_B

            weight_T, weight_B = 0.5, 0.5
            if total_mask > 0 and count_T > 0 and count_B > 0:
                weight_T = count_T / total_mask
                weight_B = count_B / total_mask

            step_T = step_side * weight_T * alpha_shift
            step_B = step_side * weight_B * alpha_shift

            if ada_kiri:
                if P_TL[0] + step_T * n_left[0] <= 0 or P_BL[0] + step_B * n_left[0] <= 0:
                    ada_kiri = False
                else:
                    P_TL[0] += step_T * n_left[0]; P_TL[1] += step_T * n_left[1]
                    P_BL[0] += step_B * n_left[0]

            if ada_kanan:
                if P_TR[0] + step_T * n_right[0] >= canvas_w-1 or P_BR[0] + step_B * n_right[0] >= canvas_w-1:
                    ada_kanan = False
                else:
                    P_TR[0] += step_T * n_right[0]; P_TR[1] += step_T * n_right[1]
                    P_BR[0] += step_B * n_right[0]

            if not ada_kiri and not ada_kanan:
                break

    n_pts = pts.copy()
    n_pts["C_TL"] = tuple(map(int, P_TL))
    n_pts["C_TR"] = tuple(map(int, P_TR))
    n_pts["C_BL"] = tuple(map(int, P_BL))
    n_pts["C_BR"] = tuple(map(int, P_BR))

    def c2i(pt): return (pt[0] - shift_x, pt[1] - shift_y)
    n_pts["P_TL"] = c2i(n_pts["C_TL"])
    n_pts["P_TR"] = c2i(n_pts["C_TR"])
    n_pts["P_BL"] = c2i(n_pts["C_BL"])
    n_pts["P_BR"] = c2i(n_pts["C_BR"])

    return n_pts
