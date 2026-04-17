"""
utils/perspective.py
─────────────────────────────────────────────────────────────────
Modul utilitas untuk rendering tile keramik perspektif (VTO system).

Upgrade dari versi lama:
  - SAM3 mask clipping (tile hanya muncul di area lantai SAM3)
  - Smart Trapezoid Fitting (mask-weighted asymmetric shift)
  - Dynamic grid sizing (cols/rows proporsional terhadap trapezoid)
  - Grout rendering sebelum warp (lebih realistis)

Sumber riset & porting:
  - tests/VTO/test_homography_vs_sam3.backup_perfect.py
  - tests/segmentasi-perfect/step2_refine_mask.py (mask smoothing)

Referensi akademis:
  - Hartley & Zisserman, "Multiple View Geometry in CV" (2004), Ch.2: Projective Geometry
  - OpenCV getPerspectiveTransform — 4-point homography estimation
"""

import cv2
import numpy as np
import math
import os


# ================================================================
# 1. DETEKSI 4 TITIK PERSPEKTIF
# ================================================================

def detect_4_points(mask_orig: np.ndarray):
    """
    Deteksi P_TL, P_TR, P_BL, P_BR dalam IMAGE coords (bukan canvas).
    Menggunakan canvas 3x untuk mengakomodasi titik yang keluar batas gambar.
    
    Return dict { P_TL, P_TR, P_BL, P_BR, C_TL, ..., canvas metadata } atau None.
    """
    h_orig, w_orig = mask_orig.shape
    canvas_h, canvas_w = h_orig * 3, w_orig * 3
    shift_y = (canvas_h - h_orig) // 2
    shift_x = (canvas_w - w_orig) // 2

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas[shift_y:shift_y + h_orig, shift_x:shift_x + w_orig] = mask_orig

    # Tepi kiri & kanan
    l_edge = np.where(mask_orig[:, 0] > 0)[0]
    tkiri  = (shift_x, shift_y + l_edge[0]) if len(l_edge) > 0 else None
    r_edge = np.where(mask_orig[:, w_orig - 1] > 0)[0]
    tkanan = (shift_x + w_orig - 1, shift_y + r_edge[0]) if len(r_edge) > 0 else None

    cnts, _ = cv2.findContours(canvas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mid_pts = []
    if cnts:
        cnt   = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True).reshape(-1, 2)
        min_y = min(approx[:, 1])
        y_thr = min_y + (0.1 * h_orig)
        for pt in approx:
            if (shift_x + 5) < pt[0] < (shift_x + w_orig - 6) and pt[1] <= y_thr:
                mid_pts.append(tuple(map(int, pt)))
    mid_pts = sorted(list(set(mid_pts)), key=lambda p: p[0])
    t1 = mid_pts[0] if mid_pts else None
    t2 = mid_pts[1] if len(mid_pts) > 1 else None

    green_lines = []
    if t1 and t2:
        green_lines.append([t1, t2, "garis atas"])
        if tkiri:  green_lines.append([tkiri, t1,     "garis pinggir kiri"])
        if tkanan: green_lines.append([t2,    tkanan,  "garis pinggir kanan"])
    elif t1:
        dist_l = math.hypot(t1[0]-tkiri[0],  t1[1]-tkiri[1])  if tkiri  else 1e9
        dist_r = math.hypot(t1[0]-tkanan[0], t1[1]-tkanan[1]) if tkanan else 1e9
        if dist_l < dist_r:
            green_lines.append([tkiri, t1, "garis atas"])
            if tkanan: green_lines.append([t1, tkanan, "garis pinggir kanan"])
        else:
            green_lines.append([t1, tkanan, "garis atas"])
            if tkiri: green_lines.append([tkiri, t1, "garis pinggir kiri"])

    for i, (p1, p2, label) in enumerate(green_lines):
        if "atas" in label and abs(p1[1] - p2[1]) < (0.05 * h_orig):
            hy = min(p1[1], p2[1])
            green_lines[i][0] = (p1[0], hy)
            green_lines[i][1] = (p2[0], hy)
            for j, (_, _, slabel) in enumerate(green_lines):
                if "pinggir kiri"  in slabel: green_lines[j][1] = green_lines[i][0]
                if "pinggir kanan" in slabel: green_lines[j][0] = green_lines[i][1]

    side_l   = next((l for l in green_lines if "pinggir kiri"  in l[2]), None)
    side_r   = next((l for l in green_lines if "pinggir kanan" in l[2]), None)
    top_line = next((l for l in green_lines if "garis atas"    in l[2]), None)

    P_TL = P_TR = P_BL = P_BR = None
    if top_line:
        P_TL, P_TR, _ = top_line
        if side_l and not side_r:
            junc, bot, gb = side_l[1], side_l[0], P_TR
            vt = (gb[0]-junc[0], gb[1]-junc[1]); vs = (bot[0]-junc[0], bot[1]-junc[1])
            ab = math.atan2(vt[0]*vs[1]-vt[1]*vs[0], vt[0]*vs[0]+vt[1]*vs[1])
            sl = math.hypot(*vs)
            vr = (junc[0]-gb[0], junc[1]-gb[1])
            ag = math.atan2(vr[1], vr[0]) - ab
            P_BL = bot; P_BR = (int(gb[0]+sl*math.cos(ag)), int(gb[1]+sl*math.sin(ag)))
        elif side_r and not side_l:
            junc, bot, gb = side_r[0], side_r[1], P_TL
            vt = (gb[0]-junc[0], gb[1]-junc[1]); vs = (bot[0]-junc[0], bot[1]-junc[1])
            ab = math.atan2(vt[0]*vs[1]-vt[1]*vs[0], vt[0]*vs[0]+vt[1]*vs[1])
            sl = math.hypot(*vs)
            vr = (junc[0]-gb[0], junc[1]-gb[1])
            ag = math.atan2(vr[1], vr[0]) - ab
            P_BL = (int(gb[0]+sl*math.cos(ag)), int(gb[1]+sl*math.sin(ag))); P_BR = bot
        elif side_l and side_r:
            P_BL, P_BR = side_l[0], side_r[1]

    # Sweeping fit-bottom
    if P_TL and P_TR and P_BL and P_BR and cnts:
        cnt = max(cnts, key=cv2.contourArea)
        cnt_pts = cnt.reshape(-1, 2)
        top_y = min(P_TL[1], P_TR[1])
        below = cnt_pts[cnt_pts[:, 1] > top_y]
        if len(below) > 0:
            dT = (float(P_TR[0]-P_TL[0]), float(P_TR[1]-P_TL[1]))
            dL = (float(P_BL[0]-P_TL[0]), float(P_BL[1]-P_TL[1]))
            dR = (float(P_BR[0]-P_TR[0]), float(P_BR[1]-P_TR[1]))
            NT = (-dT[1], dT[0])
            if NT[1] < 0: NT = (-NT[0], -NT[1])
            maxd = -float('inf'); fur = None
            for pt in below:
                d = pt[0]*NT[0]+pt[1]*NT[1]
                if d > maxd: maxd, fur = d, (float(pt[0]), float(pt[1]))
            def _ix(p1, d1, p2, d2):
                det = d1[0]*(-d2[1]) - (-d2[0])*d1[1]
                if abs(det) < 1e-6: return None
                dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                k = (dx*-d2[1] - (-d2[0])*dy) / det
                return (p1[0]+k*d1[0], p1[1]+k*d1[1])
            if fur:
                fBL = _ix(fur, dT, P_TL, dL); fBR = _ix(fur, dT, P_TR, dR)
                if fBL and fBR:
                    P_BL = (int(fBL[0]), int(fBL[1])); P_BR = (int(fBR[0]), int(fBR[1]))

    # ── Side-line adjustment ────────────────────────
    def _check_line_touches(p_top, p_bot, mask_canvas, margin_y=20):
        x0, y0 = int(p_top[0]), int(p_top[1])
        x1, y1 = int(p_bot[0]), int(p_bot[1])
        dy = abs(y1 - y0)
        n = max(100, int(dy))
        if n == 0: return True
        xs = np.linspace(x0, x1, n+1).astype(np.int32)
        ys = np.linspace(y0, y1, n+1).astype(np.int32)
        xs, ys = xs[ys >= (y0 + margin_y)], ys[ys >= (y0 + margin_y)]
        hm, wm = mask_canvas.shape[:2]
        ok = (xs >= 0) & (xs < wm) & (ys >= 0) & (ys < hm)
        xs, ys = xs[ok], ys[ok]
        if len(xs) == 0: return False
        return bool(np.any(mask_canvas[ys, xs] > 0))

    def _intersect(p1, d1, p2, d2):
        det = d1[0]*(-d2[1]) - (-d2[0])*d1[1]
        if abs(det) < 1e-6: return None
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        k = (dx*-d2[1] - (-d2[0])*dy) / det
        return (p1[0]+k*d1[0], p1[1]+k*d1[1])

    if P_TL and P_TR and P_BL and P_BR:
        MAX_ITER = 3000
        d_B = (float(P_BR[0]-P_BL[0]), float(P_BR[1]-P_BL[1]))
        base_bl = (float(P_BL[0]), float(P_BL[1]))
        base_br = (float(P_BR[0]), float(P_BR[1]))

        # Kiri
        tl_x, tl_y = float(P_TL[0]), float(P_TL[1])
        t_bl_x, t_bl_y = base_bl
        cur_bl = base_bl
        iter_l = 0
        while not _check_line_touches((int(tl_x), int(tl_y)),
                                      (int(cur_bl[0]), int(cur_bl[1])), canvas):
            if iter_l > MAX_ITER or cur_bl[0] >= (canvas_w // 2): break
            tl_x -= 1;  t_bl_x += 10
            d_L_new = (t_bl_x - tl_x, t_bl_y - tl_y)
            res = _intersect((tl_x, tl_y), d_L_new, base_bl, d_B)
            cur_bl = res if res else (t_bl_x, t_bl_y)
            iter_l += 1
        if iter_l > 0:
            P_TL = (int(tl_x), int(tl_y))
            P_BL = (int(cur_bl[0]), int(cur_bl[1]))

        # Kanan
        tr_x, tr_y = float(P_TR[0]), float(P_TR[1])
        t_br_x, t_br_y = base_br
        cur_br = base_br
        iter_r = 0
        while not _check_line_touches((int(tr_x), int(tr_y)),
                                      (int(cur_br[0]), int(cur_br[1])), canvas):
            if iter_r > MAX_ITER or cur_br[0] <= (canvas_w // 2): break
            tr_x += 1;  t_br_x -= 10
            d_R_new = (t_br_x - tr_x, t_br_y - tr_y)
            res = _intersect((tr_x, tr_y), d_R_new, base_br, d_B)
            cur_br = res if res else (t_br_x, t_br_y)
            iter_r += 1
        if iter_r > 0:
            P_TR = (int(tr_x), int(tr_y))
            P_BR = (int(cur_br[0]), int(cur_br[1]))

    if not (P_TL and P_TR and P_BL and P_BR):
        return None

    def c2i(pt): return (pt[0] - shift_x, pt[1] - shift_y)

    return {
        "P_TL": c2i(P_TL), "P_TR": c2i(P_TR),
        "P_BL": c2i(P_BL), "P_BR": c2i(P_BR),
        "C_TL": P_TL, "C_TR": P_TR,
        "C_BL": P_BL, "C_BR": P_BR,
        "shift_x": shift_x, "shift_y": shift_y,
        "canvas_w": canvas_w, "canvas_h": canvas_h,
    }


# ================================================================
# 2. SMART TRAPEZOID FITTING (Mask-Weighted Asymmetric Shift)
# ================================================================

def smart_trapezoid_fitting(mask_sam3: np.ndarray, pts: dict) -> dict:
    """
    Menyesuaikan 4 titik trapezoid agar menutupi seluruh area SAM3 mask.
    Menggunakan Mask-Weighted Asymmetric Shift:
    - Garis atas dinaikkan sampai semua mask tercover
    - Sisi kiri/kanan diperlebar dengan bobot proporsional (atas vs bawah)
    
    Referensi:
    - Adaptive bounding polygon fitting, geometric optimization
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


# ================================================================
# 3. DYNAMIC GRID SIZING
# ================================================================

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


# ================================================================
# 4. RESOLVE TILE PATH
# ================================================================

def _resolve_tile_path(tile_path: str):
    """Resolve tile asset path: try as-is, then relative to project root."""
    from pathlib import Path as _Path
    ROOT = _Path(__file__).parent.parent

    p = _Path(tile_path)
    if p.exists():
        return str(p)
    p2 = ROOT / tile_path
    if p2.exists():
        return str(p2)
    return tile_path


# ================================================================
# 5. RENDER CERAMIC PERSPECTIVE (MAIN VTO FUNCTION)
# ================================================================

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
