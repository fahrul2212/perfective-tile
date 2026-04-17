"""
utils/perspective/detect_points.py
─────────────────────────────────────────────────────────────────
Deteksi 4 titik perspektif dari mask lantai ST_RoomNet.
Menggunakan canvas 3x untuk mengakomodasi titik yang keluar batas gambar.

Referensi:
  - Hartley & Zisserman, "Multiple View Geometry in CV" (2004), Ch.2
"""
import cv2
import numpy as np
import math


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
