"""
test_homography_vs_sam3.py
─────────────────────────────────────────────────────────────────
Test untuk memvisualisasikan ALIGNMENT antara:
  - Trapezoid Homography dari ST_RoomNet (4 titik perspektif)
  - Mask SAM3 (boundary lantai organik dari microservice :8001)

TIDAK ada warp tile / keramik di sini. Ini murni visualisasi
apakah 4 titik perspektif ST_RoomNet "duduk" di atas SAM3 mask
dengan benar.

Cara pakai:
    # Pastikan SAM3 service sudah berjalan (port 8001)
    cd C:/Project/simpel
    .venv/Scripts/python tests/VTO/test_homography_vs_sam3.py --img assets/1.jpg

Output: tests/VTO/outputs/
    *_hom_vs_sam3.jpg   → overlay trapezoid di atas SAM3 mask
    *_debug_full.jpg    → 4 panel: original | mask_st | sam3_mask | overlay
"""

import sys
import argparse
import time
import math
import traceback
from pathlib import Path

import cv2
import numpy as np
import requests          # sync HTTP — tidak butuh asyncio di script test
import torch
import torch.nn.functional as F

# ── Tambah root project ke path ─────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Import HANYA untuk inference & postprocess — tidak ada perubahan ke file itu
from core.config    import Config
from core.model     import ST_RoomNet
from core.postprocess import get_largest_cc, fill_floor_bottom


# ================================================================
# BAGIAN 1: Deteksi 4 Titik Perspektif (inline, TANPA warp tile)
#           Sama dengan utils/perspective.py tapi hanya ambil P_TL..P_BR
# ================================================================

def detect_4_points(mask_orig: np.ndarray):
    """
    Deteksi P_TL, P_TR, P_BL, P_BR dalam IMAGE coords (bukan canvas).
    Return dict { P_TL, P_TR, P_BL, P_BR } atau None jika gagal.
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

    # ── 5b. SIDE-LINE ADJUSTMENT (titik mendekat) ────────────────────────
    # Sama persis dengan utils/perspective.py step 5b
    # Ketika garis sisi tidak menyentuh mask → geser titik sampai menyentuh
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

        # Kiri: geser TL ke kiri, BL ke kanan sampai garis menyentuh mask
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
            print(f"[detect_4_points] Side-adj kiri: {iter_l} iterasi → P_TL={P_TL} P_BL={P_BL}")

        # Kanan: geser TR ke kanan, BR ke kiri sampai garis menyentuh mask
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
            print(f"[detect_4_points] Side-adj kanan: {iter_r} iterasi → P_TR={P_TR} P_BR={P_BR}")

    if not (P_TL and P_TR and P_BL and P_BR):
        return None

    # Konversi dari canvas coords → image coords
    def c2i(pt): return (pt[0] - shift_x, pt[1] - shift_y)

    return {
        # image coords (untuk overlay di foto asli)
        "P_TL": c2i(P_TL), "P_TR": c2i(P_TR),
        "P_BL": c2i(P_BL), "P_BR": c2i(P_BR),
        # canvas coords (untuk canvas 3x view)
        "C_TL": P_TL, "C_TR": P_TR,
        "C_BL": P_BL, "C_BR": P_BR,
        "shift_x": shift_x, "shift_y": shift_y,
        "canvas_w": canvas_w, "canvas_h": canvas_h,
    }


# ================================================================
# BAGIAN 2: Inference ST_RoomNet (sync, minimal — hanya untuk test)
# ================================================================

def load_roomnet():
    """Load model ST_RoomNet ke device. Return model."""
    print("[TEST] Loading ST_RoomNet...")
    model = ST_RoomNet(
        ref_path=Config.REF_IMAGE_PATH,
        out_size=(Config.INPUT_SIZE, Config.INPUT_SIZE)
    )
    model.load_state_dict(
        torch.load(Config.WEIGHT_PATH, map_location=Config.DEVICE, weights_only=True)
    )
    model.to(Config.DEVICE)
    if Config.DEVICE.type == "cuda":
        model.half()
    model.eval()
    print(f"[TEST] Model siap di: {Config.DEVICE}")
    return model


def predict_mask(model, img_bgr: np.ndarray):
    """
    Jalankan inference ST_RoomNet → kembalikan mask_cleaned (H, W) uint8.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        img_gpu = (
            torch.from_numpy(img_rgb)
            .to(Config.DEVICE).float()
            .permute(2, 0, 1) / 255.0
        )
        inp = F.interpolate(
            img_gpu.unsqueeze(0),
            size=(Config.INPUT_SIZE, Config.INPUT_SIZE),
            mode="bilinear", align_corners=False,
        )
        if Config.DEVICE.type == "cuda":
            inp = inp.half()

        out = model(inp)

    dist = torch.abs(out - 4.0)
    mask_soft = torch.clamp(1.0 - dist, min=0.0)
    mask_soft = F.avg_pool2d(mask_soft, kernel_size=3, stride=1, padding=1)
    mask_up   = F.interpolate(mask_soft, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    mask_cpu  = ((mask_up[0, 0] > 0.5).byte() * 255).cpu().numpy()

    # Postprocess (persis seperti di app.py)
    mask_cleaned = get_largest_cc(mask_cpu)
    mask_cleaned = fill_floor_bottom(mask_cleaned)
    return mask_cleaned


# ================================================================
# BAGIAN 3: Panggil SAM3 (sync HTTP requests — tidak butuh asyncio)
# ================================================================

def call_sam3(img_bgr: np.ndarray, sam3_url: str = None) -> np.ndarray | None:
    """
    Kirim gambar ke SAM3 microservice, kembalikan mask (H, W) uint8 atau None.
    sam3_url default: http://localhost:8001/predict/floor
    """
    if sam3_url is None:
        sam3_url = f"{Config.SAM3_BASE_URL}/predict/floor"

    _, img_enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_bytes  = img_enc.tobytes()

    print(f"[TEST] Mengirim ke SAM3: {sam3_url} ...")
    try:
        resp = requests.post(
            sam3_url,
            files={"file": ("image.jpg", img_bytes, "image/jpeg")},
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"[TEST] SAM3 error {resp.status_code}: {resp.text[:200]}")
            return None

        # Response adalah bytes PNG/JPG dari mask
        nparr     = np.frombuffer(resp.content, np.uint8)
        mask_sam3 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if mask_sam3 is None:
            print("[TEST] SAM3: gagal decode response mask.")
            return None

        print(f"[TEST] SAM3 mask diterima: {mask_sam3.shape}  "
              f"pixel lantai: {np.sum(mask_sam3 > 0):,}")
        return mask_sam3

    except requests.exceptions.ConnectionError:
        print("[TEST] SAM3 tidak tersedia (ConnectionError). "
              "Jalankan api-sam3 terlebih dahulu (port 8001).")
        return None
    except requests.exceptions.Timeout:
        print("[TEST] SAM3 timeout.")
        return None


# ================================================================
# BAGIAN 3.5: SMART TRAPEZOID FITTING (RISET MASK-WEIGHTED SHIFT)
# ================================================================

def smart_trapezoid_fitting(mask_sam3: np.ndarray, pts: dict) -> dict:
    """
    Menyesuaikan kembali 4 titik trapezoid berdasarkan mask_sam3
    menggunakan logika Mask-Weighted Asymmetric Shift.
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
        # ── STEP 1: CEK COVERAGE ────────────────────
        poly = np.array([P_TL, P_TR, P_BR, P_BL], dtype=np.int32)
        mask_trap = np.zeros_like(mask_canvas)
        cv2.fillPoly(mask_trap, [poly], 255)

        uncovered = cv2.subtract(mask_canvas, mask_trap)
        uncovered_count = np.count_nonzero(uncovered)

        print(f"[SmartFit] Outer {outer}: Uncovered pixel = {uncovered_count}")
        if uncovered_count <= PIXEL_TOLERANCE:
            print("[SmartFit] Coverage memenuhi toleransi.")
            break

        # ── STEP 2: NAIKKAN GARIS ATAS (Standar) ────────────────
        max_iter_top = (canvas_h // step_top) + 1
        for iter_top in range(max_iter_top):
            d_top = (P_TR[0] - P_TL[0], P_TR[1] - P_TL[1])
            d_top_norm = normalize(d_top)
            # Normal ke ATAS di origin TL (+x kanan, +y bawah) => n_up = (dy, -dx)
            n_up = (d_top_norm[1], -d_top_norm[0])

            ys, xs = np.where(mask_canvas > 0)
            if len(xs) == 0: break

            # P_TL ke titik (xs,ys) DOT n_up > 0 (artinya ada mask di 'atas' trapezoid)
            dist_up = (xs - P_TL[0]) * n_up[0] + (ys - P_TL[1]) * n_up[1]
            if not np.any(dist_up > 0):
                break

            new_TL_y = P_TL[1] + step_top * n_up[1]
            new_TR_y = P_TR[1] + step_top * n_up[1]

            if new_TL_y < 0 or new_TR_y < 0:
                break # Titik atas sudah mentok batas paling atas canvas y=0

            P_TL[0] += step_top * n_up[0]; P_TL[1] += step_top * n_up[1]
            P_TR[0] += step_top * n_up[0]; P_TR[1] += step_top * n_up[1]
            # P_BL dan P_BR tidak diubah agar garis BAWAH tetap di tempat aslinya!

        # ── STEP 3 & 4: PERLEBAR SISI KIRI & KANAN (Mask-Weighted Shift) ────────────────
        for iter_side in range(2000):
            d_L = (P_BL[0] - P_TL[0], P_BL[1] - P_TL[1])
            d_L_norm = normalize(d_L)
            # n_left = arah KIRI <=> rotasi d_L (+y ke bawah) sebesar 90deg ke kiri (-dy, dx)
            n_left = (-d_L_norm[1], d_L_norm[0])

            d_R = (P_BR[0] - P_TR[0], P_BR[1] - P_TR[1])
            d_R_norm = normalize(d_R)
            # n_right = arah KANAN <=> rotasi d_R 90deg ke kanan (dy, -dx)
            n_right = (d_R_norm[1], -d_R_norm[0])

            poly = np.array([P_TL, P_TR, P_BR, P_BL], dtype=np.int32)
            mask_trap = np.zeros_like(mask_canvas)
            cv2.fillPoly(mask_trap, [poly], 255)

            uncovered = cv2.subtract(mask_canvas, mask_trap)
            ys_unc, xs_unc = np.where(uncovered > 0)
            if len(xs_unc) == 0:
                break # Tidak ada uncovered

            dist_left = (xs_unc - P_TL[0]) * n_left[0] + (ys_unc - P_TL[1]) * n_left[1]
            ada_kiri = np.sum(dist_left > 0) > 10

            dist_right = (xs_unc - P_TR[0]) * n_right[0] + (ys_unc - P_TR[1]) * n_right[1]
            ada_kanan = np.sum(dist_right > 0) > 10

            if not ada_kiri and not ada_kanan:
                break
                
            # Hitung Weight T vs B (Asymmetric Shift) untuk WIDENING
            midline_y = (P_TL[1] + P_TR[1] + P_BL[1] + P_BR[1]) / 4.0
            ys_mask, _ = np.where(mask_canvas > 0)
            count_T = np.sum(ys_mask < midline_y)
            count_B = np.sum(ys_mask >= midline_y)
            total_mask = count_T + count_B

            weight_T = 0.5
            weight_B = 0.5
            if total_mask > 0 and count_T > 0 and count_B > 0:
                weight_T = count_T / total_mask
                weight_B = count_B / total_mask

            step_T = step_side * weight_T * alpha_shift
            step_B = step_side * weight_B * alpha_shift

            if ada_kiri:
                if P_TL[0] + step_T * n_left[0] <= 0 or P_BL[0] + step_B * n_left[0] <= 0:
                    ada_kiri = False # MENTOK KIRI
                else:
                    P_TL[0] += step_T * n_left[0]; P_TL[1] += step_T * n_left[1]
                    P_BL[0] += step_B * n_left[0]
                    # P_BL[1] TETAP agar garis bawah tidak bergeser dari posisinya!

            if ada_kanan:
                if P_TR[0] + step_T * n_right[0] >= canvas_w-1 or P_BR[0] + step_B * n_right[0] >= canvas_w-1:
                    ada_kanan = False # MENTOK KANAN
                else:
                    P_TR[0] += step_T * n_right[0]; P_TR[1] += step_T * n_right[1]
                    P_BR[0] += step_B * n_right[0]
                    # P_BR[1] TETAP agar garis bawah tidak bergeser dari posisinya!
            
            if not ada_kiri and not ada_kanan:
                break

    n_pts = pts.copy()
    n_pts["C_TL"] = tuple(map(int, P_TL))
    n_pts["C_TR"] = tuple(map(int, P_TR))
    n_pts["C_BL"] = tuple(map(int, P_BL))
    n_pts["C_BR"] = tuple(map(int, P_BR))

    # Konversi balik ke coordinate gambar asli
    def c2i(pt): return (pt[0] - shift_x, pt[1] - shift_y)
    n_pts["P_TL"] = c2i(n_pts["C_TL"])
    n_pts["P_TR"] = c2i(n_pts["C_TR"])
    n_pts["P_BL"] = c2i(n_pts["C_BL"])
    n_pts["P_BR"] = c2i(n_pts["C_BR"])

    return n_pts


# ================================================================
# BAGIAN 4: Visualisasi — Overlay Trapezoid di Atas SAM3 Mask
# ================================================================

def _resolve_tile_path(tile_path: str):
    """Resolve tile asset: try as-is (relative to CWD), then relative to project ROOT."""
    from pathlib import Path as _Path
    p = _Path(tile_path)
    if p.exists():
        return str(p)
    # If absolute doesn't work, maybe it was relative
    p2 = ROOT / tile_path
    if p2.exists():
        return str(p2)
    return tile_path


def calc_cols_rows(pts):
    """
    Logika untuk menghitung berapa jumlah kolom (cols) dan baris (rows) tile 60x60 
    yang pas sesuai dengan ukuran trapezoid di layar.
    """
    C_BL = np.array(pts["C_BL"])
    C_BR = np.array(pts["C_BR"])
    C_TL = np.array(pts["C_TL"])
    C_TR = np.array(pts["C_TR"])
    
    w_b = np.linalg.norm(C_BR - C_BL)
    w_t = np.linalg.norm(C_TR - C_TL)
    h_trap = ((C_BL[1] + C_BR[1]) / 2.0) - ((C_TL[1] + C_TR[1]) / 2.0)
    
    # Resolusi gambar berbeda-beda (ada yg 4K seperti 1.jpg). 
    # Lebih baik hitung proporsional menggunakan lebar gambar aslinya.
    w_orig = pts["canvas_w"] / 3.0
    
    # Diturunkan lagi ke 2.4 agar jumlah keramik semakin sedikit (ukurannya menjadi super besar).
    # Cocok untuk menampilkan nuansa mewah ubin 60x60.
    base_cols_for_full_width = 2.4
    
    cols = max(3, int(round((w_b / w_orig) * base_cols_for_full_width)))
    if cols > 20: cols = 20
    
    # Foreshortening perspektif yang lebih akurat:
    # Rasio w_b / w_t secara matematis sangat mewakili "kedalaman perspektif".
    depth_ratio = w_b / max(1.0, w_t)
    
    # Pengali empiris 0.55 (sedikit diturunkan) agar baris ke atasnya tetap proporsional bujur sangkar 
    # namun tidak terlalu padat merapat.
    rows = max(3, int(round(cols * depth_ratio * 0.55)))
    if rows > 35: rows = 35
    
    print(f"[Logika Grid] w_b={w_b:.0f}px, w_t={w_t:.0f}px => Cols={cols}, Rows={rows}")
    return cols, rows


# ================================================================
# BAGIAN 3.7: Render Tile Perspektif + Composite ke Foto Asli
# ================================================================

def render_tiles_perspective(
    img_bgr:   np.ndarray,
    pts:       dict,
    mask_sam3: np.ndarray,
    tile_path: str = r"C:\Project\simpel\assets\tile\Concord-60x60-PGC66K001-ALASKA WHITE.jpeg",
) -> np.ndarray:
    """
    Render tile grid dengan perspective warp, dicomposite ke img_bgr,
    hanya menampilkan area yang ter-mask oleh SAM3.
    """
    cols, rows = calc_cols_rows(pts)

    canvas_h = pts["canvas_h"]
    canvas_w = pts["canvas_w"]
    shift_y  = pts["shift_y"]
    shift_x  = pts["shift_x"]
    h_orig, w_orig = img_bgr.shape[:2]

    # Pastikan mask_sam3 pas
    if mask_sam3.shape[:2] != (h_orig, w_orig):
        mask_sam3 = cv2.resize(mask_sam3, (w_orig, h_orig),
                               interpolation=cv2.INTER_NEAREST)
    _, mask_sam3 = cv2.threshold(mask_sam3, 127, 255, cv2.THRESH_BINARY)

    TILE_PX = 500
    tile_w, tile_h = TILE_PX, TILE_PX
    tex_w = cols * tile_w
    tex_h = rows * tile_h

    resolved = _resolve_tile_path(tile_path)
    tile_img = cv2.imread(resolved) if resolved else None

    if tile_img is None:
        print(f"[render_tiles] WARN: tile tidak ditemukan: {tile_path} ")
        tile_img = np.full((tile_h, tile_w, 3), (220, 220, 220), dtype=np.uint8)
    else:
        tile_img = cv2.resize(tile_img, (tile_w, tile_h),
                              interpolation=cv2.INTER_CUBIC)

    # Template texture besar (sheet)
    texture = np.zeros((tex_h, tex_w, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            texture[r * tile_h:(r + 1) * tile_h,
                    c * tile_w:(c + 1) * tile_w] = tile_img

    # Gambar grout (nat)
    GROUT_COLOR     = (80, 78, 75)
    GROUT_THICKNESS = 4
    for c in range(cols + 1):
        x = min(c * tile_w, tex_w - 1)
        cv2.line(texture, (x, 0), (x, tex_h - 1), GROUT_COLOR, GROUT_THICKNESS)
    for r in range(rows + 1):
        y = min(r * tile_h, tex_h - 1)
        cv2.line(texture, (0, y), (tex_w - 1, y), GROUT_COLOR, GROUT_THICKNESS)

    # Matrix 
    src_pts = np.float32([
        [0,         0        ],   # TL
        [tex_w - 1, 0        ],   # TR
        [0,         tex_h - 1],   # BL
        [tex_w - 1, tex_h - 1],   # BR
    ])
    dst_pts = np.float32([
        pts["C_TL"],
        pts["C_TR"],
        pts["C_BL"],
        pts["C_BR"],
    ])

    M_warp = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp texture ke bentuk trapezoid di kanvas besar
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

    # Canvas untuk original photo dan SAM3 mask
    img_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    img_canvas[shift_y:shift_y + h_orig, shift_x:shift_x + w_orig] = img_bgr

    mask_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask_canvas[shift_y:shift_y + h_orig, shift_x:shift_x + w_orig] = mask_sam3

    # Intersection antara WARP tile boundary dan SAM3 mask
    combined_mask = cv2.bitwise_and(warp_mask, mask_canvas)

    # Composite tile as layer mask onto image canvas
    result = img_canvas.copy()
    result[combined_mask > 0] = warped_ceramic[combined_mask > 0]

    # Crop ke resolusi asli
    result_cropped = result[shift_y:shift_y + h_orig, shift_x:shift_x + w_orig]

    print(f"[render_tiles] Selesai — tiles: {cols}×{rows} pixel terisi: {np.count_nonzero(combined_mask):,}")
    return result_cropped

def draw_homography_overlay(
    base_img:  np.ndarray,   # gambar dasar (GS atau BGR)
    pts:       dict,          # { P_TL, P_TR, P_BL, P_BR } dalam image coords
    alpha:     float = 0.55,  # transparansi overlay trapezoid
) -> np.ndarray:
    """
    Draw trapezoid perspektif (dari ST_RoomNet) di atas base_img.
    Menggunakan 2 layer:
      - Fill semi-transparan warna biru di dalam trapezoid
      - Garis tepi dan titik sudut berwarna mencolok
    """
    if base_img.ndim == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    overlay = base_img.copy()
    vis     = base_img.copy()

    P_TL = pts["P_TL"]; P_TR = pts["P_TR"]
    P_BL = pts["P_BL"]; P_BR = pts["P_BR"]

    poly = np.array([P_TL, P_TR, P_BR, P_BL], dtype=np.int32)

    # Fill trapezoid semi-transparan (biru muda)
    cv2.fillPoly(overlay, [poly], (255, 200, 100))
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # Garis tepi trapezoid (kuning terang)
    cv2.polylines(vis, [poly], isClosed=True, color=(0, 220, 255), thickness=4, lineType=cv2.LINE_AA)

    # Titik sudut + label
    corners = {
        "TL": (P_TL, (0,   0,   255)),   # Merah
        "TR": (P_TR, (0,   255,  0)),    # Hijau
        "BL": (P_BL, (255,  0,   0)),   # Biru
        "BR": (P_BR, (0,   255, 255)),   # Kuning
    }
    for label, (pt, color) in corners.items():
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(vis, (px, py), 18, color, -1)
        cv2.putText(vis, label, (px + 22, py + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

    return vis


def draw_canvas_view(
    mask_sam3:  np.ndarray,   # SAM3 mask dalam image coords (H, W)
    pts:        dict,          # output detect_4_points dengan canvas coords
    scale_down: float = 0.25,  # faktor resize agar canvas 3x tidak terlalu besar
) -> np.ndarray:
    """
    Buat canvas 3x, letakkan SAM3 mask di tengah,
    lalu gambar trapezoid homografi dalam canvas coords.
    Ini menunjukkan posisi P_BL/P_BR yang extend keluar gambar.
    """
    canvas_h = pts["canvas_h"]
    canvas_w = pts["canvas_w"]
    shift_y  = pts["shift_y"]
    shift_x  = pts["shift_x"]
    h_orig, w_orig = mask_sam3.shape[:2]

    # Buat canvas 3x — hitam (=luar gambar), putih di area gambar
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas[shift_y:shift_y+h_orig, shift_x:shift_x+w_orig] = mask_sam3
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # Tandai batas gambar asli dengan kotak putih tipis
    cv2.rectangle(
        canvas_bgr,
        (shift_x, shift_y),
        (shift_x + w_orig - 1, shift_y + h_orig - 1),
        (200, 200, 200), 3
    )
    # Label "batas gambar asli"
    cv2.putText(canvas_bgr, "Batas Gambar Asli",
                (shift_x + 10, shift_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3, cv2.LINE_AA)

    # Gambar trapezoid dalam CANVAS coords
    CTL = pts["C_TL"]; CTR = pts["C_TR"]
    CBL = pts["C_BL"]; CBR = pts["C_BR"]

    poly = np.array([CTL, CTR, CBR, CBL], dtype=np.int32)

    # Fill semi-transparan biru
    overlay = canvas_bgr.copy()
    cv2.fillPoly(overlay, [poly], (255, 180, 80))
    cv2.addWeighted(overlay, 0.35, canvas_bgr, 0.65, 0, canvas_bgr)

    # Garis tepi kuning terang
    cv2.polylines(canvas_bgr, [poly], isClosed=True,
                  color=(0, 220, 255), thickness=6, lineType=cv2.LINE_AA)

    # ── Garis Atas (CTL → CTR) — warna hijau toska, label di tengah ──
    mid_atas_x = (int(CTL[0]) + int(CTR[0])) // 2
    mid_atas_y = (int(CTL[1]) + int(CTR[1])) // 2
    cv2.line(canvas_bgr, (int(CTL[0]), int(CTL[1])), (int(CTR[0]), int(CTR[1])),
             (0, 255, 180), 10, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "GARIS ATAS",
                (mid_atas_x - 120, mid_atas_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 180), 4, cv2.LINE_AA)

    # ── Garis Bawah (CBL → CBR) — warna oranye, label di tengah ──
    # Klem ke batas canvas agar garis tetap tergambar walau titik keluar
    bl_x = max(0, min(canvas_w - 1, int(CBL[0])))
    bl_y = max(0, min(canvas_h - 1, int(CBL[1])))
    br_x = max(0, min(canvas_w - 1, int(CBR[0])))
    br_y = max(0, min(canvas_h - 1, int(CBR[1])))
    mid_bawah_x = (bl_x + br_x) // 2
    mid_bawah_y = (bl_y + br_y) // 2
    cv2.line(canvas_bgr, (bl_x, bl_y), (br_x, br_y),
             (0, 100, 255), 10, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "GARIS BAWAH",
                (mid_bawah_x - 140, mid_bawah_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 100, 255), 4, cv2.LINE_AA)

    # Titik sudut
    corners = [
        (CTL, (0, 0, 255),   "TL"),
        (CTR, (0, 255,  0),  "TR"),
        (CBL, (255, 0,   0), "BL"),
        (CBR, (0, 255, 255), "BR"),
    ]
    for pt, color, label in corners:
        px, py = int(pt[0]), int(pt[1])
        # Gambar hanya jika dalam canvas
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            cv2.circle(canvas_bgr, (px, py), 28, color, -1)
            cv2.putText(canvas_bgr, label, (px + 35, py + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4, cv2.LINE_AA)
        else:
            # Titik di luar canvas — gambar tanda panah dari tepi canvas
            edge_x = max(0, min(canvas_w - 1, px))
            edge_y = max(0, min(canvas_h - 1, py))
            cv2.circle(canvas_bgr, (edge_x, edge_y), 28, color, 4)
            cv2.putText(canvas_bgr, f"{label}(off:{px},{py})",
                        (edge_x + 10, edge_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)

    # Resize agar output tidak terlalu besar
    new_w = int(canvas_w * scale_down)
    new_h = int(canvas_h * scale_down)
    canvas_small = cv2.resize(canvas_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return canvas_small


def put_label(img: np.ndarray, text: str, color=(255,255,255)) -> np.ndarray:
    out = img.copy()
    if out.ndim == 2: out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return out


# ================================================================
# BAGIAN 5: Main Test
# ================================================================

def run_test(img_path: str, save_dir: str = None, sam3_url: str = None):

    save_dir = Path(save_dir or (Path(__file__).parent / "outputs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem

    # ── Load gambar ─────────────────────────────────────────────────────
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")
    h, w = img_bgr.shape[:2]
    print(f"\n[TEST] Gambar: {img_path}  ({w}x{h})")

    # ── Step 1: ST_RoomNet inference ────────────────────────────────────
    print("\n[TEST] ── Step 1: ST_RoomNet inference ──")
    t0     = time.perf_counter()
    model  = load_roomnet()
    mask_st = predict_mask(model, img_bgr)
    del model   # bebaskan VRAM segera
    ms_st = (time.perf_counter() - t0) * 1000
    print(f"[TEST] mask_st selesai: {ms_st:.0f} ms  "
          f"pixel lantai: {np.sum(mask_st > 0):,}")

    # ── Step 2: Deteksi 4 titik perspektif ─────────────────────────────
    print("\n[TEST] ── Step 2: Deteksi 4 titik perspektif ──")
    pts = detect_4_points(mask_st)
    if pts is None:
        print("[TEST] GAGAL mendeteksi 4 titik. Cek mask_st.")
    else:
        print(f"[TEST] P_TL={pts['P_TL']}  P_TR={pts['P_TR']}")
        print(f"[TEST] P_BL={pts['P_BL']}  P_BR={pts['P_BR']}")

    # ── Step 3: SAM3 mask ───────────────────────────────────────────────
    print("\n[TEST] ── Step 3: SAM3 inference ──")
    t1       = time.perf_counter()
    mask_sam3 = call_sam3(img_bgr, sam3_url)
    ms_sam3  = (time.perf_counter() - t1) * 1000
    if mask_sam3 is not None:
        # Pastikan ukuran sama dengan gambar asli
        if mask_sam3.shape != (h, w):
            mask_sam3 = cv2.resize(mask_sam3, (w, h), interpolation=cv2.INTER_NEAREST)
        _, mask_sam3 = cv2.threshold(mask_sam3, 127, 255, cv2.THRESH_BINARY)
        print(f"[TEST] SAM3 selesai: {ms_sam3:.0f} ms")
    else:
        print("[TEST] SAM3 tidak tersedia. Visualisasi overlay menggunakan mask_st sebagai fallback.")
        mask_sam3 = mask_st.copy()

    # ── Step 3.5: Smart Trapezoid Fitting ───────────────────────────────
    if pts is not None:
        print("\n[TEST] ── Step 3.5: Smart Trapezoid Fitting (Mask-Weighted Shift) ──")
        t_smart = time.perf_counter()
        pts = smart_trapezoid_fitting(mask_sam3, pts)
        ms_smart = (time.perf_counter() - t_smart) * 1000
        print(f"[TEST] Smart Fitting selesai: {ms_smart:.0f} ms")

    # ── Step 4: Overlay trapezoid homography di atas SAM3 ───────────────
    print("\n[TEST] ── Step 4: Overlay Homography di atas SAM3 mask ──")

    # Base: SAM3 mask dibuat RGB agar bisa overlay warna
    sam3_bgr = cv2.cvtColor(mask_sam3, cv2.COLOR_GRAY2BGR)

    if pts:
        # ① Overlay trapezoid di atas SAM3 mask
        overlay_on_sam3 = draw_homography_overlay(sam3_bgr, pts, alpha=0.45)

        # ② Overlay trapezoid di atas foto ASLI
        overlay_on_photo = draw_homography_overlay(img_bgr, pts, alpha=0.35)

        # ③ Canvas 3x view — trapezoid + SAM3 dalam ruang canvas penuh
        canvas_view = draw_canvas_view(mask_sam3, pts, scale_down=0.25)
        
        # ④ Render Virtual Try-On (VTO) dengan tile perspektif
        print("\n[TEST] ── Step 5: Render VTO Perspective Tile ──")
        tile_path = r"C:\Project\simpel\assets\tile\Concord-60x60-PGC66K001-ALASKA WHITE.jpeg"
        rendered_vto = render_tiles_perspective(img_bgr, pts, mask_sam3, tile_path=tile_path)
    else:
        overlay_on_sam3  = sam3_bgr.copy()
        overlay_on_photo = img_bgr.copy()
        canvas_view      = np.zeros((h // 3, w // 3, 3), dtype=np.uint8)
        rendered_vto     = img_bgr.copy()
        cv2.putText(overlay_on_sam3, "4 titik GAGAL", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # ── Simpan output — canvas 3x dan VTO ─────────────────────────────────
    out_canvas = str(save_dir / f"{stem}_canvas3x.jpg")
    cv2.imwrite(out_canvas, canvas_view, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    out_vto = str(save_dir / f"{stem}_vto_hasil.jpg")
    cv2.imwrite(out_vto, rendered_vto, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\n[TEST] Output tersimpan:")
    print(f"  → {out_canvas}")
    print(f"  → {out_vto}")
    print(f"\n  Waktu: ST_RoomNet={ms_st:.0f}ms  SAM3={ms_sam3:.0f}ms")


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualisasi overlap Homography ST_RoomNet vs SAM3 mask"
    )
    parser.add_argument("--img",     required=True, help="Path gambar ruangan")
    parser.add_argument("--out",     default=None,  help="Folder output (default: tests/VTO/outputs/)")
    parser.add_argument("--sam3_url",default=None,  help="URL SAM3 (default: http://localhost:8001/predict/floor)")
    args = parser.parse_args()

    try:
        run_test(img_path=args.img, save_dir=args.out, sam3_url=args.sam3_url)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
