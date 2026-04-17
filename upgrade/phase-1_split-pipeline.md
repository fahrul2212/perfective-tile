# Phase 1 — Split Pipeline Architecture

> **"AI berat hanya 1x. Ganti keramik = instan."**

## 1. Masalah Saat Ini

```
Setiap kali user ganti keramik:

Upload → RoomNet (GPU) → Postprocess → SAM3 (GPU) → Perspective Detect → Tile Render → Output
  ↑                                                                           ↑
  └─ MAHAL (2-5 detik)                                                        └─ MURAH (0.1 detik)
```

**Fakta:**
- RoomNet inference: ~200ms (GPU)
- SAM3 inference: ~1-3 detik (GPU, HTTP)
- Perspective detection: ~100ms (CPU)
- Smart Trapezoid Fitting: ~200ms (CPU)
- **Tile rendering (warp): ~50-100ms (CPU)** ← hanya ini yang berubah saat ganti tile

**Kesimpulan:** 90% waktu terbuang untuk menghitung ulang hal yang SAMA ketika user hanya ganti tile.

---

## 2. Solusi: Blueprint Architecture (Split Pipeline)

### Konsep Inti

Pisahkan pipeline menjadi 2 fase:

```
FASE A — "Analyze" (1x per foto, BERAT)
─────────────────────────────────────────
Upload → RoomNet → Postprocess → SAM3 → Perspective Detect
                                    ↓
                            Simpan "Scene Blueprint" (JSON + binary)

FASE B — "Render" (Nx per tile swap, RINGAN)
─────────────────────────────────────────
Load Blueprint + Tile baru → Warp → Composite → Output
```

### Scene Blueprint — Data yang Di-cache

Semua data yang dibutuhkan untuk rendering tersimpan di "Scene Blueprint":

```json
{
  "scene_id": "abc123",
  "created_at": "2026-04-17T21:30:00Z",
  "resolution": { "w": 1920, "h": 1080 },
  
  "perspective_points": {
    "P_TL": [320, 180], "P_TR": [1600, 180],
    "P_BL": [-200, 1200], "P_BR": [2100, 1200],
    "canvas_w": 5760, "canvas_h": 3240,
    "shift_x": 1920, "shift_y": 1080
  },
  
  "grid": { "cols": 3, "rows": 5 },
  
  "masks": {
    "sam3_url": "/blueprints/abc123/mask_sam3.png",
    "refined_url": "/blueprints/abc123/mask_refined.png"
  },
  
  "shadow": {
    "shadow_map_url": "/blueprints/abc123/shadow_map.png"
  },
  
  "original_photo_url": "/blueprints/abc123/original.jpg"
}
```

### Binary Data (disimpan di server, bukan JSON):
- `original.jpg` — foto asli
- `mask_sam3.png` — SAM3 binary mask
- `mask_refined.png` — refined mask
- `shadow_map.png` — shadow map
- `mask_cleaned.npy` — NumPy mask untuk perspective detection (opsional)

---

## 3. Desain API Baru

### Endpoint `/analyze` (Fase A — 1x, berat)

```
POST /analyze
Body: multipart file (foto ruangan)

Response:
{
  "status": "success",
  "scene_id": "abc123",
  "blueprint": { ... scene blueprint JSON di atas ... },
  "preview_url": "/blueprints/abc123/preview_vto.jpg",  ← preview default tile
  "inference_time_ms": 2500
}
```

**Yang terjadi di backend:**
1. RoomNet inference → mask
2. Postprocess → mask cleaned
3. Refine mask smooth
4. SAM3 → mask_sam3
5. Detect perspective points
6. Smart Trapezoid Fitting
7. Shadow extraction
8. Simpan semua ke disk/Redis
9. Render preview dengan default tile
10. Return blueprint JSON

### Endpoint `/render-tile` (Fase B — Nx, ringan)

```
POST /render-tile
Body: { "scene_id": "abc123", "tile_id": "alaska-white-60x60" }

Response:
{
  "status": "success",
  "vto_url": "/outputs/abc123_alaska-white.jpg",
  "render_time_ms": 80
}
```

**Yang terjadi di backend:**
1. Load blueprint dari cache (scene_id)
2. Load tile texture dari catalog
3. Build texture sheet + grout
4. Perspective warp (menggunakan points dari blueprint)
5. Composite dengan SAM3 mask clipping
6. Save + return URL

> **Tidak ada RoomNet. Tidak ada SAM3. Tidak ada postprocess.**
> Hanya warp + composite = ~50-100ms.

### Endpoint `/tile-catalog` (Daftar tile)

```
GET /tile-catalog

Response:
{
  "tiles": [
    { "id": "alaska-white-60x60", "name": "Alaska White", "size": "60x60", "thumbnail": "/tiles/thumb/alaska.jpg" },
    { "id": "granito-grey-80x80", "name": "Granito Grey", "size": "80x80", "thumbnail": "/tiles/thumb/granito.jpg" },
    ...
  ]
}
```

---

## 4. Perbandingan Performa

| Skenario | Sekarang | Setelah Split |
|----------|---------|---------------|
| Upload pertama | ~3-5 detik | ~3-5 detik (sama) |
| Ganti tile ke-2 | ~3-5 detik ❌ | **~100ms** ✅ |
| Ganti tile ke-3 | ~3-5 detik ❌ | **~100ms** ✅ |
| Ganti tile ke-10 | ~3-5 detik ❌ | **~100ms** ✅ |
| **Total 10 tile** | **~30-50 detik** | **~5 detik + 0.9 detik = ~6 detik** |

> **Speedup: ~5-8x lebih cepat** untuk sesi browsing tile.

---

## 5. Opsi Implementasi

### Opsi A: Server-Side Render (Rekomendasi Awal) ⭐

```
Client → POST /render-tile { scene_id, tile_id }
Server → Load blueprint → Warp → Composite → Return JPEG URL
```

**Pro:**
- Implementasi paling sederhana — hanya perlu refactor `app.py` + `vto_pipeline.py`
- Kualitas render konsisten (server kontrol penuh)
- Tidak perlu kirim mask ke browser (hemat bandwidth)
- Bisa generate high-res output

**Kontra:**
- Masih ada network round-trip (~100ms + ~50ms render = ~150ms per swap)
- Server load bertambah jika banyak user concurrent

**Cocok untuk:** MVP, user base kecil-menengah

### Opsi B: Client-Side Render (WebGL/Canvas)

```
Client → Terima blueprint + mask + shadow saat analyze
Client → User pilih tile → Browser render sendiri via Canvas2D/WebGL
Server → Tidak dihubungi sama sekali saat ganti tile
```

**Pro:**
- Zero latency tile swap (instan, no network)
- Server load minimal — hanya handle `/analyze`
- Ideal untuk mobile (offline-capable setelah analyze)

**Kontra:**
- Implementasi kompleks (perspective warp di JavaScript/WebGL)
- Kualitas render tergantung device user
- Harus kirim mask ke browser (beberapa MB)
- Debug lebih sulit

**Cocok untuk:** Skala besar, pengalaman premium

### Opsi C: Hybrid (Rekomendasi Jangka Panjang) ⭐⭐

```
Analyze → Server kirim blueprint + low-res mask + perspective matrix
Tile swap → Client render preview cepat (Canvas2D, low-res)
           + Server render high-res di background → swap ke high-res saat ready
```

**Pro:**
- User langsung lihat preview instan (client-side)
- High-res result menyusul dari server
- Best of both worlds

**Kontra:**
- Implementasi paling kompleks

---

## 6. Rekomendasi: Roadmap Implementasi

### Step 1 (Segera): Server-Side Split — Opsi A

Refactor yang dibutuhkan:

| File | Perubahan |
|------|-----------|
| `services/vto_pipeline.py` | Pisah `process()` → `analyze()` + `render_tile()` |
| `app.py` | Tambah endpoint `/analyze` dan `/render-tile` |
| `services/scene_cache.py` | **BARU** — simpan/load scene blueprint (disk-based) |
| `utils/tile_catalog.py` | **BARU** — scan folder `assets/tile/` → return daftar tile |
| `static/index.html` | Update JS: analyze sekali, lalu tile swap via `/render-tile` |

**Estimasi:** 1-2 hari kerja

### Step 2 (Phase 2): Web App dengan Tile Picker

Lihat `phase-2_web-app-interface.md`

### Step 3 (Phase 3): Client-Side Preview + Advanced

Lihat `phase-3_advanced-features.md`

---

## 7. Pseudocode: Refactor `vto_pipeline.py`

### Sekarang (Monolitik):
```python
class VTOPipeline:
    async def process(self, img_bgr) -> VTOResult:
        # SEMUA dalam 1 fungsi
        mask = roomnet.predict(img)
        mask = postprocess(mask)
        sam3 = await sam3_client.get_mask(img)
        pts = detect_4_points(mask)
        pts = smart_trapezoid_fitting(sam3, pts)
        vto = render(img, pts, sam3, tile="default.jpg")
        shadow = extract_shadow(img, sam3)
        return VTOResult(vto, mask, shadow)
```

### Setelah Split:
```python
class VTOPipeline:
    async def analyze(self, img_bgr) -> SceneBlueprint:
        """Fase A — 1x per foto, BERAT"""
        mask = roomnet.predict(img)
        mask = postprocess(mask)
        sam3 = await sam3_client.get_mask(img)
        pts = detect_4_points(mask)
        pts = smart_trapezoid_fitting(sam3, pts)
        shadow = extract_shadow(img, sam3)
        
        # Simpan ke cache
        scene_id = generate_id()
        blueprint = SceneBlueprint(
            scene_id=scene_id,
            img_bgr=img, mask_sam3=sam3,
            pts=pts, shadow=shadow
        )
        scene_cache.save(scene_id, blueprint)
        
        # Render preview dengan default tile
        preview = self.render_tile(blueprint, "default.jpg")
        return blueprint, preview

    def render_tile(self, blueprint: SceneBlueprint, tile_path: str) -> np.ndarray:
        """Fase B — Nx per tile swap, RINGAN (tanpa AI)"""
        cols, rows = calc_cols_rows(blueprint.pts)
        texture = build_texture_sheet(tile_path, cols, rows)
        warped = perspective_warp(texture, blueprint.pts)
        result = composite(blueprint.img_bgr, warped, blueprint.mask_sam3)
        return result
```

---

## 8. Referensi Riset

| Sumber | Relevansi |
|--------|-----------|
| [Split Pipeline Architecture](https://vertexaisearch.cloud.google.com) | Arsitektur decouple inference vs rendering |
| Martin Fowler, "PEAA" | Service Layer, Caching Patterns |
| OpenCV `getPerspectiveTransform` | Homography matrix caching |
| WebGL Texture Mapping | Client-side rendering untuk Phase 3 |
| Redis / Disk-based Scene Cache | Blueprint persistence strategy |

---

## 9. Kesimpulan

> **Intinya sederhana:**
> 1. AI (RoomNet + SAM3) hanya jalan **1 kali** per foto
> 2. Hasilnya disimpan sebagai **"Scene Blueprint"**
> 3. Ganti tile = **load blueprint + warp ulang** (~100ms)
> 4. Mulai dengan **server-side render** (Opsi A), upgrade ke **hybrid** nanti
