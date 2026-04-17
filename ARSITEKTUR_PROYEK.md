# 🏛️ Panduan Arsitektur & Struktur Folder Proyek

Proyek ini mengadopsi **Clean Layered Architecture** (Robert C. Martin, 2017) digabung dengan **Microservice Pattern** dan **Hexagonal Architecture** (Alistair Cockburn). Setiap layer memiliki tanggung jawab tunggal (Single Responsibility Principle).

---

## 1. Diagram Arsitektur

```
              ┌─────────────────────┐
              │    static/          │ ← Frontend (HTML/CSS/JS)
              │    index.html       │
              └────────┬────────────┘
                       │ HTTP
              ┌────────▼────────────┐
              │    app.py           │ ← Application Layer (Thin Router)
              │    (FastAPI)        │    Hanya routing + file I/O
              └────────┬────────────┘
                       │
              ┌────────▼────────────┐
              │    services/        │ ← Orchestration Layer
              │    ├ vto_pipeline   │    Mengatur alur pipeline
              │    └ sam3_client    │    Adapter HTTP (SAM3)
              └───┬────────────┬───┘
                  │            │
         ┌────────▼──┐  ┌─────▼──────────┐
         │  core/    │  │  utils/         │
         │  Domain   │  │  Infrastructure │
         │  Layer    │  │  Tools          │
         └──────────┘  └────────────────┘
```

---

## 2. Penjelasan Masing-Masing Folder

### `app.py` (Application Layer — Thin Router)
**Prinsip**: Semakin tipis, semakin baik. `app.py` HANYA bertugas:
- Menerima HTTP request
- Memanggil `services/vto_pipeline.py`
- Menyimpan file output
- Mengembalikan JSON response

### `services/` (Orchestration Layer) 🆕
Merupakan *"Konduktor Orkestra"*. Mengatur urutan pemanggilan komponen tanpa menulis logika bisnis sendiri.
- `vto_pipeline.py` : Pipeline VTO end-to-end (RoomNet → Postprocess → SAM3 → VTO Render → Shadow)
- `sam3_client.py` : *Adapter Layer* untuk HTTP ke SAM3 microservice. Dipindahkan dari `core/` karena ini adapter (ring luar), bukan domain.

### `core/` (Domain Layer — Otak)
Merupakan *"Otak"* murni. Tidak bergantung pada HTTP, database, atau framework apapun.
- `config.py` : Konfigurasi sentral
- `model.py` : Arsitektur Neural Network `ST_RoomNet`
- `inference.py` : Service GPU inference
- `postprocess/` : 🆕 **Package** (dipecah dari 1 file menjadi 3 modul):
  - `mask_cleanup.py` : `get_largest_cc()`, `fill_floor_bottom()` — pembersihan dasar
  - `mask_refinement.py` : `refine_mask_smooth()`, `generate_alpha_mask()` — smoothing & matting
  - `shadow.py` : `extract_shadow_map()` — ekstraksi bayangan
- `sam3_client.py` : *Backward compat proxy* → re-export dari `services/sam3_client.py`

### `utils/` (Infrastructure Layer — Toolbox)
Fungsi-fungsi stateless yang bisa di-reuse tanpa membawa logika bisnis.
- `perspective/` : 🆕 **Package** (dipecah dari 1 file menjadi 4 modul):
  - `detect_points.py` : `detect_4_points()` — geometri deteksi titik perspektif
  - `trapezoid_fitting.py` : `smart_trapezoid_fitting()` — adaptive fitting ke SAM3 mask
  - `grid.py` : `calc_cols_rows()` — kalkulasi grid tile dinamis
  - `renderer.py` : `render_ceramic_perspective()` — compositing tile + perspective warp

### `api-sam3/` (Microservice — Pulau Terisolasi)
Model AI SAM3 diisolasi karena dependency yang berbenturan. Punya virtual environment dan server HTTP sendiri (Port 8001).

### `tests/` (Lab Riset & Eksperimen)
Tempat R&D. Script-script riset segmentasi, shadow extraction, VTO prototype.

### `static/`, `outputs/`, `assets/`, `scripts/`, `weights/`
Frontend UI, hasil rendering, input assets, DevOps scripts, model weights.

---

## 3. Alur Cara Kerja (Flow)

```
1. User upload gambar → app.py
2. app.py → services/vto_pipeline.process()
3. Pipeline:
   a. core/inference.py → GPU inference → raw mask
   b. core/postprocess/mask_cleanup.py → bersihkan noise
   c. core/postprocess/mask_refinement.py → haluskan tepian
   d. services/sam3_client.py → HTTP ke api-sam3/:8001
   e. utils/perspective/renderer.py → render tile VTO
   f. core/postprocess/shadow.py → ekstraksi bayangan
4. app.py → simpan ke outputs/ → return JSON
5. Frontend render hasil di browser
```

---

## 4. Prinsip Arsitektur

| Prinsip | Implementasi |
|---------|-------------|
| **Single Responsibility** | Setiap file punya 1 alasan untuk berubah |
| **Dependency Rule** | `app.py` → `services/` → `core/` + `utils/` (bukan sebaliknya) |
| **Backward Compatibility** | `from core.postprocess import X` tetap bekerja via `__init__.py` |
| **Adapter Pattern** | SAM3 client adalah adapter — `core/` tidak tahu tentang HTTP |
