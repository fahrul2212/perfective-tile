# 🚀 Upgrade Roadmap — RoomVision AI

Folder ini berisi roadmap pengembangan web app RoomVision AI dalam 3 fase.

## Struktur

```
upgrade/
├── README.md                          ← File ini
├── phase-1_split-pipeline.md          ← Riset: Pisah inference dari rendering
├── phase-2_web-app-interface.md       ← Web app: Room visualizer + tile catalog
├── phase-3_advanced-features.md       ← Fitur lanjutan: lighting, grout, dll
```

## Phase Overview

| Phase | Fokus | Status |
|-------|-------|--------|
| **Phase 1** | Split Pipeline Architecture — AI sekali jalan, tile swap instan | 📋 Riset selesai |
| **Phase 2** | Web App Interface — Room catalog, tile picker, editor canvas | 📋 Belum dimulai |
| **Phase 3** | Advanced Features — Lighting, grout control, export, compare | 📋 Belum dimulai |

## Prinsip Utama

> **"AI berat hanya 1x. Ganti keramik = instan."**
> 
> Arsitektur Blueprint: server kirim "scene data" (mask + titik perspektif),
> client render tile sendiri tanpa hit API lagi.
