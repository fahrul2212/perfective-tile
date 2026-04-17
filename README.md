# ST-RoomNet Optimized by Fahrul Rozi

Proyek ini adalah pengembangan dan optimasi dari **ST-RoomNet** yang dikembangkan oleh **Fahrul Rozi**. Versi ini telah mengalami perubahan signifikan dari implementasi aslinya untuk mencapai performa tinggi dan kualitas visual yang lebih baik.

## Fitur Utama
- **High Performance**: Pipeline preprocessing penuh di GPU menggunakan PyTorch.
- **Clean Predictions**: Filter *Largest Connected Component* untuk menghapus noise di tepi gambar.
- **Smooth Edges**: *Soft probability mask* untuk hasil seleksi lantai yang anti-alias dan mulus.
- **FastAPI Backend**: Endpoint yang siap digunakan untuk integrasi sistem.
- **Clean Architecture**: Struktur pemfolderan yang rapi dan mudah di-maintain.

## Struktur Proyek Terbaru (Layered Architecture)
```text
/simpel
├── app.py                # Server FastAPI utama (Entrypoint HTTP)
├── core/                 # (Domain) Logika Bisnis & AI Inference utama
│   ├── config.py         # Seting parameter & direktori
│   ├── inference.py      # Pengaturan model ST-RoomNet & GPU Prediksi
│   ├── postprocess.py    # Pembersihan noise mask dengan Numpy OpenCV
│   └── sam3_client.py    # Adaptor HTTPS untuk Microservice SAM3
├── utils/                # (Infrastructure) Tools independen & helpers
│   └── perspective.py    # Alat bantu proses warp & rendering keramik
├── api-sam3/             # (Microservice) Standalone SAM3 Model AI Pipeline
│   ├── main.py           # Port: 8001
│   ├── env/              # Venv Python Isolasi SAM3
│   └── .env              # Kunci akses HuggingFace
├── scripts/              # Skrip Batch untuk background Runner/Launchers
├── static/               # Assets Web App (index.html)
├── outputs/              # Direktori cache respon akhir
├── tests/                # Lingkungan QC & Eksperimen laboratorium
└── run_all.bat           # MAIN LAUNCHER: Menjalankan Main + SAM3
```

## Persiapan & Instalasi

### 1. Prasyarat
- **Python 3.8+**
- **Git LFS** (untuk mendownload file model berukuran besar)

### 2. Instalasi
Clone repository dan pindah ke direktori proyek:
```bash
git clone <url-repo>
cd simpel
```

Buat virtual environment (opsional tapi sangat disarankan):
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

Install seluruh dependensi dengan perintah berikut:
```bash
pip install -r requirements.txt
```

## Cara Menjalankan

### Menjalankan Sistem Penuh (Main API + SAM3 Microservice)
Karena proyek ini mengadopsi standar **Multi-Python Process**, cara paling direkomendasikan adalah menggunakan skrip yang sudah kami susun:
```bash
run_all.bat
```
*Script ini otomatis merutekan _SAM3_ ke port `8001`, menunggu model hangat selama 45 detik, lalu meluncurkan main app di port `8000`.*

### URL Tersedia:
- **Dashboard Web**: `http://localhost:8000`
- **SAM3 Health Check**: `http://localhost:8001/health`

### Menjalankan Pengujian (Visual Check)
Untuk menjalankan pengujian pada skrip independen di dalam folder tests:
```bash
$env:PYTHONPATH = "."; python tests/test_model.py
```
Hasil akan disimpan sebagai gambar baru pada file terkait.

## Developer & Maintainer
- **Owner**: Fahrul Rozi
- **Status**: Stable / Optimized Version

---

---
*Optimized with ❤️ by Fahrul Rozi*
