# ST-RoomNet Optimized by Fahrul Rozi

Proyek ini adalah pengembangan dan optimasi dari **ST-RoomNet** yang dikembangkan oleh **Fahrul Rozi**. Versi ini telah mengalami perubahan signifikan dari implementasi aslinya untuk mencapai performa tinggi dan kualitas visual yang lebih baik.

## Fitur Utama
- **High Performance**: Pipeline preprocessing penuh di GPU menggunakan PyTorch.
- **Clean Predictions**: Filter *Largest Connected Component* untuk menghapus noise di tepi gambar.
- **Smooth Edges**: *Soft probability mask* untuk hasil seleksi lantai yang anti-alias dan mulus.
- **FastAPI Backend**: Endpoint yang siap digunakan untuk integrasi sistem.
- **Clean Architecture**: Struktur pemfolderan yang rapi dan mudah di-maintain.

## Struktur Proyek
```text
/simpel
├── app.py                # Server FastAPI utama
├── core/
│   └── model.py          # Arsitektur model ST-RoomNet
├── weights/
│   └── persfective.pth   # Bobot model (Weights)
├── assets/
│   └── ref_img2.png      # Mask referensi
├── static/
│   └── index.html        # Frontend dashboard
├── tests/
│   └── test_model.py     # Skrip pengujian bulk
├── utils/
│   └── debug_classes.py  # Utilitas bantu
├── api_outputs/          # Folder hasil prediksi
└── requirements.txt      # Daftar dependensi
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

### Menjalankan Server API
Jalankan server aplikasi utama menggunakan perintah:
```bash
python app.py
```
Aplikasi akan berjalan di `http://localhost:8000`. Anda bisa membuka browser di alamat tersebut untuk melihat dashboard demo.

### Menjalankan Pengujian (Visual Check)
Untuk menjalankan pengujian pada banyak gambar sekaligus di folder `assets/`:
```bash
$env:PYTHONPATH = "."; python tests/test_model.py
```
Hasil prediksi akan disimpan secara otomatis di folder `api_outputs/`.

## Developer & Maintainer
- **Owner**: Fahrul Rozi
- **Status**: Stable / Optimized Version

---

---
*Optimized with ❤️ by Fahrul Rozi*
