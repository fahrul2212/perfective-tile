# 🏛️ Panduan Arsitektur & Struktur Folder Proyek

Proyek ini telah mengadopsi standar **Layered Architecture (N-Tier Architecture)** yang digabungkan dengan **Microservice Pattern**. Bertujuan untuk membuat pemeliharaan kode (maintenance), pengujian (testing), dan pembagian kerja menjadi lebih jelas dan profesional.

---

## 1. Penjelasan Masing-Masing Folder

Berikut adalah fungsi dan peran masing-masing folder dalam arsitektur ini:

### `core/` (Domain Layer / Logika Inti)
Merupakan *"otak"* dari keseluruhan proyek. Tidak peduli Anda mengubah sistemnya menjadi API, Desktop App, atau Command Line, folder ini tetap utuh karena tidak bergantung pada hal eksternal (seperti Web HTTP).
- `config.py` : Tempat sentral mengatur semua konfigurasi (Path, Port, URL, Parameter Model).
- `model.py` : Arsitektur Neural Network `ST_RoomNet`.
- `inference.py` : *Service Layer* yang mengatur proses inisiasi GPU dan prediksi gambar.
- `postprocess.py` : Logika matematika murni (NumPy/OpenCV) tanpa kaitan dengan internet, seperti membersihkan noise mask.
- `sam3_client.py` : *Adapter Layer* untuk berkomunikasi via HTTP ke service SAM3 secara aman (di mana main app seakan tidak peduli SAM3 itu ada dimana).

### `utils/` (Infrastructure Layer / Utilitas)
Singkatan dari *Utilities*. Ini adalah **kotak perkakas (toolbox)**. Isinya adalah fungsi-fungsi pembantu yang berdiri sendiri, tidak peduli terhadap *"Bisnis Utama"*, tapi sangat membantu pengerjaan hal repetitif.
- `perspective.py` : Alat untuk menggambar garis, efek keramik, efek distorsi (*drawing tool*). Jika kelak Anda tidak membuat aplikasi keramik lagi tapi butuh fungsi *warp* sudut, file ini masih bisa di-_copy-paste_ langsung tanpa membawa aplikasi `core`. 

### `api-sam3/` (Microservice)
Karena model AI `SAM3` memiliki kebutuhan teknologi yang "berbenturan" (butuh library transformator super baru yang bisa merusak versi `Torch` stabil Anda), ia **diisolasi** ke sini. Folder ini seperti pulau mandiri yang punya Virtual Environment (`env/`) dan server HTTP tersendiri (Port 8001).

### `scripts/` (Operasional DevOps)
Menyimpan semua file kotor `.bat` yang menjalankan sistem di *background* seperti *background runner* atau *kill port*. Dipisahkan ke sini agar level awal (*root*) file explorer tetap enak dilihat.

### `tests/` (Pengujian / QC)
Lingkungan Lab untuk bereksperimen. Segala skrip untuk mengecek apakah `core/model.py` bisa memotong 1000 gambar sekaligus tanpa menyalakan server UI, adanya di sini (termasuk folder `laboratory`).

### `outputs/`
Berisi hasil rendering sistem untuk disaksikan oleh Frontend (API output static).

---

## 2. Alur Cara Kerja (Flow)

Saat gambar masuk, ini yang terjadi berurutan:
1. **Request via Frontend (`index.html`)** masuk ke `app.py`.
2. **`app.py`** hanya berkata: *"Tolong teruskan ini ke `core/inference.py`"*.
3. **`inference.py`** melakukan sihir GPU-nya dan mengirim hasil ke **`core/postprocess.py`** untuk dibersihkan.
4. **`app.py`** (secara bersamaan) menyuruh **`core/sam3_client.py`** menelepon `api-sam3/:8001` untuk minta tolong ke SAM3.
5. Setelah jadi mask bersih, **`app.py`** minta tolong alat gambar di **`utils/perspective.py`** untuk melukis keramiknya.
6. Gambar disimpan ke folder **`outputs/`** dan Web merespon kembali!

Itulah kenapa disebut *Clean Layered Architecture*. Setiap halangan jika error, Anda tahu mana folder yang bermasalah secara instan!
