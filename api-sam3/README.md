# SAM3 Floor Demarkation API

API berbasis FastAPI untuk deteksi dan segmentasi mask lantai (floor) secara otomatis menggunakan model SAM3 dari Hugging Face. Output berupa gambar PNG mask biner (lantai = putih, lainnya = hitam).

---

## Prasyarat (Prerequisites)

| Kebutuhan | Keterangan |
|---|---|
| **Python** | Versi 3.9 atau lebih baru |
| **GPU NVIDIA** | Sangat disarankan (inferensi 5–10x lebih cepat) |
| **CUDA Toolkit** | Versi 11.8 atau 12.1 (sesuaikan versi PyTorch) |
| **VRAM** | Minimal 6 GB untuk model SAM3 |

---

## Langkah 1 — Install CUDA Toolkit (Jika Pakai GPU)

> Lewati langkah ini jika ingin berjalan di CPU saja.

1. Cek versi CUDA yang didukung GPU Anda:
   ```cmd
   nvidia-smi
   ```
   Lihat kolom `CUDA Version` di pojok kanan atas output.

2. Download **CUDA Toolkit** sesuai versi GPU dari:
   👉 https://developer.nvidia.com/cuda-downloads

3. Setelah install, verifikasi:
   ```cmd
   nvcc --version
   ```

4. Install **PyTorch versi CUDA** (ganti `cu121` jika pakai CUDA 11.8 → `cu118`):
   ```cmd
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

5. Verifikasi PyTorch bisa mendeteksi GPU:
   ```python
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Output harus `True` dan nama GPU Anda.

---

## Langkah 2 — Aktivasi Virtual Environment

Virtual environment terisolasi sudah tersedia di folder `env/`.

**Windows — Command Prompt:**
```cmd
env\Scripts\activate.bat
```

**Windows — PowerShell:**
```powershell
.\env\Scripts\Activate.ps1
```

> Jika PowerShell menolak dengan error execution policy, jalankan dulu:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

**MacOS / Linux:**
```bash
source env/bin/activate
```

Indikator env aktif: nama `(env)` muncul di awal baris terminal.

---

## Langkah 3 — Install Dependensi

Pastikan virtual environment sudah aktif, lalu jalankan:

```cmd
pip install -r requirements.txt
```

> **Catatan:** Jika sudah install PyTorch CUDA di Langkah 1, `torch` di `requirements.txt` tidak akan menimpa instalasi CUDA Anda selama pip mendeteksi versi yang kompatibel.

---

## Langkah 4 — Jalankan Server

**Opsi 1 — Uvicorn (Direkomendasikan):**
```cmd
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Opsi 2 — Python langsung:**
```cmd
python main.py
```

Server siap ketika muncul pesan:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
SAM3 Service successfully loaded!
```

> ⚠️ **Perhatian:** Proses loading model SAM3 pertama kali membutuhkan waktu **2–10 menit** karena mendownload bobot model (~2 GB) dari Hugging Face. Tunggu hingga muncul `SAM3 Service successfully loaded!` sebelum mengirim request.

---

## Cara Menggunakan API

### Endpoint

| Method | URL | Deskripsi |
|---|---|---|
| `GET` | `http://localhost:8000/` | Health check |
| `POST` | `http://localhost:8000/predict/floor` | Kirim gambar, terima mask lantai PNG |
| `GET` | `http://localhost:8000/docs` | Swagger UI interaktif |

---

### Menggunakan Postman

1. **Buka Postman** → klik **New Request**

2. Set method ke **`POST`** dan masukkan URL:
   ```
   http://localhost:8000/predict/floor
   ```

3. Pergi ke tab **`Body`** → pilih **`form-data`**

4. Tambahkan field baru:
   - **Key**: `file`
   - **Type**: ubah dari `Text` ke **`File`** (klik dropdown di sebelah kanan kolom Key)
   - **Value**: klik **Select Files** → pilih gambar `.jpg` atau `.png` ruangan Anda

5. Klik **Send**

6. **Respons akan berupa gambar PNG** — klik tab **`Body`** → klik ikon **Save Response** → **Save to a file** untuk menyimpan mask PNG-nya.

**Cek waktu proses di Response Headers:**

| Header | Contoh Nilai | Keterangan |
|---|---|---|
| `X-Process-Time` | `1.2345 seconds` | Total durasi inferensi SAM3 |
| `content-type` | `image/png` | Format output |

---

### Menggunakan Swagger UI (Browser)

1. Buka: `http://localhost:8000/docs`
2. Klik **`POST /predict/floor`** → **Try it out**
3. Pada field `file`, klik **Choose File** → pilih gambar
4. Klik **Execute**
5. Scroll ke bawah → lihat response body (gambar mask)

---

### Menggunakan curl (Terminal)

```bash
curl -X POST http://localhost:8000/predict/floor \
  -F "file=@/path/to/gambar.jpg" \
  --output mask_lantai.png
```

---

## Response

| Kondisi | Status Code | Output |
|---|---|---|
| Berhasil | `200 OK` | File PNG gambar mask lantai |
| File bukan gambar | `400 Bad Request` | JSON error |
| Model belum selesai loading | `500` | JSON error |
| Error inferensi | `500` | JSON error dengan detail |

**Contoh mask output:**
- ⬜ **Piksel Putih (255)** = Area lantai terdeteksi
- ⬛ **Piksel Hitam (0)** = Area non-lantai

---

## Optimasi yang Diimplementasikan

| Optimasi | Keterangan |
|---|---|
| **BF16/FP16 Precision** | VRAM -50%, throughput GPU meningkat |
| **Image Resize (max 1024px)** | Komputasi encoder berkurang kuadratik |
| **torch.compile** | Aktif otomatis jika Triton tersedia (Linux/WSL) |
| **Vectorized Mask OR** | `torch.stack().any(dim=0)` tanpa loop |
| **PNG compress_level=1** | I/O encode output lebih cepat |
