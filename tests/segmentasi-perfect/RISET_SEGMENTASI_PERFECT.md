# Riset Segmentasi Sempurna (Perfect Segmentation)

## Tujuan
Mendapatkan mask segmentasi area `floor` (lantai) yang **sangat halus, akurat, dan presisi** hingga ke batas tepi terkecil (pixel-perfect) menggunakan model state-of-the-art.

## Metodologi
Penelitian ini akan dilakukan secara bertahap untuk mencapai hasil yang "perfect":

1. **Langkah 1: Ekstraksi Mask Dasar dengan SAM3 (Baseline)**
   Menggunakan `facebook/sam3` via `transformers` library dengan prompt text `"floor"`.
   *Target*: Memastikan pipeline inferensi berjalan dengan baik, menangani resolusi tinggi dengan benar, dan menghasilkan mask dasar (`0` dan `255`) sesuai resolusi asli gambar.

2. **Langkah 2: Analisis dan Penghalusan Tepi (Edge Refinement) - (Next Step)**
   Menganalisis hasil dari Langkah 1. SAM3 pada umumnya cukup tajam, namun pada batasan objek kecil (seperti kaki kursi atau nat keramik jauh) mungkin membutuhkan perlakuan khusus. 
   Metode yang bisa dieksplorasi jika diperlukan:
   - *Guided Image Filtering* (menggunakan gambar asli untuk memuluskan mask).
   - *Dense Conditional Random Fields (DenseCRF)*.
   - *Anti-aliasing boundary smoothing*.

3. **Langkah 3: Perfect Alignment & Integration - (Next Step)**
   Menggabungkan mask presisi tinggi ini ke dalam alur kerja utama (seperti *perspective warp* trapezoid) untuk memastikan tidak ada celah antara dinding dan lantai.

## Referensi Model Utama
- Model: [facebook/sam3](https://huggingface.co/facebook/sam3)
- Framework: `transformers` (HuggingFace) & `PyTorch`
- Presisi yang digunakan: `bfloat16` / `float16` dengan `torch.inference_mode()` untuk menjaga komputasi optimal.
