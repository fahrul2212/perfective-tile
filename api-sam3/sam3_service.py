import torch
from transformers import Sam3Processor, Sam3Model
from huggingface_hub import login
from PIL import Image
import numpy as np
import io
import os

# Resolusi maksimum input sebelum masuk ke encoder (Layer 4)
MAX_INFERENCE_SIZE = 1024

class Sam3Service:
    def __init__(self, hf_token: str = None):
        # Baca token dari parameter, lalu fallback ke environment variable
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError(
                "HuggingFace token tidak ditemukan. "
                "Set environment variable HF_TOKEN atau berikan via parameter."
            )
        login(token=self.hf_token)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Layer 1: Pilih dtype terbaik yang didukung hardware
        if self.device == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = torch.float32  # CPU tidak mendukung BF16 inferensi stable

        print(f"Loading SAM3 model on {self.device} with dtype={self.dtype}... (this may take a while)")

        # Load processor and model
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=self.hf_token)
        self.model = Sam3Model.from_pretrained(
            "facebook/sam3",
            token=self.hf_token,
            torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

        # Layer 2: torch.compile — hanya efektif di GPU dengan Triton (Linux/WSL)
        # Di Windows native, Triton tidak tersedia sehingga compile di-skip otomatis
        if self.device == "cuda":
            try:
                import triton  # noqa: F401
                print("Compiling SAM3 model with torch.compile (reduce-overhead)...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("torch.compile done. First inference akan warm-up.")
            except (ImportError, Exception) as e:
                print(f"torch.compile di-skip (Triton tidak tersedia: {e}). Berjalan dalam eager mode.")

    @staticmethod
    def _resize_for_inference(image: Image.Image, max_size: int = MAX_INFERENCE_SIZE) -> Image.Image:
        """Layer 4: Resize gambar ke max_size di sisi terpanjang sebelum masuk encoder."""
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        return image

    def predict_floor_mask(self, image: Image.Image) -> bytes:
        """
        Melakukan segmentasi terhadap gambar dengan text prompt "floor".
        Menggabungkan mask (jika ada lebih dari 1) dan mereturn byte Stream PNG.
        """
        # Pastikan gambar diubah ke RGB
        image = image.convert("RGB")
        original_size = image.size  # (W, H) — simpan untuk resize mask output

        # Layer 4: Resize sebelum encode — kurangi beban komputasi encoder secara kuadratik
        image = self._resize_for_inference(image)

        # Proses text prompt
        text_prompt = "floor"
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        # Layer 1+2: Inferensi dengan precision casting + torch.compile graph
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast(self.device, dtype=self.dtype):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        # Post-processing — gunakan original_size agar mask output sesuai resolusi gambar asli
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=[original_size[::-1]]  # PIL (W,H) -> (H,W) untuk target_sizes
        )[0]
        
        masks = results.get("masks", [])
        
        # Gabungkan mask jika ada lebih dari 1 instance floor, jika tidak ada kembalikan mask kosong (hitam)
        if len(masks) == 0:
            final_mask_np = np.zeros(original_size[::-1], dtype=np.uint8) # original (W,H) -> (H,W)
        else:
            # 1. OPTIMASI: Stack & Operasi logical OR secara simultan tanpa loop
            if isinstance(masks, list):
                stacked_masks = torch.stack(masks)
            else:
                stacked_masks = masks  # Jika sudah berupa tensor (N, H, W)
                
            combined_tensor = stacked_masks.any(dim=0)
            
            # 2. OPTIMASI: Casting dan perkalian matriks dikerjakan di operasi Tensor sebelum di pindah ke CPU (Numpy)
            final_mask_np = (combined_tensor.to(torch.uint8) * 255).cpu().numpy()
            
        # Konversi ke PIL Image lalu ke bytes buffer format PNG
        mask_img = Image.fromarray(final_mask_np, mode="L")
        
        buf = io.BytesIO()
        # 3. OPTIMASI: Menurunkan level kompresi bawaan PNG
        mask_img.save(buf, format="PNG", compress_level=1)
        buf.seek(0)
        
        return buf.read()
