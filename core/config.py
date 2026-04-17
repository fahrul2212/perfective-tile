import os
from pathlib import Path
import torch

class Config:
    # Directories
    ROOT_DIR = Path(__file__).parent.parent  # because it's in core/ now
    OUTPUT_DIR = ROOT_DIR / "outputs"
    
    # Model Configuration
    WEIGHT_PATH = str(ROOT_DIR / "weights/persfective.pth")
    REF_IMAGE_PATH = str(ROOT_DIR / "assets/ref_img2.png")
    INPUT_SIZE = 400
    
    # SAM3 Microservice settings
    SAM3_BASE_URL = os.getenv("SAM3_BASE_URL", "http://localhost:8001")
    SAM3_TIMEOUT = float(os.getenv("SAM3_TIMEOUT", "90.0"))
    
    # Device Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Inisialisasi folder saat diimpor
Config.setup()
