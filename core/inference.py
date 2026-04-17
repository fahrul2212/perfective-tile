import torch
import torch.nn.functional as F
import time
from core.model import ST_RoomNet
from core.config import Config

class RoomNetService:
    def __init__(self):
        self.model = None

    def initialize(self):
        print(f"[*] Starting RoomNet on device: {Config.DEVICE}")
        try:
            self.model = ST_RoomNet(ref_path=Config.REF_IMAGE_PATH, out_size=(Config.INPUT_SIZE, Config.INPUT_SIZE))
            self.model.load_state_dict(torch.load(Config.WEIGHT_PATH, map_location=Config.DEVICE, weights_only=True))
            self.model.to(Config.DEVICE)
            if Config.DEVICE.type == 'cuda':
                self.model.half()
            self.model.eval()
            
            # Warm-up phase
            print("Warming up GPU...")
            with torch.no_grad():
                dummy_input = torch.zeros((1, 3, Config.INPUT_SIZE, Config.INPUT_SIZE)).to(Config.DEVICE)
                if Config.DEVICE.type == 'cuda':
                    dummy_input = dummy_input.half()
                for _ in range(3):
                    _ = self.model(dummy_input)
            print("[*] Model loaded successfully.")
        except Exception as e:
            print(f"[!] Failed to load model: {e}")

    def predict(self, img_gpu, orig_h, orig_w):
        if self.model is None:
            raise ValueError("Model not loaded")

        inp_batch = F.interpolate(img_gpu.unsqueeze(0), size=(Config.INPUT_SIZE, Config.INPUT_SIZE), mode='bilinear', align_corners=False)
        if Config.DEVICE.type == 'cuda':
            inp_batch = inp_batch.half()
        
        start_time = time.perf_counter()
        out = self.model(inp_batch)
        inference_time = (time.perf_counter() - start_time) * 1000
        
        dist = torch.abs(out - 4.0)
        mask_soft = torch.clamp(1.0 - dist, min=0.0)
        mask_soft = F.avg_pool2d(mask_soft, kernel_size=3, stride=1, padding=1)
        mask_upscaled = F.interpolate(mask_soft, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        
        mask_final_gpu = (mask_upscaled[0, 0] > 0.5).byte() * 255
        mask_cpu = mask_final_gpu.cpu().numpy()
        
        return mask_cpu, inference_time

roomnet_service = RoomNetService()
