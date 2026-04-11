import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny
import cv2
import numpy as np

class ProjectiveTransformer(nn.Module):
    def __init__(self, out_size=(400, 400)):
        super(ProjectiveTransformer, self).__init__()
        self.out_size = out_size
        
        # Create meshgrid mapping: [-1, 1]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, out_size[0]),
            torch.linspace(-1.0, 1.0, out_size[1]),
            indexing='ij'
        )
        self.register_buffer('pixel_grid', torch.stack([xx.reshape(-1), yy.reshape(-1), torch.ones_like(xx.reshape(-1))]))

    def forward(self, x, theta):
        # theta is (B, 8)
        B = theta.shape[0]
        # Complete the 3x3 matrix
        ones = torch.ones(B, 1, device=theta.device, dtype=theta.dtype)
        matrix = torch.cat([theta, ones], dim=1).view(B, 3, 3)
        
        # Transform grid: (B, 3, 3) @ (3, H*W) -> (B, 3, H*W)
        grid_flat = self.pixel_grid.unsqueeze(0).expand(B, -1, -1)
        T_g = torch.bmm(matrix, grid_flat)
        
        # Homogeneous to Cartesian
        z_s = T_g[:, 2, :].unsqueeze(1)
        z_s = torch.clamp(z_s, min=1e-8)
        xy_s = T_g[:, :2, :] / z_s
        
        # Reshape to (B, H, W, 2)
        grid_sample = xy_s.permute(0, 2, 1).reshape(B, self.out_size[0], self.out_size[1], 2)
        
        # Bilinear sampling
        output = F.grid_sample(x, grid_sample, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return output

class ST_RoomNet(nn.Module):
    def __init__(self, ref_path="assets/ref_img2.png", out_size=(400, 400)):
        super(ST_RoomNet, self).__init__()
        self.backbone = convnext_tiny(weights=None)
        self.backbone.classifier = nn.Identity()
        
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.head = nn.Linear(768, 8)
        
        self.transformer = ProjectiveTransformer(out_size=out_size)
        
        # Load and prepare reference image
        ref = cv2.imread(ref_path, 0)
        if ref is None:
             # Create dummy if missing (for initialization)
             ref = np.zeros(out_size, dtype=np.uint8)
        else:
             ref = cv2.resize(ref, out_size, interpolation=cv2.INTER_NEAREST)
             
        ref_tensor = torch.from_numpy(ref).float() / 51.0
        self.register_buffer('ref_img', ref_tensor.unsqueeze(0).unsqueeze(0)) # (1, 1, H, W)

    def forward(self, x):
        B = x.shape[0]
        # Predict theta from input image
        features = self.backbone.features(x)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        normed = self.norm(pooled)
        theta = self.head(normed)
        
        # Transform the FIXED reference image for every batch
        ref_batch = self.ref_img.expand(B, -1, -1, -1)
        return self.transformer(ref_batch, theta)

if __name__ == "__main__":
    model = ST_RoomNet()
    test_input = torch.randn(1, 3, 400, 400)
    out = model(test_input)
    print(f"Output shape: {out.shape}") # Expect (1, 1, 400, 400)
