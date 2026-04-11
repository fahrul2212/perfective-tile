import torch
import cv2
import numpy as np
from arch_torch import ST_RoomNet

def debug_classes():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ST_RoomNet(ref_path='assets/ref_img2.png', out_size=(400, 400))
    model.load_state_dict(torch.load('persfective.pth', map_location=device))
    model.to(device).eval()
    
    img = cv2.imread('assets/1.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (400, 400))
    inp = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    
    with torch.no_grad():
        out = model(inp)
    
    out_np = np.rint(out.cpu().numpy()[0,0]).astype(np.uint8)
    for i in range(1, 6):
        mask = (out_np == i).astype(np.uint8) * 255
        cv2.imwrite(f'debug_class_{i}.png', mask)
        print(f"Saved debug_class_{i}.png")

if __name__ == "__main__":
    debug_classes()
