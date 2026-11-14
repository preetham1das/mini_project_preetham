import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import torch
from PIL import Image
import numpy as np
from src.temporal_model import TemporalUNet
from backend.main import preprocess_image, TEMPORAL_CHECKPOINT_PATH

DEVICE = torch.device("cpu")
checkpoint = torch.load(TEMPORAL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
model = TemporalUNet(in_channels=6, out_channels=1, init_features=32).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_a_dir = Path("data/Forest-Change/extracted/Forest-Change-dataset/images/test/A")
test_b_dir = Path("data/Forest-Change/extracted/Forest-Change-dataset/images/test/B")
for p in sorted([f.stem for f in test_a_dir.glob("*.png")])[:12]:
    img_a = Image.open(test_a_dir / f"{p}.png").convert("RGB")
    img_b = Image.open(test_b_dir / f"{p}.png").convert("RGB")
    img_a_tensor = preprocess_image(img_a)
    img_b_tensor = preprocess_image(img_b)
    temporal_tensor = torch.cat([img_a_tensor, img_b_tensor], dim=1).to(DEVICE)
    with torch.no_grad():
        logits = model(temporal_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    c5 = np.sum(probs > 0.5)
    print(f"{p}: {c5} pixels ({c5/probs.size*100:.2f}%)")
