from pathlib import Path
import torch
from PIL import Image
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from src.temporal_model import TemporalUNet
from backend.main import preprocess_image, TEMPORAL_CHECKPOINT_PATH

DEVICE = torch.device("cpu")
temporal_checkpoint = torch.load(TEMPORAL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
temporal_model = TemporalUNet(in_channels=6, out_channels=1, init_features=32).to(DEVICE)
temporal_model.load_state_dict(temporal_checkpoint["model_state_dict"])
temporal_model.eval()

test_a_path = "data/Forest-Change/extracted/Forest-Change-dataset/images/test/A/test_000001.png"
test_b_path = "data/Forest-Change/extracted/Forest-Change-dataset/images/test/B/test_000001.png"

img_a = Image.open(test_a_path).convert("RGB")
img_b = Image.open(test_b_path).convert("RGB")

img_a_tensor = preprocess_image(img_a)
img_b_tensor = preprocess_image(img_b)

temporal_tensor = torch.cat([img_a_tensor, img_b_tensor], dim=1).to(DEVICE)

with torch.no_grad():
    logits = temporal_model(temporal_tensor)
    probabilities = torch.sigmoid(logits)

probs_np = probabilities[0, 0].cpu().numpy()

print(f"Min probability: {probs_np.min():.6f}")
print(f"Max probability: {probs_np.max():.6f}")
print(f"Mean probability: {probs_np.mean():.6f}")

for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    count = np.sum(probs_np > t)
    print(f"Pixels above {t}: {count} ({count/probs_np.size*100:.2f}%)")
