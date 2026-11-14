import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import torch
from PIL import Image
import numpy as np

from src.temporal_model import TemporalUNet
from backend.main import preprocess_image, TEMPORAL_CHECKPOINT_PATH

DEVICE = torch.device("cpu")

temporal_checkpoint = torch.load(TEMPORAL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)

temporal_model = TemporalUNet(in_channels=6, out_channels=1, init_features=32).to(DEVICE)
temporal_model.load_state_dict(temporal_checkpoint["model_state_dict"])
temporal_model.eval()

val_a_dir = Path("data/Forest-Change/extracted/Forest-Change-dataset/images/val/A")
val_b_dir = Path("data/Forest-Change/extracted/Forest-Change-dataset/images/val/B")

print(f"{'Sample':<15} {'Mean Prob':<12} {'Max Prob':<12} {'>0.1':<8} {'>0.3':<8} {'>0.5':<8} {'>0.7':<8} {'>0.9':<8}")

for p in sorted([f.stem for f in val_a_dir.glob("*.png")])[:6]:
    img_a = Image.open(val_a_dir / f"{p}.png").convert("RGB")
    img_b = Image.open(val_b_dir / f"{p}.png").convert("RGB")
    
    img_a_tensor = preprocess_image(img_a)
    img_b_tensor = preprocess_image(img_b)
    
    temporal_tensor = torch.cat([img_a_tensor, img_b_tensor], dim=1).to(DEVICE)
    
    with torch.no_grad():
        logits = temporal_model(temporal_tensor)
        probabilities = torch.sigmoid(logits)
        
    probs = probabilities[0, 0].cpu().numpy()
    
    mean_p = probs.mean()
    max_p = probs.max()
    c1 = np.sum(probs > 0.1)
    c3 = np.sum(probs > 0.3)
    c5 = np.sum(probs > 0.5)
    c7 = np.sum(probs > 0.7)
    c9 = np.sum(probs > 0.9)
    
    print(f"{p:<15} {mean_p:<12.4f} {max_p:<12.4f} {c1:<8} {c3:<8} {c5:<8} {c7:<8} {c9:<8}")
