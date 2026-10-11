import sys
sys.path.append("src")
import torch
from temporal_dataset import get_temporal_dataloaders, compute_pos_weight
from temporal_model import TemporalUNet, WeightedBCEDiceLoss, FocalDiceLoss

# Dataset smoke test
print("--- Dataset test ---")
for diff in [False, True]:
    tl, vl, testl = get_temporal_dataloaders(batch_size=2, diff_channels=diff)
    imgs, masks, _ = next(iter(tl))
    ch = 9 if diff else 6
    assert imgs.shape == (2, ch, 256, 256), f"Shape mismatch: {imgs.shape}"
    assert masks.shape == (2, 1, 256, 256)
    print(f"  diff_channels={diff}: imgs={imgs.shape} masks={masks.shape} OK")

# pos_weight
pw = compute_pos_weight("data/Forest-Change/extracted/Forest-Change-dataset/images")
print(f"  pos_weight = {pw:.2f}")

# Model smoke test  
print("--- Model test ---")
for ch in [6, 9]:
    m = TemporalUNet(in_channels=ch, out_channels=1, init_features=32)
    dummy = torch.randn(2, ch, 256, 256)
    out = m(dummy)
    assert out.shape == (2, 1, 256, 256), f"Output shape: {out.shape}"
    params = sum(p.numel() for p in m.parameters())
    print(f"  in_channels={ch}: output={out.shape} params={params:,} OK")

# Loss smoke test
print("--- Loss test ---")
logits = torch.randn(2, 1, 256, 256)
masks  = (torch.rand(2, 1, 256, 256) > 0.95).float()  # ~5% positive
loss1 = WeightedBCEDiceLoss(pos_weight_val=20.0)(logits, masks)
loss2 = FocalDiceLoss()(logits, masks)
print(f"  WeightedBCEDice loss = {loss1.item():.4f}")
print(f"  FocalDice loss       = {loss2.item():.4f}")

print("\nAll smoke tests PASSED")
