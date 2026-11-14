import sys
from pathlib import Path
sys.path.append("src")
import torch
from PIL import Image
import numpy as np
from temporal_dataset import TemporalForestDataset

root = Path("data/Forest-Change/extracted/Forest-Change-dataset/images")

total_pos_train = 0
total_neg_train = 0
total_pos_val = 0
total_neg_val = 0

train_ds = TemporalForestDataset(root / "train", augment=False)
for _, mask, _ in train_ds:
    total_pos_train += mask.sum().item()
    total_neg_train += (mask == 0).sum().item()

val_ds = TemporalForestDataset(root / "val", augment=False)
for _, mask, _ in val_ds:
    total_pos_val += mask.sum().item()
    total_neg_val += (mask == 0).sum().item()

test_ds = TemporalForestDataset(root / "test", augment=False)
total_pos_test = 0
total_neg_test = 0
for _, mask, _ in test_ds:
    total_pos_test += mask.sum().item()
    total_neg_test += (mask == 0).sum().item()

print("=" * 60)
print("DATASET CLASS IMBALANCE ANALYSIS")
print("=" * 60)
print(f"\nTRAIN ({len(train_ds)} pairs):")
print(f"  Positive pixels (deforestation): {int(total_pos_train):,}")
print(f"  Negative pixels (background):    {int(total_neg_train):,}")
print(f"  Positive ratio:                  {total_pos_train/(total_pos_train+total_neg_train)*100:.2f}%")
print(f"  Neg/Pos ratio (pos_weight):      {total_neg_train/(total_pos_train+1e-6):.2f}")

print(f"\nVAL ({len(val_ds)} pairs):")
print(f"  Positive pixels (deforestation): {int(total_pos_val):,}")
print(f"  Negative pixels (background):    {int(total_neg_val):,}")
print(f"  Positive ratio:                  {total_pos_val/(total_pos_val+total_neg_val)*100:.2f}%")
print(f"  Neg/Pos ratio (pos_weight):      {total_neg_val/(total_pos_val+1e-6):.2f}")

print(f"\nTEST ({len(test_ds)} pairs):")
print(f"  Positive pixels (deforestation): {int(total_pos_test):,}")
print(f"  Negative pixels (background):    {int(total_neg_test):,}")
print(f"  Positive ratio:                  {total_pos_test/(total_pos_test+total_neg_test)*100:.2f}%")
print(f"  Neg/Pos ratio (pos_weight):      {total_neg_test/(total_pos_test+1e-6):.2f}")

print(f"\nImage original sizes:")
pil = Image.open(list((root / "train" / "A").glob("*.png"))[0])
print(f"  A: {pil.size}")
pil = Image.open(list((root / "train" / "label").glob("*.png"))[0])
arr = np.array(pil)
print(f"  Label unique values: {np.unique(arr)}")
