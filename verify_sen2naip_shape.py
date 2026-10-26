"""
verify_sen2naip_shape.py
========================
Verification script: runs the CURRENT production SwinIR model directly
on a real SEN2NAIP Sentinel-2 sample and validates output shape compatibility:
  LR: [1, 3, 120, 120] -> SwinIR Model -> SR: [1, 3, 480, 480] == HR: [1, 3, 480, 480]
"""

import sys
import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "swinir"))

from sen2naip_dataset import SEN2NAIPDataset
from swinir.swinir_inference import _load_model

DEVICE = torch.device("cpu")


def main():
    print("=" * 70)
    print("SWINIR MODEL SHAPE VERIFICATION ON REAL SEN2NAIP SAMPLE")
    print("=" * 70)

    # 1. Load real SEN2NAIP sample
    dataset = SEN2NAIPDataset(
        root_dir=PROJECT_ROOT / "data" / "SEN2NAIP",
        split="all",
        rgb_only=True,
        lr_crop_size=(120, 120),
        random_crop=False,
    )

    sample = dataset[0]
    roi_id = sample["roi_id"]
    lr_tensor = sample["lr"].unsqueeze(0).to(DEVICE)  # (1, 3, 120, 120)
    hr_tensor = sample["hr"].unsqueeze(0).to(DEVICE)  # (1, 3, 480, 480)

    print(f"Sample ROI:              {roi_id}")
    print(f"Actual LR shape:         {list(lr_tensor.shape)}")
    print(f"Actual HR target shape:  {list(hr_tensor.shape)}")
    print(f"LR value range:          min={lr_tensor.min():.4f}, max={lr_tensor.max():.4f}, mean={lr_tensor.mean():.4f}")
    print(f"HR value range:          min={hr_tensor.min():.4f}, max={hr_tensor.max():.4f}, mean={hr_tensor.mean():.4f}")

    # Check input range [0, 1]
    assert lr_tensor.min() >= 0.0 and lr_tensor.max() <= 1.0, "LR input outside [0, 1]!"
    assert hr_tensor.min() >= 0.0 and hr_tensor.max() <= 1.0, "HR target outside [0, 1]!"

    # 2. Load CURRENT production SwinIR model checkpoint
    model, err = _load_model(DEVICE)
    if err:
        print(f"[ERROR] Failed to load production SwinIR model: {err}")
        sys.exit(1)

    print("Loaded current model:    models/swinir_x4.pth (SwinIR-M classical SR x4)")

    # 3. Direct model forward pass on LR tensor
    model.eval()
    with torch.no_grad():
        sr_output = model(lr_tensor)

    print(f"Actual SR output shape:  {list(sr_output.shape)}")
    print(f"SR value range:          min={sr_output.min():.4f}, max={sr_output.max():.4f}, mean={sr_output.mean():.4f}")

    has_nan = torch.isnan(sr_output).any().item()
    has_inf = torch.isinf(sr_output).any().item()
    print(f"NaN present:             {has_nan}")
    print(f"Inf present:             {has_inf}")

    # 4. Strict assertions
    assert not has_nan, "NaN detected in SR output!"
    assert not has_inf, "Inf detected in SR output!"
    assert list(sr_output.shape) == [1, 3, 480, 480], f"Expected [1, 3, 480, 480], got {list(sr_output.shape)}"
    assert list(sr_output.shape) == list(hr_tensor.shape), f"SR shape {list(sr_output.shape)} != HR shape {list(hr_tensor.shape)}"

    print("=" * 70)
    print("[VERIFICATION SUCCESS] The 4x training pipeline is exactly shape-compatible:")
    print(f"  Input  LR: {list(lr_tensor.shape)} (10 m GSD)")
    print(f"  Output SR: {list(sr_output.shape)} (2.5 m GSD)")
    print(f"  Target HR: {list(hr_tensor.shape)} (2.5 m GSD)")
    print("=" * 70)


if __name__ == "__main__":
    main()
