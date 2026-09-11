"""
Compatibility module: re-exports UNet and DoubleConv from model.py.
"""
import sys
from pathlib import Path

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

try:
    from src.model import UNet, DoubleConv
except ModuleNotFoundError:
    from model import UNet, DoubleConv

if __name__ == "__main__":
    import torch
    print("=" * 50)
    print("U-NET COMPATIBILITY TEST (src/unet.py -> src/model.py)")
    print("=" * 50)

    model = UNet(in_channels=3, out_channels=1, init_features=32)
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Input shape       : {dummy_input.shape}")
    print(f"Output shape      : {output.shape}")
    print(f"Trainable params  : {total_params:,}")
    print("Compatibility verification successful!")
