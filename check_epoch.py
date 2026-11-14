import sys, json
from pathlib import Path

# Check if best_model.pth exists and read its metadata
ckpt_path = Path("outputs/experiments/exp_A/best_model.pth")
if ckpt_path.exists():
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint epoch: {ck.get(chr(101)+chr(112)+chr(111)+chr(99)+chr(104), '?')}")
    print(f"Best val IoU: {ck.get(chr(98)+chr(101)+chr(115)+chr(116)+chr(95)+chr(118)+chr(97)+chr(108)+chr(95)+chr(105)+chr(111)+chr(117), 0):.4f}")
    print(f"Config: {ck.get(chr(99)+chr(111)+chr(110)+chr(102)+chr(105)+chr(103), {})}")
else:
    print("No checkpoint yet")
