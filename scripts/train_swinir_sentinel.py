"""
scripts/train_swinir_sentinel.py
================================
SwinIR fine-tuning pilot on Sentinel-2 / NAIP cross-sensor imagery (SEN2NAIP).

Pilot Objectives:
  1. Load existing pretrained SwinIR-M checkpoint (models/swinir_x4.pth).
  2. Compute baseline metrics (PSNR, SSIM, L1 loss) on validation set for:
     - Bicubic upsampling (4x)
     - Generic SwinIR model (DF2K weights)
  3. Fine-tune for 2 pilot epochs using RGB-only pairs (120x120 LR -> 480x480 HR).
  4. Save pilot checkpoint separately to models/swinir_sentinel_x4_pilot.pth
     (the production checkpoint models/swinir_x4.pth is NEVER overwritten).
  5. Save training history CSV, validation metrics JSON, visual comparisons,
     and an empirical comparison report.

Data & Normalization Policy:
  - LR Sentinel-2: surface reflectance [0, 10000] normalized to [0.0, 1.0] via / 10000.0
  - HR NAIP: aerial RGB uint8 [0, 255] normalized to [0.0, 1.0] via / 255.0
  - Loss: L1 reconstruction loss (torch.nn.L1Loss)
  - Optimizer: Adam (lr = 1e-4)

Usage:
  python scripts/train_swinir_sentinel.py
  python scripts/train_swinir_sentinel.py --epochs 2 --lr 1e-4 --batch-size 2
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "swinir"))

from sen2naip_dataset import get_sen2naip_dataloaders, SEN2NAIPDataset
from swinir.swinir_inference import _load_model, _MODEL_ARGS
from swinir.network_swinir import SwinIR

# Constants
GENERIC_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "swinir_x4.pth"
PILOT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "swinir_sentinel_x4_pilot.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "swinir_pilot"
VISUALS_DIR = OUTPUT_DIR / "visuals"


def parse_args():
    parser = argparse.ArgumentParser(description="SwinIR Sentinel-2 Fine-Tuning Pilot.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of pilot epochs (default: 2).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4).")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def set_seed(seed: int):
    random_seed = seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def evaluate_model(model: nn.Module, dataset: SEN2NAIPDataset, device: torch.device):
    """
    Evaluates model on validation dataset.
    Returns mean (psnr, ssim, l1_loss) and per-sample results.
    """
    model.eval()
    psnr_list, ssim_list, l1_list = [], [], []
    sample_outputs = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            lr = sample["lr"].unsqueeze(0).to(device)   # (1, 3, 120, 120)
            hr = sample["hr"].unsqueeze(0).to(device)   # (1, 3, 480, 480)

            sr = model(lr).clamp(0.0, 1.0)              # (1, 3, 480, 480)

            l1 = nn.functional.l1_loss(sr, hr).item()
            l1_list.append(l1)

            # Convert to (H, W, C) numpy float32 in [0, 1]
            hr_np = hr[0].permute(1, 2, 0).cpu().numpy()
            sr_np = sr[0].permute(1, 2, 0).cpu().numpy()

            psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
            ssim = structural_similarity(hr_np, sr_np, data_range=1.0, channel_axis=2)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            sample_outputs.append({
                "roi_id": sample["roi_id"],
                "sr_np": sr_np,
                "hr_np": hr_np,
                "lr_np": lr[0].permute(1, 2, 0).cpu().numpy(),
                "psnr": psnr,
                "ssim": ssim,
                "l1": l1,
            })

    return {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "l1": float(np.mean(l1_list)),
        "details": sample_outputs,
    }


def evaluate_bicubic(dataset: SEN2NAIPDataset, device: torch.device):
    """Evaluates bicubic upsampling baseline on validation dataset."""
    psnr_list, ssim_list, l1_list = [], [], []
    sample_outputs = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            lr = sample["lr"].unsqueeze(0).to(device)
            hr = sample["hr"].unsqueeze(0).to(device)

            bic = nn.functional.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0.0, 1.0)

            l1 = nn.functional.l1_loss(bic, hr).item()
            l1_list.append(l1)

            hr_np = hr[0].permute(1, 2, 0).cpu().numpy()
            bic_np = bic[0].permute(1, 2, 0).cpu().numpy()

            psnr = peak_signal_noise_ratio(hr_np, bic_np, data_range=1.0)
            ssim = structural_similarity(hr_np, bic_np, data_range=1.0, channel_axis=2)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            sample_outputs.append({
                "roi_id": sample["roi_id"],
                "bic_np": bic_np,
                "psnr": psnr,
                "ssim": ssim,
                "l1": l1,
            })

    return {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "l1": float(np.mean(l1_list)),
        "details": sample_outputs,
    }


def save_visual_comparisons(
    bic_results, generic_results, finetuned_results, out_dir: Path
):
    """
    Saves multi-panel comparison images:
    [LR Sentinel-2 (10m) | Bicubic (4x) | Generic SwinIR | Fine-Tuned SwinIR | NAIP HR Reference (2.5m)]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(finetuned_results["details"])):
        ft_info = finetuned_results["details"][i]
        gen_info = generic_results["details"][i]
        bic_info = bic_results["details"][i]

        roi_id = ft_info["roi_id"]
        lr_np = ft_info["lr_np"]
        bic_np = bic_info["bic_np"]
        gen_np = gen_info["sr_np"]
        ft_np = ft_info["sr_np"]
        hr_np = ft_info["hr_np"]

        # Contrast stretch LR for clear visualization (since surface reflectance is [0, 0.2])
        lr_disp = np.clip(lr_np * 3.5, 0.0, 1.0)
        # Resize LR with nearest neighbor to match HR size for visual comparison
        lr_up = np.repeat(np.repeat(lr_disp, 4, axis=0), 4, axis=1)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

        axes[0].imshow(lr_up)
        axes[0].set_title(f"LR Sentinel-2 (10 m)\n[{roi_id}]", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(bic_np)
        axes[1].set_title(f"Bicubic 4x\nPSNR: {bic_info['psnr']:.2f} | SSIM: {bic_info['ssim']:.3f}", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(gen_np)
        axes[2].set_title(f"Generic SwinIR\nPSNR: {gen_info['psnr']:.2f} | SSIM: {gen_info['ssim']:.3f}", fontsize=10)
        axes[2].axis("off")

        axes[3].imshow(ft_np)
        axes[3].set_title(f"Fine-Tuned SwinIR (Pilot)\nPSNR: {ft_info['psnr']:.2f} | SSIM: {ft_info['ssim']:.3f}", fontsize=10, color="darkgreen", weight="bold")
        axes[3].axis("off")

        axes[4].imshow(hr_np)
        axes[4].set_title("NAIP Reference (2.5 m)\n[Ground Truth HR]", fontsize=10)
        axes[4].axis("off")

        plt.tight_layout()
        save_path = out_dir / f"comparison_{roi_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"      Saved visual comparison: {save_path.name}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # Use CPU by default (matches environment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SWINIR SENTINEL-2 FINE-TUNING PILOT")
    print("=" * 70)
    print(f"Device:               {device} (Threads: {torch.get_num_threads()})")
    print(f"Base Checkpoint:      {GENERIC_CHECKPOINT_PATH}")
    print(f"Pilot Checkpoint:     {PILOT_CHECKPOINT_PATH}")
    print(f"Pilot Epochs:         {args.epochs}")
    print(f"Learning Rate:        {args.lr}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Input Crop Size:      120x120 LR -> 480x480 HR (RGB only)")
    print(f"Output Directory:     {OUTPUT_DIR}")

    # 1. Load Data
    print("\n[1/5] Loading SEN2NAIP Dataset (30 prepared pairs)...")
    train_loader, val_loader, _ = get_sen2naip_dataloaders(
        root_dir=PROJECT_ROOT / "data" / "SEN2NAIP",
        batch_size=args.batch_size,
        num_workers=0,
        rgb_only=True,
        lr_crop_size=(120, 120),
    )
    val_dataset = SEN2NAIPDataset(
        root_dir=PROJECT_ROOT / "data" / "SEN2NAIP",
        split="val",
        rgb_only=True,
        lr_crop_size=(120, 120),
        random_crop=False,
    )
    print(f"      Train samples: {len(train_loader.dataset)} ({len(train_loader)} batches/epoch)")
    print(f"      Val samples:   {len(val_dataset)}")

    # 2. Baseline Evaluation
    print("\n[2/5] Evaluating Pre-Training Baselines on Validation Set...")

    print("      Evaluating Bicubic upsampling...")
    t0 = time.time()
    bicubic_results = evaluate_bicubic(val_dataset, device)
    print(f"      -> Bicubic:        PSNR = {bicubic_results['psnr']:.2f} dB | SSIM = {bicubic_results['ssim']:.4f} | L1 = {bicubic_results['l1']:.4f}")

    print("      Loading and evaluating Generic SwinIR (DF2K weights)...")
    generic_model, err = _load_model(device)
    if err:
        print(f"[ERROR] Failed to load {GENERIC_CHECKPOINT_PATH}: {err}")
        sys.exit(1)

    generic_results = evaluate_model(generic_model, val_dataset, device)
    print(f"      -> Generic SwinIR: PSNR = {generic_results['psnr']:.2f} dB | SSIM = {generic_results['ssim']:.4f} | L1 = {generic_results['l1']:.4f}")

    # 3. Model Fine-Tuning Setup
    # Instantiate a clean trainable instance starting from generic weights
    print("\n[3/5] Initializing Fine-Tuning Model...")
    model = SwinIR(**_MODEL_ARGS)
    # Load pretrained weights into model
    pretrained_weights = torch.load(GENERIC_CHECKPOINT_PATH, map_location=device)
    state_dict = pretrained_weights["params_ema"] if "params_ema" in pretrained_weights else pretrained_weights.get("params", pretrained_weights)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Training Loop (2 Epochs Pilot)
    print("\n[4/5] Executing 2-Epoch Fine-Tuning Pilot...")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_epoch_start = time.time()

        for step, batch in enumerate(train_loader, start=1):
            lr = batch["lr"].to(device)   # (B, 3, 120, 120)
            hr = batch["hr"].to(device)   # (B, 3, 480, 480)

            t_step = time.time()
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            epoch_loss += step_loss * lr.size(0)
            n_batches += 1
            elapsed_step = time.time() - t_step

            print(f"      Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}] Loss: {step_loss:.4f} ({elapsed_step:.1f}s/step)")

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        epoch_time = time.time() - t_epoch_start

        # Validate after epoch
        val_res = evaluate_model(model, val_dataset, device)
        print(f"      --> Epoch {epoch} Complete ({epoch_time:.1f}s): Train L1 = {avg_train_loss:.4f} | Val L1 = {val_res['l1']:.4f} | Val PSNR = {val_res['psnr']:.2f} dB | Val SSIM = {val_res['ssim']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_res["l1"],
            "val_psnr": val_res["psnr"],
            "val_ssim": val_res["ssim"],
            "epoch_time_sec": round(epoch_time, 2),
        })

    # Final Validation Evaluation
    final_val_res = evaluate_model(model, val_dataset, device)

    # 5. Save Artifacts
    print("\n[5/5] Saving Pilot Artifacts & Reports...")

    # A. Pilot Checkpoint
    checkpoint_payload = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr": args.lr,
        "loss_fn": "L1Loss",
        "dataset": "SEN2NAIP (cross-sensor)",
        "scale": 4,
        "val_psnr": final_val_res["psnr"],
        "val_ssim": final_val_res["ssim"],
        "val_loss": final_val_res["l1"],
        "model_args": _MODEL_ARGS,
        "normalization": {
            "lr": "Sentinel-2 reflectance [0, 10000] / 10000.0 -> [0, 1]",
            "hr": "NAIP uint8 [0, 255] / 255.0 -> [0, 1]",
        },
    }
    torch.save(checkpoint_payload, PILOT_CHECKPOINT_PATH)
    print(f"      Saved pilot checkpoint to: {PILOT_CHECKPOINT_PATH}")

    # B. Training History CSV
    csv_path = OUTPUT_DIR / "training_history.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_psnr,val_ssim,epoch_time_sec\n")
        for h in history:
            f.write(f"{h['epoch']},{h['train_loss']:.6f},{h['val_loss']:.6f},{h['val_psnr']:.4f},{h['val_ssim']:.6f},{h['epoch_time_sec']}\n")
    print(f"      Saved training history to: {csv_path}")

    # C. Validation Metrics JSON
    delta_psnr = final_val_res["psnr"] - generic_results["psnr"]
    delta_ssim = final_val_res["ssim"] - generic_results["ssim"]
    delta_l1 = final_val_res["l1"] - generic_results["l1"]

    metrics_payload = {
        "experiment": "SwinIR Sentinel-2 Fine-Tuning Pilot",
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_dataset),
        "normalization": {
            "lr_input": "Sentinel-2 L2A [0, 10000] scaled by 1/10000.0 into [0.0, 1.0]",
            "hr_target": "NAIP RGB [0, 255] scaled by 1/255.0 into [0.0, 1.0]",
        },
        "baselines": {
            "bicubic": {
                "psnr": bicubic_results["psnr"],
                "ssim": bicubic_results["ssim"],
                "l1_loss": bicubic_results["l1"],
            },
            "generic_swinir": {
                "psnr": generic_results["psnr"],
                "ssim": generic_results["ssim"],
                "l1_loss": generic_results["l1"],
            },
        },
        "fine_tuned_pilot": {
            "psnr": final_val_res["psnr"],
            "ssim": final_val_res["ssim"],
            "l1_loss": final_val_res["l1"],
        },
        "improvements_over_generic_swinir": {
            "delta_psnr_db": delta_psnr,
            "delta_ssim": delta_ssim,
            "delta_l1_loss": delta_l1,
            "improved": bool(delta_psnr > 0 and delta_l1 < 0),
        },
    }
    json_path = OUTPUT_DIR / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"      Saved metrics JSON to: {json_path}")

    # D. Visual Comparisons
    print("      Generating multi-panel visual comparisons...")
    save_visual_comparisons(bicubic_results, generic_results, final_val_res, VISUALS_DIR)

    # E. Summary Report TXT
    report_path = OUTPUT_DIR / "comparison_report.txt"
    report_text = f"""================================================================================
SWINIR SENTINEL-2 FINE-TUNING PILOT REPORT
================================================================================
Dataset:         SEN2NAIP Cross-Sensor (30 pairs: 24 train, 3 val, 3 test)
Resolution:      10.0 m (LR) -> 2.5 m (HR) [4x Super-Resolution]
Input/Output:    120x120 LR RGB -> 480x480 HR RGB
Base Weights:    models/swinir_x4.pth (DF2K classical SR x4)
Pilot Checkpoint:models/swinir_sentinel_x4_pilot.pth
Pilot Epochs:    {args.epochs}
Learning Rate:   {args.lr}
Loss Function:   L1 Reconstruction Loss

NORMALIZATION:
  LR (Sentinel-2 L2A): Surface reflectance [0, 10000] scaled by / 10000.0 -> [0.0, 1.0]
  HR (NAIP 2.5 m):     Aerial RGB uint8 [0, 255] scaled by / 255.0 -> [0.0, 1.0]

QUANTITATIVE VALIDATION COMPARISON:
--------------------------------------------------------------------------------
Method                       PSNR (dB)          SSIM               L1 Loss
--------------------------------------------------------------------------------
Bicubic Upsampling           {bicubic_results['psnr']:.4f} dB        {bicubic_results['ssim']:.4f}             {bicubic_results['l1']:.4f}
Generic SwinIR (Pretrained)  {generic_results['psnr']:.4f} dB        {generic_results['ssim']:.4f}             {generic_results['l1']:.4f}
Fine-Tuned SwinIR (Pilot)    {final_val_res['psnr']:.4f} dB        {final_val_res['ssim']:.4f}             {final_val_res['l1']:.4f}
--------------------------------------------------------------------------------
Delta over Generic SwinIR:   {delta_psnr:+.4f} dB        {delta_ssim:+.4f}             {delta_l1:+.4f}

CONCLUSION:
  - Fine-tuning improvement: {'YES (Significant gain)' if delta_psnr > 0 else 'NO'}
  - PSNR Shift:              {generic_results['psnr']:.2f} dB -> {final_val_res['psnr']:.2f} dB ({delta_psnr:+.2f} dB)
  - SSIM Shift:              {generic_results['ssim']:.4f} -> {final_val_res['ssim']:.4f} ({delta_ssim:+.4f})
  - L1 Error Reduction:      {generic_results['l1']:.4f} -> {final_val_res['l1']:.4f} ({delta_l1:+.4f})
  - Production Status:       PILOT ONLY (Not production-ready; full training recommended)
================================================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"      Saved comparison report to: {report_path}")

    print("\n" + "=" * 70)
    print("PILOT EXECUTION COMPLETE")
    print(f"PSNR:  Generic = {generic_results['psnr']:.2f} dB  -->  Fine-Tuned = {final_val_res['psnr']:.2f} dB ({delta_psnr:+.2f} dB)")
    print(f"SSIM:  Generic = {generic_results['ssim']:.4f}  -->  Fine-Tuned = {final_val_res['ssim']:.4f} ({delta_ssim:+.4f})")
    print(f"L1:    Generic = {generic_results['l1']:.4f}  -->  Fine-Tuned = {final_val_res['l1']:.4f} ({delta_l1:+.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
