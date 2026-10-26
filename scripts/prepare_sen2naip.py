"""
scripts/prepare_sen2naip.py
===========================
Dataset preparation script for Sentinel-2 / NAIP super-resolution using the
official SEN2NAIP dataset (Hugging Face: isp-uv-es/SEN2NAIP).

Dataset Reference:
  Aybar et al., 'SEN2NAIP: A large-scale dataset for Sentinel-2 Image Super-Resolution',
  Scientific Data 11, 1389 (2024). https://doi.org/10.1038/s41597-024-04214-y

Key Features:
  - Selectively downloads ONLY the cross-sensor subset (or a specified N-pair slice),
    preventing unnecessary downloads of the 165+ GB synthetic portion.
  - Uses HTTP Range requests via huggingface_hub.HfFileSystem to stream and extract
    individual ROIs directly from cross-sensor/cross-sensor.zip.
  - Validates image dimensions, band counts, value ranges, and spatial alignment:
      LR: Sentinel-2 L2A 10 m (121 x 121 x 4, int32 surface reflectance)
      HR: NAIP 2.5 m         (484 x 484 x 4, uint8 RGBN)
  - Generates deterministic train / validation / test splits (splits.json) and
    an aggregated metadata summary (dataset_summary.json).

Usage:
  # Download a 30-pair subset for pipeline smoke test:
  python scripts/prepare_sen2naip.py --num-pairs 30

  # Download 50 pairs:
  python scripts/prepare_sen2naip.py --num-pairs 50

  # Verify existing local dataset without downloading:
  python scripts/prepare_sen2naip.py --verify-only
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import logging
    import tifffile
    logging.getLogger("tifffile").setLevel(logging.ERROR)
except ImportError:
    print("[ERROR] 'tifffile' is required to read GeoTIFF files. Install via: pip install tifffile")
    sys.exit(1)

try:
    from huggingface_hub import HfFileSystem
except ImportError:
    print("[ERROR] 'huggingface_hub' is required. Install via: pip install huggingface_hub")
    sys.exit(1)

import zipfile

# Default paths and repo constants
HF_REPO_ID = "isp-uv-es/SEN2NAIP"
HF_ZIP_PATH = f"datasets/{HF_REPO_ID}/cross-sensor/cross-sensor.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "SEN2NAIP"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SEN2NAIP dataset for Sentinel-2 super-resolution.")
    parser.add_argument("--num-pairs", type=int, default=30,
                        help="Number of ROI pairs to download (default: 30, e.g. 20-50 for smoke test).")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting ROI index offset (default: 0).")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Target directory for downloaded data.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val/test splitting.")
    parser.add_argument("--train-ratio", type=float, default=0.80,
                        help="Train split fraction (default: 0.80).")
    parser.add_argument("--val-ratio", type=float, default=0.10,
                        help="Validation split fraction (default: 0.10).")
    parser.add_argument("--test-ratio", type=float, default=0.10,
                        help="Test split fraction (default: 0.10).")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip download and verify existing local dataset files.")
    return parser.parse_args()


def inspect_pair(lr_path: Path, hr_path: Path, meta_path: Path) -> dict:
    """
    Reads an LR-HR pair and extracts dimensional and radiometric properties.
    """
    lr_arr = tifffile.imread(str(lr_path))
    hr_arr = tifffile.imread(str(hr_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    lr_h, lr_w = lr_arr.shape[:2]
    lr_c = lr_arr.shape[2] if lr_arr.ndim == 3 else 1
    hr_h, hr_w = hr_arr.shape[:2]
    hr_c = hr_arr.shape[2] if hr_arr.ndim == 3 else 1

    scale_h = hr_h / lr_h
    scale_w = hr_w / lr_w

    stats = {
        "roi_id": lr_path.parent.name,
        "lr_shape": list(lr_arr.shape),
        "hr_shape": list(hr_arr.shape),
        "lr_dtype": str(lr_arr.dtype),
        "hr_dtype": str(hr_arr.dtype),
        "scale_factor_h": scale_h,
        "scale_factor_w": scale_w,
        "lr_min": int(np.min(lr_arr)),
        "lr_max": int(np.max(lr_arr)),
        "lr_mean": float(np.mean(lr_arr)),
        "hr_min": int(np.min(hr_arr)),
        "hr_max": int(np.max(hr_arr)),
        "hr_mean": float(np.mean(hr_arr)),
        "bands_count": lr_c,
        "s2_id": meta.get("s2_id"),
        "s2_date": meta.get("s2_date"),
        "naip_id": meta.get("naip_id"),
        "naip_date": meta.get("naip_date"),
        "epsg": meta.get("proj:epsg"),
    }
    return stats


def download_sen2naip_subset(output_dir: Path, num_pairs: int, start_idx: int = 0):
    """
    Selectively streams individual ROI files from the remote Hugging Face zip
    using HTTP Range requests.
    """
    output_cross = output_dir / "cross-sensor"
    output_cross.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Connecting to Hugging Face repository '{HF_REPO_ID}'...")
    hffs = HfFileSystem()

    print(f"[2/4] Opening remote archive '{HF_ZIP_PATH}'...")
    t0 = time.time()
    with hffs.open(HF_ZIP_PATH, "rb") as remote_file:
        zf = zipfile.ZipFile(remote_file)
        print(f"      Remote archive opened in {time.time() - t0:.2f}s")

        # Discover all ROIs in the archive
        all_rois = sorted(list(set(
            p.split("/")[1]
            for p in zf.namelist()
            if len(p.split("/")) > 1 and p.split("/")[1].startswith("ROI_")
        )))
        total_available = len(all_rois)
        print(f"      Total cross-sensor ROIs available in dataset: {total_available:,}")

        target_rois = all_rois[start_idx : start_idx + num_pairs]
        print(f"[3/4] Downloading {len(target_rois)} ROI pairs (indices {start_idx} to {start_idx + len(target_rois) - 1})...")

        for idx, roi in enumerate(target_rois, start=1):
            roi_dir = output_cross / roi
            roi_dir.mkdir(parents=True, exist_ok=True)

            lr_file = roi_dir / "lr.tif"
            hr_file = roi_dir / "hr.tif"
            meta_file = roi_dir / "metadata.json"

            # Check if already present on disk
            if lr_file.exists() and hr_file.exists() and meta_file.exists():
                print(f"      [{idx:02d}/{len(target_rois):02d}] {roi} already exists locally. Skipping.")
                continue

            t_start = time.time()
            lr_bytes = zf.read(f"cross-sensor/{roi}/lr.tif")
            hr_bytes = zf.read(f"cross-sensor/{roi}/hr.tif")
            meta_bytes = zf.read(f"cross-sensor/{roi}/metadata.json")

            with open(lr_file, "wb") as f:
                f.write(lr_bytes)
            with open(hr_file, "wb") as f:
                f.write(hr_bytes)
            with open(meta_file, "wb") as f:
                f.write(meta_bytes)

            elapsed = time.time() - t_start
            size_mb = (len(lr_bytes) + len(hr_bytes) + len(meta_bytes)) / (1024 * 1024)
            print(f"      [{idx:02d}/{len(target_rois):02d}] Downloaded {roi} ({size_mb:.2f} MB in {elapsed:.2f}s)")


def verify_and_split(output_dir: Path, seed: int = 42, train_r: float = 0.8, val_r: float = 0.1, test_r: float = 0.1):
    """
    Verifies downloaded local pairs, aggregates technical statistics,
    and generates train/validation/test split definitions.
    """
    cross_dir = output_dir / "cross-sensor"
    roi_dirs = sorted([d for d in cross_dir.iterdir() if d.is_dir() and d.name.startswith("ROI_")])

    if not roi_dirs:
        print(f"[ERROR] No ROI directories found in {cross_dir}")
        return None

    print(f"[4/4] Verifying and generating split manifests for {len(roi_dirs)} pairs...")

    pair_stats = []
    total_bytes = 0

    for roi_d in roi_dirs:
        lr_p = roi_d / "lr.tif"
        hr_p = roi_d / "hr.tif"
        meta_p = roi_d / "metadata.json"

        if not (lr_p.exists() and hr_p.exists() and meta_p.exists()):
            print(f"[WARN] Incomplete pair at {roi_d.name}. Skipping.")
            continue

        total_bytes += lr_p.stat().st_size + hr_p.stat().st_size + meta_p.stat().st_size

        try:
            stats = inspect_pair(lr_p, hr_p, meta_p)
            pair_stats.append(stats)
        except Exception as e:
            print(f"[ERROR] Failed to inspect {roi_d.name}: {e}")

    valid_rois = [s["roi_id"] for s in pair_stats]
    print(f"      Successfully validated {len(valid_rois)} pairs ({total_bytes / (1024*1024):.2f} MB total disk usage).")

    # Generate deterministic splits
    random.seed(seed)
    shuffled = list(valid_rois)
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(n_total * train_r))
    n_val = int(round(n_total * val_r))
    # Guarantee all pairs partitioned
    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train : n_train + n_val])
    test_ids = sorted(shuffled[n_train + n_val :])

    splits = {
        "metadata": {
            "dataset_name": "SEN2NAIP Cross-Sensor Subset",
            "seed": seed,
            "total_pairs": n_total,
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "train_ratio": train_r,
            "val_ratio": val_r,
            "test_ratio": test_r,
        },
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    splits_path = output_dir / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    # Technical summary
    sample = pair_stats[0] if pair_stats else {}
    summary = {
        "dataset_name": "SEN2NAIP (Sentinel-2 to NAIP Super-Resolution)",
        "source": f"Hugging Face: {HF_REPO_ID}",
        "subset": "cross-sensor",
        "total_cross_sensor_catalog_pairs": 2851,
        "locally_available_pairs": len(valid_rois),
        "disk_usage_mb": round(total_bytes / (1024 * 1024), 2),
        "lr_specifications": {
            "sensor": "Sentinel-2 L2A MSI",
            "spatial_resolution": "10.0 meters",
            "native_shape": sample.get("lr_shape"),
            "dtype": sample.get("lr_dtype"),
            "band_count": 4,
            "band_names": ["B04 (Red)", "B03 (Green)", "B02 (Blue)", "B08 (NIR)"],
            "rgb_channel_indices": [0, 1, 2],
            "nir_channel_index": 3,
            "value_range": "Surface Reflectance integers scaled x10000 (typical range ~0 to 3000 for vegetation/soil)",
            "normalization_strategy": "Divide by 10000.0 (or clamp [0, 10000] and scale to [0, 1])",
        },
        "hr_specifications": {
            "sensor": "NAIP (National Agriculture Imagery Program) 2.5 m",
            "spatial_resolution": "2.5 meters",
            "native_shape": sample.get("hr_shape"),
            "dtype": sample.get("hr_dtype"),
            "band_count": 4,
            "band_names": ["Red", "Green", "Blue", "NIR"],
            "rgb_channel_indices": [0, 1, 2],
            "nir_channel_index": 3,
            "value_range": "uint8 [0, 255]",
            "normalization_strategy": "Divide by 255.0 to [0.0, 1.0]",
        },
        "scale_factor": 4.0,
        "geographic_crs": "UTM projections (e.g. WGS 84 / UTM zone 13N, EPSG: 32613)",
        "swinir_adaptation_notes": {
            "channel_selection": "For first fine-tuning experiments, use RGB (channels 0, 1, 2) to match standard 3-channel SwinIR-M classical SR weights.",
            "nir_band_policy": "NIR (channel 3) is preserved on disk in lr.tif and hr.tif. It is NOT discarded from the dataset; it can be accessed by setting include_nir=True in the dataset loader.",
            "spatial_divisibility": "Native LR is 121x121 (HR 484x484). For SwinIR window_size=8, center crop to 120x120 (HR 480x480) or reflection pad to 128x128 (HR 512x512).",
        },
        "splits_summary": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
    }

    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"      Saved splits manifest to {splits_path}")
    print(f"      Saved technical summary to {summary_path}")
    return summary


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)

    print("=" * 70)
    print("SEN2NAIP DATASET PREPARATION FOR SENTINEL-2 SWINIR SUPER-RESOLUTION")
    print("=" * 70)
    print(f"Target Directory: {out_dir}")
    print(f"Subset Size:      {args.num_pairs} pairs")

    if not args.verify_only:
        download_sen2naip_subset(out_dir, num_pairs=args.num_pairs, start_idx=args.start_idx)

    summary = verify_and_split(
        out_dir,
        seed=args.seed,
        train_r=args.train_ratio,
        val_r=args.val_ratio,
        test_r=args.test_ratio,
    )

    if summary:
        print("\n" + "=" * 70)
        print("DATASET PREPARATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Total Available Pairs: {summary['locally_available_pairs']}")
        print(f"Total Disk Usage:      {summary['disk_usage_mb']} MB")
        print(f"LR Resolution & Shape: {summary['lr_specifications']['spatial_resolution']} | {summary['lr_specifications']['native_shape']}")
        print(f"HR Resolution & Shape: {summary['hr_specifications']['spatial_resolution']} | {summary['hr_specifications']['native_shape']}")
        print(f"Scale Factor:          {summary['scale_factor']}x")
        print(f"Splits Breakdown:      Train={summary['splits_summary']['train']} | Val={summary['splits_summary']['val']} | Test={summary['splits_summary']['test']}")
        print("=" * 70)


if __name__ == "__main__":
    main()
