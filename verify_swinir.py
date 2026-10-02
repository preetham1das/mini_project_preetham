"""
verify_swinir_fast.py
---------------------
Fast verification of SwinIR 4x super-resolution (CPU-friendly sizes only).
- 64x64  → 256x256  (very fast, < 30s on CPU)
- 128x128 → 512x512  (moderate, ~1-2 min on CPU)
Skips 512x512 which would take many minutes on CPU.
"""

import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "swinir"))

from swinir.swinir_inference import (
    superresolve,
    is_checkpoint_available,
    CHECKPOINT_PATH,
    _load_model,
)
import torch
from PIL import Image

DEVICE = torch.device("cpu")
SEP = "-" * 60

def ok(cond, msg):
    status = "[PASS]" if cond else "[FAIL]"
    print(f"  {status} {msg}")
    return cond

failures = []

def check(cond, msg):
    if not ok(cond, msg):
        failures.append(msg)
    return cond

# ─── 1. Checkpoint ───────────────────────────────────────────────────────────
print(SEP)
print("1. CHECKPOINT VERIFICATION")
print(f"   Path: {CHECKPOINT_PATH}")
ckpt_exists = CHECKPOINT_PATH.exists()
ckpt_mb = CHECKPOINT_PATH.stat().st_size / 1024 / 1024 if ckpt_exists else 0
check(ckpt_exists, f"Checkpoint file exists  ({ckpt_mb:.1f} MB)")
check(ckpt_mb > 50, f"Checkpoint size > 50 MB  ({ckpt_mb:.1f} MB)")

if not ckpt_exists:
    print("ABORT: checkpoint not found.")
    sys.exit(1)

# ─── 2. Model load ───────────────────────────────────────────────────────────
print(SEP)
print("2. MODEL LOAD")
t0 = time.time()
model, err = _load_model(DEVICE)
load_time = time.time() - t0
check(err is None, f"Model loaded without error  ({load_time:.2f}s)")
if err:
    print(f"   ERROR: {err}")
    sys.exit(1)
n_params = sum(p.numel() for p in model.parameters())
check(n_params > 10_000_000, f"Parameter count: {n_params:,}")

# ─── 3. Tiny image 64x64 → 256x256 (very fast) ────────────────────────────
print(SEP)
print("3. INFERENCE: 64x64 -> 256x256  (fast sanity check)")
rng = np.random.default_rng(42)
img64 = Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8), "RGB")

t0 = time.time()
res64, err64 = superresolve(img64, device=DEVICE)
t64 = time.time() - t0

check(err64 is None,    f"Inference ok  ({t64:.2f}s)")
if err64:
    print(f"   ERROR: {err64}")
    sys.exit(1)
ow, oh = res64.size
check(ow == 64*4 and oh == 64*4, f"Dimensions: 64x64 -> {ow}x{oh}  (expect 256x256)")
arr = np.array(res64)
check(arr.dtype == np.uint8,                "Output dtype: uint8")
check(arr.shape == (oh, ow, 3),             f"Shape: {arr.shape}")
check(int(arr.min()) >= 0,                  f"Min pixel >= 0  ({arr.min()})")
check(int(arr.max()) <= 255,                f"Max pixel <= 255  ({arr.max()})")
check(not np.any(np.isnan(arr.astype(np.float32))), "No NaN")
check(not np.any(np.isinf(arr.astype(np.float32))), "No Inf")
print(f"   Pixel stats: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

# ─── 4. 128x128 → 512x512 (Sentinel Sentinel thumbnail size) ─────────────
print(SEP)
print("4. INFERENCE: 128x128 -> 512x512  (Sentinel thumbnail representative)")
img128 = Image.fromarray((rng.random((128, 128, 3)) * 255).astype(np.uint8), "RGB")

t0 = time.time()
res128, err128 = superresolve(img128, device=DEVICE)
t128 = time.time() - t0

check(err128 is None,   f"Inference ok  ({t128:.2f}s)")
if err128:
    print(f"   ERROR: {err128}")
else:
    ow2, oh2 = res128.size
    check(ow2 == 128*4 and oh2 == 128*4, f"Dimensions: 128x128 -> {ow2}x{oh2}  (expect 512x512)")
    arr2 = np.array(res128)
    check(arr2.min() >= 0 and arr2.max() <= 255, f"Pixel range [0,255]  [{arr2.min()},{arr2.max()}]")
    # Save output for visual inspection
    out_path = ROOT / "outputs" / "swinir_test_output.png"
    # Stack input (upscaled for comparison) and output side by side
    in_upscaled = img128.resize((512, 512), Image.NEAREST)  # nearest for fair comparison
    comparison = Image.new("RGB", (1024, 512))
    comparison.paste(in_upscaled, (0, 0))
    comparison.paste(res128, (512, 0))
    comparison.save(str(out_path))
    print(f"   Saved side-by-side comparison: {out_path}")

# ─── 5. Real image from disk ──────────────────────────────────────────────
print(SEP)
print("5. REAL IMAGE FROM DISK (sample_pair.png, cropped to 128x128)")
cand = ROOT / "outputs" / "sample_pair.png"
real_done = False
if cand.exists():
    try:
        img_real = Image.open(cand).convert("RGB").crop((0, 0, 128, 128))
        t0 = time.time()
        res_real, err_real = superresolve(img_real, device=DEVICE)
        t_real = time.time() - t0
        check(err_real is None, f"sample_pair.png 128x128 -> 4x: ok  ({t_real:.2f}s)")
        if err_real is None:
            rw, rh = res_real.size
            check(rw == 512 and rh == 512, f"Output size: {rw}x{rh}")
        real_done = True
    except Exception as e:
        print(f"   Could not process sample_pair.png: {e}")
if not real_done:
    print("   sample_pair.png not found or skipped; synthetic tests are sufficient.")

# ─── 6. is_checkpoint_available() ────────────────────────────────────────────
print(SEP)
print("6. HELPER API")
check(is_checkpoint_available(), "is_checkpoint_available() -> True")

# ─── 7. Temporal U-Net untouched confirmation ─────────────────────────────────
print(SEP)
print("7. TEMPORAL U-NET ISOLATION CONFIRMATION")
app_text = (ROOT / "app.py").read_text(encoding="utf-8")
analyze_line = next((i+1 for i, l in enumerate(app_text.splitlines()) if "/api/sentinel/analyze" in l and "@app.route" in l), None)
sr_line      = next((i+1 for i, l in enumerate(app_text.splitlines()) if "/api/sentinel/superresolve" in l and "@app.route" in l), None)
check(analyze_line is not None, f"/api/sentinel/analyze route present  (line {analyze_line})")
check(sr_line is not None,      f"/api/sentinel/superresolve route present  (line {sr_line})")
check(analyze_line < sr_line,   f"Analyze route ({analyze_line}) comes BEFORE superresolve ({sr_line}) — no interleaving")
# Confirm superresolve / swinir is not mentioned inside analyze function
import ast
parsed = ast.parse(app_text)
analyze_fn_node = next(n for n in ast.walk(parsed) if isinstance(n, ast.FunctionDef) and n.name == "analyze_sentinel_deforestation")
analyze_fn_code = ast.get_source_segment(app_text, analyze_fn_node)
check("superresolve" not in analyze_fn_code, "Temporal U-Net analyze fn has NO reference to superresolve")
check("swinir" not in analyze_fn_code.lower(), "Temporal U-Net analyze fn has NO reference to swinir")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(SEP)
print("SUMMARY")
print(f"  Checkpoint:    {CHECKPOINT_PATH.name}  ({ckpt_mb:.1f} MB)")
print(f"  Parameters:    {n_params:,}")
print(f"  Load time:     {load_time:.2f}s")
print(f"  64x64  infer:  {t64:.2f}s  ->  256x256")
if err128 is None:
    print(f"  128x128 infer: {t128:.2f}s  ->  512x512")
print(f"  Failures:      {len(failures)}")
if failures:
    for f in failures:
        print(f"    - {f}")
else:
    print("  All checks passed.")
print(SEP)
sys.exit(0 if not failures else 1)
