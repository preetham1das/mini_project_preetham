"""
diagnose_swinir_color.py
------------------------
Diagnose color/rendering issues in the SwinIR super-resolution pipeline.
Tests questions 1-10 on real Sentinel-2 RGB imagery.
Saves outputs/swinir_debug.png and outputs/sentinel_orig_debug.png.
"""

import sys
import os
import io
import base64
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "swinir"))

import app
from swinir.swinir_inference import superresolve, _load_model

client = app.app.test_client()

print("=" * 60)
print("FETCHING REAL SENTINEL-2 RGB PREVIEW")
search_res = client.get('/api/sentinel/search?lat=-3.4653&lon=-62.2159&start_date=2023-08-01&end_date=2023-09-01&limit=3')
acqs = search_res.get_json().get('acquisitions', [])
if not acqs:
    print("No acquisitions found")
    sys.exit(1)

acq = acqs[0]
print(f"Acquisition: {acq['id']} ({acq['datetime']})")

prev_res = client.get(f"/api/sentinel/preview?id={acq['id']}&lat=-3.4653&lon=-62.2159&datetime={acq['datetime']}&format=json&width=256&height=256")
prev_data = prev_res.get_json()
if not prev_data.get('success'):
    print("Preview failed:", prev_data)
    sys.exit(1)

b64_str = prev_data['image'].split(',', 1)[1]
raw_bytes = base64.b64decode(b64_str)
orig_pil = Image.open(io.BytesIO(raw_bytes))

# Save original for comparison
orig_path = ROOT / "outputs" / "sentinel_orig_debug.png"
orig_pil.save(orig_path)
print(f"Saved original preview to: {orig_path}")

print("=" * 60)
print("1. CHECK INPUT IMAGE")
print(f"   Mode: {orig_pil.mode}")
print(f"   Format: {orig_pil.format}")
print(f"   Size: {orig_pil.size}")

orig_np = np.array(orig_pil.convert("RGB"))
print(f"   Shape: {orig_np.shape}, dtype: {orig_np.dtype}")
r_orig, g_orig, b_orig = orig_np[:,:,0], orig_np[:,:,1], orig_np[:,:,2]
print(f"   Original R: min={r_orig.min():3d}, max={r_orig.max():3d}, mean={r_orig.mean():6.2f}, std={r_orig.std():6.2f}")
print(f"   Original G: min={g_orig.min():3d}, max={g_orig.max():3d}, mean={g_orig.mean():6.2f}, std={g_orig.std():6.2f}")
print(f"   Original B: min={b_orig.min():3d}, max={b_orig.max():3d}, mean={b_orig.mean():6.2f}, std={b_orig.std():6.2f}")

diff_rg = np.abs(r_orig.astype(float) - g_orig.astype(float)).mean()
diff_gb = np.abs(g_orig.astype(float) - b_orig.astype(float)).mean()
print(f"   Original channel differences: |R-G| mean={diff_rg:.2f}, |G-B| mean={diff_gb:.2f}")
if diff_rg < 1.0 and diff_gb < 1.0:
    print("   WARNING: Input is essentially grayscale!")
else:
    print("   Input is TRUE COLOR RGB.")

print("=" * 60)
print("2. RUN SWINIR WITH CURRENT CODE")
sr_pil, err = superresolve(orig_pil)
if err:
    print(f"SwinIR failed: {err}")
    sys.exit(1)

debug_path = ROOT / "outputs" / "swinir_debug.png"
sr_pil.save(debug_path)
print(f"Saved SwinIR debug output to: {debug_path}")

print("=" * 60)
print("3. CHECK SWINIR OUTPUT CHANNELS")
sr_np = np.array(sr_pil)
print(f"   Mode: {sr_pil.mode}")
print(f"   Shape: {sr_np.shape}, dtype: {sr_np.dtype}")

r_sr, g_sr, b_sr = sr_np[:,:,0], sr_np[:,:,1], sr_np[:,:,2]
print(f"   SwinIR   R: min={r_sr.min():3d}, max={r_sr.max():3d}, mean={r_sr.mean():6.2f}, std={r_sr.std():6.2f}")
print(f"   SwinIR   G: min={g_sr.min():3d}, max={g_sr.max():3d}, mean={g_sr.mean():6.2f}, std={g_sr.std():6.2f}")
print(f"   SwinIR   B: min={b_sr.min():3d}, max={b_sr.max():3d}, mean={b_sr.mean():6.2f}, std={b_sr.std():6.2f}")

sr_diff_rg = np.abs(r_sr.astype(float) - g_sr.astype(float)).mean()
sr_diff_gb = np.abs(g_sr.astype(float) - b_sr.astype(float)).mean()
print(f"   SwinIR channel differences: |R-G| mean={sr_diff_rg:.2f}, |G-B| mean={sr_diff_gb:.2f}")

print("=" * 60)
print("4. CHANNEL ORDER COMPARISON (RGB vs BGR)")
# What if the pretrained model expects BGR input?
# Let's test SwinIR if input is fed as BGR (i.e. RGB reversed):
bgr_pil = Image.fromarray(orig_np[:, :, ::-1], "RGB")
sr_bgr_pil, _ = superresolve(bgr_pil)
sr_bgr_np = np.array(sr_bgr_pil)[:, :, ::-1] # swap back to RGB

sr_bgr_path = ROOT / "outputs" / "swinir_bgr_test.png"
Image.fromarray(sr_bgr_np).save(sr_bgr_path)

print(f"   SwinIR if BGR: R mean={sr_bgr_np[:,:,0].mean():6.2f}, G mean={sr_bgr_np[:,:,1].mean():6.2f}, B mean={sr_bgr_np[:,:,2].mean():6.2f}")

print("=" * 60)
print("5. SATURATION COMPARISON (HSV)")
from PIL import ImageEnhance
orig_hsv = orig_pil.convert("HSV")
sr_hsv   = sr_pil.convert("HSV")
s_orig = np.array(orig_hsv)[:,:,1]
s_sr   = np.array(sr_hsv)[:,:,1]
print(f"   Original Saturation: mean={s_orig.mean():.2f}, std={s_orig.std():.2f}")
print(f"   SwinIR   Saturation: mean={s_sr.mean():.2f}, std={s_sr.std():.2f}")
print(f"   Saturation change: {((s_sr.mean() - s_orig.mean()) / max(s_orig.mean(), 1e-5))*100:+.2f}%")

print("=" * 60)
print("6. TEST ENDPOINT POST /api/sentinel/superresolve")
sr_api_res = client.post('/api/sentinel/superresolve', json={
    'image': prev_data['image'],
    'bbox': prev_data['bbox']
})
sr_api_data = sr_api_res.get_json()
print(f"   API success: {sr_api_data.get('success')}")
api_b64 = sr_api_data['image'].split(',', 1)[1]
api_pil = Image.open(io.BytesIO(base64.b64decode(api_b64)))
api_np = np.array(api_pil)
print(f"   API Image size: {api_pil.size}, mode: {api_pil.mode}")
print(f"   API Image equals direct SwinIR: {np.array_equal(api_np, sr_np)}")

print("=" * 60)
print("DONE")
