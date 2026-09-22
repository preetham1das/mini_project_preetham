import base64
import io
import json
import sys
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import torch

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from model import UNet, CHECKPOINT_PATH
from temporal_model import TemporalUNet, TEMPORAL_CHECKPOINT_PATH
from metrics import THRESHOLD

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cpu")
TARGET_SIZE = (256, 256)

# GIS Spatial Constants
# Ground Sample Distance: 10 meters per pixel (Sentinel-2 standard)
GSD_METERS = 10.0
SQM_PER_PIXEL = GSD_METERS * GSD_METERS
HA_PER_PIXEL = SQM_PER_PIXEL / 10000.0
SQKM_PER_PIXEL = SQM_PER_PIXEL / 1000000.0

# Base Amazon Basin Reference Coordinates
BASE_LAT = -3.4653
BASE_LON = -62.2159

# Global variables for loaded models
stage1_model = None
stage2_model = None


def get_sample_bounds(sample_id_str):
    """Generates realistic Lat/Lon bounding box based on sample hash."""
    h = hash(str(sample_id_str)) % 1000
    lat_offset = (h / 1000.0) * 4.0 - 2.0
    lon_offset = ((h * 3) % 1000 / 1000.0) * 4.0 - 2.0

    lat_center = BASE_LAT + lat_offset
    lon_center = BASE_LON + lon_offset

    delta_deg = 0.023

    south = round(lat_center - delta_deg / 2, 6)
    north = round(lat_center + delta_deg / 2, 6)
    west = round(lon_center - delta_deg / 2, 6)
    east = round(lon_center + delta_deg / 2, 6)

    return {
        "bounds": [[south, west], [north, east]],
        "center": [round(lat_center, 6), round(lon_center, 6)],
        "south": south, "north": north, "west": west, "east": east
    }


def load_models():
    global stage1_model, stage2_model
    
    # 1. Load Stage 1 UNet
    stage1_path = Path(CHECKPOINT_PATH)
    if stage1_path.exists():
        checkpoint = torch.load(stage1_path, map_location=DEVICE, weights_only=True)
        stage1_model = UNet(in_channels=3, out_channels=1, init_features=32).to(DEVICE)
        stage1_model.load_state_dict(checkpoint["model_state_dict"])
        stage1_model.eval()
        print(f"[OK] Loaded Stage 1 U-Net from {stage1_path}")
    else:
        print(f"[WARNING] Stage 1 checkpoint not found at {stage1_path}")

    # 2. Load Stage 2 Temporal UNet (9 channels with key compatibility)
    stage2_path = Path(TEMPORAL_CHECKPOINT_PATH)
    if stage2_path.exists():
        checkpoint = torch.load(stage2_path, map_location=DEVICE, weights_only=True)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        # Remap legacy layer names (cu1 -> conv_up1, etc.)
        new_state_dict = {}
        for k, v in state_dict.items():
            nk = k.replace("cu1.", "conv_up1.").replace("cu2.", "conv_up2.").replace("cu3.", "conv_up3.").replace("cu4.", "conv_up4.")
            new_state_dict[nk] = v

        stage2_model = TemporalUNet(in_channels=9, out_channels=1, init_features=32).to(DEVICE)
        stage2_model.load_state_dict(new_state_dict)
        stage2_model.eval()
        print(f"[OK] Loaded Stage 2 Temporal U-Net from {stage2_path}")
    else:
        print(f"[WARNING] Stage 2 checkpoint not found at {stage2_path}")


def array_to_base64_png(np_img):
    """Converts numpy image array (RGB or RGBA) to base64 PNG data string."""
    if np_img.dtype != np.uint8:
        if np_img.max() <= 1.0:
            np_img = (np_img * 255).astype(np.uint8)
        else:
            np_img = np_img.astype(np.uint8)
    
    pil_img = Image.fromarray(np_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/gis/regions", methods=["GET"])
def get_gis_regions():
    """Returns sample GIS region metadata and bounding boxes."""
    amazon_dir = Path("data/Amazon Forest Dataset")
    regions = []

    test_images = list((amazon_dir / "Validation" / "images").glob("*.tiff")) + list((amazon_dir / "Validation" / "images").glob("*.tif"))
    for f in test_images[:8]:
        gis_info = get_sample_bounds(f.name)
        regions.append({
            "id": f.name,
            "name": f.name,
            "region_name": f"Amazon Sub-basin Tile {f.stem}",
            "center": gis_info["center"],
            "bounds": gis_info["bounds"],
            "gsd_meters": GSD_METERS
        })

    return jsonify({"regions": regions, "gsd_meters": GSD_METERS})


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns sample lists for quick selection in frontend."""
    amazon_dir = Path("data/Amazon Forest Dataset")
    stage1_samples = []
    
    val_images = list((amazon_dir / "Validation" / "images").glob("*.tiff")) + list((amazon_dir / "Validation" / "images").glob("*.tif"))
    for f in val_images[:10]:
        gis = get_sample_bounds(f.name)
        stage1_samples.append({
            "id": f.name,
            "name": f.name,
            "split": "Validation",
            "path": str(f),
            "gis": gis
        })

    test_images = list((amazon_dir / "Test").glob("*.tiff")) + list((amazon_dir / "Test").glob("*.tif"))
    for f in test_images[:10]:
        gis = get_sample_bounds(f.name)
        stage1_samples.append({
            "id": f.name,
            "name": f"Test / {f.name}",
            "split": "Test",
            "path": str(f),
            "gis": gis
        })

    temporal_root = Path("data/Forest-Change/extracted/Forest-Change-dataset/images")
    stage2_samples = []
    
    if (temporal_root / "test" / "A").exists():
        test_pairs = sorted([f.stem for f in (temporal_root / "test" / "A").glob("*.png")])
        for p in test_pairs[:12]:
            gis = get_sample_bounds(p)
            stage2_samples.append({
                "id": p,
                "name": f"Test Pair: {p}",
                "split": "test",
                "gis": gis
            })

    return jsonify({
        "stage1_samples": stage1_samples,
        "stage2_samples": stage2_samples,
        "gsd_meters": GSD_METERS
    })


@app.route("/api/predict/stage1", methods=["POST"])
def predict_stage1():
    """Runs Stage 1 Forest Cover Segmentation with GIS metrics."""
    if stage1_model is None:
        return jsonify({"error": "Stage 1 model is not loaded"}), 500

    threshold = float(request.form.get("threshold", THRESHOLD))
    pil_img = None
    sample_name = "Uploaded Image"

    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        sample_name = file.filename
        pil_img = Image.open(file.stream).convert("RGB")
    else:
        sample_path_str = request.form.get("sample_path")
        if sample_path_str and Path(sample_path_str).exists():
            pil_img = Image.open(sample_path_str).convert("RGB")
            sample_name = Path(sample_path_str).name
        else:
            return jsonify({"error": "No image provided"}), 400

    gis_bounds = get_sample_bounds(sample_name)

    pil_resized = pil_img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    img_np = np.array(pil_resized, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = stage1_model(img_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

    pred_mask = preds[0, 0].cpu().numpy().astype(np.uint8)

    total_pixels = int(pred_mask.size)
    forest_pixels = int(np.sum(pred_mask == 1))
    forest_pct = round((forest_pixels / total_pixels) * 100.0, 2)
    non_forest_pixels = total_pixels - forest_pixels
    non_forest_pct = round(100.0 - forest_pct, 2)

    total_hectares = round(total_pixels * HA_PER_PIXEL, 2)
    forest_hectares = round(forest_pixels * HA_PER_PIXEL, 2)
    forest_sqkm = round(forest_pixels * SQKM_PER_PIXEL, 3)

    original_b64 = array_to_base64_png(img_np)

    mask_rgb = np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)
    mask_rgb[pred_mask == 1] = [34, 197, 94]
    mask_b64 = array_to_base64_png(mask_rgb)

    overlay_rgb = img_np.copy()
    green_tint = np.array([0.133, 0.772, 0.368], dtype=np.float32)
    alpha = 0.45
    forest_indices = (pred_mask == 1)
    overlay_rgb[forest_indices] = (1 - alpha) * overlay_rgb[forest_indices] + alpha * green_tint
    overlay_b64 = array_to_base64_png(overlay_rgb)

    return jsonify({
        "sample_name": sample_name,
        "threshold": threshold,
        "total_pixels": total_pixels,
        "forest_pixels": forest_pixels,
        "forest_pct": forest_pct,
        "non_forest_pixels": non_forest_pixels,
        "non_forest_pct": non_forest_pct,
        "total_hectares": total_hectares,
        "forest_hectares": forest_hectares,
        "forest_sqkm": forest_sqkm,
        "gis": gis_bounds,
        "original_image": original_b64,
        "mask_image": mask_b64,
        "overlay_image": overlay_b64
    })


@app.route("/api/predict/stage2", methods=["POST"])
def predict_stage2():
    """Runs Stage 2 Temporal Deforestation Detection with GIS spatial metrics."""
    if stage2_model is None:
        return jsonify({"error": "Stage 2 temporal model is not loaded"}), 500

    threshold = float(request.form.get("threshold", 0.5))
    split = request.form.get("split", "test")
    sample_id = request.form.get("sample_id")

    temporal_root = Path("data/Forest-Change/extracted/Forest-Change-dataset/images")
    
    pil_a = None
    pil_b = None
    gt_mask_np = None
    pair_name = sample_id or "Custom Pair"

    if "file_t1" in request.files and "file_t2" in request.files:
        f_a = request.files["file_t1"]
        f_b = request.files["file_t2"]
        pil_a = Image.open(f_a.stream).convert("RGB")
        pil_b = Image.open(f_b.stream).convert("RGB")
    elif sample_id:
        t1_path = temporal_root / split / "A" / f"{sample_id}.png"
        t2_path = temporal_root / split / "B" / f"{sample_id}.png"
        gt_path = temporal_root / split / "label" / f"{sample_id}.png"

        if t1_path.exists() and t2_path.exists():
            pil_a = Image.open(t1_path).convert("RGB")
            pil_b = Image.open(t2_path).convert("RGB")
            if gt_path.exists():
                gt_pil = Image.open(gt_path).convert("L")
                if gt_pil.size != TARGET_SIZE:
                    gt_pil = gt_pil.resize(TARGET_SIZE, Image.Resampling.NEAREST)
                gt_mask_np = (np.array(gt_pil) > 0).astype(np.uint8)
        else:
            return jsonify({"error": f"Sample pair {sample_id} not found"}), 404
    else:
        return jsonify({"error": "No T1 and T2 images provided"}), 400

    gis_bounds = get_sample_bounds(pair_name)

    pil_a_resized = pil_a.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    pil_b_resized = pil_b.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

    img_a_np = np.array(pil_a_resized, dtype=np.float32) / 255.0
    img_b_np = np.array(pil_b_resized, dtype=np.float32) / 255.0

    # 9-channel input tensor [T1_RGB (3), T2_RGB (3), |T1-T2|_RGB (3)]
    t1_tensor = torch.tensor(img_a_np, dtype=torch.float32).permute(2, 0, 1)
    t2_tensor = torch.tensor(img_b_np, dtype=torch.float32).permute(2, 0, 1)
    diff_tensor = torch.abs(t1_tensor - t2_tensor)

    concat_tensor = torch.cat([t1_tensor, t2_tensor, diff_tensor], dim=0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = stage2_model(concat_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

    pred_defor_mask = preds[0, 0].cpu().numpy().astype(np.uint8)

    total_pixels = int(pred_defor_mask.size)
    defor_pixels = int(np.sum(pred_defor_mask == 1))
    defor_pct = round((defor_pixels / total_pixels) * 100.0, 2)
    stable_pixels = total_pixels - defor_pixels
    stable_pct = round(100.0 - defor_pct, 2)

    defor_hectares = round(defor_pixels * HA_PER_PIXEL, 2)
    defor_sqkm = round(defor_pixels * SQKM_PER_PIXEL, 3)

    t1_b64 = array_to_base64_png(img_a_np)
    t2_b64 = array_to_base64_png(img_b_np)

    defor_mask_rgb = np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)
    defor_mask_rgb[pred_defor_mask == 1] = [239, 68, 68]
    defor_mask_b64 = array_to_base64_png(defor_mask_rgb)

    overlay_rgb = img_b_np.copy()
    red_tint = np.array([0.937, 0.267, 0.267], dtype=np.float32)
    alpha = 0.5
    defor_indices = (pred_defor_mask == 1)
    overlay_rgb[defor_indices] = (1 - alpha) * overlay_rgb[defor_indices] + alpha * red_tint
    overlay_b64 = array_to_base64_png(overlay_rgb)

    gt_b64 = None
    metrics = None

    if gt_mask_np is not None:
        gt_rgb = np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)
        gt_rgb[gt_mask_np == 1] = [239, 68, 68]
        gt_b64 = array_to_base64_png(gt_rgb)

        tp = np.sum((pred_defor_mask == 1) & (gt_mask_np == 1))
        fp = np.sum((pred_defor_mask == 1) & (gt_mask_np == 0))
        fn = np.sum((pred_defor_mask == 0) & (gt_mask_np == 1))

        iou_den = tp + fp + fn
        dice_den = 2 * tp + fp + fn

        iou = float(tp / iou_den) if iou_den > 0 else (1.0 if np.sum(gt_mask_np) == 0 and defor_pixels == 0 else 0.0)
        dice = float((2 * tp) / dice_den) if dice_den > 0 else (1.0 if np.sum(gt_mask_np) == 0 and defor_pixels == 0 else 0.0)

        gt_defor_px = int(np.sum(gt_mask_np == 1))
        metrics = {
            "sample_iou": round(iou, 4),
            "sample_dice": round(dice, 4),
            "gt_defor_pixels": gt_defor_px,
            "gt_defor_pct": round((gt_defor_px / total_pixels) * 100.0, 2),
            "gt_defor_hectares": round(gt_defor_px * HA_PER_PIXEL, 2)
        }

    return jsonify({
        "pair_name": pair_name,
        "threshold": threshold,
        "total_pixels": total_pixels,
        "deforestation_pixels": defor_pixels,
        "deforestation_pct": defor_pct,
        "deforestation_hectares": defor_hectares,
        "deforestation_sqkm": defor_sqkm,
        "stable_pixels": stable_pixels,
        "stable_pct": stable_pct,
        "gis": gis_bounds,
        "t1_image": t1_b64,
        "t2_image": t2_b64,
        "deforestation_mask": defor_mask_b64,
        "deforestation_overlay": overlay_b64,
        "gt_mask": gt_b64,
        "metrics": metrics
    })


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns stored evaluation metrics for the UI dashboard."""
    stage1_report = None
    stage2_report = None

    stage1_json = Path("outputs/evaluation_metrics.json")
    if stage1_json.exists():
        with open(stage1_json, "r") as f:
            stage1_report = json.load(f)

    stage2_json = Path("outputs/temporal/temporal_test_results.json")
    if stage2_json.exists():
        with open(stage2_json, "r") as f:
            stage2_report = json.load(f)

    return jsonify({
        "stage1": stage1_report,
        "stage2": stage2_report
    })


if __name__ == "__main__":
    load_models()
    print("[START] Deforestation Detection GIS Web Server running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
