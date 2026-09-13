import base64
import io
import json
import os
import sys
from pathlib import Path

# Reconfigure sys.stdout and sys.stderr for UTF-8 compatibility on Windows consoles
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent

# Auto-discover venv dependencies if available
venv_site_packages = BASE_DIR / "venv" / "Lib" / "site-packages"
if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
    sys.path.insert(0, str(venv_site_packages))

# Ensure src directory is in sys.path
src_dir = BASE_DIR / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from flask import Flask, render_template, request, jsonify
try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False
import numpy as np
from PIL import Image
import torch


from src.model import UNet, CHECKPOINT_PATH
from src.temporal_model import TemporalUNet, TEMPORAL_CHECKPOINT_PATH
from src.metrics import THRESHOLD


app = Flask(__name__)
if HAS_CORS:
    CORS(app)

DEVICE = torch.device("cpu")
TARGET_SIZE = (256, 256)

# Global variables for loaded models
stage1_model = None
stage2_model = None


def safe_print(message: str):
    """Safely print messages to console, falling back if unicode fails."""
    try:
        print(message)
    except UnicodeEncodeError:
        ascii_msg = message.encode("ascii", "replace").decode("ascii")
        print(ascii_msg)


def load_models():
    """Loads Stage 1 and Stage 2 models into memory safely."""
    global stage1_model, stage2_model
    
    # 1. Load Stage 1 UNet
    stage1_path = BASE_DIR / CHECKPOINT_PATH
    if stage1_path.exists():
        try:
            try:
                checkpoint = torch.load(stage1_path, map_location=DEVICE, weights_only=True)
            except Exception:
                checkpoint = torch.load(stage1_path, map_location=DEVICE, weights_only=False)
            
            stage1_model = UNet(in_channels=3, out_channels=1, init_features=32).to(DEVICE)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            stage1_model.load_state_dict(state_dict)
            stage1_model.eval()
            safe_print(f"[OK] Loaded Stage 1 U-Net from {stage1_path}")
        except Exception as e:
            safe_print(f"[ERROR] Failed to load Stage 1 checkpoint at {stage1_path}: {e}")
            stage1_model = None
    else:
        safe_print(f"[WARNING] Stage 1 checkpoint not found at {stage1_path}")

    # 2. Load Stage 2 Temporal UNet
    stage2_path = BASE_DIR / TEMPORAL_CHECKPOINT_PATH
    if stage2_path.exists():
        try:
            try:
                checkpoint = torch.load(stage2_path, map_location=DEVICE, weights_only=True)
            except Exception:
                checkpoint = torch.load(stage2_path, map_location=DEVICE, weights_only=False)
                
            stage2_model = TemporalUNet(in_channels=6, out_channels=1, init_features=32).to(DEVICE)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            stage2_model.load_state_dict(state_dict)
            stage2_model.eval()
            safe_print(f"[OK] Loaded Stage 2 Temporal U-Net from {stage2_path}")
        except Exception as e:
            safe_print(f"[ERROR] Failed to load Stage 2 checkpoint at {stage2_path}: {e}")
            stage2_model = None
    else:
        safe_print(f"[WARNING] Stage 2 checkpoint not found at {stage2_path}")


def array_to_base64_png(np_img):
    """Converts numpy image array (RGB or RGBA) to base64 PNG data string safely."""
    if not isinstance(np_img, np.ndarray):
        np_img = np.array(np_img)

    if np_img.dtype != np.uint8:
        if np_img.max() <= 1.0:
            np_img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
        else:
            np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    
    pil_img = Image.fromarray(np_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns sample lists for quick selection in frontend."""
    # Stage 1 Amazon Forest Dataset samples
    amazon_dir = BASE_DIR / "data" / "Amazon Forest Dataset"
    stage1_samples = []
    
    val_dir = amazon_dir / "Validation" / "images"
    if val_dir.exists():
        val_images = sorted(list(val_dir.glob("*.tiff")) + list(val_dir.glob("*.tif")))
        for f in val_images[:10]:
            stage1_samples.append({
                "id": f.name,
                "name": f.name,
                "split": "Validation",
                "path": str(f)
            })

    test_dir = amazon_dir / "Test"
    if test_dir.exists():
        test_images = sorted(list(test_dir.glob("*.tiff")) + list(test_dir.glob("*.tif")))
        for f in test_images[:10]:
            stage1_samples.append({
                "id": f.name,
                "name": f"Test / {f.name}",
                "split": "Test",
                "path": str(f)
            })

    # Stage 2 Forest-Change dataset samples
    temporal_root = BASE_DIR / "data" / "Forest-Change" / "extracted" / "Forest-Change-dataset" / "images"
    stage2_samples = []
    
    test_a_dir = temporal_root / "test" / "A"
    if test_a_dir.exists():
        test_pairs = sorted([f.stem for f in test_a_dir.glob("*.png")])
        for p in test_pairs[:12]:
            stage2_samples.append({
                "id": p,
                "name": f"Test Pair: {p}",
                "split": "test"
            })
            
    val_a_dir = temporal_root / "val" / "A"
    if val_a_dir.exists():
        val_pairs = sorted([f.stem for f in val_a_dir.glob("*.png")])
        for p in val_pairs[:6]:
            stage2_samples.append({
                "id": p,
                "name": f"Val Pair: {p}",
                "split": "val"
            })

    return jsonify({
        "stage1_samples": stage1_samples,
        "stage2_samples": stage2_samples
    })


@app.route("/api/predict/stage1", methods=["POST"])
def predict_stage1():
    """Runs Stage 1 Forest Cover Segmentation on uploaded file or sample ID/path."""
    if stage1_model is None:
        return jsonify({"error": "Stage 1 model is not loaded. Please ensure best_unet_model.pth exists."}), 500

    try:
        threshold = float(request.form.get("threshold", THRESHOLD))
        threshold = max(0.0, min(1.0, threshold))
    except (ValueError, TypeError):
        threshold = THRESHOLD

    pil_img = None
    sample_name = "Uploaded Image"

    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        sample_name = file.filename
        try:
            pil_img = Image.open(file.stream).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Failed to parse uploaded image: {str(e)}"}), 400
    else:
        sample_path_str = request.form.get("sample_path") or request.form.get("sample_id")
        if sample_path_str:
            p = Path(sample_path_str)
            if not p.is_absolute():
                p = BASE_DIR / p
            
            # If path doesn't exist directly, try locating in Amazon dataset paths
            if not p.exists():
                amazon_dir = BASE_DIR / "data" / "Amazon Forest Dataset"
                candidates = [
                    amazon_dir / "Validation" / "images" / p.name,
                    amazon_dir / "Test" / p.name
                ]
                for cand in candidates:
                    if cand.exists():
                        p = cand
                        break

            if p.exists() and p.is_file():
                try:
                    pil_img = Image.open(p).convert("RGB")
                    sample_name = p.name
                except Exception as e:
                    return jsonify({"error": f"Failed to load sample image: {str(e)}"}), 400
            else:
                return jsonify({"error": f"Sample image file not found: {sample_path_str}"}), 404
        else:
            return jsonify({"error": "No image file or sample path provided."}), 400

    try:
        # Resize to TARGET_SIZE
        pil_resized = pil_img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        img_np = np.array(pil_resized, dtype=np.float32) / 255.0

        # Convert to PyTorch Tensor (1, 3, H, W)
        img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = stage1_model(img_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

        pred_mask = preds[0, 0].cpu().numpy().astype(np.uint8)
        
        # Calculate statistics
        total_pixels = int(pred_mask.size)
        forest_pixels = int(np.sum(pred_mask == 1))
        forest_pct = round((forest_pixels / total_pixels) * 100.0, 2)
        non_forest_pixels = total_pixels - forest_pixels
        non_forest_pct = round(100.0 - forest_pct, 2)

        # 1. Base RGB Image
        original_b64 = array_to_base64_png(img_np)

        # 2. Binary Forest Mask Visualization (Green = Forest, Black = Non-forest)
        mask_rgb = np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)
        mask_rgb[pred_mask == 1] = [34, 197, 94]  # Emerald Green
        mask_b64 = array_to_base64_png(mask_rgb)

        # 3. Green Overlay Map on original image
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
            "original_image": original_b64,
            "mask_image": mask_b64,
            "overlay_image": overlay_b64
        })
    except Exception as e:
        return jsonify({"error": f"Inference execution failed: {str(e)}"}), 500


@app.route("/api/predict/stage2", methods=["POST"])
def predict_stage2():
    """Runs Stage 2 Temporal Deforestation Detection."""
    if stage2_model is None:
        return jsonify({"error": "Stage 2 temporal model is not loaded. Please ensure best_temporal_model.pth exists."}), 500

    try:
        threshold = float(request.form.get("threshold", 0.5))
        threshold = max(0.0, min(1.0, threshold))
    except (ValueError, TypeError):
        threshold = 0.5

    split = request.form.get("split", "test")
    raw_sample_id = request.form.get("sample_id", "")
    sample_id = Path(raw_sample_id).stem if raw_sample_id else ""

    temporal_root = BASE_DIR / "data" / "Forest-Change" / "extracted" / "Forest-Change-dataset" / "images"
    
    pil_a = None
    pil_b = None
    gt_mask_np = None
    pair_name = sample_id or "Custom Pair"

    if "file_t1" in request.files and "file_t2" in request.files and request.files["file_t1"].filename != "" and request.files["file_t2"].filename != "":
        f_a = request.files["file_t1"]
        f_b = request.files["file_t2"]
        try:
            pil_a = Image.open(f_a.stream).convert("RGB")
            pil_b = Image.open(f_b.stream).convert("RGB")
            pair_name = f"{f_a.filename} vs {f_b.filename}"
        except Exception as e:
            return jsonify({"error": f"Failed to parse uploaded temporal image pair: {str(e)}"}), 400
    elif sample_id:
        t1_path = temporal_root / split / "A" / f"{sample_id}.png"
        t2_path = temporal_root / split / "B" / f"{sample_id}.png"
        gt_path = temporal_root / split / "label" / f"{sample_id}.png"

        if t1_path.exists() and t2_path.exists():
            try:
                pil_a = Image.open(t1_path).convert("RGB")
                pil_b = Image.open(t2_path).convert("RGB")
                if gt_path.exists():
                    gt_pil = Image.open(gt_path).convert("L")
                    if gt_pil.size != TARGET_SIZE:
                        gt_pil = gt_pil.resize(TARGET_SIZE, Image.Resampling.NEAREST)
                    gt_mask_np = (np.array(gt_pil) > 0).astype(np.uint8)
            except Exception as e:
                return jsonify({"error": f"Failed to read dataset pair images: {str(e)}"}), 400
        else:
            return jsonify({"error": f"Sample pair '{sample_id}' not found in split '{split}'."}), 404
    else:
        return jsonify({"error": "No T1 and T2 images or sample_id provided."}), 400

    try:
        # Resize images
        pil_a_resized = pil_a.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        pil_b_resized = pil_b.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

        img_a_np = np.array(pil_a_resized, dtype=np.float32) / 255.0
        img_b_np = np.array(pil_b_resized, dtype=np.float32) / 255.0

        # Prepare 6-channel input tensor (1, 6, H, W)
        t1_tensor = torch.tensor(img_a_np, dtype=torch.float32).permute(2, 0, 1)
        t2_tensor = torch.tensor(img_b_np, dtype=torch.float32).permute(2, 0, 1)
        concat_tensor = torch.cat([t1_tensor, t2_tensor], dim=0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = stage2_model(concat_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

        pred_defor_mask = preds[0, 0].cpu().numpy().astype(np.uint8)

        # Pixel Statistics
        total_pixels = int(pred_defor_mask.size)
        defor_pixels = int(np.sum(pred_defor_mask == 1))
        defor_pct = round((defor_pixels / total_pixels) * 100.0, 2)
        stable_pixels = total_pixels - defor_pixels
        stable_pct = round(100.0 - defor_pct, 2)

        # Base Images Base64
        t1_b64 = array_to_base64_png(img_a_np)
        t2_b64 = array_to_base64_png(img_b_np)

        # Deforestation Red Mask (Red = Deforestation, Black = No Change)
        defor_mask_rgb = np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)
        defor_mask_rgb[pred_defor_mask == 1] = [239, 68, 68]  # Vibrant Red
        defor_mask_b64 = array_to_base64_png(defor_mask_rgb)

        # Deforestation Red Overlay on T2
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

            # Compute sample metrics
            tp = int(np.sum((pred_defor_mask == 1) & (gt_mask_np == 1)))
            fp = int(np.sum((pred_defor_mask == 1) & (gt_mask_np == 0)))
            fn = int(np.sum((pred_defor_mask == 0) & (gt_mask_np == 1)))

            iou_den = tp + fp + fn
            dice_den = 2 * tp + fp + fn

            iou = float(tp / iou_den) if iou_den > 0 else (1.0 if np.sum(gt_mask_np) == 0 and defor_pixels == 0 else 0.0)
            dice = float((2 * tp) / dice_den) if dice_den > 0 else (1.0 if np.sum(gt_mask_np) == 0 and defor_pixels == 0 else 0.0)

            metrics = {
                "sample_iou": round(iou, 4),
                "sample_dice": round(dice, 4),
                "gt_defor_pixels": int(np.sum(gt_mask_np == 1)),
                "gt_defor_pct": round((int(np.sum(gt_mask_np == 1)) / total_pixels) * 100.0, 2)
            }

        return jsonify({
            "pair_name": pair_name,
            "threshold": threshold,
            "total_pixels": total_pixels,
            "deforestation_pixels": defor_pixels,
            "deforestation_pct": defor_pct,
            "stable_pixels": stable_pixels,
            "stable_pct": stable_pct,
            "t1_image": t1_b64,
            "t2_image": t2_b64,
            "deforestation_mask": defor_mask_b64,
            "deforestation_overlay": overlay_b64,
            "gt_mask": gt_b64,
            "metrics": metrics
        })
    except Exception as e:
        return jsonify({"error": f"Temporal inference execution failed: {str(e)}"}), 500


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns stored evaluation metrics for the UI dashboard."""
    stage1_report = None
    stage2_report = None

    stage1_json = BASE_DIR / "outputs" / "evaluation_metrics.json"
    if stage1_json.exists():
        try:
            with open(stage1_json, "r", encoding="utf-8") as f:
                stage1_report = json.load(f)
        except Exception as e:
            safe_print(f"[WARNING] Could not read Stage 1 evaluation metrics: {e}")

    stage2_json = BASE_DIR / "outputs" / "temporal" / "temporal_test_results.json"
    if stage2_json.exists():
        try:
            with open(stage2_json, "r", encoding="utf-8") as f:
                stage2_report = json.load(f)
        except Exception as e:
            safe_print(f"[WARNING] Could not read Stage 2 evaluation metrics: {e}")

    return jsonify({
        "stage1": stage1_report,
        "stage2": stage2_report
    })


@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "API endpoint not found"}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def handle_500(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error"}), 500
    return "Internal Server Error", 500


if __name__ == "__main__":
    load_models()
    safe_print("[START] Starting Deforestation Detection Web Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
