import time
import base64
import io
import json
import logging
import math
import os
import re
import sys
import tempfile
import tarfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# Load environment variables (.env) if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _load_env_fallback():
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

_load_env_fallback()

from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
import torch
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.io import MemoryFile

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from model import UNet, CHECKPOINT_PATH
from metrics import THRESHOLD
from temporal_model import TemporalUNet
from backend.sentinel_service import (
    BENCHMARK_METADATA,
    geodesic_area_hectares,
    forest_change_metrics,
    predict_directional_deforestation,
    refine_forest_mask,
    rgba_mask_to_data_uri,
)
from experiments.kaggle_amazon_deforestation.model import LateFusionSegNet
from experiments.kaggle_amazon_deforestation.dataset import (
    resolve_sentinel1_bands,
    resolve_sentinel2_bands,
    align_sentinel_pair,
)

app = Flask(__name__)
CORS(app)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cpu")
TARGET_SIZE = (256, 256)
TEMPORAL_CHECKPOINT_PATH = Path(__file__).resolve().parent / "models" / "best_temporal_model.pth"
KAGGLE_AMAZON_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "kaggle_amazon" / "late_fusion_segnet_v2_corrected_best.pt"
KAGGLE_AMAZON_NORMALIZATION = Path(__file__).resolve().parent / "checkpoints" / "kaggle_amazon" / "normalization.json"
TEMPORAL_MODEL_CHECKPOINT = TEMPORAL_CHECKPOINT_PATH
kaggle_amazon_model = None
temporal_model = None
temporal_model_validation_iou = None


def get_kaggle_amazon_model():
    """Load the independent Sentinel-1/Sentinel-2 experiment checkpoint lazily."""
    global kaggle_amazon_model
    if kaggle_amazon_model is not None:
        return kaggle_amazon_model
    if not KAGGLE_AMAZON_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Kaggle Amazon checkpoint not found: {KAGGLE_AMAZON_CHECKPOINT}")
    if not KAGGLE_AMAZON_NORMALIZATION.is_file():
        raise FileNotFoundError(f"Kaggle Amazon normalization config not found: {KAGGLE_AMAZON_NORMALIZATION}")

    checkpoint = torch.load(KAGGLE_AMAZON_CHECKPOINT, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model = LateFusionSegNet(s1_channels=2, s2_channels=4, base_channels=32).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    kaggle_amazon_model = model
    return kaggle_amazon_model

# Approximate spherical Earth radius used only for AOI area estimates.
EARTH_RADIUS_METERS = 6_371_008.8

# Sentinel-2 / Copernicus Data Space Ecosystem Configuration
SENTINEL_CLIENT_ID = os.environ.get("SENTINEL_CLIENT_ID", "").strip()
SENTINEL_CLIENT_SECRET = os.environ.get("SENTINEL_CLIENT_SECRET", "").strip()
SENTINEL_INSTANCE_ID = os.environ.get("SENTINEL_INSTANCE_ID", "").strip()
SENTINEL_TOKEN_URL = os.environ.get(
    "SENTINEL_TOKEN_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
).strip()
SENTINEL_BASE_URL = os.environ.get(
    "SENTINEL_BASE_URL",
    "https://sh.dataspace.copernicus.eu/api/v1"
).strip()


def is_sentinel_configured():
    """Checks whether Copernicus Data Space / Sentinel credentials are provided."""
    cid = os.environ.get("SENTINEL_CLIENT_ID", "").strip()
    sec = os.environ.get("SENTINEL_CLIENT_SECRET", "").strip()
    return bool(cid and sec)


_sentinel_token_cache = {
    "token": None,
    "expires_at": 0.0
}


def get_sentinel_token():
    """
    Retrieves an OAuth2 access token from Copernicus Data Space / Sentinel Hub.
    Uses in-memory caching to avoid repeated auth handshakes and rate limiting.
    Returns (token_str, error_dict, http_status_code).
    """
    global _sentinel_token_cache
    now = time.time()
    if _sentinel_token_cache["token"] and now < _sentinel_token_cache["expires_at"] - 60:
        return _sentinel_token_cache["token"], None, 200

    # Re-check env or fallback in case .env was added during runtime
    if not (os.environ.get("SENTINEL_CLIENT_ID") and os.environ.get("SENTINEL_CLIENT_SECRET")):
        _load_env_fallback()

    client_id = os.environ.get("SENTINEL_CLIENT_ID", "").strip()
    client_secret = os.environ.get("SENTINEL_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        return None, {
            "success": False,
            "error": "Sentinel credentials are not configured. Please set SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET in .env."
        }, 400

    token_url = os.environ.get(
        "SENTINEL_TOKEN_URL",
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    ).strip()

    try:
        resp = requests.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20
        )
        if resp.status_code == 200:
            token_data = resp.json()
            access_tok = token_data.get("access_token")
            exp_sec = float(token_data.get("expires_in", 600))
            _sentinel_token_cache["token"] = access_tok
            _sentinel_token_cache["expires_at"] = time.time() + exp_sec
            return access_tok, None, 200
        else:
            err_msg = "Invalid client or client credentials"
            try:
                err_data = resp.json()
                err_msg = err_data.get("error_description") or err_data.get("error") or str(err_data)
            except Exception:
                err_msg = resp.text[:200]
            return None, {
                "success": False,
                "error": f"Sentinel authentication failed: {err_msg}"
            }, 401
    except requests.exceptions.RequestException as e:
        return None, {
            "success": False,
            "error": f"Network error during Sentinel authentication: {str(e)}"
        }, 503


# Global variables for loaded models
stage1_model = None
temporal_model = None


def estimate_pixel_area_hectares(bbox, pixel_count):
    """Estimate hectares per prediction pixel from a real geographic AOI bbox.

    This spherical-area approximation is only appropriate when the prediction
    image spans the supplied bbox. Unreferenced samples and uploads return None.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or pixel_count <= 0:
        return None
    try:
        west, south, east, north = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in (west, south, east, north)):
        return None
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        return None

    west_rad, east_rad = np.radians([west, east])
    south_rad, north_rad = np.radians([south, north])
    area_square_meters = EARTH_RADIUS_METERS ** 2 * (east_rad - west_rad) * (
        np.sin(north_rad) - np.sin(south_rad)
    )
    if not np.isfinite(area_square_meters) or area_square_meters <= 0:
        return None
    return float(area_square_meters / pixel_count / 10_000.0)


def parse_probability_threshold(value, default=0.5):
    """Parse and validate a binary-classification probability threshold."""
    try:
        threshold = float(default if value is None else value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Threshold must be a number between 0 and 1.") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be a number between 0 and 1.")
    return threshold


def parse_geographic_bbox(value):
    """Validate a [west, south, east, north] WGS84 bounding box."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("AOI bbox must contain west, south, east, and north coordinates.")
    try:
        west, south, east, north = (float(coordinate) for coordinate in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("AOI bbox coordinates must be numeric.") from exc
    if not all(np.isfinite(coordinate) for coordinate in (west, south, east, north)):
        raise ValueError("AOI bbox coordinates must be finite numbers.")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("AOI bbox must be ordered and within WGS84 longitude/latitude ranges.")
    return [west, south, east, north]


def remap_temporal_checkpoint_keys(state_dict):
    """Translate legacy decoder names to current names without altering weights."""
    aliases = {f"cu{i}.": f"conv_up{i}." for i in range(1, 5)}
    mapped = {}
    for name, value in state_dict.items():
        for old, new in aliases.items():
            if name.startswith(old):
                name = new + name[len(old):]
                break
        if name in mapped:
            raise ValueError(f"Duplicate Temporal U-Net parameter after key mapping: {name}")
        mapped[name] = value
    return mapped


def build_temporal_input(t1, t2):
    """Build the exact training input [RGB T1, RGB T2, |T1-T2|]."""
    if t1.shape != t2.shape or t1.ndim != 4 or t1.shape[1] != 3:
        raise ValueError("Temporal inputs must be aligned (N, 3, H, W) RGB tensors.")
    return torch.cat((t1, t2, torch.abs(t1 - t2)), dim=1)


def load_models():
    global stage1_model, temporal_model, temporal_model_validation_iou

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

    temporal_model = None
    temporal_model_validation_iou = None
    if TEMPORAL_MODEL_CHECKPOINT.is_file():
        try:
            checkpoint = torch.load(TEMPORAL_MODEL_CHECKPOINT, map_location=DEVICE, weights_only=True)
            config = checkpoint.get("config") or {}
            if config and (config.get("in_channels") != 9 or not config.get("diff_channels")):
                raise ValueError("Expected the trained 9-channel [T1,T2,|T1-T2|] configuration.")
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
            if not isinstance(state_dict, dict):
                if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
                    raise ValueError("Checkpoint has no model_state_dict.")
                state_dict = checkpoint
            compatible_state = remap_temporal_checkpoint_keys(state_dict)
            candidate_model = TemporalUNet(
                in_channels=9,
                out_channels=1,
                init_features=int(config.get("init_features", 32)),
                bottleneck_dropout=float(config.get("bottleneck_dropout", 0.0)),
            ).to(DEVICE)
            candidate_model.load_state_dict(compatible_state, strict=True)
            candidate_model.eval()
            temporal_model = candidate_model
            temporal_model_validation_iou = checkpoint.get("best_val_iou")
            print(f"[OK] Loaded 9-channel Temporal U-Net from {TEMPORAL_MODEL_CHECKPOINT}")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"[WARNING] Temporal U-Net unavailable; other application features remain available: {exc}")
    else:
        print(f"[WARNING] Temporal U-Net checkpoint not found at {TEMPORAL_MODEL_CHECKPOINT}")

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


def decode_stage2_upload_pair(file_t1, file_t2):
    """Decode uploaded T1/T2 images and ensure they use the same pixel grid."""
    try:
        image_t1 = Image.open(file_t1.stream).convert("RGB")
        image_t2 = Image.open(file_t2.stream).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read both uploaded images: {exc}") from exc

    if image_t1.size != image_t2.size:
        raise ValueError(
            "Uploaded T1 and T2 images must have identical dimensions and be "
            "co-registered to the same geographic footprint."
        )

    return image_t1, image_t2


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sentinel/status", methods=["GET"])
def get_sentinel_status():
    """
    Verifies that Copernicus Data Space / Sentinel Hub API configuration is present.
    Does not make satellite requests; checks credentials readiness.
    """
    configured = is_sentinel_configured()
    client_id = os.environ.get("SENTINEL_CLIENT_ID", "").strip()

    masked_id = None
    if client_id:
        masked_id = f"{client_id[:4]}...{client_id[-4:]}" if len(client_id) > 8 else "***"

    return jsonify({
        "configured": configured,
        "provider": "Copernicus Data Space Ecosystem / Sentinel Hub",
        "auth_url": os.environ.get("SENTINEL_TOKEN_URL", SENTINEL_TOKEN_URL),
        "api_base_url": os.environ.get("SENTINEL_BASE_URL", SENTINEL_BASE_URL),
        "client_id_configured": bool(client_id),
        "client_id_preview": masked_id
    })


@app.route("/api/sentinel/search", methods=["GET"])
def search_sentinel():
    """
    Searches for available Sentinel-2 acquisitions around a target location and date range.
    Accepts:
      - lat or latitude (float, -90 to 90)
      - lon or longitude (float, -180 to 180)
      - start_date (YYYY-MM-DD)
      - end_date (YYYY-MM-DD)
      - limit (optional int, default 50, max 100)
      - bbox (optional str, "west,south,east,north"; if omitted, 0.05 deg delta around lat/lon)
    """
    # 1. Parameter extraction
    lat_raw = request.args.get("lat") or request.args.get("latitude")
    lon_raw = request.args.get("lon") or request.args.get("longitude")
    start_date_raw = request.args.get("start_date")
    end_date_raw = request.args.get("end_date")

    # 2. Check missing parameters
    if not lat_raw:
        return jsonify({"success": False, "error": "Missing required parameter: 'lat' (or 'latitude')"}), 400
    if not lon_raw:
        return jsonify({"success": False, "error": "Missing required parameter: 'lon' (or 'longitude')"}), 400
    if not start_date_raw:
        return jsonify({"success": False, "error": "Missing required parameter: 'start_date' (format: YYYY-MM-DD)"}), 400
    if not end_date_raw:
        return jsonify({"success": False, "error": "Missing required parameter: 'end_date' (format: YYYY-MM-DD)"}), 400

    # 3. Coordinate validation
    try:
        lat = float(lat_raw)
        if not (-90.0 <= lat <= 90.0):
            return jsonify({"success": False, "error": "Latitude must be between -90 and 90 degrees."}), 400
    except ValueError:
        return jsonify({"success": False, "error": f"Invalid latitude value: '{lat_raw}'. Expected float."}), 400

    try:
        lon = float(lon_raw)
        if not (-180.0 <= lon <= 180.0):
            return jsonify({"success": False, "error": "Longitude must be between -180 and 180 degrees."}), 400
    except ValueError:
        return jsonify({"success": False, "error": f"Invalid longitude value: '{lon_raw}'. Expected float."}), 400

    # 4. Date validation
    try:
        start_dt = datetime.strptime(start_date_raw.strip(), "%Y-%m-%d")
    except ValueError:
        return jsonify({"success": False, "error": f"Invalid start_date: '{start_date_raw}'. Expected YYYY-MM-DD format."}), 400

    try:
        end_dt = datetime.strptime(end_date_raw.strip(), "%Y-%m-%d")
    except ValueError:
        return jsonify({"success": False, "error": f"Invalid end_date: '{end_date_raw}'. Expected YYYY-MM-DD format."}), 400

    if start_dt > end_dt:
        return jsonify({"success": False, "error": f"'start_date' ({start_date_raw}) cannot be later than 'end_date' ({end_date_raw})."}), 400

    # 5. Sentinel credentials & authentication
    token, err_dict, status_code = get_sentinel_token()
    if not token:
        return jsonify(err_dict), status_code

    # 6. Define geographic area for search.
    #    Optional 'bbox' param: "west,south,east,north" (e.g. "-62.3,-3.5,-62.1,-3.4").
    #    If not provided, use 0.05 degree delta around lat/lon (~5.5 km bounding box).
    bbox_raw = request.args.get("bbox")
    if bbox_raw:
        try:
            parts = [float(v.strip()) for v in bbox_raw.split(",")]
            if len(parts) != 4:
                raise ValueError("bbox must have exactly 4 comma-separated values")
            search_bbox = [round(parts[0], 6), round(parts[1], 6),
                           round(parts[2], 6), round(parts[3], 6)]
        except ValueError as ve:
            return jsonify({"success": False,
                            "error": f"Invalid 'bbox' parameter: {str(ve)}. "
                                     "Expected 'west,south,east,north'."}), 400
    else:
        delta_deg = 0.05
        search_bbox = [
            round(lon - delta_deg, 6),
            round(lat - delta_deg, 6),
            round(lon + delta_deg, 6),
            round(lat + delta_deg, 6),
        ]

    # 7. Query Copernicus Data Space / Sentinel Hub STAC Catalog with pagination
    base_url = os.environ.get("SENTINEL_BASE_URL", SENTINEL_BASE_URL).rstrip("/")
    catalog_url = f"{base_url}/catalog/1.0.0/search"

    # Copernicus STAC maximum allowed limit per page is 100
    stac_page_limit = 100

    # Support optional collection parameter: sentinel-2-l2a (default), sentinel-2-l1c, or all
    coll_param = (request.args.get("collection") or "sentinel-2-l2a").strip().lower()
    if coll_param in ("sentinel-2-l1c", "l1c"):
        target_collections = ["sentinel-2-l1c"]
    elif coll_param in ("all", "both"):
        target_collections = ["sentinel-2-l2a", "sentinel-2-l1c"]
    else:
        target_collections = ["sentinel-2-l2a"]

    # Max total acquisitions to return across all pages (default 1000, covers ~10+ years)
    max_acquisitions = 1000
    if request.args.get("limit"):
        try:
            max_acquisitions = min(max(1, int(request.args.get("limit"))), 2000)
        except ValueError:
            pass

    # Inclusive UTC boundaries for filtering
    start_dt_utc = start_dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    end_dt_utc = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)

    print(f"[Sentinel Search] Query: range={start_date_raw.strip()} to {end_date_raw.strip()} | bbox={search_bbox} | collections={target_collections}")

    search_payload = {
        "collections": target_collections,
        "datetime": f"{start_date_raw.strip()}T00:00:00Z/{end_date_raw.strip()}T23:59:59Z",
        "bbox": search_bbox,
        "limit": stac_page_limit
    }

    all_features = []
    seen_ids = set()
    next_token = None
    page_count = 0
    max_pages = 25
    has_more = False

    while page_count < max_pages:
        page_count += 1
        if next_token:
            search_payload["next"] = next_token
        elif "next" in search_payload:
            del search_payload["next"]

        try:
            cat_resp = requests.post(
                catalog_url,
                json=search_payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                timeout=25
            )
        except requests.exceptions.RequestException as e:
            if not all_features:
                return jsonify({
                    "success": False,
                    "error": f"Network error connecting to Sentinel-2 catalog: {str(e)}"
                }), 503
            break

        if cat_resp.status_code != 200:
            if not all_features:
                err_detail = cat_resp.text[:300]
                try:
                    err_json = cat_resp.json()
                    err_detail = err_json.get("description") or err_json.get("error") or err_detail
                except Exception:
                    pass
                return jsonify({
                    "success": False,
                    "error": f"Sentinel catalog search failed (HTTP {cat_resp.status_code}): {err_detail}"
                }), cat_resp.status_code
            break

        try:
            cat_data = cat_resp.json()
        except Exception:
            if not all_features:
                return jsonify({"success": False, "error": "Invalid JSON received from Sentinel catalog service."}), 502
            break

        features = cat_data.get("features", [])
        if not features:
            break

        for item in features:
            item_id = item.get("id")
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                all_features.append(item)
                if len(all_features) >= max_acquisitions:
                    break

        next_token = cat_data.get("context", {}).get("next")
        if not next_token:
            for link in cat_data.get("links", []):
                if link.get("rel") == "next":
                    body = link.get("body", {})
                    next_token = body.get("next")
                    break

        if len(all_features) >= max_acquisitions:
            if next_token:
                has_more = True
            break

        if not next_token:
            break

    acquisitions = []
    for item in all_features:
        item_id = item.get("id")
        props = item.get("properties", {})
        dt_str = props.get("datetime")

        # Strictly enforce requested date range
        if dt_str:
            try:
                item_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                if not (start_dt_utc <= item_dt <= end_dt_utc):
                    continue
            except Exception:
                pass

        cloud = props.get("eo:cloud_cover")
        if cloud is None:
            cloud = props.get("cloudCover")
        if cloud is not None:
            try:
                cloud = round(float(cloud), 2)
            except (ValueError, TypeError):
                pass

        coll_name = item.get("collection")
        if not coll_name:
            coll_name = "sentinel-2-l1c" if ("_MSIL1C_" in str(item_id) or str(item_id).startswith(("S2A_MSIL1C", "S2B_MSIL1C"))) else "sentinel-2-l2a"
        is_l1c = (coll_name == "sentinel-2-l1c")
        
        acquisitions.append({
            "id": item_id,
            "datetime": dt_str,
            "cloud_coverage": cloud,
            "collection": coll_name,
            "level": "L1C" if is_l1c else "L2A"
        })

    # Sort acquisitions chronologically, oldest first, so the timeline flows left-to-right
    acquisitions.sort(key=lambda x: x.get("datetime") or "")

    earliest_dt = acquisitions[0]["datetime"] if acquisitions else "None"
    latest_dt = acquisitions[-1]["datetime"] if acquisitions else "None"
    print(f"[Sentinel Search] Completed: returned {len(acquisitions)} item(s) in {page_count} page(s). Earliest: {earliest_dt} | Latest: {latest_dt}")

    resp_payload = {
        "success": True,
        "count": len(acquisitions),
        "total_count": len(acquisitions),
        "has_more": has_more,
        "query": {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date_raw.strip(),
            "end_date": end_date_raw.strip(),
            "search_bbox": search_bbox,
            "collections": target_collections
        },
        "acquisitions": acquisitions
    }

    if len(acquisitions) == 0:
        resp_payload["message"] = "No matching Sentinel-2 acquisitions found for the requested location and date range."

    return jsonify(resp_payload), 200


@app.route("/api/sentinel/preview", methods=["GET"])
def preview_sentinel():
    """
    Given a Sentinel-2 acquisition ID returned by /api/sentinel/search,
    requests a small 512x512 RGB preview image from Sentinel Hub / CDSE Process API.
    
    Accepts:
      - acquisition_id (or id)
      - lat or latitude
      - lon or longitude
      - format (optional: 'png' [default] or 'json')
      - bbox (optional str, "west,south,east,north"; if omitted, 0.05 deg delta around lat/lon)
      - width (optional int, default 512, max 1024)
      - height (optional int, default 512, max 1024)
    """
    # 1. Parameter extraction & validation
    acquisition_id = request.args.get("acquisition_id") or request.args.get("id")
    lat_raw = request.args.get("lat") or request.args.get("latitude")
    lon_raw = request.args.get("lon") or request.args.get("longitude")
    bbox_raw = request.args.get("bbox")

    if not acquisition_id or not acquisition_id.strip():
        return jsonify({
            "success": False,
            "error": "Missing required parameter: 'acquisition_id' (or 'id')."
        }), 400

    acquisition_id = acquisition_id.strip()

    # If lat/lon are omitted but bbox is provided, compute center from bbox
    if (not lat_raw or not lon_raw) and bbox_raw:
        try:
            parts = [float(v.strip()) for v in bbox_raw.split(",")]
            if len(parts) == 4:
                lon_raw = str((parts[0] + parts[2]) / 2.0)
                lat_raw = str((parts[1] + parts[3]) / 2.0)
        except Exception:
            pass

    if not lat_raw:
        return jsonify({
            "success": False,
            "error": "Missing required parameter: 'lat' (or 'latitude')."
        }), 400
    if not lon_raw:
        return jsonify({
            "success": False,
            "error": "Missing required parameter: 'lon' (or 'longitude')."
        }), 400

    try:
        lat = float(lat_raw)
        if not (-90.0 <= lat <= 90.0):
            return jsonify({
                "success": False,
                "error": "Latitude must be between -90 and 90 degrees."
            }), 400
    except ValueError:
        return jsonify({
            "success": False,
            "error": f"Invalid latitude value: '{lat_raw}'. Expected float."
        }), 400

    try:
        lon = float(lon_raw)
        if not (-180.0 <= lon <= 180.0):
            return jsonify({
                "success": False,
                "error": "Longitude must be between -180 and 180 degrees."
            }), 400
    except ValueError:
        return jsonify({
            "success": False,
            "error": f"Invalid longitude value: '{lon_raw}'. Expected float."
        }), 400

    # 2. Authenticate
    token, err_dict, status_code = get_sentinel_token()
    if not token:
        return jsonify(err_dict), status_code

    base_url = os.environ.get("SENTINEL_BASE_URL", SENTINEL_BASE_URL).rstrip("/")

    # 3. Resolve the acquisition timestamp/date
    time_from = None
    time_to = None

    # Check if date/datetime passed explicitly in query args
    if request.args.get("datetime"):
        dt_param = request.args.get("datetime").strip()
        try:
            dt = datetime.fromisoformat(dt_param.replace("Z", "+00:00"))
            time_from = (dt - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
            time_to = (dt + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            time_from = f"{dt_param[:10]}T00:00:00Z"
            time_to = f"{dt_param[:10]}T23:59:59Z"
    elif request.args.get("date"):
        d_param = request.args.get("date").strip()
        time_from = f"{d_param}T00:00:00Z"
        time_to = f"{d_param}T23:59:59Z"

    # Query STAC catalog by ID to get the exact datetime if not yet resolved
    if not time_from:
        stac_search_url = f"{base_url}/catalog/1.0.0/search"
        try:
            stac_resp = requests.post(
                stac_search_url,
                json={"collections": ["sentinel-2-l2a"], "ids": [acquisition_id]},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                timeout=15
            )
            if stac_resp.status_code == 200:
                features = stac_resp.json().get("features", [])
                if features:
                    dt_str = features[0].get("properties", {}).get("datetime")
                    if dt_str:
                        try:
                            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                            time_from = (dt - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
                            time_to = (dt + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
                        except Exception:
                            date_part = dt_str.split("T")[0]
                            time_from = f"{date_part}T00:00:00Z"
                            time_to = f"{date_part}T23:59:59Z"
        except Exception:
            pass

    # Extract timestamp from standard Sentinel-2 product name if still not resolved
    if not time_from:
        match = re.search(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", acquisition_id)
        if match:
            y, m, d, H, M, S = match.groups()
            dt = datetime(int(y), int(m), int(d), int(H), int(M), int(S))
            time_from = (dt - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
            time_to = (dt + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            match_date = re.search(r"(\d{4})-(\d{2})-(\d{2})", acquisition_id) or re.search(r"(\d{4})(\d{2})(\d{2})", acquisition_id)
            if match_date:
                if len(match_date.groups()) == 3:
                    y, m, d = match_date.groups()
                else:
                    s = match_date.group(0)
                    y, m, d = s[:4], s[4:6], s[6:8]
                time_from = f"{y}-{m}-{d}T00:00:00Z"
                time_to = f"{y}-{m}-{d}T23:59:59Z"

    if not time_from:
        return jsonify({
            "success": False,
            "error": f"Could not determine acquisition date for ID '{acquisition_id}'. Please ensure the ID was obtained from /api/sentinel/search."
        }), 404

    # 4. Geographic bounding box for the Process API.
    #    Optional 'bbox' param: "west,south,east,north" (e.g. "-62.3,-3.5,-62.1,-3.4").
    #    If not provided, use 0.05 degree delta around lat/lon (~5.5 km bounding box).
    bbox_raw = request.args.get("bbox")
    if bbox_raw:
        try:
            parts = [float(v.strip()) for v in bbox_raw.split(",")]
            if len(parts) != 4:
                raise ValueError("bbox must have exactly 4 comma-separated values")
            bbox = [round(parts[0], 6), round(parts[1], 6),
                    round(parts[2], 6), round(parts[3], 6)]
        except ValueError as ve:
            return jsonify({"success": False,
                            "error": f"Invalid 'bbox' parameter: {str(ve)}. "
                                     "Expected 'west,south,east,north'."}), 400
    else:
        delta_deg = 0.05
        bbox = [
            round(lon - delta_deg, 6),
            round(lat - delta_deg, 6),
            round(lon + delta_deg, 6),
            round(lat + delta_deg, 6),
        ]

    # 4b. Optional output dimensions (default 512x512, max 1024)
    try:
        out_width = min(max(1, int(request.args.get("width", 512))), 1024)
    except (ValueError, TypeError):
        out_width = 512
    try:
        out_height = min(max(1, int(request.args.get("height", 512))), 1024)
    except (ValueError, TypeError):
        out_height = 512

    # 5. Determine collection type (sentinel-2-l2a vs sentinel-2-l1c)
    proc_type = "sentinel-2-l2a"
    coll_hint = (request.args.get("collection") or "").lower()
    if coll_hint in ("sentinel-2-l1c", "l1c") or "_MSIL1C_" in acquisition_id or acquisition_id.startswith(("S2A_MSIL1C", "S2B_MSIL1C")):
        proc_type = "sentinel-2-l1c"

    print(f"[Sentinel Preview] Rendering: id={acquisition_id} | timeRange={time_from} to {time_to} | type={proc_type} | bbox={bbox}")

    process_url = f"{base_url}/process"

    evalscript = """//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3 }
  };
}

function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
}
"""

    process_payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                }
            },
            "data": [
                {
                    "type": proc_type,
                    "dataFilter": {
                        "timeRange": {
                            "from": time_from,
                            "to": time_to
                        },
                        "mosaickingOrder": "mostRecent"
                    }
                }
            ]
        },
        "output": {
            "width": out_width,
            "height": out_height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/png"
                    }
                }
            ]
        },
        "evalscript": evalscript
    }

    try:
        proc_resp = requests.post(
            process_url,
            json=process_payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "image/png"
            },
            timeout=30
        )
    except requests.exceptions.RequestException as e:
        return jsonify({
            "success": False,
            "error": f"Network error connecting to Sentinel Hub Process API: {str(e)}"
        }), 503

    if proc_resp.status_code != 200:
        err_msg = "Sentinel Hub cannot provide the requested image."
        try:
            err_json = proc_resp.json()
            err_msg = err_json.get("error", {}).get("message") or err_json.get("description") or err_msg
        except Exception:
            err_msg = proc_resp.text[:300] or err_msg

        return jsonify({
            "success": False,
            "error": f"Sentinel Hub cannot provide the requested image (HTTP {proc_resp.status_code}): {err_msg}"
        }), proc_resp.status_code

    # 6. Verify image content with PIL
    try:
        test_img = Image.open(io.BytesIO(proc_resp.content))
        test_img.verify()
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Invalid or corrupted image received from Sentinel Hub: {str(e)}"
        }), 502

    # 7. Return in browser-friendly format
    req_format = (request.args.get("format") or "").lower()
    accept_header = request.headers.get("Accept", "")

    if req_format == "json" or ("application/json" in accept_header and req_format != "png"):
        b64_str = base64.b64encode(proc_resp.content).decode("utf-8")
        return jsonify({
            "success": True,
            "acquisition_id": acquisition_id,
            "collection": proc_type,
            "level": "L1C" if proc_type == "sentinel-2-l1c" else "L2A",
            "latitude": lat,
            "longitude": lon,
            "bbox": bbox,
            "width": out_width,
            "height": out_height,
            "image": f"data:image/png;base64,{b64_str}",
            "time_range": {
                "from": time_from,
                "to": time_to
            }
        }), 200

    # Direct browser-friendly PNG response
    return Response(
        proc_resp.content,
        mimetype="image/png",
        headers={
            "Content-Type": "image/png",
            "Content-Disposition": f"inline; filename={acquisition_id}_preview.png",
            "X-Acquisition-ID": acquisition_id,
            "X-Sentinel-BBox": f"{bbox}"
        }
    )



@app.route("/api/sentinel/info", methods=["GET"])
def get_sentinel_info():
    """Returns metadata about the Sentinel-2 integration."""
    return jsonify({
        "configured": is_sentinel_configured(),
        "provider": "Copernicus Data Space Ecosystem / Sentinel Hub",
        "collections": ["sentinel-2-l2a"],
        "rgb_resolution_meters": 10,
        "bands": {"R": "B04", "G": "B03", "B": "B02"},
        "note": ("Sentinel-2 imagery is for visualization only and is NOT "
                 "connected to the ML model pipeline.")
    })


# ── Shared helpers (used by preview and cloud-stats) ──────────────────────────

def _resolve_acquisition_time_range(acquisition_id, token, base_url,
                                    datetime_param=None, date_param=None):
    """
    Resolve a Sentinel-2 acquisition ID to a (time_from, time_to) pair.

    Resolution order:
      1. Caller-provided datetime/date strings.
      2. STAC catalog lookup by acquisition_id.
      3. Regex parse of timestamp embedded in the acquisition_id itself.

    Returns (time_from_str, time_to_str) or raises ValueError on failure.
    """
    time_from = time_to = None

    if datetime_param:
        dt_param = datetime_param.strip()
        try:
            dt = datetime.fromisoformat(dt_param.replace("Z", "+00:00"))
            time_from = (dt - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
            time_to   = (dt + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            time_from = f"{dt_param[:10]}T00:00:00Z"
            time_to   = f"{dt_param[:10]}T23:59:59Z"

    elif date_param:
        d = date_param.strip()
        time_from = f"{d}T00:00:00Z"
        time_to   = f"{d}T23:59:59Z"

    if not time_from:
        try:
            stac_resp = requests.post(
                f"{base_url}/catalog/1.0.0/search",
                json={"collections": ["sentinel-2-l2a"], "ids": [acquisition_id]},
                headers={"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"},
                timeout=15,
            )
            if stac_resp.status_code == 200:
                feats = stac_resp.json().get("features", [])
                if feats:
                    dt_str = feats[0].get("properties", {}).get("datetime", "")
                    if dt_str:
                        try:
                            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                            time_from = (dt - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
                            time_to   = (dt + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
                        except Exception:
                            dp = dt_str.split("T")[0]
                            time_from = f"{dp}T00:00:00Z"
                            time_to   = f"{dp}T23:59:59Z"
        except Exception:
            pass

    if not time_from:
        m = re.search(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", acquisition_id)
        if m:
            y, mo, d, H, M, S = m.groups()
            dt = datetime(int(y), int(mo), int(d), int(H), int(M), int(S))
            time_from = (dt - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
            time_to   = (dt + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            m2 = (re.search(r"(\d{4})-(\d{2})-(\d{2})", acquisition_id) or
                  re.search(r"(\d{4})(\d{2})(\d{2})", acquisition_id))
            if m2:
                y, mo, d = m2.groups()
                time_from = f"{y}-{mo}-{d}T00:00:00Z"
                time_to   = f"{y}-{mo}-{d}T23:59:59Z"

    if not time_from:
        raise ValueError(
            f"Cannot determine acquisition date for '{acquisition_id}'. "
            "Pass 'datetime' or 'date' param, or use an ID from /api/sentinel/search."
        )
    return time_from, time_to


def _parse_bbox_param(bbox_raw, lat=None, lon=None, delta_deg=0.05):
    """
    Parse a 'west,south,east,north' bbox string.
    Falls back to a square of `delta_deg` around (lat, lon) if bbox_raw is empty.
    Returns [west, south, east, north] as floats, or raises ValueError.
    """
    if bbox_raw:
        parts = [float(v.strip()) for v in bbox_raw.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must have exactly 4 comma-separated values.")
        return [round(p, 6) for p in parts]
    if lat is None or lon is None:
        raise ValueError("Either 'bbox' or both 'lat'/'lon' are required.")
    return [
        round(lon - delta_deg, 6),
        round(lat - delta_deg, 6),
        round(lon + delta_deg, 6),
        round(lat + delta_deg, 6),
    ]


# ── SCL class definitions ──────────────────────────────────────────────────────
# Sentinel-2 L2A Scene Classification Layer values (20 m resolution)
SCL_CLASSES = {
    0:  "no_data",
    1:  "saturated_or_defective",
    2:  "dark_area_pixels",
    3:  "cloud_shadows",
    4:  "vegetation",
    5:  "bare_soils",
    6:  "water",
    7:  "unclassified",
    8:  "cloud_medium_probability",
    9:  "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow_or_ice",
}

# Classes considered cloud/obscured for AOI cloud stats
# 3=shadow, 7=unclassified, 8=medium-prob cloud, 9=high-prob cloud, 10=cirrus
SCL_CLOUD_CLASSES = {3, 7, 8, 9, 10}
# Classes considered clear/usable for analysis
SCL_CLEAR_CLASSES = {4, 5, 6}  # vegetation, bare soil, water


@app.route("/api/sentinel/cloud-stats", methods=["GET"])
def sentinel_cloud_stats():
    """
    Compute AOI-specific cloud / obstruction statistics for a Sentinel-2 L2A
    acquisition using the official SCL (Scene Classification Layer).

    This is SEPARATE from the scene-level cloud cover reported by the STAC catalog.
    The STAC cloud_cover value covers the entire granule (100x100 km tile);
    this endpoint returns cloud percentage INSIDE the user's drawn AOI bbox.

    Accepts (all query parameters):
      - acquisition_id (str, required) — from /api/sentinel/search
      - bbox (str, required*)  — "west,south,east,north" of the AOI
      - lat / lon (float)      — fallback center if bbox not provided
      - datetime (str)         — ISO 8601 acquisition datetime (optional, speeds lookup)
      - date (str)             — YYYY-MM-DD (optional fallback)
      - resolution (int)       — SCL pixel size in metres; 20 (default) or 60

    Returns JSON:
      {
        "success": true,
        "acquisition_id": "...",
        "bbox": [west, south, east, north],
        "scl_resolution_meters": 20,
        "time_range": {"from": "...", "to": "..."},
        "total_pixels": 65536,
        "valid_pixels": 63000,          // pixels with dataMask=1 (exclude no_data)
        "cloud_pixels": 12000,
        "clear_pixels": 51000,
        "cloud_pct": 19.05,             // cloud / valid * 100
        "clear_pct": 80.95,
        "scl_class_counts": {
          "vegetation": 45000, "cloud_high_probability": 8000, ...
        },
        "cloud_classes_used": [3, 7, 8, 9, 10],
        "note": "Cloud percentage is AOI-specific, not scene-level."
      }

    The SCL band is provided at 20 m native resolution. We request a fixed
    512x512 grid over the AOI bbox; the Process API resamples accordingly.
    Cloud percentage is computed purely from pixel counts, not area weighting
    (all pixels in the requested grid are assumed equal in area within the AOI).
    """
    if not is_sentinel_configured():
        return jsonify({
            "success": False,
            "error": "Sentinel credentials are not configured.",
        }), 400

    # ── 1. Parameters ──────────────────────────────────────────────────────────
    acquisition_id = (request.args.get("acquisition_id") or
                      request.args.get("id") or "").strip()
    if not acquisition_id:
        return jsonify({
            "success": False,
            "error": "Missing required parameter: 'acquisition_id'.",
        }), 400

    lat_raw = request.args.get("lat") or request.args.get("latitude")
    lon_raw = request.args.get("lon") or request.args.get("longitude")
    lat = lon = None
    if lat_raw:
        try:
            lat = float(lat_raw)
        except ValueError:
            return jsonify({"success": False,
                            "error": f"Invalid latitude: '{lat_raw}'."}), 400
    if lon_raw:
        try:
            lon = float(lon_raw)
        except ValueError:
            return jsonify({"success": False,
                            "error": f"Invalid longitude: '{lon_raw}'."}), 400

    try:
        bbox = _parse_bbox_param(request.args.get("bbox"), lat=lat, lon=lon)
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400

    try:
        scl_res = int(request.args.get("resolution", 20))
        if scl_res not in (10, 20, 60):
            scl_res = 20
    except (ValueError, TypeError):
        scl_res = 20

    # ── 2. Authenticate ────────────────────────────────────────────────────────
    token, err_dict, status_code = get_sentinel_token()
    if not token:
        return jsonify(err_dict), status_code

    base_url = os.environ.get("SENTINEL_BASE_URL", SENTINEL_BASE_URL).rstrip("/")

    # ── 3. Resolve acquisition time range ─────────────────────────────────────
    try:
        time_from, time_to = _resolve_acquisition_time_range(
            acquisition_id, token, base_url,
            datetime_param=request.args.get("datetime"),
            date_param=request.args.get("date"),
        )
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 404

    print(f"[cloud-stats] acq={acquisition_id}  bbox={bbox}  time={time_from}..{time_to}")

    # ── 4. Build SCL evalscript ────────────────────────────────────────────────
    # Output: 2-band PNG — band 0 = SCL value [0..11], band 1 = dataMask [0 or 1]
    # We request a grayscale PNG (L mode) from the SCL band only and a separate
    # dataMask band. Since PNG can only write combined channels cleanly, we use
    # a 2-band output (RGBA PNG: R=SCL, G=0, B=0, A=dataMask*255).
    scl_evalscript = """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["SCL", "dataMask"]
    }],
    output: [
      {
        id: "scl_out",
        bands: 1,
        sampleType: "UINT8"
      },
      {
        id: "mask_out",
        bands: 1,
        sampleType: "UINT8"
      }
    ]
  };
}

function evaluatePixel(sample) {
  return {
    scl_out:  [sample.SCL],
    mask_out: [sample.dataMask * 255]
  };
}
"""

    # Request two single-band PNGs as a multipart response.
    # Sentinel Hub Process API supports multiple responses via multipart/form-data.
    process_url = f"{base_url}/process"

    # Grid size: 512x512 gives ~20 m/pixel for a 0.1-deg AOI, adequate for SCL.
    GRID = 512

    process_payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                }
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": time_from,
                            "to":   time_to
                        },
                        "mosaickingOrder": "mostRecent"
                    }
                }
            ]
        },
        "output": {
            "width":  GRID,
            "height": GRID,
            "responses": [
                {
                    "identifier": "scl_out",
                    "format": {"type": "image/png"}
                },
                {
                    "identifier": "mask_out",
                    "format": {"type": "image/png"}
                }
            ]
        },
        "evalscript": scl_evalscript
    }

    # ── 5. Call Process API ────────────────────────────────────────────────────
    try:
        proc_resp = requests.post(
            process_url,
            json=process_payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/tar",
            },
            timeout=45,
        )
    except requests.exceptions.RequestException as e:
        return jsonify({
            "success": False,
            "error": f"Network error during SCL request: {str(e)}",
        }), 503

    # ── 6. Parse multipart/tar response ───────────────────────────────────────
    scl_arr = None
    mask_arr = None

    if proc_resp.status_code == 200:
        content_type = proc_resp.headers.get("Content-Type", "")
        if "tar" in content_type or "octet-stream" in content_type:
            # Parse tar archive containing scl_out.png and mask_out.png
            import tarfile
            try:
                tar_buf = io.BytesIO(proc_resp.content)
                with tarfile.open(fileobj=tar_buf) as tar:
                    for member in tar.getmembers():
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        img_data = f.read()
                        pil = Image.open(io.BytesIO(img_data)).convert("L")
                        arr = np.array(pil, dtype=np.uint8)
                        name = member.name.lower()
                        if "scl" in name:
                            scl_arr = arr
                        elif "mask" in name:
                            mask_arr = arr
            except Exception as te:
                # Fallback: try interpreting as a single PNG (some Process API
                # versions return single response even if 2 requested)
                try:
                    pil = Image.open(io.BytesIO(proc_resp.content)).convert("L")
                    scl_arr = np.array(pil, dtype=np.uint8)
                except Exception:
                    pass
        elif "image/png" in content_type or "image" in content_type:
            # Single-band PNG fallback
            try:
                pil = Image.open(io.BytesIO(proc_resp.content)).convert("L")
                scl_arr = np.array(pil, dtype=np.uint8)
            except Exception as ie:
                return jsonify({
                    "success": False,
                    "error": f"Could not decode SCL PNG: {ie}",
                }), 502
        else:
            # Try to get a useful error from the body
            err_msg = proc_resp.text[:400]
            try:
                ej = proc_resp.json()
                err_msg = ej.get("error", {}).get("message") or ej.get("description") or err_msg
            except Exception:
                pass
            return jsonify({
                "success": False,
                "error": f"Unexpected content-type from Process API ({content_type}): {err_msg}",
            }), 502
    else:
        err_msg = proc_resp.text[:300]
        try:
            ej = proc_resp.json()
            err_msg = ej.get("error", {}).get("message") or ej.get("description") or err_msg
        except Exception:
            pass
        return jsonify({
            "success": False,
            "error": f"SCL Process API call failed (HTTP {proc_resp.status_code}): {err_msg}",
        }), proc_resp.status_code

    # ── 6b. Fallback: if only mask_arr came through, request SCL only ─────────
    if scl_arr is None:
        # Retry with a simpler single-output request
        simple_payload = dict(process_payload)
        simple_payload["output"] = {
            "width":  GRID,
            "height": GRID,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        }
        simple_payload["evalscript"] = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["SCL", "dataMask"] }],
    output: { bands: 2, sampleType: "UINT8" }
  };
}
function evaluatePixel(sample) {
  return [sample.SCL, sample.dataMask * 255];
}
"""
        try:
            r2 = requests.post(
                process_url,
                json=simple_payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "image/png",
                },
                timeout=45,
            )
            if r2.status_code == 200:
                pil2 = Image.open(io.BytesIO(r2.content))
                arr2 = np.array(pil2)
                if arr2.ndim == 3 and arr2.shape[2] >= 2:
                    scl_arr  = arr2[:, :, 0]
                    mask_arr = arr2[:, :, 1]
                elif arr2.ndim == 2:
                    scl_arr = arr2
                elif arr2.ndim == 3:
                    scl_arr = arr2[:, :, 0]
        except Exception:
            pass

    if scl_arr is None:
        return jsonify({
            "success": False,
            "error": "Could not retrieve SCL band from Sentinel Hub Process API.",
        }), 502

    # ── 7. Compute cloud statistics ────────────────────────────────────────────
    total_pixels = int(scl_arr.size)

    # Valid pixels: those with dataMask=1 (non-no-data).
    # If mask_arr is available, use it; otherwise count all non-zero SCL pixels
    # (SCL=0 means no_data in Sentinel-2 L2A).
    if mask_arr is not None:
        valid_mask = mask_arr > 127    # dataMask=1 means valid
    else:
        valid_mask = scl_arr != 0      # SCL=0 = no_data

    valid_pixels = int(np.sum(valid_mask))
    if valid_pixels == 0:
        return jsonify({
            "success": False,
            "error": (
                "No valid SCL pixels found for this acquisition and AOI. "
                "The requested bbox may be outside the scene footprint."
            ),
        }), 422

    # Count per SCL class (over the full grid)
    class_counts = {}
    for cls_val, cls_name in SCL_CLASSES.items():
        cnt = int(np.sum(scl_arr == cls_val))
        if cnt > 0:
            class_counts[cls_name] = cnt

    # Count cloud pixels (restricted to valid data area)
    cloud_pixels = int(np.sum(valid_mask & np.isin(scl_arr, list(SCL_CLOUD_CLASSES))))
    clear_pixels = int(np.sum(valid_mask & np.isin(scl_arr, list(SCL_CLEAR_CLASSES))))

    cloud_pct = round(cloud_pixels / valid_pixels * 100.0, 2)
    clear_pct = round(clear_pixels / valid_pixels * 100.0, 2)

    # Sanity clamp (floating-point edge cases)
    cloud_pct = max(0.0, min(100.0, cloud_pct))
    clear_pct = max(0.0, min(100.0, clear_pct))

    print(
        f"[cloud-stats] valid={valid_pixels} cloud={cloud_pixels} "
        f"({cloud_pct}%) clear={clear_pixels} ({clear_pct}%)"
    )

    return jsonify({
        "success": True,
        "acquisition_id": acquisition_id,
        "bbox": bbox,
        "scl_resolution_meters": scl_res,
        "grid_size": GRID,
        "time_range": {"from": time_from, "to": time_to},
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "cloud_pixels": cloud_pixels,
        "clear_pixels": clear_pixels,
        "cloud_pct": cloud_pct,
        "clear_pct": clear_pct,
        "scl_class_counts": class_counts,
        "cloud_classes_used": sorted(SCL_CLOUD_CLASSES),
        "cloud_class_names": {
            str(c): SCL_CLASSES[c] for c in sorted(SCL_CLOUD_CLASSES)
        },
        "note": (
            "AOI-specific cloud percentage computed from Sentinel-2 L2A SCL band. "
            "This is distinct from the scene-level eo:cloud_cover in the STAC catalog. "
            "Cloud classes used: 3=shadow, 7=unclassified, 8=medium-prob, "
            "9=high-prob, 10=cirrus."
        ),
    })


def _fetch_sentinel2_analysis_image(acquisition, bbox, output_size=(480, 480)):
    """Fetch selected Sentinel-2 acquisition pixels, not the browser preview image.

    Returns RGB reflectance rendered with the same 2.5 gain used by the map's
    existing Process API evalscript, plus a validity mask and grid metadata.
    """
    if not isinstance(acquisition, dict) or not acquisition.get("id") or not acquisition.get("datetime"):
        raise ValueError("Each selected scene must include its acquisition id and exact datetime.")

    try:
        acquired_at = datetime.fromisoformat(str(acquisition["datetime"]).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("The selected scene has an invalid acquisition datetime.") from exc
    if acquired_at.tzinfo is None:
        acquired_at = acquired_at.replace(tzinfo=timezone.utc)
    time_from = (acquired_at - timedelta(minutes=10)).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    time_to = (acquired_at + timedelta(minutes=10)).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    collection = str(acquisition.get("collection") or "sentinel-2-l2a").lower()
    if collection in {"l2a", "sentinel-2-l2a"}:
        collection = "sentinel-2-l2a"
        with_scl = True
    else:
        raise ValueError("Live model analysis requires Sentinel-2 L2A so cloud and shadow pixels can be masked.")

    token, auth_error, auth_status = get_sentinel_token()
    if not token:
        raise RuntimeError((auth_error or {}).get("error", "Sentinel authentication failed."))

    scl_test = " && ![0, 1, 3, 7, 8, 9, 10, 11].includes(sample.SCL)" if with_scl else ""

    def make_evalscript(include_nir):
        bands = ["B04", "B03", "B02"]
        units = ["REFLECTANCE", "REFLECTANCE", "REFLECTANCE"]
        if include_nir:
            bands.append("B08")
            units.append("REFLECTANCE")
        bands.append("dataMask")
        units.append("DN")
        if with_scl:
            bands.append("SCL")
            units.append("DN")
        output_bands = 5 if include_nir else 4
        nir_output = "sample.B08, " if include_nir else ""
        return f"""//VERSION=3
function setup() {{
  return {{input: [{{bands: {json.dumps(bands)}, units: {json.dumps(units)}}}],
          output: {{bands: {output_bands}, sampleType: "FLOAT32", nodataValue: -9999}}}};
}}
function evaluatePixel(sample) {{
  var valid = sample.dataMask === 1{scl_test};
  return [sample.B04, sample.B03, sample.B02, {nir_output}valid ? 1 : 0];
}}
"""

    base_url = os.environ.get("SENTINEL_BASE_URL", SENTINEL_BASE_URL).rstrip("/")
    width, height = output_size
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": collection,
                "dataFilter": {
                    "timeRange": {"from": time_from, "to": time_to},
                    "mosaickingOrder": "mostRecent",
                },
            }],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": make_evalscript(include_nir=True),
    }
    try:
        response = requests.post(
            f"{base_url}/process", json=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "image/tiff"},
            timeout=90,
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not retrieve the selected Sentinel-2 scene: {exc}") from exc
    nir_available = True
    if response.status_code == 400:
        logger.warning("Sentinel Hub rejected B08; retrying the selected scene with RGB only.")
        rgb_payload = {**payload, "evalscript": make_evalscript(include_nir=False)}
        try:
            response = requests.post(
                f"{base_url}/process", json=rgb_payload,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "image/tiff"},
                timeout=90,
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Could not retrieve the selected Sentinel-2 scene: {exc}") from exc
        nir_available = False
    if not response.ok:
        raise RuntimeError(f"Sentinel Hub could not retrieve the selected scene (HTTP {response.status_code}).")

    try:
        with MemoryFile(response.content) as mem, mem.open() as src:
            expected_bands = 5 if nir_available else 4
            if src.count != expected_bands or src.width != width or src.height != height or src.crs is None:
                raise ValueError("Sentinel Hub returned an unexpected image grid or band count.")
            data = src.read().astype(np.float32)
            raster_valid = np.all(src.read_masks(list(range(1, expected_bands + 1))) > 0, axis=0)
            metadata = {
                "crs": src.crs.to_string(),
                "transform": [float(value) for value in src.transform[:6]],
                "width": src.width,
                "height": src.height,
                "bounds": [float(value) for value in src.bounds],
            }
    except (rasterio.errors.RasterioError, OSError) as exc:
        raise RuntimeError("Sentinel Hub returned an unreadable analysis GeoTIFF.") from exc

    validity_band = 4 if nir_available else 3
    source_band_count = 4 if nir_available else 3
    valid = (
        raster_valid
        & (data[validity_band] > 0.5)
        & np.all(np.isfinite(data[:source_band_count]), axis=0)
    )
    # Convert physical reflectance to the app's existing RGB display range using
    # its preview gain; the model's training PNG rendering is not independently
    # calibrated to this reflectance source, so live predictions stay exploratory.
    rgb = np.clip(np.moveaxis(data[:3], 0, -1) * 2.5, 0.0, 1.0)
    # Keep masked cloud/NoData values from becoming artificial black inputs;
    # predictions over these pixels are excluded using the returned valid mask.
    if valid.any():
        rgb[~valid] = np.median(rgb[valid], axis=0)
    else:
        rgb[:] = 0.0
    red = data[0].copy()
    nir = data[3].copy() if nir_available else None
    scene_metadata = {
        "id": str(acquisition["id"]),
        "acquisition_id": str(acquisition["id"]),
        "datetime": acquired_at.isoformat(),
        "collection": collection,
        "cloud_mask_applied": with_scl,
        "nir_available": nir_available,
    }
    return rgb, nir, red, valid, metadata, scene_metadata


def _resize_rgb_for_temporal_model(rgb, valid, target_size=TARGET_SIZE):
    """Resize RGB using only valid source pixels and return a conservative mask."""
    rgb = np.asarray(rgb, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool).copy()
    valid &= np.all(np.isfinite(rgb), axis=-1)
    rgb = np.where(valid[..., None], np.clip(rgb, 0.0, 1.0), 0.0)
    coverage = Image.fromarray(valid.astype(np.float32), mode="F").resize(
        target_size, Image.Resampling.BILINEAR
    )
    coverage = np.asarray(coverage, dtype=np.float32)
    resized = np.zeros((*target_size[::-1], 3), dtype=np.float32)
    for channel in range(3):
        numerator = Image.fromarray(rgb[..., channel] * valid, mode="F").resize(
            target_size, Image.Resampling.BILINEAR
        )
        numerator = np.asarray(numerator, dtype=np.float32)
        resized[..., channel] = np.divide(
            numerator, coverage, out=np.zeros_like(numerator), where=coverage > 1e-6
        )
    # Mark an output pixel valid only when its bilinear source footprint is clear.
    resized_valid = coverage >= (1.0 - 1e-6)
    resized[~resized_valid] = 0.0
    image = np.rint(np.clip(resized, 0.0, 1.0) * 255.0).astype(np.uint8).astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0), resized_valid, image


def _resize_spectral_band(band, valid, target_size=TARGET_SIZE):
    """Resize one reflectance band while excluding masked source pixels."""
    band = np.asarray(band, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool).copy()
    if band.ndim != 2 or band.shape != valid.shape:
        raise ValueError("Spectral bands and validity masks must be matching 2D arrays.")
    valid &= np.isfinite(band)
    coverage = Image.fromarray(valid.astype(np.float32), mode="F").resize(
        target_size, Image.Resampling.BILINEAR
    )
    coverage = np.asarray(coverage, dtype=np.float32)
    numerator = Image.fromarray(np.where(valid, band, 0.0), mode="F").resize(
        target_size, Image.Resampling.BILINEAR
    )
    numerator = np.asarray(numerator, dtype=np.float32)
    resized = np.divide(
        numerator, coverage, out=np.zeros_like(numerator), where=coverage > 1e-6
    )
    resized_valid = coverage >= (1.0 - 1e-6)
    resized[~resized_valid] = 0.0
    return resized, resized_valid


def _prediction_gsd_metadata(bbox, size=TARGET_SIZE):
    """Estimate live model-pixel ground dimensions from an EPSG:4326 bbox."""
    if bbox is None:
        return None
    west, south, east, north = bbox
    width, height = size
    center_lat = (south + north) / 2.0
    semi_major = 6_378_137.0
    eccentricity_sq = 6.6943799901413165e-3
    sin_latitude = math.sin(math.radians(center_lat))
    denominator = math.sqrt(1.0 - eccentricity_sq * sin_latitude ** 2)
    prime_vertical_radius = semi_major / denominator
    meridional_radius = semi_major * (1.0 - eccentricity_sq) / denominator ** 3
    horizontal_m = (
        prime_vertical_radius * math.cos(math.radians(center_lat))
        * math.radians(east - west)
    )
    vertical_m = meridional_radius * math.radians(north - south)
    horizontal_gsd = abs(horizontal_m) / width
    vertical_gsd = abs(vertical_m) / height
    reported_training_model_gsd = 30.0 * 480.0 / 256.0
    return {
        "horizontal_m_per_pixel_at_center_latitude": horizontal_gsd,
        "vertical_m_per_pixel": vertical_gsd,
        "calculation": "WGS84 ellipsoid local-scale approximation at bbox center divided by prediction dimensions",
        "training_source_gsd_meters": None,
        "training_source_gsd_reported_meters_approx": 30.0,
        "training_model_input_gsd_estimate_meters_approx": reported_training_model_gsd,
        "live_to_reported_training_model_gsd_estimate_ratio": {
            "horizontal": horizontal_gsd / reported_training_model_gsd,
            "vertical": vertical_gsd / reported_training_model_gsd,
        },
        "training_scale_comparable": False,
        "training_scale_note": "Project documentation reports approximately 30 m source pixels, but local PNGs have no georeferencing; after 480-to-256 resizing that implies about 56.25 m per model pixel only if the reported source scale is accurate.",
    }


def _sentinel_analysis_error(message, status, requested_method=None, resolved_method=None):
    """Keep the analysis response contract stable when a prediction cannot run."""
    model_used = {
        "temporal": "temporal_unet_with_stage1_loss_gate",
        "baseline": "stage1_post_classification",
        "compare": "temporal_unet_and_stage1_post_classification",
    }.get(resolved_method)
    checkpoint_info = {}
    if resolved_method in {"temporal", "compare"}:
        checkpoint_info["temporal_unet"] = {
            "name": TEMPORAL_MODEL_CHECKPOINT.name,
            "loaded": temporal_model is not None,
            "validation_iou": temporal_model_validation_iou,
        }
    if resolved_method in {"temporal", "baseline", "compare"}:
        checkpoint_info["stage1"] = {"name": Path(CHECKPOINT_PATH).name, "loaded": stage1_model is not None}
    return jsonify({
        "success": False,
        "error": message,
        "detection_method": resolved_method,
        "method_selection": {
            "requested": requested_method,
            "resolved": resolved_method,
            "default_applied": requested_method is None and resolved_method is not None,
        },
        "model_used": model_used,
        "model_status": "not run",
        "checkpoint_info": checkpoint_info,
        "prediction_output": None,
        "warnings": [],
        "limitations": ["No prediction was produced for this request."],
    }), status


def _geodesic_mask_area_hectares(mask, valid, bbox):
    """Sum WGS84 ellipsoid pixel areas row by row for an EPSG:4326 bbox grid."""
    height, width = mask.shape
    west, south, east, north = bbox
    lon_step = (east - west) / width
    lat_step = (north - south) / height
    # Exact area between two parallels on the WGS84 ellipsoid. Using a
    # per-row latitude term matters because longitude degrees shrink poleward.
    semi_major = 6_378_137.0
    eccentricity_sq = 6.6943799901413165e-3
    eccentricity = eccentricity_sq ** 0.5

    def area_primitive(latitude_radians):
        sin_latitude = math.sin(latitude_radians)
        return (
            sin_latitude / (1.0 - eccentricity_sq * sin_latitude ** 2)
            + math.atanh(eccentricity * sin_latitude) / eccentricity
        )

    longitude_width = math.radians(lon_step)
    area_constant = semi_major ** 2 * (1.0 - eccentricity_sq) * longitude_width / 2.0
    area_m2 = 0.0
    for row in range(height):
        north_edge = math.radians(north - row * lat_step)
        south_edge = math.radians(north - (row + 1) * lat_step)
        pixel_area = area_constant * (area_primitive(north_edge) - area_primitive(south_edge))
        area_m2 += pixel_area * int(np.count_nonzero((mask[row] == 1) & valid[row]))
    return area_m2 / 10_000.0


def _deforestation_area_hectares(changed_pixels, total_pixels, total_area_hectares):
    if changed_pixels == 0:
        return 0.0
    if total_pixels <= 0:
        raise ValueError("total_pixels must be positive when deforestation pixels are present.")
    if total_area_hectares is None:
        return round(changed_pixels * 0.01, 3)
    return round(total_area_hectares * (changed_pixels / total_pixels), 3)


def _analyze_live_temporal_pair(data):
    """Analyze selected Sentinel-2 scenes with the temporal model and optional baseline.

    Positive Temporal U-Net pixels follow the Forest-Change training mask mapping
    (class 1 = dataset-labelled deforestation). Live outputs remain exploratory
    because the live Sentinel-2 radiometry differs from the PNG training domain.
    """
    requested_method = data.get("detection_method")
    method = "temporal" if requested_method is None else str(requested_method).strip().lower()
    try:
        bbox = parse_geographic_bbox(data.get("bbox"))
        threshold = parse_probability_threshold(data.get("threshold", 0.5))
        if method not in {"temporal", "baseline", "compare"}:
            raise ValueError("Detection method must be temporal, baseline, or compare.")
        if stage1_model is None:
            return _sentinel_analysis_error(
                "The Stage-1 U-Net is required to classify forest cover for deforestation detection.",
                503, requested_method, method,
            )
        if method != "baseline" and temporal_model is None:
            return _sentinel_analysis_error(
                "The trained 9-channel Temporal U-Net checkpoint is not available or could not be loaded.",
                503, requested_method, method,
            )
        if method in {"baseline", "compare"} and stage1_model is None:
            return _sentinel_analysis_error(
                "The Stage-1 U-Net is required for the post-classification baseline but is not loaded.",
                503, requested_method, method,
            )

        acquisition_before = data.get("acquisition_before")
        acquisition_after = data.get("acquisition_after")
        if not isinstance(acquisition_before, dict) or not isinstance(acquisition_after, dict):
            raise ValueError("Select both a before and after Sentinel-2 acquisition.")
        before_at = datetime.fromisoformat(str(acquisition_before.get("datetime", "")).replace("Z", "+00:00"))
        after_at = datetime.fromisoformat(str(acquisition_after.get("datetime", "")).replace("Z", "+00:00"))
        if before_at.tzinfo is None:
            before_at = before_at.replace(tzinfo=timezone.utc)
        if after_at.tzinfo is None:
            after_at = after_at.replace(tzinfo=timezone.utc)
        if before_at >= after_at:
            raise ValueError("The T1 acquisition must be earlier than the T2 acquisition.")
        rgb_before, nir_before, red_before, valid_before, grid_before, meta_before = _fetch_sentinel2_analysis_image(acquisition_before, bbox)
        rgb_after, nir_after, red_after, valid_after, grid_after, meta_after = _fetch_sentinel2_analysis_image(acquisition_after, bbox)
        if grid_before["crs"] != grid_after["crs"] or grid_before["transform"] != grid_after["transform"] or (grid_before["width"], grid_before["height"]) != (grid_after["width"], grid_after["height"]):
            raise ValueError("The selected Sentinel-2 scenes did not return the same spatial grid.")

        t1, valid_t1, display_t1 = _resize_rgb_for_temporal_model(rgb_before, valid_before)
        t2, valid_t2, display_t2 = _resize_rgb_for_temporal_model(rgb_after, valid_after)
        red_t1, red_valid_t1 = _resize_spectral_band(red_before, valid_before)
        red_t2, red_valid_t2 = _resize_spectral_band(red_after, valid_after)
        if nir_before is not None:
            nir_t1, nir_valid_t1 = _resize_spectral_band(nir_before, valid_before)
        else:
            nir_t1, nir_valid_t1 = None, None
        if nir_after is not None:
            nir_t2, nir_valid_t2 = _resize_spectral_band(nir_after, valid_after)
        else:
            nir_t2, nir_valid_t2 = None, None
        valid_t1 &= red_valid_t1
        valid_t2 &= red_valid_t2
        if nir_valid_t1 is not None:
            valid_t1 &= nir_valid_t1
        if nir_valid_t2 is not None:
            valid_t2 &= nir_valid_t2
        valid = valid_t1 & valid_t2
        if not valid.any():
            raise ValueError("The selected dates have no mutually clear, valid pixels in this AOI.")
        if float(valid.mean()) < 0.25:
            raise ValueError("Less than 25% of the AOI is clear on both dates. Choose different acquisitions or a smaller AOI.")

        t1 = t1.to(DEVICE)
        t2 = t2.to(DEVICE)
        temporal_mask = np.zeros(TARGET_SIZE, dtype=np.uint8)
        baseline_mask = np.zeros(TARGET_SIZE, dtype=np.uint8)
        before_forest = after_forest = None
        with torch.inference_mode():
            if method != "baseline":
                temporal_input = build_temporal_input(t1, t2)
                logits = temporal_model(temporal_input)
                temporal_mask = (torch.sigmoid(logits.float())[0, 0].cpu().numpy() > threshold).astype(np.uint8)
            logits_t1 = stage1_model(t1)
            logits_t2 = stage1_model(t2)
            before_forest = refine_forest_mask(
                torch.sigmoid(logits_t1.float())[0, 0].cpu().numpy(),
                display_t1,
                nir=nir_t1,
                red_band=red_t1,
            )
            after_forest = refine_forest_mask(
                torch.sigmoid(logits_t2.float())[0, 0].cpu().numpy(),
                display_t2,
                nir=nir_t2,
                red_band=red_t2,
            )
            baseline_mask = ((before_forest == 1) & (after_forest == 0)).astype(np.uint8)

        temporal_mask[~valid] = 0
        baseline_mask[~valid] = 0
        if before_forest is not None:
            before_forest[~valid] = 0
            after_forest[~valid] = 0
        if method != "baseline":
            temporal_mask &= baseline_mask
        chosen_mask = baseline_mask if method == "baseline" else temporal_mask
        total_pixels = int(valid.sum())
        loss_pixels = int(np.count_nonzero((chosen_mask == 1) & valid))
        loss_pct = round(100.0 * loss_pixels / total_pixels, 2)
        total_area_ha = geodesic_area_hectares(tuple(bbox)) if bbox is not None else None
        loss_ha = _deforestation_area_hectares(
            loss_pixels, total_pixels, total_area_ha
        )
        overlay = display_t2.copy()
        loss_pixels_mask = (chosen_mask == 1) & valid
        if before_forest is None:
            overlay[loss_pixels_mask] = 0.45 * overlay[loss_pixels_mask] + 0.55 * np.array([0.937, 0.267, 0.267], dtype=np.float32)
        overlay[~valid] = 0.45 * overlay[~valid] + 0.55 * np.array([0.5, 0.5, 0.5], dtype=np.float32)

        def color_mask(mask, color=(239, 68, 68)):
            canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
            canvas[(mask == 1) & valid] = color
            canvas[~valid] = (128, 128, 128)
            return array_to_base64_png(canvas)

        temporal_pixels = int(np.count_nonzero((temporal_mask == 1) & valid))
        baseline_pixels = int(np.count_nonzero((baseline_mask == 1) & valid))
        temporal_pct = round(100.0 * temporal_pixels / total_pixels, 2) if total_pixels else 0.0
        baseline_pct = round(100.0 * baseline_pixels / total_pixels, 2) if total_pixels else 0.0
        temporal_area_ha = _geodesic_mask_area_hectares(temporal_mask, valid, bbox)
        baseline_area_ha = _geodesic_mask_area_hectares(baseline_mask, valid, bbox)
        change_metrics = None
        forest_change_overlay = None
        if before_forest is not None and after_forest is not None:
            change_metrics = forest_change_metrics(
                before_forest,
                after_forest,
                total_area_hectares=geodesic_area_hectares(tuple(bbox)),
                valid_mask=valid,
                deforestation_mask=chosen_mask,
            )
            forest_t1_area_ha = _geodesic_mask_area_hectares(before_forest, valid, bbox)
            forest_t2_area_ha = _geodesic_mask_area_hectares(after_forest, valid, bbox)
            deforested_area_ha = _geodesic_mask_area_hectares(chosen_mask, valid, bbox)
            change_metrics.update({
                "area_unit": "ha",
                "forest_area_t1": round(forest_t1_area_ha, 4),
                "forest_area_t2": round(forest_t2_area_ha, 4),
                "total_forest_area_t1": round(forest_t1_area_ha, 4),
                "total_forest_area_t2": round(forest_t2_area_ha, 4),
                "deforested_area": round(deforested_area_ha, 4),
                "forest_area_t1_hectares": round(forest_t1_area_ha, 4),
                "forest_area_t2_hectares": round(forest_t2_area_ha, 4),
                "deforested_area_hectares": round(deforested_area_ha, 4),
            })
            forest_change_overlay = rgba_mask_to_data_uri(chosen_mask)
            active_deforestation = chosen_mask == 1
            overlay[active_deforestation] = (
                0.45 * overlay[active_deforestation]
                + 0.55 * np.array([0.937, 0.267, 0.267], dtype=np.float32)
            )
        warnings = []
        if method in {"temporal", "compare"}:
            warnings.append(
                "Temporal U-Net predicts class 1 is dataset-labelled deforestation and was trained on "
                "Forest-Change RGB PNGs; its source-rendering equivalence "
                "to Sentinel-2 L2A reflectance and live performance have not been validated. The existing "
                "2.5 gain clips any band above 0.4 reflectance; the saturation impact on live scenes is unknown."
            )
            warnings.append(
                "The model-pixel ground scale is AOI-dependent and the local training PNGs have no "
                "georeferencing for a verified scale comparison."
            )
        warnings.append(
            "The Stage-1 U-Net was trained on a different image domain; its forest mask is "
            "constrained by NDVI when B08 is available or Excess Green for RGB-only scenes. "
            "Independent Sentinel-2 ground-truth validation has not been performed."
        )
        model_note = " ".join(warnings)
        model_used = {
            "temporal": "temporal_unet_with_stage1_loss_gate",
            "baseline": "stage1_post_classification",
            "compare": "temporal_unet_and_stage1_post_classification",
        }[method]
        selected_checkpoint_info = {
            "temporal_unet": {
                "name": TEMPORAL_MODEL_CHECKPOINT.name,
                "loaded": temporal_model is not None,
                "validation_iou": temporal_model_validation_iou,
            },
            "stage1": {"name": Path(CHECKPOINT_PATH).name, "loaded": stage1_model is not None},
        }
        if method == "baseline":
            checkpoint_info = {"stage1": selected_checkpoint_info["stage1"]}
        else:
            checkpoint_info = selected_checkpoint_info
        return jsonify({
            "success": True,
            "date_before": meta_before["datetime"][:10],
            "date_after": meta_after["datetime"][:10],
            "acquisition_before": meta_before,
            "acquisition_after": meta_after,
            "threshold": threshold,
            "detection_method": method,
            "method_selection": {
                "requested": data.get("detection_method"),
                "resolved": method,
                "default_applied": data.get("detection_method") is None,
            },
            "model_used": model_used,
            "model_name": (
                "Stage 1 post-classification baseline" if method == "baseline" else
                "Temporal U-Net (9-channel) + Stage 1 loss gate" if method == "temporal" else
                "Temporal U-Net + Stage 1 post-classification comparison"
            ),
            "model_status": "inference complete",
            "checkpoint_info": checkpoint_info,
            "temporal_validation_iou": temporal_model_validation_iou if method != "baseline" else None,
            "total_pixels": total_pixels,
            "valid_pixels": total_pixels,
            "valid_pixel_pct": round(100.0 * valid.mean(), 2),
            "forest_mask_refinement": {
                "method": "NDVI" if meta_before["nir_available"] and meta_after["nir_available"] else "Excess Green",
                "nir_available": bool(meta_before["nir_available"] and meta_after["nir_available"]),
            },
            "defor_pixels": loss_pixels,
            "deforestation_pct": loss_pct,
            "stable_pixels": total_pixels - loss_pixels,
            "stable_pct": round(100.0 - loss_pct, 2),
            "deforestation_hectares": loss_ha,
            "area_loss_hectares": loss_ha,
            "signal": change_metrics["signal"] if change_metrics else None,
            "area_unit": change_metrics["area_unit"] if change_metrics else None,
            "forest_area_t1": change_metrics["forest_area_t1"] if change_metrics else None,
            "forest_area_t2": change_metrics["forest_area_t2"] if change_metrics else None,
            "total_forest_area_t1": change_metrics["total_forest_area_t1"] if change_metrics else None,
            "total_forest_area_t2": change_metrics["total_forest_area_t2"] if change_metrics else None,
            "deforested_area": change_metrics["deforested_area"] if change_metrics else None,
            "forest_area_t1_hectares": change_metrics["forest_area_t1_hectares"] if change_metrics else None,
            "forest_area_t2_hectares": change_metrics["forest_area_t2_hectares"] if change_metrics else None,
            "deforested_area_hectares": change_metrics["deforested_area_hectares"] if change_metrics else None,
            "forest_pixels_t1": change_metrics["forest_pixels_t1"] if change_metrics else None,
            "forest_pixels_t2": change_metrics["forest_pixels_t2"] if change_metrics else None,
            "deforested_pixels": change_metrics["deforested_pixels"] if change_metrics else None,
            "status": change_metrics["status"] if change_metrics else "No Deforestation Detected",
            "area_estimate_method": "WGS84 geodesic pixel area summed row-by-row on the selected AOI grid; estimate, not ground truth",
            "temporal_deforestation_pixels": temporal_pixels,
            "baseline_deforestation_pixels": baseline_pixels,
            "temporal_deforestation_pct": temporal_pct,
            "baseline_deforestation_pct": baseline_pct,
            "temporal_deforestation_hectares": round(temporal_area_ha, 4),
            "baseline_deforestation_hectares": round(baseline_area_ha, 4),
            "before_forest_mask": color_mask(before_forest) if before_forest is not None else None,
            "after_forest_mask": color_mask(after_forest) if after_forest is not None else None,
            "temporal_deforestation_mask": color_mask(temporal_mask) if method != "baseline" else None,
            "baseline_deforestation_mask": color_mask(baseline_mask) if method in {"baseline", "compare"} else None,
            "before_image": array_to_base64_png(np.rint(display_t1 * 255).astype(np.uint8)),
            "after_image": array_to_base64_png(np.rint(display_t2 * 255).astype(np.uint8)),
            "deforestation_mask": color_mask(chosen_mask),
            "forest_change_mask": forest_change_overlay,
            "forest_change_overlay": forest_change_overlay,
            "valid_mask": array_to_base64_png((valid.astype(np.uint8) * 255)),
            "deforestation_overlay": array_to_base64_png(np.rint(np.clip(overlay, 0, 1) * 255).astype(np.uint8)),
            "difference_image": array_to_base64_png(np.rint(np.abs(display_t2 - display_t1) * 255).astype(np.uint8)),
            "valid_mask_pct": round(100.0 * valid.mean(), 2),
            "geospatial": {
                "crs": grid_before["crs"],
                "bounds": bbox,
                "source_transform": grid_before["transform"],
                "source_size": [grid_before["width"], grid_before["height"]],
                "prediction_size": [TARGET_SIZE[0], TARGET_SIZE[1]],
                "prediction_transform": [bbox[0], (bbox[2] - bbox[0]) / TARGET_SIZE[0], 0.0, bbox[3], 0.0, -(bbox[3] - bbox[1]) / TARGET_SIZE[1]],
                "prediction_gsd": _prediction_gsd_metadata(bbox),
            },
            "prediction_output": {
                "format": "PNG data URL",
                "size": [TARGET_SIZE[0], TARGET_SIZE[1]],
                "class_encoding": {"0": "no predicted deforestation", "1": "predicted deforestation"},
                "invalid_pixel_encoding": {"mask_rgb": [128, 128, 128], "valid_mask_value": 0},
                "valid_pixels": total_pixels,
                "total_pixels": int(valid.size),
            },
            "validity": {
                "t1_valid_pixel_pct": round(100.0 * valid_t1.mean(), 2),
                "t2_valid_pixel_pct": round(100.0 * valid_t2.mean(), 2),
                "joint_valid_pixel_pct": round(100.0 * valid.mean(), 2),
                "area_excludes_invalid_pixels": True,
            },
            "gis": {"bounds": [[bbox[1], bbox[0]], [bbox[3], bbox[2]]], "center": [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]},
            "model_note": model_note,
            "warnings": warnings,
            "limitations": [
                "No independent Sentinel-2 labelled-pair evaluation is available for this response.",
                "Pixel scale depends on the AOI; training source resolution cannot be verified from local PNG metadata.",
            ],
        }), 200
    except ValueError as exc:
        return _sentinel_analysis_error(str(exc), 400, requested_method, method)
    except RuntimeError as exc:
        return _sentinel_analysis_error(str(exc), 502, requested_method, method)
    except Exception as exc:
        app.logger.exception("Live Temporal U-Net inference failed")
        return _sentinel_analysis_error(f"Live analysis failed: {exc}", 500, requested_method, method)


@app.route("/api/fetch-satellite-data", methods=["POST"])
@app.route("/api/sentinel/analyze", methods=["POST"])
def analyze_sentinel_deforestation():
    """Analyze selected Sentinel-2 acquisitions or a legacy uploaded image pair.

    The direct live path runs the trained nine-channel model and can optionally
    compute the retained Stage-1 post-classification baseline for comparison.
    """
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict) or not data:
        return _sentinel_analysis_error("Expected JSON body.", 400)
    raw_method = data.get("detection_method")
    method = "temporal" if raw_method is None else str(raw_method).strip().lower()
    if method not in {"temporal", "baseline", "compare"}:
        return _sentinel_analysis_error(
            "detection_method must be one of: temporal, baseline, compare.", 400, raw_method
        )

    has_acquisition_before = bool(data.get("acquisition_before"))
    has_acquisition_after = bool(data.get("acquisition_after"))
    if has_acquisition_before != has_acquisition_after:
        has_image_pair = bool(data.get("image_before")) and bool(data.get("image_after"))
        if not has_image_pair:
            return _sentinel_analysis_error(
                "Provide both acquisition_before and acquisition_after, or both image_before and image_after.",
                400, raw_method, method,
            )
    if has_acquisition_before and has_acquisition_after:
        return _analyze_live_temporal_pair(data)

    legacy_directional = (
        raw_method is None and request.path == "/api/fetch-satellite-data"
    )
    if legacy_directional and (stage1_model is None or temporal_model is None):
        return _sentinel_analysis_error(
            "The Stage-1 and Temporal U-Net checkpoints are required for the default directional analysis.",
            503, raw_method, method,
        )
    if stage1_model is None:
        return _sentinel_analysis_error(
            "The Stage-1 U-Net is required to classify forest cover for deforestation detection.",
            503, raw_method, method,
        )
    if method in {"temporal", "compare"} and temporal_model is None:
        return _sentinel_analysis_error(
            "The Temporal U-Net checkpoint is not loaded; the requested detection method cannot run.",
            503, raw_method, method,
        )
    if method in {"baseline", "compare"} and stage1_model is None:
        return _sentinel_analysis_error(
            "The Stage-1 checkpoint is not loaded; the requested detection method cannot run.",
            503, raw_method, method,
        )

    # -- Parse JSON body -------------------------------------------------------
    # The legacy route retains the original directional inference by default;
    # the current route defaults to the direct Temporal U-Net prediction.

    img_before_b64 = data.get("image_before", "")
    img_after_b64  = data.get("image_after",  "")
    date_before    = data.get("date_before", "Unknown")
    date_after     = data.get("date_after",  "Unknown")
    bbox           = data.get("bbox", None)
    try:
        threshold = parse_probability_threshold(data.get("threshold", 0.5))
    except ValueError as exc:
        return _sentinel_analysis_error(str(exc), 400, raw_method, method)
    if bbox is not None:
        try:
            bbox = parse_geographic_bbox(bbox)
        except ValueError as exc:
            return _sentinel_analysis_error(str(exc), 400, raw_method, method)

    if not img_before_b64 or not img_after_b64:
        return _sentinel_analysis_error(
            "Provide both acquisition_before and acquisition_after, or both image_before and image_after.",
            400, raw_method, method,
        )

    # -- Decode base64 PNG images -----------------------------------------------
    def decode_b64_image(b64_str, label):
        """Strip data-URI prefix and decode base64 to PIL Image (RGB)."""
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return pil_img, None
        except Exception as exc:
            return None, f"Failed to decode {label} image: {str(exc)}"

    pil_before, err = decode_b64_image(img_before_b64, "image_before")
    if err:
        return _sentinel_analysis_error(err, 400, raw_method, method)

    pil_after, err = decode_b64_image(img_after_b64, "image_after")
    if err:
        return _sentinel_analysis_error(err, 400, raw_method, method)
    # Both images are resized to TARGET_SIZE (256, 256) below, so small
    # preview dimension variations (e.g. from aspect ratio rounding) are handled safely.
    if pil_before.size != pil_after.size:
        print(f"[Sentinel Analyze] Dimension variance: T1={pil_before.size}, T2={pil_after.size}. Resampling both to {TARGET_SIZE}.")

    # -- Preprocessing ----------------------------------------------------------
    pil_before_r = pil_before.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    pil_after_r  = pil_after.resize(TARGET_SIZE,  Image.Resampling.BILINEAR)

    img_t1 = np.asarray(pil_before_r, dtype=np.float32) / np.float32(255.0)
    img_t2 = np.asarray(pil_after_r, dtype=np.float32) / np.float32(255.0)
    t1_tensor = torch.from_numpy(np.ascontiguousarray(img_t1.transpose(2, 0, 1))).unsqueeze(0)
    t2_tensor = torch.from_numpy(np.ascontiguousarray(img_t2.transpose(2, 0, 1))).unsqueeze(0)

    inference = None
    temporal_mask = None
    baseline_mask = None
    before_mask = None
    after_mask = None
    if legacy_directional:
        try:
            inference = predict_directional_deforestation(
                stage1_model,
                temporal_model,
                t1_tensor,
                t2_tensor,
                DEVICE,
                threshold,
            )
        except (RuntimeError, ValueError) as exc:
            logger.exception("Live Sentinel temporal inference failed.")
            return _sentinel_analysis_error(f"Temporal inference failed: {exc}", 500, raw_method, method)
        pred_mask = inference["final_mask"]
        temporal_mask = inference["temporal_mask"]
        before_mask = inference["mask_t1"]
        after_mask = inference["mask_t2"]
    else:
        with torch.inference_mode():
            if method in {"temporal", "compare"}:
                temporal_input = build_temporal_input(t1_tensor.to(DEVICE), t2_tensor.to(DEVICE))
                temporal_logits = temporal_model(temporal_input)
                temporal_mask = (torch.sigmoid(temporal_logits.float())[0, 0].cpu().numpy() > threshold).astype(np.uint8)
            if stage1_model is not None:
                before_logits = stage1_model(t1_tensor.to(DEVICE))
                after_logits = stage1_model(t2_tensor.to(DEVICE))
                before_mask = refine_forest_mask(
                    torch.sigmoid(before_logits.float())[0, 0].cpu().numpy(),
                    img_t1,
                )
                after_mask = refine_forest_mask(
                    torch.sigmoid(after_logits.float())[0, 0].cpu().numpy(),
                    img_t2,
                )
                baseline_mask = ((before_mask == 1) & (after_mask == 0)).astype(np.uint8)
                if method != "baseline":
                    temporal_mask &= baseline_mask
        pred_mask = baseline_mask if method == "baseline" else temporal_mask

    # -- Metrics ---------------------------------------------------------------
    total_pixels  = int(pred_mask.size)
    defor_pixels  = int(np.sum(pred_mask == 1))
    stable_pixels = total_pixels - defor_pixels
    defor_pct     = round((defor_pixels / total_pixels) * 100.0, 2)
    stable_pct    = round(100.0 - defor_pct, 2)
    total_area_ha = (
        geodesic_area_hectares(tuple(bbox))
        if bbox is not None
        else None
    )
    pixel_area_ha = total_area_ha / total_pixels if total_area_ha is not None else None
    defor_ha = _deforestation_area_hectares(
        defor_pixels, total_pixels, total_area_ha
    )
    defor_sqkm = round(defor_ha / 100.0, 6) if defor_ha is not None else None
    change_metrics = None
    forest_change_overlay = None
    if before_mask is not None and after_mask is not None:
        change_metrics = forest_change_metrics(
            before_mask,
            after_mask,
            total_area_hectares=total_area_ha,
            deforestation_mask=pred_mask,
        )
        forest_change_overlay = rgba_mask_to_data_uri(pred_mask)

    # -- GIS bounds for overlay ------------------------------------------------
    if bbox and len(bbox) == 4:
        west, south, east, north = bbox
        gis_bounds_obj = {
            "bounds": [[south, west], [north, east]],
            "center": [(south + north) / 2, (west + east) / 2],
            "west": west, "east": east, "south": south, "north": north,
        }
    else:
        gis_bounds_obj = None

    # -- Encode output images --------------------------------------------------
    # Before image (resized)
    before_b64 = array_to_base64_png(img_t1)

    # After image (resized)
    after_b64 = array_to_base64_png(img_t2)

    mask_b64 = rgba_mask_to_data_uri(pred_mask)

    # Overlay: after image with directional forest transitions
    overlay_np = img_t2.copy()
    red_tint   = np.array([0.937, 0.267, 0.267], dtype=np.float32)
    alpha_v    = 0.55
    if change_metrics is not None:
        defor_idx = change_metrics["active_deforestation_mask"] == 1
        overlay_np[defor_idx] = (1 - alpha_v) * overlay_np[defor_idx] + alpha_v * red_tint
    else:
        defor_idx = (pred_mask == 1)
        overlay_np[defor_idx] = (1 - alpha_v) * overlay_np[defor_idx] + alpha_v * red_tint
    overlay_b64 = array_to_base64_png(overlay_np)

    # Difference image (absolute change)
    diff_np = np.abs(img_t2 - img_t1)
    diff_b64 = array_to_base64_png(diff_np)

    warnings = []
    if method in {"temporal", "compare"}:
        warnings.append(
            "Legacy image-payload inputs have no Sentinel-2 SCL validity mask; all pixels are treated as valid. "
            "Independent Sentinel-2 ground-truth validation has not been performed."
        )
        warnings.append(
            "Uploaded RGB rendering and ground sampling distance are not calibrated against the Temporal U-Net training PNGs."
        )
    warnings.append(
        "The Stage-1 U-Net was trained on a different image domain; its forest-loss gate uses "
        "Excess Green for RGB-only inputs. Independent Sentinel-2 ground-truth validation has not been performed."
    )
    if legacy_directional:
        warnings.append(
            "The default legacy image-payload path constrains temporal predictions to pixels "
            "classified as forest at T1 and non-forest at T2."
        )
    model_used = {
        "temporal": "temporal_unet_with_stage1_loss_gate",
        "baseline": "stage1_post_classification",
        "compare": "temporal_unet_and_stage1_post_classification",
    }[method]
    checkpoint_info = {}
    if method in {"temporal", "compare"}:
        checkpoint_info["temporal_unet"] = {
            "name": TEMPORAL_MODEL_CHECKPOINT.name,
            "loaded": temporal_model is not None,
            "validation_iou": temporal_model_validation_iou,
        }
    if method in {"temporal", "baseline", "compare"}:
        checkpoint_info["stage1"] = {"name": Path(CHECKPOINT_PATH).name, "loaded": stage1_model is not None}
    response = {
        "success": True,
        "detection_method": method,
        "method_selection": {
            "requested": raw_method,
            "resolved": method,
            "default_applied": raw_method is None,
        },
        "model_used": model_used,
        "model_name": (
            "Stage 1 post-classification baseline" if method == "baseline" else
            "Temporal U-Net (9-channel) + Stage 1 loss gate" if method == "temporal" else
            "Temporal U-Net + Stage 1 post-classification comparison"
        ),
        "model_status": "inference complete",
        "checkpoint_info": checkpoint_info,
        "temporal_validation_iou": temporal_model_validation_iou if method != "baseline" else None,
        "date_before": date_before,
        "date_after":  date_after,
        "threshold": threshold,
        "total_pixels":  total_pixels,
        "defor_pixels":  defor_pixels,
        "changed_pixels": defor_pixels,
        "stable_pixels": stable_pixels,
        "deforestation_pct":  defor_pct,
        "deforestation_percentage": defor_pct,
        "stable_pct":         stable_pct,
        "deforestation_hectares": defor_ha,
        "area_loss_hectares": defor_ha,
        "signal": change_metrics["signal"] if change_metrics else "NO_DEFORESTATION",
        "status": change_metrics["status"] if change_metrics else "No Deforestation Detected",
        "area_unit": change_metrics["area_unit"] if change_metrics else None,
        "forest_area_t1": change_metrics["forest_area_t1"] if change_metrics else None,
        "forest_area_t2": change_metrics["forest_area_t2"] if change_metrics else None,
        "total_forest_area_t1": change_metrics["total_forest_area_t1"] if change_metrics else None,
        "total_forest_area_t2": change_metrics["total_forest_area_t2"] if change_metrics else None,
        "deforested_area": change_metrics["deforested_area"] if change_metrics else None,
        "forest_area_t1_hectares": change_metrics["forest_area_t1_hectares"] if change_metrics else None,
        "forest_area_t2_hectares": change_metrics["forest_area_t2_hectares"] if change_metrics else None,
        "deforested_area_hectares": change_metrics["deforested_area_hectares"] if change_metrics else None,
        "forest_pixels_t1": change_metrics["forest_pixels_t1"] if change_metrics else None,
        "forest_pixels_t2": change_metrics["forest_pixels_t2"] if change_metrics else None,
        "deforested_pixels": change_metrics["deforested_pixels"] if change_metrics else None,
        "deforestation_sqkm":     defor_sqkm,
        "total_area_ha": total_area_ha,
        "area_estimate_method": "WGS84 geodesic area of AOI bounding box" if pixel_area_ha is not None else None,
        "benchmark_metadata": BENCHMARK_METADATA,
        "gis": gis_bounds_obj,
        "before_image":        before_b64,
        "after_image":         after_b64,
        "deforestation_mask":  mask_b64,
        "forest_change_mask": forest_change_overlay,
        "forest_change_overlay": forest_change_overlay,
        "deforestation_overlay": overlay_b64,
        "difference_image":    diff_b64,
        "temporal_deforestation_mask": (
            array_to_base64_png(np.where(temporal_mask[..., None] == 1, [239, 68, 68], [0, 0, 0]).astype(np.uint8))
            if temporal_mask is not None else None
        ),
        "baseline_deforestation_mask": (
            array_to_base64_png(np.where(baseline_mask[..., None] == 1, [239, 68, 68], [0, 0, 0]).astype(np.uint8))
            if baseline_mask is not None else None
        ),
        "prediction_output": {
            "format": "PNG data URL",
            "size": [TARGET_SIZE[0], TARGET_SIZE[1]],
            "class_encoding": {"0": "no predicted deforestation", "1": "predicted deforestation"},
            "invalid_pixel_encoding": None,
            "all_pixels_treated_as_valid": True,
            "geospatial_bounds": gis_bounds_obj["bounds"] if gis_bounds_obj else None,
            "gsd": _prediction_gsd_metadata(bbox),
        },
        "warnings": warnings,
        "limitations": [
            "Image-payload requests contain no per-pixel cloud/shadow validity mask; all pixels are included in statistics and area estimates.",
            "Image payloads do not establish CRS, co-registration, or source ground sampling distance.",
        ],
        "model_note": " ".join(warnings),
    }
    if method == "compare":
        response["temporal_deforestation_pixels"] = int(np.count_nonzero(temporal_mask))
        response["baseline_deforestation_pixels"] = int(np.count_nonzero(baseline_mask))
    if before_mask is not None:
        response["before_forest_mask"] = array_to_base64_png(
            np.where(before_mask[..., None] == 1, [34, 197, 94], [0, 0, 0]).astype(np.uint8)
        )
        response["after_forest_mask"] = array_to_base64_png(
            np.where(after_mask[..., None] == 1, [34, 197, 94], [0, 0, 0]).astype(np.uint8)
        )
    if inference is not None:
        response.update({
            "directional_gate_applied": True,
            "directional_gate_pixels": inference["logical_gate_pixels"],
            "temporal_prediction_pixels": inference["temporal_prediction_pixels"],
            "temporal_probability_min": inference["temporal_probability_min"],
            "temporal_probability_max": inference["temporal_probability_max"],
            "temporal_probability_mean": inference["temporal_probability_mean"],
            "temporal_threshold_used": inference["temporal_threshold_used"],
        })
    else:
        response["directional_gate_applied"] = False
    return jsonify(response)


# ── SwinIR super-resolution (visualization only) ──────────────────────────────
# IMPORTANT: SwinIR output is NEVER fed into the Temporal U-Net.
# It is used solely for enhanced pixel-density display of Sentinel-2 RGB imagery.

def _get_swinir():
    """Lazy-import the SwinIR helper to avoid loading it at startup."""
    swinir_dir = Path(__file__).resolve().parent / "src" / "swinir"
    import sys as _sys
    if str(swinir_dir) not in _sys.path:
        _sys.path.insert(0, str(swinir_dir))
    parent_src = str(swinir_dir.parent)
    if parent_src not in _sys.path:
        _sys.path.insert(0, parent_src)
    from swinir.swinir_inference import superresolve, is_checkpoint_available, CHECKPOINT_PATH
    return superresolve, is_checkpoint_available, CHECKPOINT_PATH


@app.route("/api/sentinel/superresolve", methods=["POST"])
def sentinel_superresolve():
    """
    4x super-resolution of a Sentinel-2 RGB image using SwinIR.

    VISUALIZATION ONLY. The super-resolved image:
      - is 4x wider and taller than the input (pixel density only)
      - covers the SAME geographic AOI (bbox unchanged)
      - is NEVER fed into the Temporal U-Net or any analysis model

    Accepts JSON:
      { "image": "data:image/png;base64,...", "bbox": [w, s, e, n] }

    Returns:
      { "success": true, "image": "data:image/png;base64,...",
        "width": int, "height": int, "bbox": [...], "scale": 4, "note": str }
    """
    try:
        superresolve_fn, is_ckpt_ok, ckpt_path = _get_swinir()
    except Exception as imp_err:
        return jsonify({"success": False, "error": f"SwinIR module load error: {imp_err}"}), 500

    if not is_ckpt_ok():
        return jsonify({
            "success": False,
            "error": (
                "SwinIR checkpoint not found. "
                "Download 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth from "
                "https://github.com/JingyunLiang/SwinIR/releases and save it as "
                "models/swinir_x4.pth in the project root."
            ),
            "checkpoint_path": str(ckpt_path),
        }), 503

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"success": False, "error": "JSON body required."}), 400

    image_b64 = data.get("image", "")
    bbox = data.get("bbox", None)

    if not image_b64:
        return jsonify({"success": False, "error": "Field 'image' is required."}), 400

    try:
        _, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
        raw_bytes = base64.b64decode(encoded)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as dec_err:
        return jsonify({"success": False, "error": f"Invalid image data: {dec_err}"}), 400

    orig_w, orig_h = pil_img.size
    print(f"[SwinIR] Input: {orig_w}x{orig_h} px  Running 4x SR...")

    sr_img, err = superresolve_fn(pil_img, device=DEVICE)
    if err:
        print(f"[SwinIR] Error: {err}")
        return jsonify({"success": False, "error": err}), 500

    out_w, out_h = sr_img.size
    print(f"[SwinIR] Output: {out_w}x{out_h} px  Done.")

    buf = io.BytesIO()
    sr_img.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    sr_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    return jsonify({
        "success": True,
        "image":  sr_b64,
        "width":  out_w,
        "height": out_h,
        "bbox":   bbox,
        "scale":  4,
        "orig_width":  orig_w,
        "orig_height": orig_h,
        "note": (
            "SwinIR 4x super-resolved visualization. "
            "Pixel density is increased 4x (approx. 2.5 m/px scale from 10 m/px Sentinel-2). "
            "This does NOT represent true 2.5 m satellite measurements. "
            "For visualization purposes only. "
            "Geographic extent (bbox) is identical to the original image."
        ),
    })


@app.route("/api/sentinel/superresolve/status", methods=["GET"])
def sentinel_superresolve_status():
    """Returns whether the SwinIR checkpoint is available."""
    try:
        _, is_ckpt_ok, ckpt_path = _get_swinir()
        available = is_ckpt_ok()
    except Exception as exc:
        return jsonify({"available": False, "error": str(exc)})
    return jsonify({
        "available": available,
        "checkpoint_path": str(ckpt_path),
        "note": (
            "SwinIR 4x super-resolution checkpoint is available."
            if available
            else "SwinIR checkpoint not found. Download swinir_x4.pth to enable."
        ),
    })


@app.route("/api/gis/regions", methods=["GET"])
def get_gis_regions():
    """Return only regions with real spatial metadata (none are bundled yet)."""
    return jsonify({
        "regions": [],
        "message": "Bundled samples do not include georeferenced region metadata."
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns sample lists for quick selection in frontend."""
    amazon_dir = Path("data/Amazon Forest Dataset")
    stage1_samples = []
    
    val_images = list((amazon_dir / "Validation" / "images").glob("*.tiff")) + list((amazon_dir / "Validation" / "images").glob("*.tif"))
    for f in val_images[:10]:
        stage1_samples.append({
            "id": f.name,
            "name": f.name,
            "split": "Validation",
            "path": str(f),
            "gis": None
        })

    test_images = list((amazon_dir / "Test").glob("*.tiff")) + list((amazon_dir / "Test").glob("*.tif"))
    for f in test_images[:10]:
        stage1_samples.append({
            "id": f.name,
            "name": f"Test / {f.name}",
            "split": "Test",
            "path": str(f),
            "gis": None
        })

    temporal_root = Path("data/Forest-Change/extracted/Forest-Change-dataset/images")
    stage2_samples = []
    
    if (temporal_root / "test" / "A").exists():
        test_pairs = sorted([f.stem for f in (temporal_root / "test" / "A").glob("*.png")])
        for p in test_pairs[:12]:
            stage2_samples.append({
                "id": p,
                "name": f"Test Pair: {p}",
                "split": "test",
                "gis": None
            })

    return jsonify({
        "stage1_samples": stage1_samples,
        "stage2_samples": stage2_samples,
        "gsd_meters": None
    })


@app.route("/api/predict/stage1", methods=["POST"])
def predict_stage1():
    """Runs Stage 1 Forest Cover Segmentation with GIS metrics."""
    if stage1_model is None:
        return jsonify({"error": "Stage 1 model is not loaded"}), 500

    try:
        threshold = parse_probability_threshold(request.form.get("threshold"), THRESHOLD)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    pil_img = None
    sample_name = "Uploaded Image"

    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        sample_name = file.filename
        try:
            pil_img = Image.open(file.stream).convert("RGB")
        except (OSError, ValueError) as exc:
            return jsonify({"error": f"Could not read uploaded image: {exc}"}), 400
    else:
        sample_path_str = request.form.get("sample_path")
        if not sample_path_str:
            return jsonify({"error": "No image provided"}), 400
        try:
            candidate = Path(sample_path_str).resolve()
            allowed_roots = [
                (Path("data/Amazon Forest Dataset/Validation/images")).resolve(),
                (Path("data/Amazon Forest Dataset/Test")).resolve(),
            ]
            is_allowed_sample = any(candidate.is_relative_to(root) for root in allowed_roots)
        except (OSError, ValueError):
            is_allowed_sample = False
            candidate = None
        if not is_allowed_sample or candidate is None or not candidate.is_file() or candidate.suffix.lower() not in {".tif", ".tiff"}:
            return jsonify({"error": "Sample path must refer to a bundled validation or test TIFF."}), 400
        try:
            pil_img = Image.open(candidate).convert("RGB")
        except (OSError, ValueError) as exc:
            return jsonify({"error": f"Could not read sample image: {exc}"}), 400
        sample_name = candidate.name

    gis_bounds = None

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

    total_hectares = None
    forest_hectares = None
    forest_sqkm = None

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
        "area_estimate_method": None,
        "gis": gis_bounds,
        "original_image": original_b64,
        "mask_image": mask_b64,
        "overlay_image": overlay_b64
    })


@app.route("/api/predict/stage2", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict_stage2():
    """Runs 9-channel temporal change inference with a directional forest gate."""
    if stage1_model is None or temporal_model is None:
        return jsonify({"error": "Both the forest-cover and temporal models must be loaded"}), 500

    try:
        threshold = parse_probability_threshold(request.form.get("threshold"), 0.5)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    bbox = None
    bbox_value = request.form.get("bbox")
    if bbox_value:
        try:
            bbox_coordinates = (
                json.loads(bbox_value)
                if bbox_value.lstrip().startswith("[")
                else bbox_value.split(",")
            )
            bbox = parse_geographic_bbox(bbox_coordinates)
        except (json.JSONDecodeError, ValueError) as exc:
            return jsonify({"error": f"Invalid bbox: {exc}"}), 400
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
        try:
            pil_a, pil_b = decode_stage2_upload_pair(f_a, f_b)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    elif sample_id:
        if split not in {"train", "val", "test"} or not re.fullmatch(r"(?:train|val|test)_\d{6}", sample_id):
            return jsonify({"error": "Invalid sample pair or split."}), 400
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

    pil_a_resized = pil_a.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    pil_b_resized = pil_b.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

    img_a_np = np.asarray(pil_a_resized, dtype=np.float32) / np.float32(255.0)
    img_b_np = np.asarray(pil_b_resized, dtype=np.float32) / np.float32(255.0)

    t1_tensor = torch.from_numpy(np.ascontiguousarray(img_a_np.transpose(2, 0, 1))).unsqueeze(0)
    t2_tensor = torch.from_numpy(np.ascontiguousarray(img_b_np.transpose(2, 0, 1))).unsqueeze(0)
    try:
        inference = predict_directional_deforestation(
            stage1_model,
            temporal_model,
            t1_tensor,
            t2_tensor,
            DEVICE,
            threshold,
        )
    except (RuntimeError, ValueError) as exc:
        logger.exception("Stage 2 temporal inference failed.")
        return jsonify({"error": f"Temporal inference failed: {exc}"}), 500
    pred_defor_mask = inference["final_mask"]

    total_pixels = int(pred_defor_mask.size)
    defor_pixels = int(np.sum(pred_defor_mask == 1))
    defor_pct = round((defor_pixels / total_pixels) * 100.0, 2)
    stable_pixels = total_pixels - defor_pixels
    stable_pct = round(100.0 - defor_pct, 2)
    total_area_ha = (
        geodesic_area_hectares(tuple(bbox))
        if bbox is not None
        else None
    )
    change_metrics = forest_change_metrics(
        inference["mask_t1"],
        inference["mask_t2"],
        total_area_hectares=total_area_ha,
        deforestation_mask=pred_defor_mask,
    )
    forest_change_mask_b64 = rgba_mask_to_data_uri(pred_defor_mask)
    pixel_area_ha = total_area_ha / total_pixels if total_area_ha is not None else None
    defor_hectares = _deforestation_area_hectares(
        defor_pixels, total_pixels, total_area_ha
    )
    defor_sqkm = round(defor_hectares / 100.0, 6) if defor_hectares is not None else None
    gis_bounds = None
    if bbox is not None:
        west, south, east, north = bbox
        gis_bounds = {
            "bounds": [[south, west], [north, east]],
            "center": [(south + north) / 2, (west + east) / 2],
            "west": west,
            "east": east,
            "south": south,
            "north": north,
        }

    t1_b64 = array_to_base64_png(img_a_np)
    t2_b64 = array_to_base64_png(img_b_np)

    defor_mask_b64 = rgba_mask_to_data_uri(pred_defor_mask)

    overlay_rgb = img_b_np.copy()
    red_tint = np.array([0.937, 0.267, 0.267], dtype=np.float32)
    alpha = 0.5
    defor_indices = change_metrics["active_deforestation_mask"] == 1
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
            "gt_defor_hectares": None
        }

    return jsonify({
        "success": True,
        "pair_name": pair_name,
        "threshold": threshold,
        "total_pixels": total_pixels,
        "changed_pixels": defor_pixels,
        "deforestation_pixels": defor_pixels,
        "deforestation_pct": defor_pct,
        "deforestation_percentage": defor_pct,
        "deforestation_hectares": defor_hectares,
        "area_loss_hectares": defor_hectares,
        **{
            key: value
            for key, value in change_metrics.items()
            if not key.endswith("_mask")
        },
        "deforestation_sqkm": defor_sqkm,
        "total_area_ha": total_area_ha,
        "area_estimate_method": "WGS84 geodesic area of AOI bounding box" if pixel_area_ha is not None else None,
        "area_note": (
            None
            if total_area_ha is not None
            else "Pass bbox=[west,south,east,north] to calculate geodesic area loss."
        ),
        "bbox": bbox,
        "stable_pixels": stable_pixels,
        "stable_pct": stable_pct,
        "gis": gis_bounds,
        "t1_image": t1_b64,
        "t2_image": t2_b64,
        "deforestation_mask": defor_mask_b64,
        "forest_change_mask": forest_change_mask_b64,
        "forest_change_overlay": forest_change_mask_b64,
        "deforestation_overlay": overlay_b64,
        "gt_mask": gt_b64,
        "metrics": metrics,
        "directional_gate_pixels": inference["logical_gate_pixels"],
        "temporal_prediction_pixels": inference["temporal_prediction_pixels"],
        "temporal_probability_min": inference["temporal_probability_min"],
        "temporal_probability_max": inference["temporal_probability_max"],
        "temporal_probability_mean": inference["temporal_probability_mean"],
        "temporal_threshold_used": inference["temporal_threshold_used"],
        "benchmark_metadata": BENCHMARK_METADATA,
    })


@app.route("/api/kaggle-amazon/sentinel1", methods=["POST"])
def retrieve_kaggle_sentinel1():
    """Download numeric VV/VH backscatter on the uploaded S2 raster grid.

    Authentication is shared with the existing Copernicus integration. The
    output is deliberately tagged as calibration-unverified because the V2
    Kaggle training files do not record the SAR coefficient/units they used.
    """
    s2_upload = request.files.get("sentinel2")
    start_date = (request.form.get("start_date") or "").strip()
    end_date = (request.form.get("end_date") or "").strip()
    if not s2_upload:
        return jsonify({"error": "Upload the four-band Sentinel-2 GeoTIFF first; its footprint defines the area of interest and output grid."}), 400
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start > end:
            raise ValueError("Start date must be on or before end date.")
        if (end - start).days > 90:
            raise ValueError("Choose an acquisition window of 90 days or less.")

        handle = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        s2_path = Path(handle.name)
        handle.close()
        try:
            s2_upload.save(str(s2_path))
            with rasterio.open(s2_path) as s2_src:
                if s2_src.count != 4 or s2_src.crs is None:
                    raise ValueError("AOI reference must be a four-band Sentinel-2 GeoTIFF with a defined CRS.")
                if s2_src.width * s2_src.height > 25_000_000 or max(s2_src.width, s2_src.height) > 5000:
                    raise ValueError("Sentinel-2 grid is too large for one Process API request (limit: 5,000 pixels per side and 25 million pixels total). Clip the raster to the selected AOI first.")
                epsg = s2_src.crs.to_epsg()
                if epsg is None:
                    raise ValueError("The Sentinel-2 reference CRS must resolve to an EPSG code for Process API alignment.")
                source_bounds = tuple(s2_src.bounds)
                west, south, east, north = transform_bounds(s2_src.crs, "EPSG:4326", *source_bounds, densify_pts=21)
                if west >= east or south >= north:
                    raise ValueError("Sentinel-2 AOI bounds are invalid.")
                target = {
                    "width": s2_src.width, "height": s2_src.height,
                    "crs": s2_src.crs, "transform": s2_src.transform,
                    "profile": s2_src.profile.copy(),
                }
        finally:
            s2_path.unlink(missing_ok=True)

        token, auth_error, auth_status = get_sentinel_token()
        if not token:
            return jsonify(auth_error or {"error": "Sentinel OAuth authentication failed."}), auth_status
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        time_range = f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z"

        # Catalog lookup prevents treating a syntactically valid but empty
        # Process API request as successful. Process API remains authoritative.
        catalog = requests.post(
            "https://sh.dataspace.copernicus.eu/catalog/v1/search",
            headers=headers,
            json={"bbox": [west, south, east, north], "datetime": time_range,
                  "collections": ["sentinel-1-grd"], "limit": 100},
            timeout=45,
        )
        if not catalog.ok:
            app.logger.warning("Sentinel-1 Catalog API returned HTTP %s", catalog.status_code)
            return jsonify({"error": "Sentinel-1 availability search failed. Check the date range and try again."}), 502
        features = (catalog.json() or {}).get("features") or []
        if not features:
            return jsonify({"error": "No Sentinel-1 GRD acquisitions cover this AOI and date range."}), 422
        dual_features = []
        for feature in features:
            props = feature.get("properties") or {}
            polarization = props.get("s1:polarization") or props.get("sar:polarizations") or props.get("polarization")
            if polarization is None:
                dual_features.append(feature)  # The Process API DV filter performs the final check.
            elif isinstance(polarization, str) and polarization.upper() in {"DV", "VV+VH", "VH+VV"}:
                dual_features.append(feature)
            elif isinstance(polarization, (list, tuple)) and {str(v).upper() for v in polarization} >= {"VV", "VH"}:
                dual_features.append(feature)
        if not dual_features:
            return jsonify({"error": "Scenes were found, but none report the required dual VV+VH (DV) polarization for this AOI and date range."}), 422

        evalscript = """//VERSION=3
function setup() {
  return {
    input: [{bands: [\"VV\", \"VH\", \"dataMask\"], units: [\"LINEAR_POWER\", \"LINEAR_POWER\", \"DN\"]}],
    mosaicking: Mosaicking.ORBIT,
    output: {id: \"default\", bands: 2, sampleType: \"FLOAT32\", nodataValue: -9999}
  };
}
function preProcessScenes(collections) {
  collections.scenes.orbits.sort(function(a, b) {
    return new Date(b.dateFrom).getTime() - new Date(a.dateFrom).getTime();
  });
  return collections;
}
function evaluatePixel(samples) {
  for (var i = 0; i < samples.length; i++) {
    if (samples[i].dataMask === 1) return [samples[i].VV, samples[i].VH];
  }
  return [-9999, -9999];
}
function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  outputMetadata.userData = {orbits: scenes.orbits};
}
"""
        process_payload = {
            "input": {
                "bounds": {
                    "bbox": list(source_bounds),
                    "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"},
                },
                "data": [{
                    "type": "sentinel-1-grd",
                    "dataFilter": {
                        "timeRange": {"from": f"{start.isoformat()}T00:00:00Z", "to": f"{end.isoformat()}T23:59:59Z"},
                        "acquisitionMode": "IW", "polarization": "DV",
                        "resolution": "HIGH", "mosaickingOrder": "mostRecent",
                    },
                    "processing": {"backCoeff": "GAMMA0_ELLIPSOID", "orthorectify": True},
                }],
            },
            "output": {
                "width": target["width"], "height": target["height"],
                "responses": [
                    {"identifier": "default", "format": {"type": "image/tiff"}},
                    {"identifier": "userdata", "format": {"type": "application/json"}},
                ],
            },
            "evalscript": evalscript,
        }
        process = requests.post(
            "https://sh.dataspace.copernicus.eu/process/v1",
            headers={**headers, "Accept": "application/tar"},
            json=process_payload, timeout=180,
        )
        if not process.ok:
            app.logger.warning("Sentinel-1 Process API returned HTTP %s", process.status_code)
            return jsonify({"error": "Sentinel Hub could not create a VV/VH GeoTIFF for this AOI and date range. Try a smaller AOI or a shorter date range."}), 422 if process.status_code in (400, 404, 422) else 502

        with tarfile.open(fileobj=io.BytesIO(process.content), mode="r:*") as archive:
            tiff_member = next((member for member in archive.getmembers() if member.isfile() and member.name.lower().endswith((".tif", ".tiff"))), None)
            metadata_member = next((member for member in archive.getmembers() if member.isfile() and member.name.rsplit("/", 1)[-1] == "userdata.json"), None)
            if tiff_member is None:
                raise ValueError("Sentinel Hub returned no GeoTIFF for the selected AOI and dates.")
            tiff_bytes = archive.extractfile(tiff_member).read()
            scene_metadata = json.load(archive.extractfile(metadata_member)) if metadata_member else {}

        with MemoryFile(tiff_bytes) as memory_file, memory_file.open() as source:
            if source.count != 2:
                raise ValueError(f"Sentinel Hub returned {source.count} bands; exactly VV and VH are required.")
            downloaded = source.read().astype(np.float32)
            downloaded_valid = np.all(source.read_masks() > 0, axis=0) & np.all(np.isfinite(downloaded), axis=0) & np.all(downloaded != -9999, axis=0)
            if not downloaded_valid.any():
                return jsonify({"error": "Sentinel-1 catalog items exist, but the Process API returned no valid VV/VH pixels. Try another date range or AOI."}), 422
            aligned = np.full((2, target["height"], target["width"]), -9999.0, dtype=np.float32)
            for band in range(2):
                reproject(
                    source=downloaded[band], destination=aligned[band],
                    src_transform=source.transform, src_crs=source.crs,
                    src_nodata=-9999, dst_transform=target["transform"],
                    dst_crs=target["crs"], dst_nodata=-9999,
                    resampling=Resampling.bilinear,
                )
            output_valid = np.all(np.isfinite(aligned), axis=0) & np.all(aligned != -9999, axis=0)
            if not output_valid.any():
                return jsonify({"error": "The downloaded VV/VH data do not overlap the Sentinel-2 reference grid."}), 422
            profile = target["profile"]
            profile.update(driver="GTiff", count=2, dtype="float32", nodata=-9999, compress="deflate")
            output = io.BytesIO()
            with MemoryFile() as aligned_file:
                with aligned_file.open(**profile) as destination:
                    destination.write(aligned)
                    destination.set_band_description(1, "VV")
                    destination.set_band_description(2, "VH")
                    destination.update_tags(
                        source="Sentinel Hub Process API", collection="sentinel-1-grd",
                        polarization="DV", band_order="VV,VH",
                        backscatter_coefficient="GAMMA0_ELLIPSOID",
                        units="LINEAR_POWER (unitless)",
                        sentinel1_calibration_status="UNVERIFIED_FOR_KAGGLE_V2",
                        training_ranges="checkpoints/kaggle_amazon/normalization.json",
                        acquisition_window=time_range,
                        catalog_item_count=str(len(dual_features)),
                    )
                output.write(aligned_file.read())
            output.seek(0)

        orbit_dates = []
        for orbit in (scene_metadata.get("orbits") or []):
            for tile in orbit.get("tiles") or []:
                date_value = tile.get("date")
                if date_value and date_value not in orbit_dates:
                    orbit_dates.append(date_value)
        app.logger.info("Retrieved Sentinel-1 VV/VH on S2 grid; catalog items=%d, valid pixels=%d", len(dual_features), int(output_valid.sum()))
        response = send_file(output, mimetype="image/tiff", as_attachment=True, download_name="sentinel1_VV_VH_GAMMA0_ELLIPSOID.tif")
        response.headers["X-Sentinel1-Metadata"] = json.dumps({
            "bands": ["VV", "VH"], "backscatter_coefficient": "GAMMA0_ELLIPSOID",
            "units": "LINEAR_POWER (unitless)", "polarization": "DV",
            "acquisition_window": time_range, "acquisition_dates": orbit_dates,
            "catalog_item_count": len(dual_features), "valid_pixels": int(output_valid.sum()),
            "width": target["width"], "height": target["height"], "crs": target["crs"].to_string(),
            "aligned_to_sentinel2": True, "calibration_compatible_with_kaggle_v2": False,
            "calibration_note": "The V2 training rasters contain no recorded backscatter coefficient or units. The saved training ranges include negative values, unlike Sentinel Hub LINEAR_POWER; do not run this file through V2 inference until the training source calibration is verified.",
        })
        return response
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (tarfile.TarError, rasterio.errors.RasterioError, OSError, KeyError, StopIteration) as exc:
        app.logger.exception("Sentinel-1 Process API result validation failed")
        return jsonify({"error": f"Could not validate the Sentinel-1 GeoTIFF response: {exc}"}), 502
    except requests.RequestException as exc:
        app.logger.warning("Sentinel-1 service request failed: %s", exc.__class__.__name__)
        return jsonify({"error": "Sentinel-1 service is temporarily unreachable. Please retry shortly."}), 502
    except Exception as exc:
        app.logger.exception("Sentinel-1 retrieval failed")
        return jsonify({"error": f"Sentinel-1 retrieval failed: {exc}"}), 500


@app.route("/api/predict/kaggle-amazon", methods=["POST"])
def predict_kaggle_amazon():
    """Infer with the separate dual-stream model on aligned S1/S2 GeoTIFFs."""
    s1_upload = request.files.get("sentinel1")
    s2_upload = request.files.get("sentinel2")
    if not s1_upload or not s2_upload:
        return jsonify({"error": "Upload both Sentinel-1 and Sentinel-2 GeoTIFF files."}), 400
    try:
        threshold = parse_probability_threshold(request.form.get("threshold"), 0.5)
        model = get_kaggle_amazon_model()
        normalization = json.loads(KAGGLE_AMAZON_NORMALIZATION.read_text(encoding="utf-8"))
        low1 = np.asarray(normalization["s1"]["low"], dtype=np.float32)[:, None, None]
        high1 = np.asarray(normalization["s1"]["high"], dtype=np.float32)[:, None, None]
        low2 = np.asarray(normalization["s2"]["low"], dtype=np.float32)[:, None, None]
        high2 = np.asarray(normalization["s2"]["high"], dtype=np.float32)[:, None, None]

        temp_paths = []
        try:
            for upload in (s1_upload, s2_upload):
                handle = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                temp_paths.append(Path(handle.name))
                handle.close()
                upload.save(str(temp_paths[-1]))

            with rasterio.open(temp_paths[0]) as s1_src, rasterio.open(temp_paths[1]) as s2_src:
                if s1_src.width < 1 or s1_src.height < 1 or s2_src.width < 1 or s2_src.height < 1:
                    raise ValueError("Input rasters cannot be empty.")

                # Resolve and validate bands based on metadata and physics
                if s1_src.tags().get("sentinel1_calibration_status") == "UNVERIFIED_FOR_KAGGLE_V2":
                    raise ValueError(
                        "This retrieved Sentinel-1 file is not calibrated to the V2 training data. "
                        "The download is GAMMA0_ELLIPSOID LINEAR_POWER, but the original V2 training rasters do not record their coefficient/units and their saved ranges do not match linear power. "
                        "Use a Sentinel-1 file processed with the verified training configuration before running inference."
                    )
                s1_raw, s1_valid, s1_info = resolve_sentinel1_bands(s1_src)
                s2_raw, s2_valid, s2_info = resolve_sentinel2_bands(s2_src)
                app.logger.info("Kaggle Amazon S1: %s", s1_info)
                app.logger.info("Kaggle Amazon S2: %s", s2_info)

                # Validate geospatial coverage and align rasters
                s1, s2, valid, was_resampled, align_note = align_sentinel_pair(
                    s1_src, s2_src, s1_raw, s2_raw, s1_valid, s2_valid
                )
                app.logger.info("Kaggle Amazon Alignment: %s", align_note)

                if not valid.any():
                    raise ValueError("The rasters contain no mutually valid pixels in the overlapping area.")

                # Train-only robust normalization (1st and 99th percentiles)
                s1 = np.clip((s1 - low1) / np.maximum(high1 - low1, 1e-6), 0.0, 1.0)
                s2 = np.clip((s2 - low2) / np.maximum(high2 - low2, 1e-6), 0.0, 1.0)
                s1[:, ~valid] = 0.0
                s2[:, ~valid] = 0.0
                height, width = valid.shape
                prediction = np.zeros((height, width), dtype=np.uint8)
                window_batch1, window_batch2, windows = [], [], []
                batch_size = 2

                def run_window_batch():
                    if not windows:
                        return
                    x1 = torch.from_numpy(np.stack(window_batch1)).to(DEVICE)
                    x2 = torch.from_numpy(np.stack(window_batch2)).to(DEVICE)
                    with torch.inference_mode():
                        logits = model(x1, x2)
                        masks = (torch.sigmoid(logits.float()) >= threshold).cpu().numpy()[:, 0]
                    for (row, col, rows, cols), mask in zip(windows, masks):
                        prediction[row:row + rows, col:col + cols] = mask[:rows, :cols].astype(np.uint8)
                    windows.clear(); window_batch1.clear(); window_batch2.clear()

                for row in range(0, height, 256):
                    for col in range(0, width, 256):
                        rows, cols = min(256, height - row), min(256, width - col)
                        patch1 = np.zeros((2, 256, 256), dtype=np.float32)
                        patch2 = np.zeros((4, 256, 256), dtype=np.float32)
                        patch1[:, :rows, :cols] = s1[:, row:row + rows, col:col + cols]
                        patch2[:, :rows, :cols] = s2[:, row:row + rows, col:col + cols]
                        window_batch1.append(patch1); window_batch2.append(patch2)
                        windows.append((row, col, rows, cols))
                        if len(windows) == batch_size:
                            run_window_batch()
                run_window_batch()
                # 0=predicted non-deforestation, 1=predicted deforestation, 255=NoData
                prediction[~valid] = 255

                valid_pixels = int(valid.sum())
                predicted_pixels = int(((prediction == 1) & valid).sum())
                percent = 100.0 * predicted_pixels / max(valid_pixels, 1)
                pixel_area_ha = None
                if s1_src.crs.is_projected and s1_src.crs.linear_units.casefold() in {"metre", "meter", "metres", "meters"}:
                    pixel_area_ha = abs(s1_src.transform.a * s1_src.transform.e - s1_src.transform.b * s1_src.transform.d) / 10000.0
                hectares = predicted_pixels * pixel_area_ha if pixel_area_ha is not None else None

                prediction_id = uuid.uuid4().hex
                prediction_dir = Path(__file__).resolve().parent / "outputs" / "kaggle_amazon" / "predictions"
                prediction_dir.mkdir(parents=True, exist_ok=True)
                prediction_path = prediction_dir / f"{prediction_id}_deforestation_mask.tif"
                result_profile = s1_src.profile.copy()
                result_profile.update(
                    driver="GTiff", count=1, dtype="uint8", nodata=255,
                    compress="deflate", photometric="MINISBLACK",
                    width=width, height=height, crs=s1_src.crs, transform=s1_src.transform
                )
                with rasterio.open(prediction_path, "w", **result_profile) as result_dst:
                    result_dst.write(prediction, 1)
                    result_dst.update_tags(
                        model="Kaggle Amazon Dual-Stream Late-Fusion SegNet V2",
                        threshold=str(threshold),
                        class_0="predicted non-deforestation",
                        class_1="predicted deforestation",
                        nodata_255="invalid or NoData pixel",
                    )
                    result_dst.update_tags(**s1_src.tags())

                west, south, east, north = transform_bounds(
                    s1_src.crs, "EPSG:4326", *s1_src.bounds, densify_pts=21
                )

                # Build a bounded preview response to avoid sending full rasters as JSON.
                scale = min(1.0, 1024.0 / max(height, width))
                out_size = (max(1, round(width * scale)), max(1, round(height * scale)))

                # RGB natural color preview: Sentinel-2 Band 3 (Red), Band 2 (Green), Band 1 (Blue)
                display = np.moveaxis(s2[[2, 1, 0]], 0, -1)
                rgb = np.zeros_like(display, dtype=np.uint8)
                for channel in range(3):
                    values = display[..., channel][valid]
                    if values.size > 0:
                        lo, hi = np.percentile(values, (2, 98))
                        rgb[..., channel] = (np.clip((display[..., channel] - lo) / max(float(hi - lo), 1e-6), 0, 1) * 255).astype(np.uint8)
                image_pil = Image.fromarray(rgb).resize(out_size, Image.Resampling.BILINEAR)
                display_mask = np.where(prediction == 1, 255, 0).astype(np.uint8)
                mask_pil = Image.fromarray(display_mask).resize(out_size, Image.Resampling.NEAREST)
                overlay = np.asarray(image_pil).copy()
                overlay_mask = np.asarray(mask_pil) > 0
                overlay[overlay_mask] = (0.55 * overlay[overlay_mask] + 0.45 * np.array([255, 48, 64])).astype(np.uint8)
                transparent_overlay = np.zeros((height, width, 4), dtype=np.uint8)
                transparent_overlay[prediction == 1] = (255, 48, 64, 190)
                map_overlay = Image.fromarray(transparent_overlay, mode="RGBA").resize(out_size, Image.Resampling.NEAREST)
                return jsonify({
                    "success": True,
                    "model": "Kaggle Amazon dual-stream SegNet V2",
                    "threshold": threshold,
                    "width": width,
                    "height": height,
                    "crs": s1_src.crs.to_string(),
                    "valid_pixels": valid_pixels,
                    "deforestation_pixels": predicted_pixels,
                    "deforestation_pct": round(percent, 4),
                    "deforestation_hectares": round(hectares, 4) if hectares is not None else None,
                    "area_estimate_method": "GeoTIFF affine pixel area and projected CRS units" if hectares is not None else None,
                    "s1_info": s1_info,
                    "s2_info": s2_info,
                    "alignment_note": align_note,
                    "was_resampled": was_resampled,
                    "gis": {"bounds": [[south, west], [north, east]]},
                    "download_url": f"/api/predict/kaggle-amazon/{prediction_id}/download",
                    "preview_image": array_to_base64_png(np.asarray(image_pil)),
                    "prediction_mask": array_to_base64_png(np.asarray(mask_pil)),
                    "overlay_image": array_to_base64_png(overlay),
                    "map_overlay_image": array_to_base64_png(np.asarray(map_overlay)),
                })
        finally:
            for path in temp_paths:
                path.unlink(missing_ok=True)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except (rasterio.errors.RasterioError, OSError) as exc:
        return jsonify({"error": f"Could not read the uploaded GeoTIFFs: {exc}"}), 400
    except Exception as exc:
        app.logger.exception("Kaggle Amazon inference failed")
        return jsonify({"error": f"Kaggle Amazon inference failed: {exc}"}), 500


@app.route("/api/predict/kaggle-amazon/<prediction_id>/download", methods=["GET"])
def download_kaggle_amazon_prediction(prediction_id):
    """Download a prediction GeoTIFF with the source grid and CRS preserved."""
    if not re.fullmatch(r"[0-9a-f]{32}", prediction_id):
        return jsonify({"error": "Invalid prediction identifier."}), 400
    prediction_dir = Path(__file__).resolve().parent / "outputs" / "kaggle_amazon" / "predictions"
    prediction_path = prediction_dir / f"{prediction_id}_deforestation_mask.tif"
    if not prediction_path.is_file() or prediction_path.resolve().parent != prediction_dir.resolve():
        return jsonify({"error": "Prediction file not found. Run the analysis again."}), 404
    return send_file(prediction_path, as_attachment=True, download_name="amazon_deforestation_prediction.tif")


@app.route("/api/kaggle-amazon/research-metrics", methods=["GET"])
def get_kaggle_amazon_research_metrics():
    """Return the recorded Colab research evaluation, separate from live inference."""
    metrics_path = Path(__file__).resolve().parent / "experiments" / "kaggle_amazon_deforestation" / "research_metrics.json"
    if not metrics_path.is_file():
        return jsonify({"available": False, "message": "Recorded research metrics are unavailable."}), 404
    return jsonify(json.loads(metrics_path.read_text(encoding="utf-8")))


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
