import base64
import io
import json
import os
import re
import sys
from datetime import datetime, timedelta
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


def get_sentinel_token():
    """
    Retrieves an OAuth2 access token from Copernicus Data Space / Sentinel Hub.
    Returns (token_str, error_dict, http_status_code).
    """
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
            timeout=15
        )
        if resp.status_code == 200:
            token_data = resp.json()
            return token_data.get("access_token"), None, 200
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

    # 6. Define small geographic area around location (~0.01 deg ~= 1.1 km bounding box)
    delta_deg = 0.01
    min_lon = round(lon - delta_deg, 6)
    max_lon = round(lon + delta_deg, 6)
    min_lat = round(lat - delta_deg, 6)
    max_lat = round(lat + delta_deg, 6)
    bbox = [min_lon, min_lat, max_lon, max_lat]

    # 7. Query Copernicus Data Space / Sentinel Hub STAC Catalog
    base_url = os.environ.get("SENTINEL_BASE_URL", SENTINEL_BASE_URL).rstrip("/")
    catalog_url = f"{base_url}/catalog/1.0.0/search"

    limit_val = 50
    if request.args.get("limit"):
        try:
            limit_val = min(max(1, int(request.args.get("limit"))), 100)
        except ValueError:
            pass

    search_payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start_date_raw.strip()}T00:00:00Z/{end_date_raw.strip()}T23:59:59Z",
        "bbox": bbox,
        "limit": limit_val
    }

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
        return jsonify({
            "success": False,
            "error": f"Network error connecting to Sentinel-2 catalog: {str(e)}"
        }), 503

    if cat_resp.status_code != 200:
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

    try:
        cat_data = cat_resp.json()
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON received from Sentinel catalog service."}), 502

    features = cat_data.get("features", [])

    acquisitions = []
    for item in features:
        item_id = item.get("id")
        props = item.get("properties", {})
        dt_str = props.get("datetime")
        cloud = props.get("eo:cloud_cover")
        if cloud is None:
            cloud = props.get("cloudCover")
        if cloud is not None:
            try:
                cloud = round(float(cloud), 2)
            except (ValueError, TypeError):
                pass

        acquisitions.append({
            "id": item_id,
            "datetime": dt_str,
            "cloud_coverage": cloud
        })

    # Sort acquisitions newest first
    acquisitions.sort(key=lambda x: x.get("datetime") or "", reverse=True)

    resp_payload = {
        "success": True,
        "count": len(acquisitions),
        "query": {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date_raw.strip(),
            "end_date": end_date_raw.strip(),
            "bbox": bbox
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
    """
    # 1. Parameter extraction & validation
    acquisition_id = request.args.get("acquisition_id") or request.args.get("id")
    lat_raw = request.args.get("lat") or request.args.get("latitude")
    lon_raw = request.args.get("lon") or request.args.get("longitude")

    if not acquisition_id or not acquisition_id.strip():
        return jsonify({
            "success": False,
            "error": "Missing required parameter: 'acquisition_id' (or 'id')."
        }), 400

    acquisition_id = acquisition_id.strip()

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

    # 4. Small geographic bounding box around coordinates (~2.5 km box)
    delta_deg = 0.023
    min_lon = round(lon - delta_deg / 2, 6)
    max_lon = round(lon + delta_deg / 2, 6)
    min_lat = round(lat - delta_deg / 2, 6)
    max_lat = round(lat + delta_deg / 2, 6)
    bbox = [min_lon, min_lat, max_lon, max_lat]

    # 5. Request 512x512 RGB from Sentinel Hub Process API
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
                    "type": "sentinel-2-l2a",
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
            "width": 512,
            "height": 512,
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
            "latitude": lat,
            "longitude": lon,
            "bbox": bbox,
            "width": 512,
            "height": 512,
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
