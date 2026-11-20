import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

# Configure remote raster access before rasterio/GDAL is imported.
os.environ["GDAL_HTTP_TIMEOUT"] = "15"
os.environ["GDAL_HTTP_MAX_RETRY"] = "2"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

import cv2
import numpy as np
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
ASSET_KEYS = ("B04", "B03", "B02", "B08", "SCL")


def _read_resized_band(
    item: Any,
    asset_key: str,
    bbox: list[float],
    size: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    asset = item.assets[asset_key]
    with rasterio.open(asset.href) as src:
        if src.crs is None:
            raise ValueError(f"Sentinel-2 asset {asset_key} has no coordinate reference system.")
        projected_bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(*projected_bounds, transform=src.transform)
        band = src.read(1, window=window, boundless=True, fill_value=0)

    height, width = size
    return cv2.resize(band, (width, height), interpolation=interpolation)


def fetch_s2_patch(
    bbox: list[float],
    target_date_str: str,
    window_days: int = 60,
    size: tuple[int, int] = (256, 256),
) -> dict:
    """Fetch cloud-minimized, aligned Sentinel-2 L2A bands for a geographic bbox."""
    if len(bbox) != 4 or not all(np.isfinite(value) for value in bbox):
        raise ValueError("bbox must contain four finite [min_lon, min_lat, max_lon, max_lat] values.")
    min_lon, min_lat, max_lon, max_lat = bbox
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError("bbox must be a non-empty EPSG:4326 bounding box.")
    if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 0:
        raise ValueError("window_days must be a non-negative integer.")
    if (
        len(size) != 2
        or any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0 for dimension in size)
    ):
        raise ValueError("size must contain two positive integers (height, width).")

    try:
        target_date = date.fromisoformat(target_date_str)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_date_str must be an ISO-8601 date (YYYY-MM-DD).") from exc

    start_date = target_date - timedelta(days=window_days)
    end_date = target_date + timedelta(days=window_days)
    start_datetime = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end_datetime = datetime.combine(end_date, time.max, tzinfo=timezone.utc)
    date_range = f"{start_datetime.isoformat()}/{end_datetime.isoformat()}"

    catalog = Client.open(STAC_API_URL)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 20}},
    )
    items = list(search.items())
    if not items:
        raise LookupError("No Sentinel-2 L2A scenes found for the requested bbox and date range.")

    item = min(items, key=lambda candidate: float(candidate.properties["eo:cloud_cover"]))
    planetary_computer.sign_inplace(item)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            key: executor.submit(
                _read_resized_band,
                item,
                key,
                bbox,
                size,
                cv2.INTER_NEAREST if key == "SCL" else cv2.INTER_LINEAR,
            )
            for key in ASSET_KEYS
        }
        bands = {key: futures[key].result() for key in ASSET_KEYS}

    valid_mask = np.isin(bands["SCL"], [4, 5, 6]).astype(np.uint8)

    def normalize(asset_key: str) -> np.ndarray:
        reflectance = np.clip(bands[asset_key].astype(np.float32) / 10000.0, 0.0, 1.0)
        return reflectance * valid_mask

    scene_datetime = item.datetime
    scene_date = scene_datetime.date().isoformat() if scene_datetime else item.properties["datetime"][:10]
    return {
        "red": normalize("B04"),
        "green": normalize("B03"),
        "blue": normalize("B02"),
        "nir": normalize("B08"),
        "valid_mask": valid_mask,
        "scene_date": scene_date,
    }
