"""GeoTIFF pairing, spatial windows, normalization, and PyTorch data loading."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import random

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import ColorInterp
from rasterio.warp import transform_bounds, reproject, Resampling
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PatchRecord:
    tile_id: str
    split: str
    s1_path: str
    s2_path: str
    mask_path: str
    row: int
    col: int
    patch_size: int
    valid_fraction: float
    positive_fraction: float

    def to_dict(self):
        return asdict(self)


def canonical_tile_id(path: str | Path) -> str:
    """Get matching grid id from names such as RASTER_12.tif or MASK_RASTER_12.tif."""
    import re
    stem = Path(path).stem
    match = re.search(r"raster[_-]?(\d+)", stem, flags=re.IGNORECASE)
    return f"RASTER_{int(match.group(1))}" if match else stem.casefold()


def identify_modality(path: str | Path) -> tuple[str | None, str]:
    parts = [part.casefold().replace("-", "_").replace(" ", "_") for part in Path(path).parts]
    joined = "_".join(parts)
    if any("mask" in part or "training_label" in part or "prodes" in part for part in parts):
        return "mask", "shared"
    if "sentinel1" in joined or "sentinel_1" in joined or "sar" in joined:
        modality = "s1"
    elif "sentinel2" in joined or "sentinel_2" in joined or "optical" in joined:
        modality = "s2"
    else:
        return None, "unknown"
    if "cloud_free" in joined or "cloudfree" in joined:
        domain = "cloud_free"
    elif "cloudy" in joined or "cloud" in joined:
        domain = "cloudy"
    else:
        domain = "unspecified"
    return modality, domain


def raster_signature(path: str | Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "path": str(path), "width": src.width, "height": src.height,
            "count": src.count, "dtype": ",".join(src.dtypes),
            "crs": src.crs.to_string() if src.crs else None,
            "transform": tuple(src.transform)[:6], "bounds": tuple(src.bounds),
            "nodata": src.nodata, "descriptions": src.descriptions,
            "block_shapes": src.block_shapes,
        }


def validate_pair_alignment(s1_path: str | Path, s2_path: str | Path,
                            mask_path: str | Path) -> tuple[bool, str]:
    signatures = [raster_signature(p) for p in (s1_path, s2_path, mask_path)]
    base = signatures[0]
    for other in signatures[1:]:
        if (other["width"], other["height"]) != (base["width"], base["height"]):
            return False, "different raster dimensions"
        if other["crs"] != base["crs"]:
            return False, "different or missing CRS"
        if not np.allclose(other["transform"], base["transform"], rtol=0, atol=1e-8):
            return False, "different affine transform/pixel alignment"
    return True, "same dimensions, CRS, and affine transform"


def resolve_sentinel1_bands(src: rasterio.DatasetReader) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Validates and extracts Sentinel-1 SAR bands (VV, VH) from an opened rasterio dataset.
    
    The Kaggle V2 Late-Fusion SegNet model requires exactly 2 SAR input channels:
      Channel 1: VV polarization (vertical transmit, vertical receive)
      Channel 2: VH polarization (vertical transmit, horizontal receive)
    
    Returns:
        (s1_data, valid_mask, info_message)
        where s1_data is float32 with shape (2, height, width) and valid_mask is boolean (height, width).
    Raises:
        ValueError if the input is incompatible or cannot be reliably resolved.
    """
    descriptions = [str(d or '').strip().lower() for d in (src.descriptions or ())]
    colorinterp = list(src.colorinterp or ())
    dtypes = list(src.dtypes)

    # Check for obvious RGB optical image masquerading as Sentinel-1
    is_rgb = False
    if len(colorinterp) >= 3:
        if (colorinterp[0] == ColorInterp.red and 
            colorinterp[1] == ColorInterp.green and 
            colorinterp[2] == ColorInterp.blue):
            is_rgb = True
    if not is_rgb and src.count == 3 and dtypes[0] == 'uint8':
        joined_desc = ' '.join(descriptions)
        if any(w in joined_desc for w in ('red', 'green', 'blue', 'tile', 'amazon', 'rgb')):
            is_rgb = True

    if is_rgb:
        raise ValueError(
            f"The uploaded Sentinel-1 raster appears to be a 3-band RGB optical image "
            f"(color interpretation: Red/Green/Blue, dtype: {dtypes[0]}), not Sentinel-1 SAR data. "
            f"The Kaggle Amazon V2 model requires Sentinel-1 C-band SAR with 2 polarizations: "
            f"Band 1 = VV polarization, Band 2 = VH polarization (normalized float backscatter, ~[-2, +3]). "
            f"Please upload a valid Sentinel-1 SAR GeoTIFF."
        )

    if src.count < 2:
        raise ValueError(
            f"Sentinel-1 raster has only {src.count} band. "
            f"The Kaggle V2 model requires 2 SAR channels: Band 1 = VV polarization, Band 2 = VH polarization."
        )

    # Check for metadata identification of VV and VH
    vv_idx = None
    vh_idx = None
    alpha_idx = None

    for i, desc in enumerate(descriptions):
        if 'vv' in desc and vv_idx is None:
            vv_idx = i
        elif 'vh' in desc and vh_idx is None:
            vh_idx = i
        elif any(k in desc for k in ('alpha', 'mask', 'valid')) and alpha_idx is None:
            alpha_idx = i

    for i, ci in enumerate(colorinterp):
        if ci == ColorInterp.alpha and alpha_idx is None:
            alpha_idx = i

    if src.count == 2:
        # Standard 2-band S1 product
        if vv_idx is not None and vh_idx is not None and (vv_idx, vh_idx) == (1, 0):
            # Inverted order in metadata: VH is band 1, VV is band 2
            raw = src.read([2, 1]).astype(np.float32)  # Read VV first, then VH
            info = "Read 2-band Sentinel-1 raster; reordered to (VV, VH) based on band descriptions."
        else:
            raw = src.read([1, 2]).astype(np.float32)
            info = "Read 2-band Sentinel-1 raster (Band 1: VV, Band 2: VH)."

        valid = np.all(src.read_masks([1, 2]) > 0, axis=0) & np.all(np.isfinite(raw), axis=0)
        return raw, valid, info

    if src.count == 3:
        # 3-band raster: could be (VV, VH, angle), (VV, VH, ratio), or (VV, VH, alpha)
        if vv_idx is not None and vh_idx is not None:
            selected = [vv_idx + 1, vh_idx + 1]
            raw = src.read(selected).astype(np.float32)
            excluded_idx = [i for i in range(3) if i not in (vv_idx, vh_idx)][0]
            excluded_desc = src.descriptions[excluded_idx] if src.descriptions and len(src.descriptions) > excluded_idx else f"Band {excluded_idx+1}"
            info = f"Extracted verified VV and VH channels (bands {vv_idx+1}, {vh_idx+1}) from 3-band Sentinel-1 raster; excluded auxiliary band '{excluded_desc}'."
            valid = np.all(src.read_masks(selected) > 0, axis=0) & np.all(np.isfinite(raw), axis=0)
            if alpha_idx is not None:
                alpha_data = src.read(alpha_idx + 1)
                valid &= (alpha_data > 0)
            return raw, valid, info

        if alpha_idx == 2:
            raw = src.read([1, 2]).astype(np.float32)
            alpha_data = src.read(3)
            valid = (alpha_data > 0) & np.all(src.read_masks([1, 2]) > 0, axis=0) & np.all(np.isfinite(raw), axis=0)
            info = "Extracted Bands 1 and 2 as (VV, VH) from 3-band Sentinel-1 raster; Band 3 used as alpha mask."
            return raw, valid, info

        desc_str = f"descriptions: {src.descriptions}" if src.descriptions else "no band descriptions"
        raise ValueError(
            f"Sentinel-1 raster contains 3 bands ({desc_str}, colorinterp: {[ci.name for ci in colorinterp]}), "
            f"but the required VV and VH polarizations could not be unambiguously verified from metadata. "
            f"The Kaggle V2 model requires exactly 2 SAR channels: Band 1 = VV polarization, Band 2 = VH polarization. "
            f"Please provide a 2-band (VV, VH) GeoTIFF or ensure bands are tagged with 'VV' and 'VH'."
        )

    if vv_idx is not None and vh_idx is not None:
        selected = [vv_idx + 1, vh_idx + 1]
        raw = src.read(selected).astype(np.float32)
        info = f"Extracted verified VV (band {vv_idx+1}) and VH (band {vh_idx+1}) from {src.count}-band Sentinel-1 raster."
        valid = np.all(src.read_masks(selected) > 0, axis=0) & np.all(np.isfinite(raw), axis=0)
        return raw, valid, info

    raise ValueError(
        f"Sentinel-1 raster contains {src.count} bands. "
        f"Expected exactly 2 bands (Band 1: VV, Band 2: VH) or tagged 'VV' and 'VH' channels."
    )


def resolve_sentinel2_bands(src: rasterio.DatasetReader) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Validates and extracts Sentinel-2 optical bands (B2, B3, B4, B8) from an opened rasterio dataset.
    
    The Kaggle V2 Late-Fusion SegNet model requires exactly 4 optical input channels in order:
      Channel 1: Blue (Band 2, ~490 nm)
      Channel 2: Green (Band 3, ~560 nm)
      Channel 3: Red (Band 4, ~665 nm)
      Channel 4: Near-Infrared (Band 8, ~842 nm)
    
    Returns:
        (s2_data, valid_mask, info_message)
        where s2_data is float32 with shape (4, height, width) and valid_mask is boolean (height, width).
    Raises:
        ValueError if the input is incompatible or cannot be reliably resolved.
    """
    descriptions = [str(d or '').strip().lower() for d in (src.descriptions or ())]
    colorinterp = list(src.colorinterp or ())
    dtypes = list(src.dtypes)

    if src.count == 3:
        raise ValueError(
            f"Sentinel-2 raster contains only 3 bands (RGB). "
            f"The Kaggle V2 model requires 4 spectral bands: Blue (B2), Green (B3), Red (B4), and Near-Infrared (B8). "
            f"The Near-Infrared band is strictly required because vegetation has high NIR reflectance; "
            f"deforestation cannot be detected accurately without the NIR band."
        )

    if src.count == 4:
        if len(colorinterp) >= 4 and colorinterp[3] == ColorInterp.alpha:
            raise ValueError(
                f"The uploaded Sentinel-2 raster is an RGBA image (Red, Green, Blue, Alpha), "
                f"where Band 4 is an alpha transparency mask, NOT Sentinel-2 Near-Infrared (Band 8). "
                f"The Kaggle V2 model requires 4 spectral channels: B2 (Blue), B3 (Green), B4 (Red), and B8 (NIR). "
                f"Please provide a multispectral GeoTIFF with NIR in Band 4."
            )
        if dtypes[0] == 'uint8' and len(descriptions) >= 4 and any(k in descriptions[3] for k in ('alpha', 'mask', 'transparency')):
            raise ValueError(
                f"The uploaded Sentinel-2 raster appears to be an RGBA image with an alpha mask in Band 4 ({src.descriptions[3]}). "
                f"Band 4 must be Sentinel-2 Near-Infrared (B8), not an alpha channel."
            )

        b2_idx, b3_idx, b4_idx, b8_idx = None, None, None, None
        for i, desc in enumerate(descriptions):
            if any(k == desc or k in desc.split() or f'_{k}' in desc or f'-{k}' in desc for k in ('b2', 'blue', 'b02')):
                b2_idx = i
            elif any(k == desc or k in desc.split() or f'_{k}' in desc or f'-{k}' in desc for k in ('b3', 'green', 'b03')):
                b3_idx = i
            elif any(k == desc or k in desc.split() or f'_{k}' in desc or f'-{k}' in desc for k in ('b4', 'red', 'b04')):
                b4_idx = i
            elif any(k == desc or k in desc.split() or f'_{k}' in desc or f'-{k}' in desc for k in ('b8', 'nir', 'b08')) and 'b8a' not in desc:
                b8_idx = i

        if b2_idx is not None and b3_idx is not None and b4_idx is not None and b8_idx is not None:
            order = [b2_idx + 1, b3_idx + 1, b4_idx + 1, b8_idx + 1]
            raw = src.read(order).astype(np.float32)
            valid = np.all(src.read_masks(order) > 0, axis=0) & np.all(np.isfinite(raw), axis=0)
            info = f"Extracted verified Sentinel-2 bands [B2, B3, B4, B8] in order: {order}."
            return raw, valid, info

        raw = src.read([1, 2, 3, 4]).astype(np.float32)
        valid = np.all(src.read_masks([1, 2, 3, 4]) > 0, axis=0) & np.all(np.isfinite(raw), axis=0)
        info = "Read 4-band Sentinel-2 raster (Band 1: B2 Blue, Band 2: B3 Green, Band 3: B4 Red, Band 4: B8 NIR)."
        return raw, valid, info

    if src.count == 5:
        alpha_idx = None
        for i in range(5):
            if colorinterp and len(colorinterp) > i and colorinterp[i] == ColorInterp.alpha:
                alpha_idx = i
            elif len(descriptions) > i and any(k in descriptions[i] for k in ('alpha', 'mask', 'scl', 'aod')):
                alpha_idx = i

        if alpha_idx == 4:
            raw = src.read([1, 2, 3, 4]).astype(np.float32)
            alpha_data = src.read(5)
            valid = (alpha_data > 0) & np.all(src.read_masks([1, 2, 3, 4]) > 0, axis=0) & np.all(np.isfinite(raw), axis=0)
            info = "Extracted Bands 1-4 as [B2, B3, B4, B8] from 5-band Sentinel-2 raster; Band 5 used as mask."
            return raw, valid, info

    if src.count >= 12:
        b2_idx, b3_idx, b4_idx, b8_idx = None, None, None, None
        for i, desc in enumerate(descriptions):
            if any(k == desc or k in desc.split() or f'_{k}' in desc for k in ('b2', 'blue', 'b02')):
                b2_idx = i
            elif any(k == desc or k in desc.split() or f'_{k}' in desc for k in ('b3', 'green', 'b03')):
                b3_idx = i
            elif any(k == desc or k in desc.split() or f'_{k}' in desc for k in ('b4', 'red', 'b04')):
                b4_idx = i
            elif any(k == desc or k in desc.split() or f'_{k}' in desc for k in ('b8', 'nir', 'b08')) and 'b8a' not in desc:
                b8_idx = i

        if b2_idx is not None and b3_idx is not None and b4_idx is not None and b8_idx is not None:
            order = [b2_idx + 1, b3_idx + 1, b4_idx + 1, b8_idx + 1]
            raw = src.read(order).astype(np.float32)
            valid = np.all(src.read_masks(order) > 0, axis=0) & np.all(np.isfinite(raw), axis=0)
            info = f"Extracted [B2, B3, B4, B8] from {src.count}-band full Sentinel-2 product (band indices: {order})."
            return raw, valid, info

    raise ValueError(
        f"Sentinel-2 raster contains {src.count} bands (descriptions: {src.descriptions}). "
        f"The Kaggle V2 model requires 4 bands: B2 (Blue), B3 (Green), B4 (Red), and B8 (NIR). "
        f"Please provide a 4-band GeoTIFF with these spectral bands."
    )


def align_sentinel_pair(
    s1_src: rasterio.DatasetReader,
    s2_src: rasterio.DatasetReader,
    s1_data: np.ndarray,
    s2_data: np.ndarray,
    s1_valid: np.ndarray,
    s2_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, str]:
    """
    Validates geospatial coverage between Sentinel-1 and Sentinel-2 and
    ensures both rasters share the same pixel grid (s1_src grid).
    
    Returns:
        (s1_aligned, s2_aligned, combined_valid, was_resampled, alignment_note)
    Raises:
        ValueError if CRSs are missing or spatial extents do not intersect.
    """
    if s1_src.crs is None or s2_src.crs is None:
        raise ValueError("Both Sentinel-1 and Sentinel-2 rasters must have a defined Coordinate Reference System (CRS).")

    b1 = transform_bounds(s1_src.crs, "EPSG:4326", *s1_src.bounds, densify_pts=21)
    b2 = transform_bounds(s2_src.crs, "EPSG:4326", *s2_src.bounds, densify_pts=21)

    int_west = max(b1[0], b2[0])
    int_south = max(b1[1], b2[1])
    int_east = min(b1[2], b2[2])
    int_north = min(b1[3], b2[3])

    if int_west >= int_east or int_south >= int_north:
        raise ValueError(
            f"Sentinel-1 and Sentinel-2 rasters do not cover the same geographic area (spatial extents do not intersect). "
            f"Sentinel-1 extent (WGS84): {[round(x, 4) for x in b1]}, "
            f"Sentinel-2 extent (WGS84): {[round(x, 4) for x in b2]}. "
            f"Please upload co-registered rasters covering the same area."
        )

    s1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    s2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
    int_area = (int_east - int_west) * (int_north - int_south)
    overlap_ratio = int_area / max(min(s1_area, s2_area), 1e-12)

    if overlap_ratio < 0.20:
        raise ValueError(
            f"Sentinel-1 and Sentinel-2 rasters have insufficient geographic overlap ({overlap_ratio * 100:.1f}%). "
            f"Co-registered rasters covering the same area are required."
        )

    same_crs = (s1_src.crs == s2_src.crs)
    same_dims = (s1_src.width == s2_src.width and s1_src.height == s2_src.height)
    same_transform = np.allclose(s1_src.transform, s2_src.transform, atol=1e-3, rtol=1e-3)

    if same_crs and same_dims and same_transform:
        combined_valid = s1_valid & s2_valid & np.all(np.isfinite(s1_data), axis=0) & np.all(np.isfinite(s2_data), axis=0)
        return s1_data, s2_data, combined_valid, False, "Identical CRS, transform, and dimensions; no resampling needed."

    s2_resampled = np.zeros((s2_data.shape[0], s1_src.height, s1_src.width), dtype=np.float32)
    reproject(
        source=s2_data,
        destination=s2_resampled,
        src_transform=s2_src.transform,
        src_crs=s2_src.crs,
        dst_transform=s1_src.transform,
        dst_crs=s1_src.crs,
        resampling=Resampling.bilinear,
    )

    s2_valid_resampled = np.zeros((s1_src.height, s1_src.width), dtype=np.float32)
    reproject(
        source=s2_valid.astype(np.float32),
        destination=s2_valid_resampled,
        src_transform=s2_src.transform,
        src_crs=s2_src.crs,
        dst_transform=s1_src.transform,
        dst_crs=s1_src.crs,
        resampling=Resampling.nearest,
    )

    combined_valid = s1_valid & (s2_valid_resampled > 0.5) & np.all(np.isfinite(s1_data), axis=0) & np.all(np.isfinite(s2_resampled), axis=0)
    note = f"Sentinel-2 bilinearly resampled from ({s2_src.width}x{s2_src.height}, {s2_src.crs}) to match Sentinel-1 grid ({s1_src.width}x{s1_src.height}, {s1_src.crs})."
    return s1_data, s2_resampled, combined_valid, True, note


def assign_spatial_splits(samples: list[dict], seed: int = 42,
                          ratios: dict[str, float] | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """Assign whole adjacent 2x2 raster blocks to splits using georeferenced centers.

    This intentionally groups nearby grid tiles before splitting. With only four
    spatial blocks, the realized ratio may be 50/25/25 rather than 70/15/15.
    Returns tile_id -> split and tile_id -> geographic block id.
    """
    if ratios is None:
        ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    if len(samples) < 3:
        raise ValueError("At least three aligned cloud-free tiles are needed for spatial train/val/test splits")
    crss = {sample["crs"] for sample in samples}
    if len(crss) != 1 or None in crss:
        raise ValueError("Spatial splitting requires one shared, defined CRS")
    widths = [sample["bounds"][2] - sample["bounds"][0] for sample in samples]
    heights = [sample["bounds"][3] - sample["bounds"][1] for sample in samples]
    x_step = float(np.median(widths)) * 2.0
    y_step = float(np.median(heights)) * 2.0
    min_x = min(sample["bounds"][0] for sample in samples)
    min_y = min(sample["bounds"][1] for sample in samples)
    max_y = max(sample["bounds"][3] for sample in samples)
    blocks = {}
    for sample in samples:
        xmin, ymin, xmax, ymax = sample["bounds"]
        center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
        col = int(np.floor((center_x - min_x) / max(x_step, 1e-9)))
        row = int(np.floor((max_y - center_y) / max(y_step, 1e-9)))
        block_id = f"block_r{row}_c{col}"
        blocks[sample["tile_id"]] = block_id
    group_to_tiles: dict[str, list[str]] = {}
    for tile, block in blocks.items():
        group_to_tiles.setdefault(block, []).append(tile)
    if len(group_to_tiles) < 3:
        raise ValueError(f"Only {len(group_to_tiles)} spatial blocks found; cannot form three independent splits")
    rng = random.Random(seed)
    groups = list(group_to_tiles)
    rng.shuffle(groups)
    groups.sort(key=lambda group: len(group_to_tiles[group]), reverse=True)
    targets = {name: ratios[name] * len(samples) for name in ("train", "val", "test")}
    counts = {name: 0 for name in targets}
    group_split = {}
    for group in groups:
        destination = min(targets, key=lambda name: counts[name] / max(targets[name], 1e-9))
        group_split[group] = destination
        counts[destination] += len(group_to_tiles[group])
    tile_splits = {tile: group_split[block] for tile, block in blocks.items()}
    return tile_splits, blocks


def _read_valid(path: str | Path, window: Window) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        values = src.read(window=window).astype(np.float32)
        band_valid = src.read_masks(window=window) > 0
        valid = np.all(band_valid, axis=0) & np.all(np.isfinite(values), axis=0)
        return values, valid


def _read_label(path: str | Path, window: Window, positive_value: int,
                background_value: int, ignore_values: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        raw = src.read(1, window=window)
        valid = np.isin(raw, [positive_value, background_value])
        if src.nodata is not None:
            valid &= raw != src.nodata
        for value in ignore_values:
            valid &= raw != value
        target = raw == positive_value
        return target.astype(np.float32), valid


def build_patch_records(samples: list[dict], tile_splits: dict[str, str],
                        patch_size: int = 256, stride: int = 256,
                        positive_value: int = 1, background_value: int = 0,
                        ignore_values: Iterable[int] = (),
                        min_valid_fraction: float = 0.90) -> list[PatchRecord]:
    """Index aligned non-overlapping windows; never split one source tile across sets."""
    records = []
    for sample in samples:
        tile = sample["tile_id"]
        split = tile_splits[tile]
        with rasterio.open(sample["mask_path"]) as mask_src:
            height, width = mask_src.height, mask_src.width
        for row in range(0, height - patch_size + 1, stride):
            for col in range(0, width - patch_size + 1, stride):
                window = Window(col, row, patch_size, patch_size)
                s1, valid1 = _read_valid(sample["s1_path"], window)
                s2, valid2 = _read_valid(sample["s2_path"], window)
                target, valid_mask = _read_label(sample["mask_path"], window,
                                                 positive_value, background_value, ignore_values)
                valid = valid1 & valid2 & valid_mask
                fraction = float(valid.mean())
                if fraction < min_valid_fraction:
                    continue
                pos_fraction = float(target[valid].mean()) if valid.any() else 0.0
                records.append(PatchRecord(tile, split, sample["s1_path"], sample["s2_path"],
                                           sample["mask_path"], row, col, patch_size,
                                           fraction, pos_fraction))
    return records


def fit_robust_ranges(records: list[PatchRecord], percentiles=(1.0, 99.0),
                      max_windows: int = 800, points_per_window: int = 512,
                      seed: int = 42, positive_value: int = 1,
                      background_value: int = 0,
                      ignore_values: Iterable[int] = ()) -> dict[str, dict[str, list[float]]]:
    """Fit per-band percentile ranges on TRAIN windows only."""
    train = [record for record in records if record.split == "train"]
    if not train:
        raise ValueError("No training windows available to fit normalization")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(train), size=min(max_windows, len(train)), replace=False)
    buckets = {"s1": None, "s2": None}
    for index in chosen:
        record = train[int(index)]
        window = Window(record.col, record.row, record.patch_size, record.patch_size)
        arrays = {}
        valid = np.ones((record.patch_size, record.patch_size), dtype=bool)
        for modality, path in (("s1", record.s1_path), ("s2", record.s2_path)):
            arrays[modality], band_valid = _read_valid(path, window)
            valid &= band_valid
        _, mask_valid = _read_label(record.mask_path, window, positive_value,
                                    background_value, ignore_values)
        valid &= mask_valid
        for modality, array in arrays.items():
            if buckets[modality] is None:
                buckets[modality] = [[] for _ in range(array.shape[0])]
            coordinates = np.flatnonzero(valid)
            if coordinates.size == 0:
                continue
            if coordinates.size > points_per_window:
                coordinates = rng.choice(coordinates, points_per_window, replace=False)
            flat = array.reshape(array.shape[0], -1)
            for band in range(array.shape[0]):
                buckets[modality][band].append(flat[band, coordinates])
    result = {}
    for modality, bands in buckets.items():
        if bands is None:
            raise ValueError(f"No valid {modality} values found in training windows")
        lows, highs = [], []
        for chunks in bands:
            values = np.concatenate(chunks)
            lo, hi = np.percentile(values, percentiles)
            if not np.isfinite(lo + hi):
                raise ValueError(f"Non-finite normalization range for {modality}")
            if hi <= lo:
                hi = lo + 1.0
            lows.append(float(lo)); highs.append(float(hi))
        result[modality] = {"low": lows, "high": highs,
                            "percentiles": [float(percentiles[0]), float(percentiles[1])]}
    return result


class RasterPatchDataset(Dataset):
    def __init__(self, records: list[PatchRecord], ranges: dict,
                 augment: bool = False, positive_value: int = 1,
                 background_value: int = 0, ignore_values: Iterable[int] = ()):
        self.records = list(records)
        self.ranges = ranges
        self.augment = augment
        self.positive_value = positive_value
        self.background_value = background_value
        self.ignore_values = tuple(ignore_values)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        window = Window(record.col, record.row, record.patch_size, record.patch_size)
        s1, valid1 = _read_valid(record.s1_path, window)
        s2, valid2 = _read_valid(record.s2_path, window)
        target, valid_mask = _read_label(record.mask_path, window, self.positive_value,
                                         self.background_value, self.ignore_values)
        valid = valid1 & valid2 & valid_mask
        for modality, array in (("s1", s1), ("s2", s2)):
            lo = np.asarray(self.ranges[modality]["low"], dtype=np.float32)[:, None, None]
            hi = np.asarray(self.ranges[modality]["high"], dtype=np.float32)[:, None, None]
            array = np.clip((array - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
            array[:, ~valid] = 0.0
            if modality == "s1":
                s1 = array
            else:
                s2 = array
        if self.augment:
            if random.random() < 0.5:
                s1 = np.flip(s1, axis=2); s2 = np.flip(s2, axis=2)
                target = np.flip(target, axis=1); valid = np.flip(valid, axis=1)
            if random.random() < 0.5:
                s1 = np.flip(s1, axis=1); s2 = np.flip(s2, axis=1)
                target = np.flip(target, axis=0); valid = np.flip(valid, axis=0)
            turns = random.randrange(4)
            if turns:
                s1 = np.rot90(s1, turns, axes=(1, 2)); s2 = np.rot90(s2, turns, axes=(1, 2))
                target = np.rot90(target, turns, axes=(0, 1)); valid = np.rot90(valid, turns, axes=(0, 1))
        return (
            torch.from_numpy(np.ascontiguousarray(s1)).float(),
            torch.from_numpy(np.ascontiguousarray(s2)).float(),
            torch.from_numpy(np.ascontiguousarray(target[None])).float(),
            torch.from_numpy(np.ascontiguousarray(valid[None])).float(),
        )
