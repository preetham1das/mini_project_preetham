"""GeoTIFF pairing, spatial windows, normalization, and PyTorch data loading."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import random

import numpy as np
import rasterio
from rasterio.windows import Window
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
