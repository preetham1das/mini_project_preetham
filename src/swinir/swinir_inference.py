"""
swinir_inference.py
-------------------
Thin wrapper around the official SwinIR network for 4x classical super-resolution.

This module is ISOLATED from the Temporal U-Net deforestation pipeline.
It is used ONLY for visualization: it enlarges Sentinel-2 RGB images 4x
for display.  The result is NEVER fed into the ML analysis models.

Checkpoint expected at:   <project_root>/models/swinir_x4.pth
Architecture used:        SwinIR-M classical SR x4
  img_size         = 64
  window_size      = 8
  embed_dim        = 180
  depths           = [6, 6, 6, 6, 6, 6]
  num_heads        = [6, 6, 6, 6, 6, 6]
  mlp_ratio        = 2.0
  upsampler        = 'pixelshuffle'
  upscale          = 4

This matches the official pretrained weight:
  001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure the swinir sub-package directory is on sys.path so that
# 'import network_swinir' works regardless of the caller's cwd.
_SWINIR_DIR = Path(__file__).resolve().parent
if str(_SWINIR_DIR) not in sys.path:
    sys.path.insert(0, str(_SWINIR_DIR))

from network_swinir import SwinIR  # noqa: E402  (after sys.path patch)

# ── constants ─────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "models" / "swinir_x4.pth"

# SwinIR-M classical SR x4 architecture matching the pretrained checkpoint
_MODEL_ARGS = dict(
    img_size=64,
    patch_size=1,
    in_chans=3,
    embed_dim=180,
    depths=[6, 6, 6, 6, 6, 6],
    num_heads=[6, 6, 6, 6, 6, 6],
    window_size=8,
    mlp_ratio=2.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
    upscale=4,
    img_range=1.0,
    upsampler='pixelshuffle',
    resi_connection='1conv',
)

_WINDOW_SIZE = 8   # must be consistent with model window size

# ── module-level model cache ──────────────────────────────────────────────────

_model = None
_model_error = None   # str if load failed


def _load_model(device: torch.device):
    """Load (once) and cache the SwinIR model.  Thread-safety is delegated to Flask."""
    global _model, _model_error

    if _model is not None:
        return _model, None
    if _model_error is not None:
        return None, _model_error

    if not CHECKPOINT_PATH.exists():
        _model_error = (
            f"SwinIR checkpoint not found at {CHECKPOINT_PATH}. "
            "Download 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth from "
            "https://github.com/JingyunLiang/SwinIR/releases and rename it to swinir_x4.pth."
        )
        return None, _model_error

    try:
        model = SwinIR(**_MODEL_ARGS)
        pretrained = torch.load(CHECKPOINT_PATH, map_location=device)
        # The official checkpoint may be wrapped under 'params' or 'params_ema'
        if "params_ema" in pretrained:
            state_dict = pretrained["params_ema"]
        elif "params" in pretrained:
            state_dict = pretrained["params"]
        else:
            state_dict = pretrained
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(device)
        _model = model
        return model, None
    except Exception as exc:
        _model_error = f"SwinIR model load error: {exc}"
        return None, _model_error


def _pad_to_multiple(img: np.ndarray, window: int) -> tuple:
    """Pad H and W to multiples of `window` (reflect-pad). Returns (padded, (h_orig, w_orig))."""
    h, w = img.shape[:2]
    pad_h = (window - h % window) % window
    pad_w = (window - w % window) % window
    if pad_h == 0 and pad_w == 0:
        return img, (h, w)
    padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    return padded, (h, w)


def superresolve(pil_img: Image.Image, device: torch.device = None) -> tuple:
    """
    Run SwinIR 4x super-resolution on *pil_img*.

    Parameters
    ----------
    pil_img : PIL.Image.Image
        Input RGB image (any size; will be up-scaled 4x).
    device  : torch.device, optional
        Defaults to CPU.

    Returns
    -------
    (result_pil, error_str)
        On success : (PIL.Image.Image of shape 4H x 4W, None)
        On failure : (None, "error message")
    """
    if device is None:
        device = torch.device("cpu")

    model, err = _load_model(device)
    if err:
        return None, err

    try:
        # Convert to numpy float32 [0..1], shape (H, W, 3)
        img_np = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0

        orig_h, orig_w = img_np.shape[:2]

        # Pad so H and W are multiples of the window size
        padded, (h_orig, w_orig) = _pad_to_multiple(img_np, _WINDOW_SIZE)

        # HWC → CHW → BCHW
        tensor = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).to(device)  # (1,3,H,W)

        with torch.no_grad():
            out = model(tensor)          # (1,3,4H,4W) float32 [0..1]

        out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()   # (4H, 4W, 3)
        out_np = np.clip(out_np, 0.0, 1.0)

        # Crop back to 4 * original dimensions (remove padding)
        out_np = out_np[: orig_h * 4, : orig_w * 4, :]

        # Convert to uint8 PIL image
        out_uint8 = (out_np * 255.0).round().astype(np.uint8)
        result = Image.fromarray(out_uint8, "RGB")
        return result, None

    except MemoryError:
        return None, "Out of memory during SwinIR inference. Try a smaller AOI."
    except Exception as exc:
        return None, f"SwinIR inference error: {exc}"


def is_checkpoint_available() -> bool:
    """Returns True if the SwinIR checkpoint file exists."""
    return CHECKPOINT_PATH.exists()
