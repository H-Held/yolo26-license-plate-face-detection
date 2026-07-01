"""Image I/O + geometry helpers.

Prefers OpenCV (fast, used on the server) but transparently falls back to
Pillow so the CPU pipeline is runnable on a laptop without cv2 installed.
All arrays are HxWx3 uint8 in **RGB** order regardless of backend.
"""
from __future__ import annotations
import numpy as np

try:
    import cv2  # type: ignore
    _HAVE_CV2 = True
except Exception:  # pragma: no cover - depends on host
    cv2 = None
    _HAVE_CV2 = False

from PIL import Image  # Pillow is a hard, lightweight dependency


def have_cv2() -> bool:
    return _HAVE_CV2


def imread(path) -> np.ndarray:
    """Read an image as HxWx3 uint8 RGB. Raises on failure (no silent None)."""
    if _HAVE_CV2:
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr is None:
            raise IOError(f"could not read image: {path}")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def imwrite(path, rgb: np.ndarray) -> None:
    """Write an HxWx3 uint8 RGB array to disk (format inferred from extension)."""
    if _HAVE_CV2:
        ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise IOError(f"could not write image: {path}")
        return
    Image.fromarray(rgb).save(str(path))


def resize(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    if _HAVE_CV2:
        interp = cv2.INTER_AREA if (w < rgb.shape[1]) else cv2.INTER_LINEAR
        return cv2.resize(rgb, (w, h), interpolation=interp)
    return np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))


def letterbox(rgb: np.ndarray, size: int, pad: int = 114):
    """Aspect-preserving resize into a size x size square, padded with `pad`.

    Returns (out_img, scale, pad_x, pad_y) so callers can map boxes:
        x_out = x_in * scale + pad_x ;  y_out = y_in * scale + pad_y
    """
    h, w = rgb.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = resize(rgb, nw, nh)
    out = np.full((size, size, 3), pad, dtype=np.uint8)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, scale, pad_x, pad_y
