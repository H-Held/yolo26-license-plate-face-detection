"""Tiling — turns one source Sample into training chips (all imgsz x imgsz).

FOUR regimes (each independently toggleable per dataset in `tiling:`):

  (a) native_tiles     : slide an imgsz x imgsz window over the image and keep it
                         at native resolution. Best for small objects (full detail).
  (b) scaled_tiles [k] : slide a (k*imgsz) SQUARE window and downscale it to imgsz.
                         Gives the model lower-res context / larger objects.
  (c) fixed_windows [[w,h],...] : slide a FIXED-PIXEL, non-square window (e.g. a
                         1920x1080 "screen-shaped" crop) over the image; each crop is
                         padded on its short side ("seitlich erweitert") to a square
                         of side max(w,h), THEN downscaled to imgsz. Use this to teach
                         the model on realistic camera/screen aspect ratios rather
                         than only square crops.
  (d) whole_if_span_tiles N : if ANY single box intersects more than N native
                         (imgsz-grid) tiles, the object is too big to survive native
                         tiling intact, so ALSO emit the whole image letterboxed to
                         imgsz (this pads-to-square + downscales the ENTIRE image, the
                         same "extend to square, then compress" idea applied globally).
                         Only that whole-image copy is added (per the spec:
                         "only when a label would span more than N small tiles").
                         `whole_image: true` forces it unconditionally (whole-image
                         datasets, e.g. an extra face set that isn't tiled at all).

Boxes are clipped to each window; a box kept only if enough of it remains
(`min_visibility`), so the model never learns on a thin sliver of an object.

Only the TRAIN split is tiled. val/test use a single whole->imgsz letterbox
(see build_dataset.write_eval_chip) to match normal inference.
"""
from __future__ import annotations
import random
from typing import List, Tuple
import numpy as np

from .adapters.base import Sample, Box
from . import imageio_util as io

Chip = Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]  # img, yolo boxes


def _yolo(cls: int, x1, y1, x2, y2, W, H):
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    return (cls, cx, cy, (x2 - x1) / W, (y2 - y1) / H)


def _clip_boxes(boxes, wx1, wy1, wx2, wy2, min_vis):
    """Clip boxes to a window; keep those retaining >= min_vis of their area.
    Returns list of boxes in WINDOW-local coordinates."""
    out = []
    for b in boxes:
        ix1, iy1 = max(b.x1, wx1), max(b.y1, wy1)
        ix2, iy2 = min(b.x2, wx2), min(b.y2, wy2)
        iw, ih = ix2 - ix1, iy2 - iy1
        if iw <= 1 or ih <= 1:
            continue
        orig = max(b.w * b.h, 1e-6)
        if (iw * ih) / orig < min_vis:
            continue
        out.append(Box(b.cls, ix1 - wx1, iy1 - wy1, ix2 - wx1, iy2 - wy1))
    return out


def _pad_to_square_and_map(crop: np.ndarray, boxes_local, pad_value: int):
    """Pad a rectangular crop on its short side to a side x side square
    ("seitlich erweitert"), centered. Returns (square_img, mapped_boxes) where
    mapped_boxes are Box objects shifted into the padded square's coordinates."""
    h, w = crop.shape[:2]
    side = max(w, h)
    square = np.full((side, side, 3), pad_value, dtype=np.uint8)
    offx, offy = (side - w) // 2, (side - h) // 2
    square[offy:offy + h, offx:offx + w] = crop
    mapped = [Box(b.cls, b.x1 + offx, b.y1 + offy, b.x2 + offx, b.y2 + offy) for b in boxes_local]
    return square, mapped


def _windows(extent: int, win: int, stride: int) -> List[int]:
    """Start offsets covering [0, extent) with a window of `win`, last flush to edge."""
    if win >= extent:
        return [0]
    starts = list(range(0, extent - win + 1, stride))
    if starts[-1] != extent - win:
        starts.append(extent - win)
    return starts


def _tiles_spanned(b: Box, imgsz: int) -> int:
    """How many cells of the non-overlapping imgsz grid this box intersects."""
    cols = int(b.x2 // imgsz) - int(b.x1 // imgsz) + 1
    rows = int(b.y2 // imgsz) - int(b.y1 // imgsz) + 1
    return max(cols, 1) * max(rows, 1)


def tile_sample(sample: Sample, img: np.ndarray, tcfg: dict, imgsz: int,
                pad: int = 114, rng: random.Random = None) -> List[Chip]:
    rng = rng or random.Random(0)
    H, W = img.shape[:2]
    min_vis = float(tcfg.get("min_visibility", 0.2))
    stride = max(1, int(round(float(tcfg.get("stride_frac", 0.8)) * imgsz)))
    max_pos = int(tcfg.get("max_pos_tiles", 8))
    hardneg_frac = float(tcfg.get("hardneg_frac", 0.05))

    positives: List[Chip] = []
    negatives: List[Chip] = []

    def add_window(wx1, wy1, wsize, out_size):
        wx2, wy2 = wx1 + wsize, wy1 + wsize
        crop = img[wy1:wy2, wx1:wx2]
        if out_size != wsize:
            crop = io.resize(crop, out_size, out_size)
            sc = out_size / wsize
        else:
            sc = 1.0
        local = _clip_boxes(sample.boxes, wx1, wy1, wx2, wy2, min_vis)
        yb = [_yolo(b.cls, b.x1 * sc, b.y1 * sc, b.x2 * sc, b.y2 * sc,
                    out_size, out_size) for b in local]
        (positives if yb else negatives).append((crop, yb))

    # (a) native imgsz tiles
    if tcfg.get("native_tiles", True) and W >= imgsz and H >= imgsz:
        for wy in _windows(H, imgsz, stride):
            for wx in _windows(W, imgsz, stride):
                add_window(wx, wy, imgsz, imgsz)

    # (b) scaled k*imgsz tiles -> imgsz
    for k in (tcfg.get("scaled_tiles") or []):
        k = int(k)
        win = k * imgsz
        if win > W or win > H:
            continue                      # window bigger than image -> that's the whole image
        for wy in _windows(H, win, stride):
            for wx in _windows(W, win, stride):
                add_window(wx, wy, win, imgsz)

    # (c) fixed-pixel non-square windows (e.g. 1920x1080 / 1080x1920 "screen" crops):
    # pad the short side to a square, THEN downscale to imgsz.
    for fw_fh in (tcfg.get("fixed_windows") or []):
        fw, fh = int(fw_fh[0]), int(fw_fh[1])
        if fw > W or fh > H:
            continue                      # window bigger than the image on some axis
        stride_x = max(1, int(round(float(tcfg.get("stride_frac", 0.8)) * fw)))
        stride_y = max(1, int(round(float(tcfg.get("stride_frac", 0.8)) * fh)))
        for wy in _windows(H, fh, stride_y):
            for wx in _windows(W, fw, stride_x):
                wx2, wy2 = wx + fw, wy + fh
                crop = img[wy:wy2, wx:wx2]
                local = _clip_boxes(sample.boxes, wx, wy, wx2, wy2, min_vis)
                square, mapped = _pad_to_square_and_map(crop, local, pad)
                scale = imgsz / square.shape[0]
                resized = io.resize(square, imgsz, imgsz)
                yb = [_yolo(b.cls, b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale,
                           imgsz, imgsz) for b in mapped]
                (positives if yb else negatives).append((resized, yb))

    # cap positives (disk/RAM guard), keep the rest deterministic
    if len(positives) > max_pos:
        rng.shuffle(positives)
        positives = positives[:max_pos]

    # (c) whole-image letterbox. Emitted when ANY of:
    #   - whole_image: true            -> ALWAYS (the "whole-image dataset" case, e.g. faces)
    #   - a box spans > whole_if_span_tiles native tiles (too big to tile intact)
    #   - the image is smaller than one tile (native/scaled produced nothing)
    span_thresh = int(tcfg.get("whole_if_span_tiles", 0) or 0)
    force_whole = bool(tcfg.get("whole_image", False))
    small_image = (W < imgsz or H < imgsz)
    big_box = span_thresh and any(_tiles_spanned(b, imgsz) > span_thresh
                                  for b in sample.boxes)
    if sample.boxes and (force_whole or big_box or small_image):
        lb, scale, px, py = io.letterbox(img, imgsz, pad)
        yb = []
        for b in sample.boxes:
            yb.append(_yolo(b.cls, b.x1 * scale + px, b.y1 * scale + py,
                            b.x2 * scale + px, b.y2 * scale + py, imgsz, imgsz))
        positives.append((lb, yb))

    # hard negatives, capped as a small fraction of positives
    keep_neg = int(round(hardneg_frac * max(len(positives), 1)))
    if negatives and keep_neg:
        rng.shuffle(negatives)
        positives.extend(negatives[:keep_neg])

    return positives
