"""Augmentation of the merged TRAIN split (chips are imgsz x imgsz).

TWO independent mechanisms (the spec asks for both):

  1. GLOBAL variety   -> geometric (rotate 90/180/270, h/v flip = upside down) and
                         photometric (over/under-expose, gamma, contrast, weird
                         colour palette). Adds robustness for every image.
  2. CLASS BOOST      -> oversample the WEAK class specifically (e.g. plates vs
                         faces). `class_boost` can be set manually per dataset, or
                         computed automatically from measured label counts
                         (see build_dataset._auto_class_boost) — augment.py doesn't
                         care which; it just receives the final per-class factor.

HARD CAP: a single source chip contributes AT MOST `max_versions` copies to the
output (default 5, incl. the original) — "maximal 5 Versionen eines Bildes". If the
requested variety + boost would need more, the cap wins; boost is prioritised over
extra plain variety since it's the one fixing class imbalance.

All geometric transforms are square rotations/flips, so bbox math is exact and is
verified by tests/run_smoke.py overlay checks.
"""
from __future__ import annotations
import random
from typing import List, Tuple
import numpy as np

Chip = Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]


# ---------------- geometric (image is square; coords normalised) ----------------
def _hflip(img, boxes):
    return img[:, ::-1].copy(), [(c, 1 - cx, cy, w, h) for (c, cx, cy, w, h) in boxes]


def _vflip(img, boxes):   # upside down
    return img[::-1, :].copy(), [(c, cx, 1 - cy, w, h) for (c, cx, cy, w, h) in boxes]


def _rot180(img, boxes):
    return np.rot90(img, 2).copy(), [(c, 1 - cx, 1 - cy, w, h) for (c, cx, cy, w, h) in boxes]


def _rot90(img, boxes):   # np.rot90 k=1 (CCW)
    return np.rot90(img, 1).copy(), [(c, cy, 1 - cx, h, w) for (c, cx, cy, w, h) in boxes]


def _rot270(img, boxes):  # np.rot90 k=3 (CW)
    return np.rot90(img, 3).copy(), [(c, 1 - cy, cx, h, w) for (c, cx, cy, w, h) in boxes]


GEOM_OPS = [_hflip, _vflip, _rot180, _rot90, _rot270]


# ---------------- photometric (boxes unchanged) ----------------
def _clip8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    f = img.astype(np.float32)
    # exposure (includes deliberate over/under exposure extremes)
    f *= rng.choice([0.45, 0.6, 0.8, 1.2, 1.5, 1.8])
    # gamma
    g = rng.choice([0.6, 0.8, 1.25, 1.6])
    f = np.power(np.clip(f / 255.0, 0, 1), g) * 255.0
    # contrast around mid-grey
    c = rng.choice([0.7, 0.85, 1.2, 1.4])
    f = (f - 128.0) * c + 128.0
    out = _clip8(f)
    # "weird colour palette": occasional channel permutation / inversion
    if rng.random() < 0.35:
        perm = rng.choice([(2, 1, 0), (1, 2, 0), (2, 0, 1)])
        out = out[:, :, perm]
    if rng.random() < 0.10:
        out = 255 - out
    return out


def _geom_variant(chip: Chip, rng: random.Random) -> Chip:
    op = rng.choice(GEOM_OPS)
    img, boxes = op(chip[0], chip[1])
    return (img, boxes)


def _combined_variant(chip: Chip, rng: random.Random) -> Chip:
    img, boxes = _geom_variant(chip, rng)
    if rng.random() < 0.8:
        img = photometric(img, rng)
    return (img, boxes)


def _has_class(chip: Chip, cls_id: int) -> bool:
    return any(b[0] == cls_id for b in chip[1])


DEFAULT_MAX_VERSIONS = 5


def augment_train(chips: List[Chip], aug_cfg: dict, name2id: dict,
                  rng: random.Random) -> List[Chip]:
    """Return originals + variants, capped at `max_versions` copies per source chip."""
    max_versions = max(1, int(aug_cfg.get("max_versions", DEFAULT_MAX_VERSIONS)))
    n_photo = int(aug_cfg.get("photometric_variants", 0) or 0)
    do_geom = bool(aug_cfg.get("geometric", False))
    boost = aug_cfg.get("class_boost") or {}
    for cls_name in boost:
        if cls_name not in name2id:
            raise ValueError(f"class_boost names unknown class '{cls_name}'")

    out: List[Chip] = []
    for chip in chips:
        chip_boost = 1
        for cls_name, factor in boost.items():
            if _has_class(chip, name2id[cls_name]):
                chip_boost = max(chip_boost, int(factor))
        plain_variety = 1 + n_photo + (1 if do_geom else 0)
        target = min(max_versions, max(chip_boost, plain_variety))

        versions = [chip]                 # the original always survives
        while len(versions) < target:
            versions.append(_combined_variant(chip, rng) if do_geom
                            else (photometric(chip[0], rng), chip[1]))
        out.extend(versions)
    return out
