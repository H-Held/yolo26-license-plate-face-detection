"""Deterministic per-dataset split, then merge.

THE LEAKAGE INVARIANT (do not break):
  * The split is decided on the SOURCE image id (`Sample.sid`) only.
  * Splitting happens BEFORE any tiling/augmentation.
  * Every tile/variant later inherits its parent sample's split.
So no derived crop of a train image can ever land in val/test.

Determinism: the split is a stable hash of sid (md5), NOT random shuffling, so a
given sid always maps to the same bucket regardless of set size or ordering, and
re-running after adding images does not reshuffle existing ones.
"""
from __future__ import annotations
import hashlib
from typing import List, Dict
from .adapters.base import Sample

SPLITS = ("train", "val", "test")


def _bucket(sid: str, seed: int) -> float:
    """Stable value in [0,1) for this sid+seed."""
    h = hashlib.md5(f"{seed}:{sid}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_dataset(samples: List[Sample], ratios: dict) -> Dict[str, List[Sample]]:
    """Split one dataset's samples into train/val/test by stable hash of sid."""
    tr = float(ratios.get("train", 0.7))
    va = float(ratios.get("val", 0.15))
    seed = int(ratios.get("seed", 42))
    out = {s: [] for s in SPLITS}
    for smp in samples:
        b = _bucket(smp.sid, seed)
        if b < tr:
            out["train"].append(smp)
        elif b < tr + va:
            out["val"].append(smp)
        else:
            out["test"].append(smp)
    return out


def merge_splits(per_dataset: List[Dict[str, List[Sample]]]) -> Dict[str, List[Sample]]:
    """Merge each dataset's already-split buckets: train+train, val+val, test+test."""
    merged = {s: [] for s in SPLITS}
    for d in per_dataset:
        for s in SPLITS:
            merged[s].extend(d.get(s, []))
    return merged


def assert_no_leakage(merged: Dict[str, List[Sample]]) -> None:
    """Fail loudly if any source id appears in more than one split."""
    seen = {}
    for s in SPLITS:
        for smp in merged[s]:
            if smp.sid in seen and seen[smp.sid] != s:
                raise AssertionError(
                    f"LEAKAGE: {smp.sid} is in both '{seen[smp.sid]}' and '{s}'"
                )
            seen[smp.sid] = s
