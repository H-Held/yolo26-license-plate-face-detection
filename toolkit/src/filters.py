"""Inclusion filter driven by a 'kombiniert.json' presence file.

The presence file marks each image 0/1 (e.g. 0 = no vehicle for a faces-only set,
0 = no person for a plates-only set). We keep only images whose flag == keep_value,
so an included image can never contain an UNLABELLED object of the other class.

Format accepted (both are handled):
    {"annotations": {"<path>": 0|1, ...}}          # presence-annotation tool output
    {"<path>": 0|1, ...}                            # flat map

Robustness:
- path keys are normalised (backslashes -> '/', configured prefixes stripped).
- if the SAME normalised key appears as both 0 and 1 (annotation conflict), it is
  treated as 1 (the unsafe value) and therefore EXCLUDED — never guess it clean.
"""
from __future__ import annotations
import json
import os


def _norm(p: str, strip_prefixes) -> str:
    p = p.replace("\\", "/").lstrip("/")
    for pre in strip_prefixes or []:
        pre = pre.replace("\\", "/").strip("/")
        if pre and p.startswith(pre + "/"):
            p = p[len(pre) + 1:]
    return p


def load_presence(json_path: str, keep_value: int, strip_prefixes=None):
    """Return (keep_set, drop_set) of normalised relative paths.

    keep_set = keys whose value == keep_value and never conflicts.
    drop_set = everything else that was annotated (for stats).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ann = raw.get("annotations", raw) if isinstance(raw, dict) else {}
    if not isinstance(ann, dict):
        raise ValueError(f"presence file {json_path} has no usable annotations map")

    seen = {}   # norm_key -> set of values
    for k, v in ann.items():
        if not isinstance(v, int):
            continue
        nk = _norm(k, strip_prefixes)
        seen.setdefault(nk, set()).add(v)

    keep, drop = set(), set()
    for nk, vals in seen.items():
        if vals == {keep_value}:      # unanimous and safe
            keep.add(nk)
        else:                          # conflicting or the unsafe value
            drop.add(nk)
    return keep, drop


def match_key(image_rel_path: str, strip_prefixes=None) -> str:
    """Normalise an image path the same way keys are normalised, for lookup."""
    return _norm(image_rel_path, strip_prefixes)
