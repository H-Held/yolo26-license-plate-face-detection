"""Registry / config loader.

Generic on purpose: it scans two folders for dataset configs and treats them
identically, so NOTHING in the committed tree names a private dataset:

    <toolkit>/config/datasets/*.yaml     -> public ("official") datasets
    <repo>/.hiden/datasets/*.yaml        -> private ("hidden") datasets (git-ignored)

Add a dataset by dropping a YAML file in one of those folders. No code change.
"""
from __future__ import annotations
import os
import glob
import yaml

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLKIT_DIR = os.path.dirname(_SRC_DIR)
REPO_DIR = os.path.dirname(TOOLKIT_DIR)

PUBLIC_DS_DIR = os.path.join(TOOLKIT_DIR, "config", "datasets")
HIDDEN_DS_DIR = os.path.join(REPO_DIR, ".hiden", "datasets")
GLOBAL_CFG = os.path.join(TOOLKIT_DIR, "config", "global.yaml")

_ADAPTERS = None


def _adapters():
    global _ADAPTERS
    if _ADAPTERS is None:
        from .adapters.coco import CocoAdapter
        from .adapters.facebbx import FaceBbxAdapter
        from .adapters.imagefolder import ImageFolderAdapter
        from .adapters.yolo_txt import YoloTxtAdapter
        _ADAPTERS = {"coco": CocoAdapter, "facebbx": FaceBbxAdapter,
                     "imagefolder": ImageFolderAdapter, "yolo_txt": YoloTxtAdapter}
    return _ADAPTERS


def load_global(path: str = GLOBAL_CFG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        g = yaml.safe_load(f)
    # env override for the data root (used for local tests / different hosts)
    g["data_root"] = os.environ.get("YOLO_TOOLKIT_DATA_ROOT", g["data_root"])
    g["name2id"] = {name: i for i, name in enumerate(g["classes"])}
    return g


def _load_ds_dir(folder: str, default_visibility: str) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.yaml"))):
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg.setdefault("visibility", default_visibility)
        cfg["_config_path"] = p
        out.append(cfg)
    return out


def discover_datasets(include_disabled: bool = False) -> list:
    """Return dataset config dicts from both public and hidden folders."""
    cfgs = (_load_ds_dir(PUBLIC_DS_DIR, "official")
            + _load_ds_dir(HIDDEN_DS_DIR, "hidden"))
    seen = {}
    for c in cfgs:
        if "name" not in c:
            raise ValueError(f"dataset config missing `name`: {c.get('_config_path')}")
        if c["name"] in seen:
            raise ValueError(f"duplicate dataset name '{c['name']}' "
                             f"({seen[c['name']]} and {c['_config_path']})")
        seen[c["name"]] = c["_config_path"]
    if not include_disabled:
        cfgs = [c for c in cfgs if c.get("enabled", True)]
    return cfgs


def build_adapter(cfg: dict, g: dict):
    fmt = cfg.get("format")
    adapters = _adapters()
    if fmt not in adapters:
        raise ValueError(f"[{cfg.get('name')}] unknown format '{fmt}'; "
                         f"available: {list(adapters)}")
    return adapters[fmt](cfg, g["data_root"], g["name2id"])
