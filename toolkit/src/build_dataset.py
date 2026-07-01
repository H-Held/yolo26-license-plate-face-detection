"""Orchestrator: split-per-dataset -> merge -> tile -> augment -> write YOLO set.

Order is fixed and matches the requirement:
    1) read each dataset            (adapters)
    2) SPLIT each dataset SEPARATELY (split.split_dataset)   <-- before augment
    3) MERGE the splits             (split.merge_splits)
    4) assert no leakage
    5) TRAIN: tile (3 regimes) then augment (variety + class boost), streamed to disk
       VAL/TEST: single whole->imgsz letterbox (normal inference geometry)
    6) write dataset.yaml + print stats

Streamed per-sample so peak RAM stays ~one image's chips, not the whole set.

Run:  python -m src.build_dataset            (from the toolkit/ folder)
"""
from __future__ import annotations
import hashlib
import os
import random
import sys

from . import imageio_util as io
from .registry import load_global, discover_datasets, build_adapter
from .split import split_dataset, merge_splits, assert_no_leakage, SPLITS
from .tiling import tile_sample
from .augment import augment_train


def _yolo_from_letterbox(sample, scale, px, py, size):
    out = []
    for b in sample.boxes:
        x1, y1 = b.x1 * scale + px, b.y1 * scale + py
        x2, y2 = b.x2 * scale + px, b.y2 * scale + py
        out.append((b.cls, (x1 + x2) / 2 / size, (y1 + y2) / 2 / size,
                    (x2 - x1) / size, (y2 - y1) / size))
    return out


def _write_chip(out_dir, split, stem, img, boxes):
    io.imwrite(os.path.join(out_dir, "images", split, stem + ".jpg"), img)
    lbl = os.path.join(out_dir, "labels", split, stem + ".txt")
    with open(lbl, "w", encoding="utf-8") as f:
        for (c, cx, cy, w, h) in boxes:
            if w <= 0 or h <= 0:
                continue
            f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def _stem(sid, idx):
    h = hashlib.md5(sid.encode()).hexdigest()[:12]
    return f"{h}_{idx:04d}"


def build(g: dict = None, verbose: bool = True):
    g = g or load_global()
    imgsz = int(g["imgsz"])
    pad = int(g.get("pad_value", 114))
    out_dir = os.path.join(g["data_root"], g["out_dataset"])
    from .preflight import check_disk
    check_disk(out_dir, float(g.get("min_disk_gb", 5)))   # clear error before we fill disk
    for split in SPLITS:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    # 1-4: read, split each, merge, verify
    per_dataset = []
    ds_cfgs = discover_datasets()
    if not ds_cfgs:
        raise SystemExit("no datasets found in config/datasets/ or .hiden/datasets/")
    for cfg in ds_cfgs:
        adapter = build_adapter(cfg, g)
        samples = adapter.load()
        buckets = split_dataset(samples, cfg.get("split", {}))
        per_dataset.append(buckets)
        if verbose:
            vis = cfg.get("visibility", "official")
            print(f"[{cfg['name']:<16} {vis:<8}] samples={len(samples):<6} "
                  f"train={len(buckets['train'])} val={len(buckets['val'])} "
                  f"test={len(buckets['test'])}")
    merged = merge_splits(per_dataset)
    assert_no_leakage(merged)

    # 5: write chips
    cfg_by_name = {c["name"]: c for c in ds_cfgs}
    counts = {s: 0 for s in SPLITS}
    box_counts = {s: [0] * len(g["classes"]) for s in SPLITS}
    rng = random.Random(int(g.get("seed", 42)))

    for split in SPLITS:
        for sample in merged[split]:
            try:
                img = io.imread(sample.image_path)
            except Exception as e:
                if verbose:
                    print(f"  skip unreadable {sample.image_path}: {e}", file=sys.stderr)
                continue
            if not sample.width:
                sample.height, sample.width = img.shape[:2]

            if split == "train":
                tcfg = cfg_by_name[sample.dataset].get("tiling", {}) or {}
                chips = tile_sample(sample, img, tcfg, imgsz, pad, rng)
                acfg = cfg_by_name[sample.dataset].get("augment", {}) or {}
                chips = augment_train(chips, acfg, g["name2id"], rng)
            else:
                lb, scale, pxo, pyo = io.letterbox(img, imgsz, pad)
                chips = [(lb, _yolo_from_letterbox(sample, scale, pxo, pyo, imgsz))]

            for i, (chip_img, boxes) in enumerate(chips):
                _write_chip(out_dir, split, _stem(sample.sid, i), chip_img, boxes)
                counts[split] += 1
                for (c, *_r) in boxes:
                    box_counts[split][c] += 1

    _write_dataset_yaml(out_dir, g)
    if verbose:
        print("\n=== build complete ===")
        for s in SPLITS:
            print(f"  {s:<5} chips={counts[s]:<7} boxes/class={dict(zip(g['classes'], box_counts[s]))}")
        print(f"  dataset.yaml -> {os.path.join(out_dir, 'dataset.yaml')}")
    return {"out_dir": out_dir, "counts": counts, "box_counts": box_counts}


def _write_dataset_yaml(out_dir, g):
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(g["classes"]))
    txt = (f"path: {out_dir}\n"
           f"train: images/train\n"
           f"val: images/val\n"
           f"test: images/test\n"
           f"nc: {len(g['classes'])}\n"
           f"names:\n{names}\n")
    with open(os.path.join(out_dir, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    build()
