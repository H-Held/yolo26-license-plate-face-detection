"""Pseudo-label a box-less image set (GPU step) -> boxes sidecar json.

Some image sets ship images but no boxes. This runs the CURRENT trained
model over the (presence-filtered) images and writes a sidecar
    {"<relpath>": [[x1,y1,x2,y2], ...], ...}
that the euplate adapter reads. Only detections >= conf and of the plate class are
kept. Runs on the GPU node.

Usage:
    python -m src.pseudo_label --images <images_dir> --weights <model.pt> \
        --out <sidecar.json> [--conf 0.35] [--filter <kombiniert.json>] [--keep 0]
"""
from __future__ import annotations
import argparse
import glob
import json
import os

from .filters import load_presence, match_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--plate-class", default="license-plate")
    ap.add_argument("--filter", default=None, help="kombiniert.json presence file")
    ap.add_argument("--keep", type=int, default=0)
    ap.add_argument("--strip", nargs="*", default=None)
    args = ap.parse_args()

    keep_set = None
    if args.filter:
        keep_set, _ = load_presence(args.filter, args.keep, args.strip)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names  # id -> name
    plate_ids = {i for i, n in names.items() if n == args.plate_class}
    if not plate_ids:
        raise SystemExit(f"model has no class '{args.plate_class}'; classes={names}")

    files = []
    for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        files += glob.glob(os.path.join(args.images, "**", e), recursive=True)
    files.sort()

    sidecar, kept, skipped = {}, 0, 0
    for path in files:
        rel = os.path.relpath(path, args.images).replace("\\", "/")
        if keep_set is not None and match_key(rel, args.strip) not in keep_set:
            skipped += 1
            continue
        r = model.predict(path, conf=args.conf, verbose=False)[0]
        boxes = []
        for b in r.boxes:
            if int(b.cls) in plate_ids:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                boxes.append([x1, y1, x2, y2])
        if boxes:
            sidecar[rel] = boxes
            kept += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(sidecar, open(args.out, "w"))
    print(f"pseudo-labelled {kept} images (skipped {skipped} filtered/empty) -> {args.out}")


if __name__ == "__main__":
    main()
