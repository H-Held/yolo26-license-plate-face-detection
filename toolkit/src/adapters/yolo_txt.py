"""yolo_txt adapter — images with sidecar YOLO-format .txt label files.

Each image `foo.jpg` has a matching `foo.txt` alongside it (or in a parallel
labels_dir with the same relative path), one line per box:
    <source_class_id> <cx> <cy> <w> <h>      (all normalised 0..1)

`class_map` maps the SOURCE numeric class id (as a string key, e.g. "0") to a
unified class NAME from global.yaml. Classes not listed are dropped. Images
with no matching .txt, or whose .txt has zero valid lines, are skipped.

Pairs with the usual include_filter (presence json) so a hidden set can be
restricted to images that can't contain an unlabelled object of another class.
Purely a format reader — which specific dataset uses it is set in its config.
"""
from __future__ import annotations
import glob
import os
from typing import List

from .base import Adapter, Sample, Box
from ..filters import load_presence, match_key
from .. import imageio_util as io


class YoloTxtAdapter(Adapter):
    def load(self) -> List[Sample]:
        cls_map_raw = self.cfg.get("class_map") or {}
        name2id = self.name2id
        id_map = {}   # source class id (int) -> unified class id
        for src_id, uni_name in cls_map_raw.items():
            if uni_name not in name2id:
                raise ValueError(
                    f"[{self.name}] class_map maps '{src_id}' -> '{uni_name}', "
                    f"but '{uni_name}' is not in global.yaml classes {list(name2id)}")
            id_map[int(src_id)] = name2id[uni_name]
        if not id_map:
            raise ValueError(f"[{self.name}] class_map is empty; nothing to keep")

        images_dir = self.abspath(self.cfg["images_dir"])
        labels_dir = self.abspath(self.cfg.get("labels_dir", self.cfg["images_dir"]))

        keep_set = None
        strip = None
        flt = self.cfg.get("include_filter")
        if flt:
            strip = flt.get("strip_prefixes")
            keep_set, _ = load_presence(
                self.abspath(flt["json"]), int(flt["keep_value"]), strip)

        files = []
        for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            files += glob.glob(os.path.join(images_dir, "**", e), recursive=True)

        samples: List[Sample] = []
        for path in sorted(files):
            rel = os.path.relpath(path, images_dir).replace("\\", "/")
            if keep_set is not None and match_key(rel, strip) not in keep_set:
                continue
            txt_path = os.path.join(labels_dir, os.path.splitext(rel)[0] + ".txt")
            if not os.path.exists(txt_path):
                continue
            try:
                h, w = io.imread(path).shape[:2]
            except Exception:
                continue
            boxes = []
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    src_id = int(float(parts[0]))
                    if src_id not in id_map:
                        continue
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                    x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                    boxes.append(Box(id_map[src_id], x1, y1, x2, y2))
            if not boxes:
                continue
            samples.append(Sample(sid=f"{self.name}/{rel}", image_path=path,
                                  width=w, height=h, boxes=boxes, dataset=self.name))
        return samples
