"""imagefolder adapter — a folder of images that carry NO boxes of their own.

Single target class (taken from class_map, which must map exactly one source name).
Because the images have no annotations, `label_source` decides where boxes come from:

  pseudo_label        : read boxes from a sidecar the GPU step `pseudo_label.py` made
                        with the current model (confidence-filtered).  [recommended]
  bbox_sidecar        : read boxes from a sidecar json you supply.
  whole_image_is_object: the whole frame IS one box (only for tight crops).

sidecar json format:  {"<relpath>": [[x1,y1,x2,y2], ...], ...}

Pairs with an include_filter (presence json, keep_value) so an included image can
never contain an UNLABELLED object of the other class. Purely a format reader —
which specific dataset uses it is set in its config file.
"""
from __future__ import annotations
import glob
import json
import os
from typing import List

from .base import Adapter, Sample, Box
from ..filters import load_presence, match_key
from .. import imageio_util as io


class ImageFolderAdapter(Adapter):
    def load(self) -> List[Sample]:
        cls_map = self._class_map_to_ids()
        if len(cls_map) != 1:
            raise ValueError(
                f"[{self.name}] imagefolder needs exactly ONE class in class_map "
                f"(these images carry a single object type); got {list(cls_map)}")
        target_id = next(iter(cls_map.values()))
        images_dir = self.abspath(self.cfg["images_dir"])
        mode = self.cfg.get("label_source", "pseudo_label")

        keep_set = None
        strip = None
        flt = self.cfg.get("include_filter")
        if flt:
            strip = flt.get("strip_prefixes")
            keep_set, _ = load_presence(
                self.abspath(flt["json"]), int(flt["keep_value"]), strip)

        sidecar = {}
        if mode in ("pseudo_label", "bbox_sidecar"):
            sc = self.cfg.get("boxes_sidecar")
            if not sc:
                raise ValueError(
                    f"[{self.name}] label_source={mode} needs `boxes_sidecar: <json>`. "
                    f"For pseudo_label, run `python -m src.pseudo_label ...` on the GPU "
                    f"node first (these images ship without boxes).")
            sc_abs = self.abspath(sc)
            if not os.path.exists(sc_abs):
                raise FileNotFoundError(
                    f"[{self.name}] boxes_sidecar not found: {sc_abs}\n"
                    f"  -> create it with the pseudo-label step, or switch label_source.")
            sidecar = json.load(open(sc_abs, "r", encoding="utf-8"))

        files = []
        for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            files += glob.glob(os.path.join(images_dir, "**", e), recursive=True)

        samples: List[Sample] = []
        for path in sorted(files):
            rel = os.path.relpath(path, images_dir).replace("\\", "/")
            if keep_set is not None and match_key(rel, strip) not in keep_set:
                continue
            try:
                h, w = io.imread(path).shape[:2]
            except Exception:
                continue
            if mode == "whole_image_is_object":
                boxes = [Box(target_id, 0, 0, w, h)]
            else:
                boxes = [Box(target_id, b[0], b[1], b[2], b[3])
                         for b in sidecar.get(rel, [])]
                if not boxes:
                    continue
            samples.append(Sample(sid=f"{self.name}/{rel}", image_path=path,
                                  width=w, height=h, boxes=boxes, dataset=self.name))
        return samples
