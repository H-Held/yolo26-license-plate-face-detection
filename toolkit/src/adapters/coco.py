"""COCO adapter — for your own exported photos (face + license-plate)."""
from __future__ import annotations
import json
import os
from typing import List

from .base import Adapter, Sample, Box
from ..filters import load_presence, match_key


class CocoAdapter(Adapter):
    def load(self) -> List[Sample]:
        cls_map = self._class_map_to_ids()          # source name -> unified id
        labels_path = self.abspath(self.cfg["labels"])
        images_dir = self.abspath(self.cfg["images_dir"])
        with open(labels_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        catid2name = {c["id"]: c["name"] for c in coco.get("categories", [])}
        # keep only categories present in class_map
        keep_catid = {cid: cls_map[nm] for cid, nm in catid2name.items() if nm in cls_map}
        if not keep_catid:
            raise ValueError(
                f"[{self.name}] none of class_map names {list(cls_map)} found in "
                f"COCO categories {list(catid2name.values())}"
            )

        # optional presence filter
        keep_set = None
        strip = None
        flt = self.cfg.get("include_filter")
        if flt:
            strip = flt.get("strip_prefixes")
            keep_set, _ = load_presence(
                self.abspath(flt["json"]), int(flt["keep_value"]), strip
            )

        imgs = {im["id"]: im for im in coco["images"]}
        by_img = {}
        for a in coco.get("annotations", []):
            cid = a["category_id"]
            if cid not in keep_catid:
                continue
            by_img.setdefault(a["image_id"], []).append(a)

        samples: List[Sample] = []
        for img_id, im in imgs.items():
            fn = im["file_name"]
            if keep_set is not None and match_key(fn, strip) not in keep_set:
                continue
            path = os.path.join(images_dir, fn)
            if not os.path.exists(path):
                continue
            w = int(im.get("width", 0)) or None
            h = int(im.get("height", 0)) or None
            boxes = []
            for a in by_img.get(img_id, []):
                x, y, bw, bh = a["bbox"]           # COCO xywh, absolute
                boxes.append(Box(keep_catid[a["category_id"]], x, y, x + bw, y + bh))
            samples.append(Sample(sid=f"{self.name}/{fn}", image_path=path,
                                  width=w or 0, height=h or 0, boxes=boxes,
                                  dataset=self.name))
        return samples
