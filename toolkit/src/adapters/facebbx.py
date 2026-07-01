"""facebbx adapter — line-based face bounding-box GT (a common public txt layout).

File layout:
    <relative/image/path>
    <count>
    x y w h [blur expr illum invalid occl pose]     (count lines; extra cols optional)

Keeps faces only; drops faces flagged invalid (col 7 == 1) and faces smaller than
`min_face_px`. Pairs with an include_filter so only "clean" images are used (e.g.
images with no object of the OTHER class, which would otherwise be unlabelled).
Purely a format reader — which specific dataset uses it is set in its config file.
"""
from __future__ import annotations
import os
from typing import List

from .base import Adapter, Sample, Box
from ..filters import load_presence, match_key


class FaceBbxAdapter(Adapter):
    def load(self) -> List[Sample]:
        cls_map = self._class_map_to_ids()
        if "face" not in cls_map:
            raise ValueError(f"[{self.name}] facebbx adapter needs 'face' in class_map")
        face_id = cls_map["face"]
        min_face = int(self.cfg.get("min_face_px", 8))

        gt_path = self.abspath(self.cfg["labels"])
        images_dir = self.abspath(self.cfg["images_dir"])

        keep_set = None
        strip = None
        flt = self.cfg.get("include_filter")
        if flt:
            strip = flt.get("strip_prefixes")
            keep_set, _ = load_presence(
                self.abspath(flt["json"]), int(flt["keep_value"]), strip
            )

        with open(gt_path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]

        samples: List[Sample] = []
        i, n = 0, len(lines)
        while i < n:
            rel = lines[i].strip()
            i += 1
            if not rel:
                continue
            if i >= n:
                break
            try:
                count = int(lines[i].strip())
            except ValueError:
                count = 0
            i += 1
            boxes = []
            for _ in range(max(count, 0)):
                if i >= n:
                    break
                parts = lines[i].split()
                i += 1
                if len(parts) < 4:
                    continue
                x, y, w, h = (float(parts[0]), float(parts[1]),
                              float(parts[2]), float(parts[3]))
                invalid = int(parts[7]) if len(parts) > 7 else 0
                if invalid == 1 or w < min_face or h < min_face:
                    continue
                boxes.append(Box(face_id, x, y, x + w, y + h))

            if keep_set is not None and match_key(rel, strip) not in keep_set:
                continue
            path = os.path.join(images_dir, rel.replace("\\", "/"))
            if not os.path.exists(path):
                continue
            samples.append(Sample(sid=f"{self.name}/{rel}", image_path=path,
                                  width=0, height=0, boxes=boxes, dataset=self.name))
        return samples
