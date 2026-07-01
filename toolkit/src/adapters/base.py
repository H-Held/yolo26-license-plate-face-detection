"""Unified sample model + adapter base.

Every dataset, whatever its native format, is read into a list of `Sample`s.
Boxes are stored as ABSOLUTE pixel corners (x1,y1,x2,y2) plus a unified class id
(index into global.yaml `classes`). Absolute corners are what tiling needs; the
YOLO writer converts to normalised cx,cy,w,h at the very end.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Box:
    cls: int          # unified class id (index into global classes)
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


@dataclass
class Sample:
    sid: str              # STABLE unique id of the SOURCE image (drives the split)
    image_path: str       # absolute path on disk
    width: int
    height: int
    boxes: List[Box] = field(default_factory=list)
    dataset: str = ""     # owning dataset name (for stats / provenance)


class Adapter:
    """Base class. Subclasses implement load() -> List[Sample].

    `cfg` is the dataset config dict; `data_root` is the global data root;
    `name2id` maps unified class NAME -> unified class id.
    """

    def __init__(self, cfg: dict, data_root: str, name2id: dict):
        self.cfg = cfg
        self.data_root = data_root
        self.name2id = name2id
        self.name = cfg["name"]

    # -- helpers shared by concrete adapters --
    def abspath(self, *parts) -> str:
        return os.path.join(self.data_root, *[str(p) for p in parts])

    def _class_map_to_ids(self) -> dict:
        """source-class-name -> unified-class-id, from cfg['class_map']."""
        cm = self.cfg.get("class_map") or {}
        out = {}
        for src_name, uni_name in cm.items():
            if uni_name not in self.name2id:
                raise ValueError(
                    f"[{self.name}] class_map maps '{src_name}' -> '{uni_name}', "
                    f"but '{uni_name}' is not in global.yaml classes {list(self.name2id)}"
                )
            out[src_name] = self.name2id[uni_name]
        if not out:
            raise ValueError(f"[{self.name}] class_map is empty; nothing to keep")
        return out

    def load(self) -> List[Sample]:  # pragma: no cover - interface
        raise NotImplementedError
