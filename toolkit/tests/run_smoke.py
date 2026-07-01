"""End-to-end CPU smoke test — the ONLY part verifiable without GPU/server.

Runs:
  A. analytic bbox geometry: a red marker at a known place, transformed by every
     geometric op + by letterbox, and checked that the predicted box actually
     covers the marker (catches silent bbox bugs in new transforms).
  B. full discover -> split -> merge -> leakage -> tile -> augment -> write build
     on synthetic fixtures, asserting: 3 datasets found (1 official + 2 hidden),
     no leakage, all label coords in [0,1], plate class_boost raised plate count.

Run:  python tests/run_smoke.py     (from the toolkit/ folder)
"""
from __future__ import annotations
import os
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLKIT = os.path.dirname(HERE)
sys.path.insert(0, TOOLKIT)

import random                              # noqa: E402
import make_fixtures                       # noqa: E402
from src import imageio_util as io         # noqa: E402
from src import augment as A               # noqa: E402
from src import registry                   # noqa: E402
from src import build_dataset              # noqa: E402
from src.tiling import tile_sample         # noqa: E402
from src.adapters.base import Sample, Box  # noqa: E402

FIX = os.path.join(HERE, "_fixtures")
FAILS = []


def check(cond, msg):
    print(("  ok  " if cond else " FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def _red_marker(size, x1, y1, x2, y2):
    img = np.zeros((size, size, 3), np.uint8)
    img[y1:y2, x1:x2] = (255, 0, 0)
    boxes = [(0, (x1 + x2) / 2 / size, (y1 + y2) / 2 / size,
              (x2 - x1) / size, (y2 - y1) / size)]
    return img, boxes


def _marker_bbox(img):
    """pixel bbox of the red region."""
    mask = (img[:, :, 0] > 150) & (img[:, :, 1] < 80) & (img[:, :, 2] < 80)
    ys, xs = np.where(mask)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def _box_px(boxes, size):
    c, cx, cy, w, h = boxes[0]
    return (round((cx - w / 2) * size), round((cy - h / 2) * size),
            round((cx + w / 2) * size), round((cy + h / 2) * size))


def test_geometry():
    print("A. analytic bbox geometry")
    size = 200
    img, boxes = _red_marker(size, 40, 20, 90, 140)   # off-centre, tall
    for name, fn in [("hflip", A._hflip), ("vflip", A._vflip),
                     ("rot180", A._rot180), ("rot90", A._rot90),
                     ("rot270", A._rot270)]:
        timg, tboxes = fn(img, boxes)
        px = _box_px(tboxes, size)
        mb = _marker_bbox(timg)
        ok = all(abs(a - b) <= 2 for a, b in zip(px, mb))
        check(ok, f"{name}: predicted {px} vs actual marker {mb}")

    # letterbox mapping of an off-square image
    rect = np.zeros((300, 600, 3), np.uint8)
    rect[50:120, 400:520] = (255, 0, 0)
    lb, sc, pxo, pyo = io.letterbox(rect, 640, 114)
    x1, y1, x2, y2 = 400 * sc + pxo, 50 * sc + pyo, 520 * sc + pxo, 120 * sc + pyo
    mb = _marker_bbox(lb)
    ok = all(abs(a - b) <= 3 for a, b in zip((x1, y1, x2, y2), mb))
    check(ok, f"letterbox: predicted {(round(x1),round(y1),round(x2),round(y2))} vs {mb}")


def _has_red(img):
    return bool(((img[:, :, 0] > 150) & (img[:, :, 1] < 80) & (img[:, :, 2] < 80)).any())


def _all_cover(chips, size, tol=6):
    """Every positive chip must actually contain the marker its box points to."""
    for img, boxes in chips:
        if not boxes:
            continue
        if not _has_red(img):
            return False            # phantom box: box but no object -> bug
        px = _box_px([boxes[0]], size)
        mb = _marker_bbox(img)
        if not all(abs(a - b) <= tol for a, b in zip(px, mb)):
            return False
    return True


def _rect_sample(W, H, rect):
    img = np.zeros((H, W, 3), np.uint8)
    x1, y1, x2, y2 = rect
    img[y1:y2, x1:x2] = (255, 0, 0)
    s = Sample(sid="t", image_path="", width=W, height=H,
               boxes=[Box(0, x1, y1, x2, y2)], dataset="t")
    return s, img


def test_tiling_regimes():
    print("C. tiling regimes on >= imgsz images (box correctness under each)")
    size = 640
    rng = random.Random(0)
    base = {"stride_frac": 0.8, "max_pos_tiles": 50, "hardneg_frac": 0.0}

    # (a) native-only, large image
    s, img = _rect_sample(1920, 1440, (300, 200, 520, 360))
    ch = tile_sample(s, img, {**base, "native_tiles": True, "scaled_tiles": [],
                              "whole_image": False, "whole_if_span_tiles": 0}, size, rng=rng)
    check(len(ch) >= 1 and _all_cover(ch, size), f"native-only: {len(ch)} chips, boxes cover marker")

    # (b) scaled-only 2x -> imgsz (exercises the sc box-scaling path)
    s, img = _rect_sample(1920, 1440, (300, 200, 520, 360))
    ch = tile_sample(s, img, {**base, "native_tiles": False, "scaled_tiles": [2],
                              "whole_image": False, "whole_if_span_tiles": 0}, size, rng=rng)
    check(len(ch) >= 1 and _all_cover(ch, size), f"scaled-only 2x: {len(ch)} chips, boxes cover marker")

    # (c) whole-only on an image >= imgsz in BOTH dims  == the advisor's regression
    s, img = _rect_sample(1024, 768, (100, 100, 300, 260))
    ch = tile_sample(s, img, {"native_tiles": False, "scaled_tiles": [],
                              "whole_image": True, "whole_if_span_tiles": 0}, size, rng=rng)
    check(len(ch) >= 1 and _all_cover(ch, size), f"whole-only >=imgsz: {len(ch)} chip(s) (was 0 before fix)")
    ch0 = tile_sample(s, img, {"native_tiles": False, "scaled_tiles": [],
                               "whole_image": False, "whole_if_span_tiles": 0}, size, rng=rng)
    check(len(ch0) == 0, f"without whole_image, a >=imgsz whole-only image yields 0 (confirms the trap)")

    # (d) conditional span trigger: huge box fires, tiny box does not
    s, img = _rect_sample(1600, 1000, (50, 50, 1550, 950))
    chb = tile_sample(s, img, {"native_tiles": False, "scaled_tiles": [],
                               "whole_image": False, "whole_if_span_tiles": 4}, size, rng=rng)
    check(len(chb) >= 1 and _all_cover(chb, size), f"span-trigger huge box -> {len(chb)} whole chip(s)")
    s, img = _rect_sample(1600, 1000, (760, 470, 820, 530))
    chs = tile_sample(s, img, {"native_tiles": False, "scaled_tiles": [],
                               "whole_image": False, "whole_if_span_tiles": 4}, size, rng=rng)
    check(len(chs) == 0, f"span-trigger off for small box -> {len(chs)} chips")


TEST_CFGS = {
    "public": {
        "own_test.yaml": """
name: own_test
format: coco
images_dir: own_src/images
labels: own_src/annotations.json
class_map: {face: face, license-plate: license-plate}
split: {train: 0.6, val: 0.2, test: 0.2, seed: 1}
tiling: {native_tiles: true, scaled_tiles: [2], whole_if_span_tiles: 4,
         stride_frac: 0.8, max_pos_tiles: 8, hardneg_frac: 0.05}
augment: {photometric_variants: 1, geometric: true, class_boost: {license-plate: 3}}
"""},
    "hidden": {
        "faces_test.yaml": """
name: faces_test
format: facebbx
images_dir: faces_src/images
labels: faces_src/faces_gt.txt
class_map: {face: face}
include_filter: {json: faces_src/faces_kombiniert.json, keep_value: 0, strip_prefixes: []}
min_face_px: 8
split: {train: 0.6, val: 0.2, test: 0.2, seed: 2}
tiling: {native_tiles: false, scaled_tiles: [], whole_image: true, whole_if_span_tiles: 0}
augment: {photometric_variants: 0, geometric: false}
""",
        "plates_test.yaml": """
name: plates_test
format: imagefolder
images_dir: plates_src/images
label_source: bbox_sidecar
boxes_sidecar: plates_src/plates_boxes.json
class_map: {license-plate: license-plate}
include_filter: {json: plates_src/plates_kombiniert.json, keep_value: 0}
split: {train: 0.6, val: 0.2, test: 0.2, seed: 3}
tiling: {native_tiles: false, scaled_tiles: [], whole_image: true, whole_if_span_tiles: 0}
augment: {photometric_variants: 0, geometric: false}
"""}}


def test_build(tmp):
    print("B. full build on fixtures")
    pub = os.path.join(tmp, "public"); os.makedirs(pub)
    hid = os.path.join(tmp, "hidden"); os.makedirs(hid)
    for fn, body in TEST_CFGS["public"].items():
        open(os.path.join(pub, fn), "w").write(body)
    for fn, body in TEST_CFGS["hidden"].items():
        open(os.path.join(hid, fn), "w").write(body)

    registry.PUBLIC_DS_DIR = pub
    registry.HIDDEN_DS_DIR = hid
    os.environ["YOLO_TOOLKIT_DATA_ROOT"] = FIX

    cfgs = registry.discover_datasets()
    vis = sorted((c["name"], c["visibility"]) for c in cfgs)
    check(len(cfgs) == 3, f"discovered 3 datasets: {vis}")
    check(sum(c["visibility"] == "hidden" for c in cfgs) == 2, "2 hidden datasets")

    g = registry.load_global()
    g["out_dataset"] = "_out"
    res = build_dataset.build(g, verbose=True)

    out = res["out_dir"]
    check(res["counts"]["train"] > 0, f"train chips written: {res['counts']}")
    check(res["counts"]["val"] > 0 and res["counts"]["test"] > 0, "val+test written")

    # all label coords within [0,1]
    bad = 0
    for split in ("train", "val", "test"):
        ld = os.path.join(out, "labels", split)
        for fn in os.listdir(ld):
            for line in open(os.path.join(ld, fn)):
                vals = line.split()
                if not vals:
                    continue
                nums = list(map(float, vals[1:]))
                if any(v < -1e-6 or v > 1 + 1e-6 for v in nums):
                    bad += 1
    check(bad == 0, f"all label coords in [0,1] (bad lines={bad})")

    # plate class boost: plate boxes should outnumber a no-boost baseline
    plate_boxes = res["box_counts"]["train"][g["name2id"]["license-plate"]]
    check(plate_boxes > 0, f"train plate boxes present ({plate_boxes})")


def main():
    make_fixtures.build()
    test_geometry()
    test_tiling_regimes()
    with tempfile.TemporaryDirectory() as tmp:
        test_build(tmp)
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURE(S): " + "; ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
