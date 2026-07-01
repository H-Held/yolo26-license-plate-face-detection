"""Create tiny synthetic fixtures so the CPU pipeline is runnable without any
real data / GPU / server. Draws coloured rectangles as 'objects' at KNOWN pixel
locations so bbox transforms can be checked exactly.

Writes under  tests/_fixtures/  :
    own_src/images/*.jpg + own_src/annotations.json   (coco: face + license-plate)
    faces_src/images/... + faces_gt.txt + faces_kombiniert.json   (facebbx)
    plates_src/images/*.jpg + plates_boxes.json + plates_kombiniert.json (euplate)
"""
from __future__ import annotations
import json
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "_fixtures")


def _img(w, h, rects):
    """rects: list of (x1,y1,x2,y2,(r,g,b)). Returns HxWx3 uint8."""
    a = np.full((h, w, 3), 200, np.uint8)
    a[:, :, 0] = (np.linspace(30, 220, w)[None, :]).astype(np.uint8)  # gradient bg
    for (x1, y1, x2, y2, col) in rects:
        a[y1:y2, x1:x2] = col
    return a


def _save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


def build():
    # ---------- OWN (coco): 6 images, faces + plates, some large-object ----------
    own_img = os.path.join(FIX, "own_src", "images")
    coco = {"images": [], "annotations": [], "categories": [
        {"id": 1, "name": "face"}, {"id": 2, "name": "license-plate"}]}
    aid = 1
    for i in range(6):
        w, h = 1600, 1000
        rects, anns = [], []
        # a normal face
        fx, fy = 200 + i * 30, 300
        rects.append((fx, fy, fx + 90, fy + 120, (240, 200, 180)))
        anns.append((1, fx, fy, 90, 120))
        # a normal plate
        px, py = 900, 640
        rects.append((px, py, px + 240, py + 70, (250, 250, 250)))
        anns.append((2, px, py, 240, 70))
        # image 0 also gets a HUGE plate spanning >4 native 640 tiles -> whole-image trigger
        if i == 0:
            rects.append((100, 100, 1500, 900, (180, 250, 180)))
            anns.append((2, 100, 100, 1400, 800))
        arr = _img(w, h, rects)
        fn = f"own_{i}.jpg"
        _save(os.path.join(own_img, fn), arr)
        coco["images"].append({"id": i, "file_name": fn, "width": w, "height": h})
        for (cid, x, y, bw, bh) in anns:
            coco["annotations"].append({"id": aid, "image_id": i, "category_id": cid,
                                        "bbox": [x, y, bw, bh], "iscrowd": 0,
                                        "area": bw * bh})
            aid += 1
    with open(os.path.join(FIX, "own_src", "annotations.json"), "w") as f:
        json.dump(coco, f)

    # ---------- FACES (facebbx): 8 images, half clean(0) half other-class(1) ----------
    faces_img = os.path.join(FIX, "faces_src", "images", "grp")
    gt_lines, komb = [], {}
    for i in range(8):
        w, h = 800, 600
        fx, fy = 100 + i * 10, 150
        arr = _img(w, h, [(fx, fy, fx + 60, fy + 80, (240, 210, 190))])
        rel = f"grp/face_{i}.jpg"
        _save(os.path.join(FIX, "faces_src", "images", f"face_{i}.jpg"), arr)
        # note: images_dir points at faces_src/images ; rel path uses grp/ subdir below
        gt_lines.append(rel)
        gt_lines.append("1")
        invalid = 0
        gt_lines.append(f"{fx} {fy} 60 80 0 0 0 {invalid} 0 0")
        komb[rel] = 0 if i % 2 == 0 else 1          # even = no vehicle (usable)
    # move images into grp/ subdir to match rel path
    grp_dir = os.path.join(FIX, "faces_src", "images", "grp")
    os.makedirs(grp_dir, exist_ok=True)
    for i in range(8):
        src = os.path.join(FIX, "faces_src", "images", f"face_{i}.jpg")
        dst = os.path.join(grp_dir, f"face_{i}.jpg")
        if os.path.exists(src):
            os.replace(src, dst)
    with open(os.path.join(FIX, "faces_src", "faces_gt.txt"), "w") as f:
        f.write("\n".join(gt_lines) + "\n")
    with open(os.path.join(FIX, "faces_src", "faces_kombiniert.json"), "w") as f:
        json.dump({"annotations": komb}, f)

    # ---------- PLATES (euplate, bbox_sidecar): 6 images, half no-person(0) ----------
    plates_img = os.path.join(FIX, "plates_src", "images")
    sidecar, komb2 = {}, {}
    for i in range(6):
        w, h = 700, 500
        px, py = 200, 250
        arr = _img(w, h, [(px, py, px + 180, py + 55, (250, 250, 250))])
        rel = f"plate_{i}.jpg"
        _save(os.path.join(plates_img, rel), arr)
        sidecar[rel] = [[px, py, px + 180, py + 55]]
        komb2[rel] = 0 if i % 2 == 0 else 1          # even = no person (usable)
    with open(os.path.join(FIX, "plates_src", "plates_boxes.json"), "w") as f:
        json.dump(sidecar, f)
    with open(os.path.join(FIX, "plates_src", "plates_kombiniert.json"), "w") as f:
        json.dump({"annotations": komb2}, f)

    print("fixtures written under", FIX)


if __name__ == "__main__":
    build()
