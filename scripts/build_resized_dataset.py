#!/usr/bin/env python
"""Build a down-scaled ("compressed") copy of a YOLO dataset.

Takes an existing tiled dataset (e.g. dataset_face_lp at 1280x1280) and writes a
new dataset with every image resized to a smaller size (e.g. 640x640). YOLO
labels are normalised (0..1), so they are copied unchanged. Images are written
as real PNG files (not symlinks) so the smaller set caches cheaply in RAM.

Usage:
  python build_resized_dataset.py <src_dataset_dir> <dst_dataset_dir> <size>
Example:
  python build_resized_dataset.py \
    /home/jovyan/shared/s0598584/dataset_face_lp \
    /home/jovyan/shared/s0598584/dataset_face_lp_640  640
"""
import os, sys, glob, shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def _resize_one(args):
    src_img, dst_img, size = args
    im = cv2.imread(src_img)
    if im is None:
        return False
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst_img, im, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return True


def build(src, dst, size, workers=16):
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    tasks = []
    for split in ("train", "val", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for lbl in glob.glob(str(src / "labels" / split / "*.txt")):
            shutil.copy(lbl, dst / "labels" / split / Path(lbl).name)
        for img in glob.glob(str(src / "images" / split / "*")):
            name = Path(img).name
            tasks.append((os.path.realpath(img),
                          str(dst / "images" / split / name), size))
    ok = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_resize_one, t) for t in tasks]
        for f in as_completed(futs):
            ok += bool(f.result())
    names = "\n".join(f"  {i}: {C.CLASSES[i]}" for i in sorted(C.CLASSES))
    (dst / "dataset.yaml").write_text(
        f"path: {dst}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {C.num_classes()}\nnames:\n{names}\n")
    print(f"built {dst} @ {size}px : {ok}/{len(tasks)} images, dataset.yaml written")
    return dst / "dataset.yaml"


if __name__ == "__main__":
    src = sys.argv[1]; dst = sys.argv[2]; size = int(sys.argv[3])
    build(src, dst, size)
