# YOLO26n — Face & License-Plate Detection

[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-YOLO26n-success)](MODEL_CARD.md)
[![Input](https://img.shields.io/badge/Tiles-1280×1280%20PNG%20(lossless)-orange)]()

A compact **YOLO26-nano** detector that finds **faces** and **vehicle license-plates**,
trained by HTW Berlin students for the **"Privacy by Upload — Automated Image
Anonymization"** project (in collaboration with Parry). The model is used to locate the
regions that must be blurred before an uploaded image is stored.

This repository exists to satisfy the **AGPL-3.0** share-alike obligation of
YOLO26/Ultralytics: it publishes the **training code** and the **resulting model
weights**. It deliberately contains **no training images, no annotations, no API keys,
and no server credentials** — only code and the model.

---

## Detected classes

| Index | Label | Description |
|------:|-------|-------------|
| `0` | `face` | Human faces |
| `1` | `license-plate` | Vehicle registration plates |

See **[MODEL_CARD.md](MODEL_CARD.md)** for the full, human-readable description of the
data, tile sizes (1280×1280, lossless PNG, **no compression**), and every training
hyper-parameter.

---

## Download the model

The trained weights are tracked with **Git LFS** under [`models/`](models/).

**Always get the latest model from one stable link** (the current `main`):

```
https://github.com/H-Held/yolo26-license-plate-face-detection/raw/main/models/yolo26n_face_lp.pt
```

```bash
# command-line download of the newest model
curl -L -o yolo26n_face_lp.pt \
  https://github.com/H-Held/yolo26-license-plate-face-detection/raw/main/models/yolo26n_face_lp.pt
```

**Pinned / reproducible versions** are published as Git **tags** (e.g. `v0.1.0`) and as
**[GitHub Releases](https://github.com/H-Held/yolo26-license-plate-face-detection/releases/latest)**
(the model weights are attached to each release as a downloadable asset). To get the
model exactly as it was at a given version, swap `main` for the tag:

```
https://github.com/H-Held/yolo26-license-plate-face-detection/raw/v0.1.0/models/yolo26n_face_lp.pt
```

If you cloned the repo, make sure LFS content is pulled:

```bash
git lfs install
git lfs pull
```

---

## Quick start (inference)

```python
from ultralytics import YOLO

model = YOLO("yolo26n_face_lp.pt")

# IMPORTANT: the model was trained on 1280-px tiles with NO resize.
# Run inference at imgsz=1280 for best small-object recall.
results = model.predict("your_image.jpg", imgsz=1280, conf=0.25)

for r in results:
    for box in r.boxes:
        cls = model.names[int(box.cls)]   # "face" or "license-plate"
        print(cls, box.xyxy[0].tolist(), float(box.conf))
```

For anonymisation you would blur every returned box. *In case of doubt, prefer recall*:
lower `conf` catches more faces/plates at the cost of a few extra blurs.

---

## Results (test split)

<!-- RESULTS:BEGIN -->
_Populated after evaluation (`notebooks/06_evaluate.ipynb`)._
<!-- RESULTS:END -->

Targets: `face` recall ≥ 0.95 & precision ≥ 0.85; `license-plate` recall ≥ 0.90 &
precision ≥ 0.90 (production goal: plate recall ≥ 0.98).

---

## Repository layout

```
.
├── LICENSE                 # AGPL-3.0
├── README.md               # this file
├── MODEL_CARD.md           # human-readable model + training description
├── .gitattributes          # Git LFS rules (*.pt, *.onnx, …)
├── .gitignore              # keeps images / data / secrets OUT
├── models/                 # trained weights (Git LFS)
│   └── yolo26n_face_lp.pt   # newest model (this path is the stable download link)
├── notebooks/              # the full, reproducible training pipeline
│   ├── 00_setup.ipynb              # environment / CUDA check
│   ├── 01_crop.ipynb               # annotation-aware 1280² lossless cropping
│   ├── 02_photometric_augment.ipynb# pixel-level augmentation (PNG, lossless)
│   ├── 03_coco_to_yolo_split.ipynb # COCO→YOLO, leakage-free 70/15/15 split
│   ├── 03b_build_face_lp_dataset.ipynb # derive the 2-class (face+plate) dataset
│   ├── 04_batch_finder.ipynb       # REAL VRAM batch-size probe
│   ├── 05_train_yolo26n.ipynb      # training (400 ep, 20 warmup, patience 60)
│   └── 06_evaluate.ipynb           # test-split metrics + PASS/FAIL vs targets
└── scripts/                # helpers used by the notebooks
    ├── image_crop_augment.py
    ├── photometric_worker.py
    ├── batch_finder.py
    └── piheif_fix.py
```

> The notebooks reference a `ROOT` working directory (the machine they were trained on).
> Set the `ROOT` environment variable / edit the first cell to point at your own data
> directory. No dataset is shipped — bring your own annotated images in COCO format.

---

## How it was trained (short version)

1. Crop full photos into **1280×1280 lossless PNG tiles** (no resize, no compression).
2. **Photometric augmentation** (20× for face/plate tiles, 10× otherwise).
3. **Leakage-free** 70/15/15 split, stratified so both classes are in val & test.
4. Reduce to **2 classes** (`face`, `license-plate`) + ~12 % hard-negative backgrounds.
5. **Measure** the optimal batch size for the GPU (real probe).
6. Train **YOLO26n** from official pretrained weights: 400 epochs, 20 warmup, early stop
   at patience 60, SGD + cosine LR + AMP.
7. Evaluate on the **independent test split** and check against the target metrics.

Full details and exact hyper-parameters: **[MODEL_CARD.md](MODEL_CARD.md)**.

---

## License

This project is licensed under the **GNU AGPL-3.0** — see [LICENSE](LICENSE). If you run a
modified version of this model as a network service, AGPL requires you to make your
source available to its users.
