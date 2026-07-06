# YOLO26 — Face & License-Plate Detection

[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-YOLO26m-success)](MODEL_CARD.md)
[![Input](https://img.shields.io/badge/Input-640×640-orange)]()

A **YOLO26** detector that finds **faces** and **vehicle license-plates**, trained by
HTW Berlin students for the **"Privacy by Upload — Automated Image Anonymization"**
project (in collaboration with Parry). The model locates the regions that must be blurred
before an uploaded image is stored.

This repository exists to satisfy the **AGPL-3.0** share-alike obligation of
YOLO26/Ultralytics: it publishes the **training code** and the **resulting model
weights**. It deliberately contains **no training images, no annotations, no API keys,
and no server credentials** — only code and the model.

> **Current release: `v2.1.0`** — YOLO26 **medium** (`yolo26m`), trained on 640×640 tiles
> (infer at `imgsz=640`). New: `CHECK_FULL_IMAGE` flag in `.env` — consumers can now run
> inference on the **complete image** (letterboxed) in addition to tiles, catching
> large objects that span multiple tiles. Still a **base model** (data-limited,
> targets not yet met — see Results).

---

## Detected classes

| Index | Label | Description |
|------:|-------|-------------|
| `0` | `face` | Human faces |
| `1` | `license-plate` | Vehicle registration plates |

See **[MODEL_CARD.md](MODEL_CARD.md)** for the full, human-readable description of the
data, tiling/compression, and every training hyper-parameter.

---

## Download the model

The trained weights are tracked with **Git LFS** under [`models/`](models/) and published
as **GitHub Releases**. The model filename is the **size-neutral, stable**
`yolo26m_face_lp.pt` — the version lives in the git tag / release, so these links never
break.

**Always get the latest model from one stable link** (no login required):

```
https://github.com/H-Held/yolo26-license-plate-face-detection/releases/latest/download/yolo26m_face_lp.pt
```

```bash
# command-line download of the newest model
curl -L -o yolo26m_face_lp.pt \
  https://github.com/H-Held/yolo26-license-plate-face-detection/releases/latest/download/yolo26m_face_lp.pt
```

**Pinned / reproducible versions** are published as Git **tags** / Releases (e.g.
`v1.1.0`):

```
https://github.com/H-Held/yolo26-license-plate-face-detection/releases/download/v2.1.0/yolo26m_face_lp.pt
https://github.com/H-Held/yolo26-license-plate-face-detection/raw/v2.1.0/models/yolo26m_face_lp.pt
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

model = YOLO("yolo26m_face_lp.pt")

# This model was trained on 640 px tiles -> infer at imgsz=640.
# Recommended per-class confidence (recall-first: better over- than under-detect):
#   face = 0.391   license-plate = 0.625
results = model.predict("your_image.jpg", imgsz=640, conf=0.391)

CONF = {"face": 0.391, "license-plate": 0.625}
for r in results:
    for box in r.boxes:
        cls = model.names[int(box.cls)]      # "face" or "license-plate"
        if float(box.conf) >= CONF[cls]:     # apply the per-class threshold
            print(cls, box.xyxy[0].tolist(), float(box.conf))
```

For anonymisation you blur every returned box. *In case of doubt, prefer recall*: these
thresholds are deliberately low so more faces/plates are caught at the cost of a few extra
blurs. The exact per-class thresholds also ship in the release `.env` as
`CONF_THRESHOLDS` (`index:conf`, parallel to `CLASSES`) and are logged for every training
run in `runs/best_conf_log.csv`.

### Full-image inference (`CHECK_FULL_IMAGE`)

Since `v2.1.0`, the `.env` ships a `CHECK_FULL_IMAGE=true` flag. Consumers that support
this flag should, **in addition to tiled inference**, letterbox the **entire source image**
to `imgsz=640` and run the model on that single frame as well. This catches large
objects (e.g. a face that fills most of the photo) that would be fragmented across tiles
and possibly missed. The full-image pass produces the same YOLO output format — merge it
with the tile detections and apply NMS/deduplication as needed.

---

## Results (test split)

<!-- RESULTS:BEGIN -->
**This is a BASE model (`v2.1.0`, yolo26m @ 640)** trained on a small dataset
(~306 source photos) as a starting point — it will be retrained as more images are
collected. Evaluated on the independent test split (`notebooks/06_evaluate.ipynb`):

| Class | Precision | Recall | AP@50 | AP@50-95 |
|---|---|---|---|---|
| `face` | 0.503 | 0.500 | — | — |
| `license-plate` | 0.538 | 0.438 | — | — |
| **overall (mAP)** | | | **0.414** | **0.220** |

> P/R are reported at Ultralytics' evaluation confidence. For anonymisation use the
> **per-class thresholds** above (face 0.391 / plate 0.625), which trade precision for recall.

**The acceptance targets below are NOT yet met** — expected at this stage. The dataset is
small and dominated by tiny objects (≈47 % of faces and 40 % of plates are < 32 px), which
caps recall. More (and larger) annotated faces/plates are the main lever; that is the
purpose of the next data-collection + retraining round.
<!-- RESULTS:END -->

Targets (future goal): `face` recall ≥ 0.95 & precision ≥ 0.85; `license-plate` recall ≥
0.90 & precision ≥ 0.90 (production goal: plate recall ≥ 0.98).

---

## Configuring & retraining (one file)

The whole pipeline is driven by **[`scripts/config.py`](scripts/config.py)** — edit that
one file, no internals needed:

- **model size** `MODEL_SIZE = n | s | m | l | x`, or build on an existing checkpoint via `INIT_FROM`;
- **classes** as a dict (`nc` is counted automatically);
- **image size**, **augmentation**, schedule, and `GPUS = "all"` (use every GPU of the host).

Training (notebook `05`) runs in two phases automatically: **coarse** training until early
stop, then an **automatic fine-tune** of the best checkpoint. Evaluation (notebook `06`)
reports test metrics **and** the best confidence threshold **per class** (from test+val
combined). For an unattended, disconnect-proof run of multiple models on all GPUs, use
`scripts/run_campaign.py` (notebook `07`).

---

## Repository layout

```
.
├── LICENSE                 # AGPL-3.0
├── README.md               # this file
├── MODEL_CARD.md           # human-readable model + training description
├── .env                    # release metadata (classes, tiling, conf thresholds, CHECK_FULL_IMAGE, metrics)
├── .gitattributes          # Git LFS rules (*.pt, *.onnx, …)
├── models/
│   └── yolo26m_face_lp.pt    # newest model (this path is the stable download link)
├── metrics/metrics_history.csv  # one row per release (accuracy over versions)
├── notebooks/              # the full, reproducible pipeline (00–07)
│   ├── 00_setup … 03b      # data prep (crop, augment, split, 2-class dataset)
│   ├── 04_batch_finder     # REAL VRAM batch-size probe (DDP-aware)
│   ├── 05_train            # config-driven training + auto fine-tune
│   ├── 06_evaluate         # test metrics + per-class best confidence
│   └── 07_campaign         # unattended multi-model campaign (all GPUs)
└── scripts/                # config + engine used by the notebooks
    ├── config.py           # <- the one file you edit
    ├── train_pipeline.py   # multi-GPU train/finetune/eval/per-class-conf engine
    ├── batch_finder.py     # real DDP VRAM probe
    ├── build_resized_dataset.py
    ├── run_campaign.py · supervise_campaign.sh
    ├── image_crop_augment.py · photometric_worker.py · piheif_fix.py
```

> The notebooks reference a `ROOT` working directory. Edit `scripts/config.py` to point at
> your own data directory. No dataset is shipped — bring your own annotated images in COCO format.

---

## License

This project is licensed under the **GNU AGPL-3.0** — see [LICENSE](LICENSE). If you run a
modified version of this model as a network service, AGPL requires you to make your
source available to its users.
