# YOLO26 — Face & License-Plate Detection

[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-YOLO26m-success)](MODEL_CARD.md)
[![Input](https://img.shields.io/badge/Input-640×640-orange)]()
[![mAP50](https://img.shields.io/badge/mAP50-0.681-brightgreen)](MODEL_CARD.md)

A **YOLO26** detector that finds **faces** and **vehicle license-plates**, trained by
HTW Berlin students for the **"Privacy by Upload — Automated Image Anonymization"**
project (in collaboration with Parry). The model locates the regions that must be blurred
before an uploaded image is stored.

This repository exists to satisfy the **AGPL-3.0** share-alike obligation of
YOLO26/Ultralytics: it publishes the **training code** and the **resulting model
weights**. It deliberately contains **no training images, no annotations, no API keys,
and no server credentials** — only code and the model.

> **Current release: `v3.0.0`** (Sprint 3) — YOLO26 **medium** (`yolo26m`), trained on
> 1280 px tiles down-scaled to `imgsz=640`. Retrained on a much larger, merged dataset
> and a self-healing training pipeline (auto VRAM-ladder, crash-safe resume — see
> [MODEL_CARD.md](MODEL_CARD.md)). **mAP50 jumped 0.414 → 0.681** and every per-class
> recall/precision improved substantially over `v2.1.1` — see Results below. Still a
> **base model**: face recall is on track but not yet at the acceptance target.

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
https://github.com/H-Held/yolo26-license-plate-face-detection/releases/download/v3.0.0/yolo26m_face_lp.pt
https://github.com/H-Held/yolo26-license-plate-face-detection/raw/v3.0.0/models/yolo26m_face_lp.pt
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

# This model was trained on 1280 px tiles downscaled to 640 -> infer at imgsz=640.
# Recommended per-class confidence (balanced: highest F1, precision & recall together):
#   face = 0.222   license-plate = 0.263
results = model.predict("your_image.jpg", imgsz=640, conf=0.222)

CONF = {"face": 0.222, "license-plate": 0.263}
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
**This is a BASE model (`v3.0.0`, yolo26m @ 640, Sprint 3)**, retrained on a much larger,
merged dataset than `v2.1.1`. Evaluated on the independent test split
(`toolkit/run.py train`'s evaluation stage):

| Class | Precision | Recall | AP@50 | AP@50-95 |
|---|---|---|---|---|
| `face` | 0.765 | 0.611 | — | — |
| `license-plate` | 0.924 | 0.703 | — | — |
| **overall (mAP)** | | | **0.681** | **0.361** |

> P/R are reported at Ultralytics' evaluation confidence. For anonymisation use the
> **per-class thresholds** above (face 0.222 / plate 0.263), which trade precision for recall.

**Accuracy across every release:**

![Model accuracy across releases](metrics/accuracy_history.png)

`license-plate` precision now clears the acceptance target and recall is close; `face`
recall is still below target. More annotated faces (especially close/large ones) remain
the main lever for the next round — see [MODEL_CARD.md §7](MODEL_CARD.md#7-results) for
the full history.
<!-- RESULTS:END -->

Targets (future goal): `face` recall ≥ 0.95 & precision ≥ 0.85; `license-plate` recall ≥
0.90 & precision ≥ 0.90 (production goal: plate recall ≥ 0.98).

---

## Configuring & retraining (one file)

The whole pipeline is driven by **[`toolkit/config/global.yaml`](toolkit/config/global.yaml)**
— edit that one file, no internals needed:

- **model size** `model_size: n | s | m | l | x`, or warm-start from an existing checkpoint via `init_weights`;
- **classes** as a list (`nc` is counted automatically);
- **image size**, augmentation cap, VRAM ladder, and dataset registry (`toolkit/config/datasets/*.yaml`,
  add a dataset = drop a YAML file, no code change).

A single CLI drives everything: `python toolkit/run.py {check,build,find-batch,train,status,all}`.
Training runs in two phases automatically: **coarse** until early-stop, then an
**automatic fine-tune** of the best checkpoint. The batch size is chosen by a **real VRAM
probe** under actual multi-GPU DDP, with a self-lowering ladder if training later hits a
genuine out-of-memory. See **[toolkit/README.md](toolkit/README.md)** and
**[toolkit/docs/HOWTO_TRAIN_YOUR_OWN.md](toolkit/docs/HOWTO_TRAIN_YOUR_OWN.md)** for the
full walkthrough.

---

## Repository layout

```
.
├── LICENSE                 # AGPL-3.0
├── README.md               # this file
├── MODEL_CARD.md           # human-readable model + training description
├── CHANGELOG.md            # per-release change log
├── .env                    # release metadata (classes, tiling, conf thresholds, CHECK_FULL_IMAGE, metrics)
├── .gitattributes          # Git LFS rules (*.pt, *.onnx, …)
├── models/
│   └── yolo26m_face_lp.pt    # newest model (this path is the stable download link)
├── metrics/
│   ├── metrics_history.csv     # one row per release (accuracy over versions)
│   ├── accuracy_history.png    # chart of the above, regenerated by plot_history.py
│   └── plot_history.py         # local, offline chart generator (no external services)
└── toolkit/                 # the full, reproducible, config-driven training pipeline
    ├── run.py                  # <- the CLI: check / build / find-batch / train / status / all
    ├── config/global.yaml      # <- the one file you edit for the model itself
    ├── config/datasets/        # one YAML per dataset (registry, format-agnostic)
    ├── src/                    # adapters, tiling, augmentation, batch-finder, train engine
    ├── recovery/                # self-healing: server watchdog + Windows resume watchdog
    └── docs/                   # setup, training and release guides
```

> Edit `toolkit/config/global.yaml` (`data_root`) to point at your own data directory. No
> dataset is shipped — bring your own annotated images (COCO, YOLO-txt, or face-bbox format).

---

## License

This project is licensed under the **GNU AGPL-3.0** — see [LICENSE](LICENSE). If you run a
modified version of this model as a network service, AGPL requires you to make your
source available to its users.
