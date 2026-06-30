# Model Card — YOLO26 Face & License-Plate Detector

A human-readable description of **what this model detects, how it was trained, and on
what kind of data** — written so a person (not a machine) can understand and reproduce
the setup. No images, annotations, API keys, or server credentials are part of this
repository (privacy by design).

> **Version `v2.0.0`** — YOLO26 **medium** (`yolo26m`), trained on 640×640 tiles.
> Infer at `imgsz=640`.

---

## 1. What the model does

| | |
|---|---|
| **Architecture** | YOLO26 **medium** (`yolo26m`) |
| **Purpose** | Locate **faces** and **vehicle license-plates** in photos so they can be anonymized (blurred) automatically on upload |
| **Project** | Part of *"Privacy by Upload — Automated Image Anonymization"* (HTW Berlin, in collaboration with Parry) |
| **Framework** | [Ultralytics](https://docs.ultralytics.com) `8.4.70`, PyTorch `2.1.2` (CUDA) |

### Classes / labels

The model outputs exactly **two** classes. The class **index is the integer** in the
detection output; the **name** is what it means:

| Index | Label | Meaning |
|------:|-------|---------|
| `0` | `face` | A human face (any orientation typical for street imagery) |
| `1` | `license-plate` | A vehicle registration plate |

> The underlying annotation export contains more object categories (car, person,
> truck, …). For this model **all other categories were deliberately dropped** — only
> `face` and `license-plate` are trained. Tiles that contained *only* other objects are
> kept as **hard-negative backgrounds** (they teach the model *not* to fire on cars or
> people), at roughly a 12–13 % background ratio.

---

## 2. Recommended inference settings

| Property | Value | Why |
|---|---|---|
| **`imgsz`** | **640** | The model was trained on tiles down-scaled to 640 px |
| **conf `face`** | **0.391** | Recall-first: catch as many faces as possible |
| **conf `license-plate`** | **0.625** | Recall-first |
| **Color** | RGB | |

The per-class thresholds are chosen **recall-first** ("better over- than under-detect" —
a missed face/plate is a privacy leak): for each class we take the **highest-recall**
confidence whose precision still stays **≥ 0.5** on the test+validation set combined. They
ship in the release `.env` as `CONF_THRESHOLDS` (`index:conf`, parallel to `CLASSES`) and
are logged for every training run in `runs/best_conf_log.csv`.

---

## 3. Input / tiling / compression

| Property | Value | Why |
|---|---|---|
| **Source tile size** | **640 × 640 px** | Tiles used as-is for training |
| **Compression** | **none** | No down-scaling applied |
| **Model input (`imgsz`)** | **640** | Native tile size |

Full-resolution source photos are first **cropped into 640×640 tiles** (not shrunk), so a
tiny license-plate keeps its pixels. This model is trained directly on 640 px tiles without
further compression.

---

## 4. How the data was prepared

The data pipeline (notebooks `01`–`03b`) turns raw annotated photos into a training set:

1. **Annotation-aware cropping** (`01`): each source photo is cut into lossless
   1280×1280 PNG tiles, placed so every annotated object is fully covered.
2. **Photometric augmentation** (`02`): pixel-level variants (brightness, noise, blur,
   fog) — files stay lossless PNG, bounding boxes unchanged. Configurable factors in
   `config.py`.
3. **COCO → YOLO + split** (`03`): YOLO label format, split **70 / 15 / 15** into
   train / val / test **per source photo** (no leakage), stratified so both classes
   appear in val *and* test.
4. **2-class dataset** (`03b`): labels reduced to the configured classes + ~12 %
   hard-negative backgrounds → `dataset_face_lp` (`nc=2`).
5. **No down-scale needed**: tiles are already 640×640 → `dataset_face_lp`, the dataset
    this model was trained on.

> **No source images or labels are published.** Only the resulting model weights are
> shared, under AGPL-3.0.

---

## 5. How the model was trained

Config-driven (`scripts/config.py`); run via the campaign (`scripts/run_campaign.py`,
notebook `07`). Training is **two-phase and automatic**: a **coarse** phase until early
stop, then an **automatic fine-tune** of the best checkpoint.

| Hyper-parameter | Value |
|---|---|
| Model | YOLO26m (init from official `yolo26m.pt`) |
| Image size (`imgsz`) | 640 (native tiles) |
| Coarse phase | up to 400 epochs, early-stop `patience=50` |
| Fine-tune phase | **75 epochs** from the coarse `best.pt`, `lr0=0.005`, cosine tail |
| Batch size | **measured under real DDP** by the batch-finder (`04`): per-GPU 5 × GPUs |
| Optimizer | SGD, `lr0=0.01` (coarse), momentum `0.937`, weight-decay `5e-4`, cosine LR, AMP |
| Augmentation | `fliplr=0.5`, **`flipud=0.0`**, `degrees=10`, `scale=0.3` (coarse) / `0.1` (fine-tune), `translate=0.1`, `shear=2`, `perspective=2e-4`; **mosaic/mixup/copy_paste off**; HSV `0.015/0.7/0.4` |
| Hardware | **3× NVIDIA V100-PCIE-16GB** (DDP across all GPUs) |

> **This is a BASE model.** It was trained on a small dataset (~306 source photos) as a
> starting point; the plan is to expand the dataset and retrain. See §7 for why the
> acceptance targets are not yet met.

### The batch-finder (why it matters)

The batch size is determined by a **real VRAM probe** (`04` + `scripts/batch_finder.py`)
that runs genuine 1-epoch passes **under real multi-GPU DDP** at increasing per-GPU sizes
and keeps the largest that does not run out of memory — so the result already accounts for
DDP overhead (single-GPU probing over-promises and OOMs).

---

## 6. Acceptance criteria (target metrics)

Evaluated on the independent **test** split. *In case of doubt, recall is prioritised over
precision* — a missed face/plate (privacy leak) is worse than an extra blur.

| Category | Min. Recall | Min. Precision |
|---|---|---|
| `face` | ≥ 0.95 | ≥ 0.85 |
| `license-plate` | ≥ 0.90 (MVP) / ≥ 0.98 (production goal) | ≥ 0.90 |

---

## 7. Results

<!-- RESULTS:BEGIN -->
Model **v2.0.0** (`yolo26m` @ 640), evaluated on the independent **test** split:

| Class | Precision | Recall | AP@50 | AP@50-95 |
|---|---|---|---|---|
| `face` | 0.503 | 0.500 | — | — |
| `license-plate` | 0.538 | 0.438 | — | — |
| **overall (mAP)** | | | **0.414** | **0.220** |

**Target check: FAIL** (face recall 0.500 < 0.95; plate recall 0.438 < 0.90). This is
expected for a base model: the training data is small (~306 source photos) and dominated by
tiny objects (≈47 % of faces and 40 % of plates are < 32 px), which caps recall. The next
iteration will add more annotated images (especially larger/closer faces and plates) and
retrain. P/R are reported at the evaluation confidence; at inference use the **per-class
thresholds** in §2 (face 0.391 / plate 0.625) to raise recall.

*(Compared to v1.1.0 — yolo26l @ 640: face R 0.165→0.500, plate R 0.407→0.438,
mAP50 0.302→0.414.)*
<!-- RESULTS:END -->

---

## 8. License & intended use

- **License:** AGPL-3.0 (this code and the trained weights). YOLO26/Ultralytics is
  itself AGPL-3.0; this repository exists to fulfil that share-alike obligation by
  publishing the training code and the resulting model.
- **Intended use:** automated anonymisation (blurring) of faces and license-plates.
- **Not intended for:** identification, tracking, or any surveillance use.
