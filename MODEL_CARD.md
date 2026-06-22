# Model Card — YOLO26n Face & License-Plate Detector

A human-readable description of **what this model detects, how it was trained, and on
what kind of data** — written so a person (not a machine) can understand and reproduce
the setup. No images, annotations, API keys, or server credentials are part of this
repository (privacy by design).

---

## 1. What the model does

| | |
|---|---|
| **Architecture** | YOLO26 **nano** (`yolo26n`), the smallest/fastest YOLO26 variant |
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

## 2. Input format (important — no compression)

| Property | Value | Why |
|---|---|---|
| **Tile size** | **1280 × 1280 px** | High enough resolution that small/distant plates and faces survive |
| **`imgsz` at train/infer** | **1280** | Equals the tile size — **no runtime resize** |
| **File format** | **PNG, lossless** | **No JPEG/lossy compression** anywhere in the pipeline — compression artifacts would erase the very small plate/face details we need |
| **Color** | RGB | |

Full-resolution source photos are **cropped into 1280×1280 tiles** rather than shrunk,
so a tiny license-plate stays as many pixels as it had in the original photo.

---

## 3. How the data was prepared

The data pipeline (notebooks `01`–`03`) turns raw annotated photos into a training set:

1. **Annotation-aware cropping** (`01`): each source photo is cut into lossless
   1280×1280 PNG tiles. Crops are placed so every annotated object is fully covered;
   objects less than 10 % visible at a tile edge are dropped/blacked-out.
2. **Photometric augmentation** (`02`): each tile gets several pixel-level variants
   (brightness, noise, blur, fog, simulated compression *as a pixel effect only* —
   files stay lossless PNG). Tiles containing a face/plate are augmented **20×**, other
   tiles **10×**, to enrich the two target classes. Bounding boxes are unchanged.
3. **COCO → YOLO + split** (`03`): converted to YOLO label format and split
   **70 / 15 / 15** into train / val / test. The split is done **per source photo**
   (all tiles/variants of one photo stay in a single split) so there is **no leakage**
   between train and test, and it is **stratified** so both classes appear in val *and*
   test.
4. **2-class filtering**: labels are reduced to `face` + `license-plate`; the dataset
   used for this model is `dataset_face_lp` (`nc=2`).

> **No source images or labels are published.** Only the resulting model weights are
> shared, under AGPL-3.0.

---

## 4. How the model was trained

Notebook `05_train_yolo26n.ipynb`. This base model is a **fine-tune of the previous
training run's `best.pt`** (warm start), with a short fully-decaying schedule.

| Hyper-parameter | Value |
|---|---|
| Model | YOLO26n |
| Init weights | **previous run's `best.pt`** (warm start / fine-tune, not the raw pretrained) |
| Image size (`imgsz`) | 1280 |
| Epochs | **50** (short schedule, run in full so the cosine LR fully decays) |
| Warmup epochs | **1** (then the rest of the run is the cosine fine-tuning tail) |
| Early stopping (`patience`) | **50** = epochs (no premature stop — we want the full LR decay) |
| Batch size | **measured per GPU** by the real batch-finder (notebook `04`) — never guessed (here: 30) |
| Optimizer | SGD, `lr0=0.005` (gentle fine-tune), `lrf=0.01`, momentum `0.937`, weight-decay `5e-4` |
| LR schedule | cosine (`cos_lr=True`) |
| Mixed precision | AMP on |
| Image caching | `cache='ram'` (decode once → GPU-bound, fast epochs) |
| Geometric augmentation | `fliplr=0.5`, **`flipud=0.0`** (faces/plates are never upside-down), `degrees=10`, **`scale=0.1`** (small — nearly half the objects are < 32 px), `translate=0.1`, `shear=2`, `perspective=2e-4`; **mosaic/mixup/copy_paste disabled** (they downscale and hurt small-object recall, and multiply I/O) |
| Color augmentation | `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4` |
| Hardware | 1× NVIDIA A30 (MIG slice, ~24 GB) |

> **This is a BASE model.** It was trained on a small dataset (~306 source photos) as a
> starting point; the plan is to expand the dataset with more images and retrain. See
> §6 for why the acceptance targets below are not yet met.

### The batch-finder (why it matters)

The batch size is determined by a **real VRAM probe** (notebook `04` +
`scripts/batch_finder.py`): it runs one genuine training pass at increasing batch sizes
in isolated subprocesses and keeps the largest that does not run out of memory. (An
earlier version of the pipeline shipped a *fake* finder that just echoed a hard-coded
value calibrated for a larger model — this has been replaced.)

---

## 5. Acceptance criteria (target metrics)

Evaluated on the independent **test** split. *In case of doubt, recall is prioritised
over precision* — a missed face/plate (privacy leak) is worse than an extra blur.

| Category | Min. Recall | Min. Precision |
|---|---|---|
| `face` | ≥ 0.95 | ≥ 0.85 |
| `license-plate` | ≥ 0.90 (MVP) / ≥ 0.98 (production goal) | ≥ 0.90 |

---

## 6. Results

<!-- RESULTS:BEGIN -->
Base model **v1.0.0**, evaluated on the independent **test** split (673 tiles):

| Class | Precision | Recall | AP@50 | AP@50-95 |
|---|---|---|---|---|
| `face` | 0.855 | 0.124 | 0.163 | 0.068 |
| `license-plate` | 0.720 | 0.401 | 0.413 | 0.227 |
| **overall (mAP)** | | | **0.288** | **0.147** |

**Target check: FAIL** (face recall 0.124 < 0.95; plate recall 0.401 < 0.90). This is
expected for a base model: the training data is small (~306 source photos; photometric
augmentation adds variety but not new scenes) and dominated by tiny objects (≈47 % of
faces and 40 % of plates are < 32 px), which caps a *nano* model's recall. The next
iteration will add more annotated images (especially larger/closer faces and plates) and
retrain. Recall is reported at the max-F1 confidence; lowering `conf` at inference raises
recall (preferred for anonymisation).
<!-- RESULTS:END -->

---

## 7. License & intended use

- **License:** AGPL-3.0 (this code and the trained weights). YOLO26/Ultralytics is
  itself AGPL-3.0; this repository exists to fulfil that share-alike obligation by
  publishing the training code and the resulting model.
- **Intended use:** automated anonymisation (blurring) of faces and license-plates.
- **Not intended for:** identification, tracking, or any surveillance use.
