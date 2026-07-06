# How to train your own YOLO model — step by step

Written so that a non-expert (or a simple AI agent) can follow it top to bottom.
You never edit code. You edit **config files** and run **`python run.py …`**.

The pipeline always runs in this fixed order (this order matters and is enforced):

```
   split each dataset  →  merge splits  →  tile + augment  →  find batch  →  train
   (BEFORE augment, so no image leaks between train/val)
```

---

## Which machine do I use?
- **Making the dataset** (`build`) is CPU work → use the **small (no-GPU) node**. It is
  cheaper and enough.
- **`find-batch` and `train`** need a GPU → use the **GPU-large node**. The batch finder
  MUST run on the same GPU you will train on, or its number is meaningless.

---

## STEP 0 — Install (once, on each node)
```
cd toolkit
pip install -r requirements.txt
```
The data-prep code also runs with just numpy + Pillow if OpenCV is missing.

## STEP 1 — Set the model
Open `config/global.yaml` and set:
- `model_size` (n/s/m/l/x), `imgsz` (e.g. 640), `epochs`, `patience`
- `classes` — the list of things to detect (index = class id)
- `data_root` — the base folder your data lives under (or set env `YOLO_TOOLKIT_DATA_ROOT`)
- `run_name` — a name for this training run

## STEP 2 — Point the toolkit at your data
Copy `config/datasets/own_export.yaml` and edit it for your dataset:
- `format`: `coco` (a COCO json), `facebbx` (line-based face boxes), or
  `imagefolder` (a folder of images; boxes come from a sidecar / pseudo-label / whole-image)
- `images_dir`, `labels`
- `class_map`: map each of THIS dataset's class names onto a name in `global.yaml`
- `split`: train/val fractions + a fixed `seed`
- `tiling` and `augment`: see the two boxes below

You can add as many dataset files as you like. Each is split **separately** then merged.

### Tiling (three regimes — all optional per dataset)
| key | what it does |
|-----|--------------|
| `native_tiles: true` | cut the image into imgsz-sized tiles (keeps small-object detail) |
| `scaled_tiles: [2,3]` | cut 2×/3× windows and shrink each to imgsz (adds context) |
| `whole_image: true` | **always** add the whole image shrunk to imgsz — use this for datasets you do NOT want tiled (e.g. an extra face set) |
| `whole_if_span_tiles: 4` | if a box is so big it would span **more than 4** tiles, also add the **whole image** shrunk to imgsz (so the model still sees the whole object) |

> A "whole-image dataset" (no tiling) needs `native_tiles: false`, `scaled_tiles: []`,
> **`whole_image: true`**. Without `whole_image: true`, images larger than imgsz would be
> dropped entirely.

### Augment (two independent things)
- **Global variety**: `geometric: true` (rotate/flip/upside-down) and
  `photometric_variants: N` (over/under-expose, gamma, weird colour palette).
- **Weak-class boost**: `class_boost: {your-rare-class: 3}` makes 3× as many samples that
  contain that rare class — this **rebalances** a lopsided dataset (variety alone won't).

### Build on an existing model instead of training from zero (warm-start)
Set `init_weights:` in `config/global.yaml` to a `.pt` file to **start a fresh run from your
own model** (transfer learning / continue improving) instead of from COCO. Example: point it at
a previous run's `runs/faces/<old_run>/weights/best.pt` to fine-tune that model on new/extra
data. Leave it `""` to start from the official pretrained weights. (An interrupted run always
auto-resumes from its own `last.pt` — warm-start only affects a brand-new run.)

## STEP 3 — Sanity check
```
python run.py check
```
Shows the model settings, every dataset it found (official + hidden), and your disk / RAM /
GPU. If a dataset you expect is missing, its yaml is in the wrong folder or `enabled: false`.

## STEP 4 — Build the dataset (small node)
```
python run.py build
```
It prints per-dataset split counts, checks there is **no leakage**, writes the YOLO dataset
under `data_root/<out_dataset>/`, and prints how many chips + boxes-per-class you got.
Watch the box counts: that is where you confirm the weak class got boosted.

## STEP 5 — Find the batch size (GPU node)
```
python run.py find-batch
```
Trains short probes on your **real** dataset while watching **true GPU memory across every
GPU**, and picks the largest batch that stays under `vram_target_pct` (default 85%). Works
with 1 GPU or many. If it can't fit even the smallest batch, it tells you to lower
`imgsz`/`model_size`. The result is cached in `runs/optimal_batch_<imgsz>.json`.

## STEP 6 — Train (GPU node, crash-proof)
Do **not** just run `python run.py train` in a bare terminal for a long run — a pod cull
would kill it. Instead start the self-healing launcher:
```
bash recovery/server/start_robust.sh
```
This starts (1) a **keep-alive** so the hub doesn't cull you, and (2) an **in-pod watchdog**
that runs training, and if it ever crashes, diagnoses why and **resumes from the last
checkpoint**. For the second safety net (pod gets culled entirely), run **Program 2** on your
PC — see `HANDOFF.md`.

## STEP 7 — Done
Training writes `runs/faces/<run_name>/pipeline_state.json` with `stages.complete = true`.
Both watchdogs see that and stop. Your weights are in
`runs/faces/<run_name>/weights/best.pt`. To ship them, see `RELEASE_GUIDE.md`.

---

## Adding a brand-new dataset later (the whole point)
1. Put its images + labels under `data_root`.
2. Drop a new `config/datasets/<name>.yaml` (copy an existing one, change the paths + `format`
   + `class_map`).
3. `python run.py build` → `python run.py train`. That's it — the split/merge/leakage/tiling
   all happen automatically for the new set too.

## If some images have NO boxes
Use `format: imagefolder` with `label_source: pseudo_label`, then run the GPU pseudo-labeler
once to create the boxes sidecar:
```
python -m src.pseudo_label --images <dir> --weights <model.pt> --out <sidecar.json> \
    --conf 0.35 --plate-class <class> [--filter <presence.json> --keep 0]
```
`build` then reads that sidecar automatically.

## If some images would contain an UNLABELLED object
Give the dataset an `include_filter` pointing at a presence file (`{path: 0|1}`) and keep only
the safe value. That drops images that carry an object your labels don't cover, which would
otherwise teach the model that the object is "background".
