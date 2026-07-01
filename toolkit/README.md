# YOLO training toolkit

A small, generic pipeline to train a YOLO detector from **any number of datasets**.
You add data by dropping a config file — no code changes. It splits each dataset
**before** augmenting (leakage-free), merges the splits, tiles + augments, finds the
biggest safe batch, and trains with **double crash-recovery**.

## The only two things you edit
| File | Controls |
|------|----------|
| `config/global.yaml` | the model: size, image size, epochs, classes, run name |
| `config/datasets/*.yaml` | each dataset: where it is, its labels, split, tiling, augment |

Private datasets go in `../.hiden/datasets/*.yaml` (git-ignored) instead — same format.

## The only command you run
```
python run.py check        # show config + resources (run this first)
python run.py build        # make the training dataset (CPU — small node is fine)
python run.py find-batch   # biggest batch under the VRAM limit (GPU node only)
python run.py train        # train, auto-resumes after a crash (GPU node)
python run.py all          # build + train
```

## Where to read next
- **`docs/HOWTO_TRAIN_YOUR_OWN.md`** — step-by-step, start to finish.
- **`docs/HANDOFF.md`** — the operating contract (what runs where, recovery, gotchas).
- **`docs/RELEASE_GUIDE.md`** — how to ship the trained model.

## Verify it works (no GPU needed)
```
python tests/run_smoke.py   # builds a synthetic dataset + checks bbox math end-to-end
```
