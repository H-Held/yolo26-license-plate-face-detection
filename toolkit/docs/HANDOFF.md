# Handoff — operating contract for the toolkit

For the next operator or AI agent. Read this before touching a live run.

## Mental model
- **One config decides the model** (`config/global.yaml`); **one config per dataset** decides
  the data (`config/datasets/*.yaml`, plus git-ignored `../.hiden/datasets/*.yaml`).
- **One CLI** does everything (`python run.py {check,build,find-batch,train,all}`).
- The pipeline order is fixed and leakage-safe: **split each dataset → merge → tile+augment →
  batch → train**. Splitting is a stable hash of the source image id, done before augment, so
  no crop of a train image can appear in val/test.

## What runs where
| Task | Node | Why |
|------|------|-----|
| `build` (make dataset) | small / no-GPU | pure CPU; cheaper |
| `find-batch`, `train` | GPU-large | needs the real GPU; the finder must run on the training GPU |

## The two datasets categories
- **official** = configs in `config/datasets/` — may appear in git.
- **hidden** = configs in `../.hiden/datasets/` — git-ignored. The committed code contains only
  **format** readers (`coco`, `facebbx`, `imagefolder`) and never names a specific external
  dataset. Anything private (which set, where, its presence-file meaning, extraction and
  pseudo-label commands) lives only in `../.hiden/HIDDEN_DATASETS.md`. Keep it that way: never
  put a hidden dataset's name, URL, or path into a committed file.

## Double crash-recovery (the "two restarts")
1. **In-pod (restart #1):** `recovery/server/start_robust.sh` launches
   `hub_keepalive.py` (anti-cull) + `watchdog.py` (runs training, diagnoses crashes to
   `runs/crashreports/`, resumes from `last.pt`, stops on `stages.complete`).
2. **Your PC (restart #2):** `recovery/windows/program2_resume_watchdog.py` reads
   `recovery.json` (copy `recovery.json.example`, set `run_name` + paths), and if the whole pod
   is culled it restarts the GPU server and re-launches the in-pod system. It stops when the run
   reports complete.
Both are idempotent: re-running never starts a second training.

## Completion signal
`data_root/runs/faces/<run_name>/pipeline_state.json` → `{"stages":{"complete":true}}`.
Everything (both watchdogs) keys off this. Delete/rename it to force a fresh run.

## Gotchas / must-not-break
- **Leakage invariant:** never tile or augment before the split; never split on anything but the
  source image id. `build` asserts no leakage and will abort if broken.
- **Batch finder measures TRUE board memory** (nvidia-smi peak across all GPUs), not torch's
  reserved number — the latter under-reads by ~1.5 GB and once put a "verified 84%" run at a
  real 97%. Keep `vram_probe_buffer` (~0.83) and `probe_iters` (~40); do not "simplify" the
  probe to a tiny synthetic subset.
- **`src/batch_finder.py` + `src/train.py` are RECONSTRUCTIONS** from the known-good spec and
  were **not runnable on GPU in the session that wrote them**. A battle-tested equivalent runs
  on the server (`deepseek/scripts/`). **Diff against that before trusting the toolkit copy on
  the cluster**; do not silently replace a working server script.
- **Dataset-size explosion:** native × scaled × photometric × class_boost multiplies fast. Watch
  `build`'s printed chip counts and keep `max_pos_tiles` / `hardneg_frac` caps. `preflight`
  checks disk/RAM and gives a plain-language error before you run out.
- **The `imagefolder` pseudo-label path needs the GPU model** to create the boxes sidecar first;
  `build` will raise a clear error pointing at `src/pseudo_label.py` if the sidecar is missing.

## Verifying without a GPU
`python tests/run_smoke.py` builds a synthetic 3-dataset set and asserts bbox transforms are
pixel-exact, discovery finds official+hidden, there is no leakage, all labels are in [0,1], and
the weak-class boost fired. Run it after any change to the data-prep code.
