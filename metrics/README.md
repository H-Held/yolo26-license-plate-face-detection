# Metrics history

`metrics_history.csv` tracks **one row per released model version** so accuracy can be
plotted over time. It is committed to the repo (the running history), separate from the
per-release `.env` (which describes a single version).

**Append a new row at every release** (after `06_evaluate`). Columns:

| Column | Meaning |
|---|---|
| `version` | release tag, e.g. `v0.2.0` |
| `date` | release date `YYYY-MM-DD` |
| `base_model` | always `yolo26` |
| `model_size` | `n` / `s` / `m` / `l` / `x` |
| `tiles_used` | `true` / `false` |
| `tile_size` | px (only meaningful if `tiles_used=true`) |
| `tile_overlap_pct` | tile overlap in % (only meaningful if `tiles_used=true`) |
| `compressed` | `true` / `false` |
| `compression_format` | e.g. `jpeg` (only if `compressed=true`) |
| `compression_quality` | e.g. `85` (only if `compressed=true`) |
| `num_classes` | number of classes |
| `face_recall`, `face_precision` | per-class metrics on the test split |
| `license_plate_recall`, `license_plate_precision` | per-class metrics on the test split |
| `map50`, `map50_95` | overall mAP@0.5 and mAP@0.5:0.95 |

Leave metric cells **empty** until measured. New metrics/classes → add new columns (keep
the header in sync). Values mirror the `METRIC_*` fields of that version's `.env`.

Quick plot:
```python
import pandas as pd
df = pd.read_csv("metrics/metrics_history.csv")
df.plot(x="version", y=["face_recall", "license_plate_recall", "map50_95"], marker="o")
```
