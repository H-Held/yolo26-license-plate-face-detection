# Changelog

All notable changes to the YOLO26 Face & License-Plate Detector model and metadata.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] — 2026-07-13

### Added

- Retrained on a substantially **expanded, merged dataset** (the project's own photos plus
  additional annotated sources — external dataset identities are not disclosed, per the
  project's privacy policy) using the generic config-driven `toolkit/` pipeline.
- Self-healing training infrastructure: automatic **VRAM ladder** that lowers itself and
  re-probes the batch size if real training hits a genuine out-of-memory; a server-side
  watchdog that auto-resumes training after a crash; a Windows-side watchdog that can
  resume the JupyterHub server itself if the pod is culled mid-training.
- `metrics/plot_history.py` + `metrics/accuracy_history.png`: a locally-rendered chart of
  accuracy across every release (no external services), embedded in `README.md` and
  `MODEL_CARD.md`.
- `toolkit/run.py status`: one-shot, read-only training-progress report.

### Changed

- Tile pipeline switched back to **1280 px source tiles down-scaled to 640 px** (was native
  640 px tiles in v2.0.0/v2.1.x) — full resolution is preserved before the down-scale.
- Per-class confidence thresholds now optimise for **highest F1** (balanced precision/recall)
  instead of recall-first: `face` 0.391 → **0.222**, `license-plate` 0.625 → **0.263**.
- `README.md`, `MODEL_CARD.md`: version references updated to v3.0.0; repository layout
  section now documents the `toolkit/` pipeline (superseding the old `notebooks/`/`scripts/`
  layout, which is no longer part of the published pipeline).
- `metrics/metrics_history.csv`: added v3.0.0 row; fixed a missing `compression_quality`
  column in the header that had silently misaligned every value after `OUTPUT_QUALITY` in
  prior rows.
- `.env`: `VERSION` bumped to v3.0.0, `TILE_SIZE` to 1280, `CONF_THRESHOLDS` and all
  `METRIC_*` values updated.

### Metrics (vs v2.1.1)

| Class | Recall | Precision | mAP50 | mAP50-95 |
|---|---|---|---|---|
| face | 0.500 → **0.611** | 0.503 → **0.765** | | |
| license-plate | 0.438 → **0.703** | 0.538 → **0.924** | | |
| **overall** | | | 0.414 → **0.681** | 0.220 → **0.361** |

---

## [2.1.1] — 2026-07-06

### Fixed

- `.env`: corrected a metadata field-naming mistake (`COMPRESSION_FORMAT`/`COMPRESSION_QUALITY`
  swapped `640`/`50`) introduced in v2.1.0. Model weights and evaluation metrics unchanged.

---

## [2.1.0] — 2026-07-06

### Added

- `CHECK_FULL_IMAGE=true` flag in `.env` release metadata. Consumers that support this
  flag should run inference on the **complete source image** (letterboxed to `imgsz=640`)
  **in addition to** the normal tiled inference, then merge and deduplicate detections.
  This catches large objects (e.g. a face filling most of the photo) that would be
  fragmented across tiles and possibly missed.
- `CHANGELOG.md` (this file) documenting all releases.
- Full-image inference documentation in `README.md`.

### Changed

- `README.md`: version references updated from v2.0.0 to v2.1.0.
- `MODEL_CARD.md`: version references updated from v2.0.0 to v2.1.0.
- `metrics/metrics_history.csv`: added v2.1.0 row (metrics identical to v2.0.0 — model
  weights unchanged, metadata-only release).
- `.env`: `VERSION` and `lowest_compatible_SoftwareVersion` bumped to v2.1.0.

---

## [2.0.0] — 2026-06-30

### Changed

- Switched from YOLO26 **large** (`yolo26l`) to YOLO26 **medium** (`yolo26m`).
- **Tile size** reduced from 1280 to **640 px** for native tiles (no compression).
- Tile overlap reduced from 20% to 15%.
- Disabled compression (`COMPRESSED=false`); previously v1.1.0 downscaled to 640 px.

### Metrics (vs v1.1.0)

| Class | Recall | Precision | mAP50 | mAP50-95 |
|---|---|---|---|---|
| face | 0.165 → **0.500** | 0.781 → 0.503 | | |
| license-plate | 0.407 → **0.438** | 0.700 → 0.538 | | |
| **overall** | | | 0.302 → **0.414** | 0.160 → **0.220** |

---

## [1.1.0] — 2026-06-23

### Changed

- Upgraded from YOLO26 **nano** (`yolo26n`) to YOLO26 **large** (`yolo26l`).
- Added compression: 1280 px tiles **downscaled to 640 px** (`COMPRESSED=true`,
  `COMPRESSION_FORMAT=downscale`, `COMPRESSION_QUALITY=640`).
- Tile overlap reduced from 20% to 20% (unchanged).

### Metrics (vs v1.0.0)

| Class | Recall | Precision | mAP50 | mAP50-95 |
|---|---|---|---|---|
| face | 0.124 → **0.165** | 0.855 → 0.781 | | |
| license-plate | 0.401 → **0.407** | 0.720 → 0.700 | | |
| **overall** | | | 0.288 → **0.302** | 0.147 → **0.160** |

---

## [1.0.0] — 2026-06-22

### Added

- Initial public release: YOLO26 **nano** (`yolo26n`) face & license-plate detector.
- 2 classes: `face` (0), `license-plate` (1).
- Tiling at 1280×1280 px with 20% overlap, no compression.
- Training pipeline: notebooks 00–07, config-driven multi-GPU training.
- AGPL-3.0 license.
- Base metrics: face R=0.124 P=0.855, plate R=0.401 P=0.720, mAP50=0.288, mAP50-95=0.147.

---

[3.0.0]: https://github.com/H-Held/yolo26-license-plate-face-detection/releases/tag/v3.0.0
[2.1.1]: https://github.com/H-Held/yolo26-license-plate-face-detection/releases/tag/v2.1.1
[2.1.0]: https://github.com/H-Held/yolo26-license-plate-face-detection/releases/tag/v2.1.0
[2.0.0]: https://github.com/H-Held/yolo26-license-plate-face-detection/releases/tag/v2.0.0
[1.1.0]: https://github.com/H-Held/yolo26-license-plate-face-detection/releases/tag/v1.1.0
[1.0.0]: https://github.com/H-Held/yolo26-license-plate-face-detection/releases/tag/v1.0.0