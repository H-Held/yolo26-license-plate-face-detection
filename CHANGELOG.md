# Changelog

All notable changes to the YOLO26 Face & License-Plate Detector model and metadata.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[2.1.0]: https://github.com/H-Held/yolo26-license-plate-face-detection/releases/tag/v2.1.0
[2.0.0]: https://github.com/H-Held/yolo26-license-plate-face-detection/releases/tag/v2.0.0
[1.1.0]: https://github.com/H-Held/yolo26-license-plate-face-detection/releases/tag/v1.1.0
[1.0.0]: https://github.com/H-Held/yolo26-license-plate-face-detection/releases/tag/v1.0.0