#!/usr/bin/env python
"""Real VRAM batch-size probe for YOLO26.

Runs ONE genuine 1-epoch training pass on a small probe subset with the full
Ultralytics pipeline (model + AMP + optimizer + augmentation) at a given batch
size, and reports peak reserved GPU memory. Exits with code 2 on CUDA OOM.

This is intentionally run as an isolated subprocess *per candidate batch size*
(see notebook 04) so that an OOM at one size cannot leave fragmented/leaked
memory that would corrupt the measurement of the next size. The previous
"batch finder" never probed anything -- it just echoed a stale cached value.

Usage: python batch_finder.py <batch> <imgsz> <weights> <data.yaml>
"""
import sys, os, json

os.environ.setdefault("YOLO_CONFIG_DIR", "/home/jovyan/shared/s0598584/ultralytics_cfg")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/jovyan/shared/s0598584/scripts")
import piheif_fix  # noqa: F401  (must precede ultralytics import)

import torch
from ultralytics import YOLO

bs = int(sys.argv[1]); imgsz = int(sys.argv[2]); weights = sys.argv[3]; data = sys.argv[4]

torch.cuda.reset_peak_memory_stats()
try:
    model = YOLO(weights)
    model.train(
        data=data, epochs=1, imgsz=imgsz, batch=bs, workers=2, device=0,
        cache=False, val=False, plots=False, verbose=False, amp=True,
        optimizer="SGD", warmup_epochs=0, close_mosaic=0,
        project="/home/jovyan/shared/s0598584/runs", name="_batch_probe", exist_ok=True,
    )
    peak = torch.cuda.max_memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("PROBE_OK " + json.dumps({"batch": bs, "peak_gb": round(peak, 2),
                                    "total_gb": round(total, 2)}))
except torch.cuda.OutOfMemoryError:
    print("PROBE_OOM " + str(bs)); sys.exit(2)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("PROBE_OOM " + str(bs)); sys.exit(2)
    raise
