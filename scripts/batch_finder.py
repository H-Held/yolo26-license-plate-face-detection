#!/usr/bin/env python
"""Real VRAM batch-size probe for YOLO26 — single GPU OR full multi-GPU (DDP).

Runs ONE genuine 1-epoch training pass with the full Ultralytics pipeline
(model + AMP + optimizer + augmentation) at a given TOTAL batch size, and reports
whether it fits. Run as an isolated subprocess *per candidate size* (see
train_pipeline.find_optimal_batch) so an OOM at one size cannot leave
fragmented/leaked memory that corrupts the next measurement.

Crucially, when given several GPUs it probes under **real DDP across all of
them** — so the measured maximum already includes the per-GPU DDP overhead
(NCCL buffers + gradient buckets). That is why the finder can correctly land on,
say, per-GPU batch 5 when a single-GPU probe would have over-promised batch 6.

Usage: python batch_finder.py <total_batch> <imgsz> <weights> <data.yaml> [devices]
  devices = "0"      -> single GPU
            "0,1,2"  -> DDP across GPUs 0,1,2 (total_batch split across them)
"""
import sys, os, json
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/home/jovyan/shared/s0598584/ultralytics_cfg")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import piheif_fix  # noqa: F401  (must precede ultralytics import)

import torch
from ultralytics import YOLO

total_batch = int(sys.argv[1]); imgsz = int(sys.argv[2])
weights = sys.argv[3]; data = sys.argv[4]
devices = sys.argv[5] if len(sys.argv) > 5 else "0"
device = [int(x) for x in devices.split(",")] if "," in devices else int(devices)
single = not isinstance(device, list)

if single:
    torch.cuda.set_device(device)              # init context before querying stats
    torch.cuda.reset_peak_memory_stats(device)

try:
    model = YOLO(weights)
    model.train(
        data=data, epochs=1, imgsz=imgsz, batch=total_batch, device=device,
        workers=2, cache=False, val=False, plots=False, verbose=False, amp=True,
        optimizer="SGD", warmup_epochs=0, close_mosaic=0,
        project="/home/jovyan/shared/s0598584/runs", name="_batch_probe", exist_ok=True,
    )
    if single:
        peak = torch.cuda.max_memory_reserved(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print("PROBE_OK " + json.dumps({"batch": total_batch, "peak_gb": round(peak, 2),
                                        "total_gb": round(total, 2)}))
    else:
        # Under DDP the real training runs in child processes, so peak memory is
        # not visible here — a clean exit means it FIT across all GPUs.
        print("PROBE_OK " + json.dumps({"batch": total_batch, "peak_gb": None,
                                        "ddp": True, "n_gpus": len(device)}))
except torch.cuda.OutOfMemoryError:
    print("PROBE_OOM " + str(total_batch)); sys.exit(2)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("PROBE_OOM " + str(total_batch)); sys.exit(2)
    raise
except Exception as e:
    # A DDP child that OOMs surfaces as a non-zero subprocess exit (CalledProcessError)
    # rather than OutOfMemoryError. Treat any such failure as "did not fit".
    print("PROBE_FAIL " + repr(e)[:300]); sys.exit(3)
