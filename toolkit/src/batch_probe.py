"""One isolated batch-size probe (spawned by batch_finder.py).

Trains ONE short pass at a given TOTAL batch on the REAL dataset while a
background thread samples true board memory via nvidia-smi (NOT torch's
`reserved`, which under-reads by ~1.5 GB and once put a "verified 84%" run at a
real 97%). Prints a single result line the parent parses:

    PROBE_OK   {json}      # peak_mib_max, peak_mib_per_gpu, total_mib, per_gpu_batch, n_gpus
    PROBE_OOM  {json}
    PROBE_FAIL {json}

Usage (internal):
    python -m src.batch_probe <total_batch> <imgsz> <weights> <data_yaml> <devices> <fraction>
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import time

# Reduce fragmentation so the probed peak reflects real training (matches the proven
# server probe). YOLO_CONFIG_DIR must be writable; override via env on shared setups.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if os.environ.get("YOLO_CONFIG_DIR_OVERRIDE"):
    os.environ["YOLO_CONFIG_DIR"] = os.environ["YOLO_CONFIG_DIR_OVERRIDE"]


def _sample_gpu(indices, peaks, stop):
    q = "--query-gpu=index,memory.used,memory.total"
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", q, "--format=csv,noheader,nounits"],
                text=True, timeout=5)
            for line in out.strip().splitlines():
                idx, used, total = [int(x) for x in line.split(",")]
                if idx in indices:
                    peaks[idx] = (max(peaks.get(idx, (0, total))[0], used), total)
        except Exception:
            pass
        time.sleep(0.25)


def main():
    total = int(sys.argv[1]); imgsz = int(sys.argv[2]); weights = sys.argv[3]
    data = sys.argv[4]; devices = sys.argv[5]; fraction = float(sys.argv[6])
    idx_list = [int(x) for x in devices.split(",") if x != ""]
    n_gpus = max(len(idx_list), 1)

    # ultralytics wants an int for one GPU, a list for DDP (not the raw string)
    device = idx_list if n_gpus > 1 else (idx_list[0] if idx_list else 0)

    peaks, stop = {}, threading.Event()
    t = threading.Thread(target=_sample_gpu, args=(set(idx_list), peaks, stop), daemon=True)
    t.start()
    try:
        import torch
        from ultralytics import YOLO
        # deterministic, representative probe params (match the proven server probe):
        # amp on, fixed SGD, no warmup/mosaic-close so 1 epoch reflects steady VRAM.
        YOLO(weights).train(
            data=data, imgsz=imgsz, epochs=1, batch=total, device=device,
            fraction=fraction, cache=False, val=False, plots=False, verbose=False,
            save=False, workers=4, amp=True, optimizer="SGD", warmup_epochs=0,
            close_mosaic=0, name="_probe", exist_ok=True)
        stop.set(); t.join(timeout=2)
        per = {i: peaks.get(i, (0, 0))[0] for i in idx_list}
        totals = {i: peaks.get(i, (0, 0))[1] for i in idx_list}
        print("PROBE_OK " + json.dumps({
            "peak_mib_max": max(per.values()) if per else 0,
            "peak_mib_per_gpu": per,
            "total_mib": max(totals.values()) if totals else 0,
            "per_gpu_batch": total // n_gpus, "n_gpus": n_gpus}))
    except Exception as e:
        stop.set()
        msg = str(e).lower()
        is_oom = ("out of memory" in msg or "cuda oom" in msg
                  or e.__class__.__name__ == "OutOfMemoryError")
        kind = "PROBE_OOM" if is_oom else "PROBE_FAIL"
        print(f"{kind} " + json.dumps({"error": str(e)[:400], "batch": total}))
        sys.exit(0)


if __name__ == "__main__":
    main()
