"""Find the largest batch whose LIVE board usage stays <= vram_target_pct on
EVERY GPU, measured on the REAL dataset, for 1 GPU or N GPUs.

>>> CANONICAL NOTE <<<
The battle-tested version of this logic currently runs on the server
(deepseek/scripts/batch_finder.py + train_pipeline._probe_total). This is a clean
generic reconstruction from the `feedback-real-batch-finder` spec. Before trusting
it on the cluster, DIFF it against the server copy — do not silently replace the
working one. Key invariants it must keep:
  * measure nvidia-smi board peak (`peak_mib_max`), NEVER torch reserved
  * probe on the real dataset for ~probe_iters iterations (dense batches must show)
  * target = vram_probe_buffer * board  (85% minus a DDP/aug margin)

Result written to  runs/optimal_batch_<imgsz>.json  and returned.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys

from .registry import load_global, discover_datasets
from .preflight import check_vram, PreflightError

TOOLKIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_init_weights(g):
    """Starting weights for a FRESH run: init_weights if set (warm-start), else the
    official COCO-pretrained yolo26<size>.pt. Relative paths resolve under data_root."""
    iw = (g.get("init_weights") or "").strip()
    if not iw:
        return f"yolo26{g['model_size']}.pt"
    if not os.path.isabs(iw):
        iw = os.path.join(g["data_root"], iw)
    if not os.path.exists(iw):
        raise FileNotFoundError(
            f"init_weights not found: {iw}\n  -> fix the path in config/global.yaml "
            f"or set init_weights: \"\" to start from COCO.")
    return iw


def _count_train_images(data_yaml):
    import yaml
    d = yaml.safe_load(open(data_yaml))
    train_dir = os.path.join(d.get("path", ""), d.get("train", "images/train"))
    try:
        return sum(1 for f in os.listdir(train_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    except Exception:
        return 5000


def _probe(total, imgsz, weights, data_yaml, devices, fraction):
    cmd = [sys.executable, "-m", "src.batch_probe", str(total), str(imgsz),
           weights, data_yaml, devices, f"{fraction:.6f}"]
    out = subprocess.run(cmd, cwd=TOOLKIT, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        for tag in ("PROBE_OK", "PROBE_OOM", "PROBE_FAIL"):
            if line.startswith(tag):
                return tag, json.loads(line[len(tag):].strip())
    return "PROBE_FAIL", {"error": (out.stderr or "no probe output")[-400:]}


def find_optimal_batch(g=None, weights=None, data_yaml=None, devices=None, force=False):
    g = g or load_global()
    imgsz = int(g["imgsz"])
    weights = weights or resolve_init_weights(g)   # probe with the real starting weights
    data_yaml = data_yaml or os.path.join(g["data_root"], g["out_dataset"], "dataset.yaml")
    # devices: default to all visible GPUs
    if devices is None:
        board = check_vram(min_gib_per_gpu=6)   # clear error if no GPU
        devices = ",".join(str(i) for i in range(len(board)))
    n_gpus = max(len([d for d in devices.split(",") if d != ""]), 1)

    cache_path = os.path.join(g["data_root"], "runs", f"optimal_batch_{imgsz}.json")
    if os.path.exists(cache_path) and not force:
        return json.load(open(cache_path))

    board = check_vram(min_gib_per_gpu=6, n_gpus_expected=n_gpus)
    board_mib = min(board)
    target_mib = float(g.get("vram_probe_buffer", 0.83)) * board_mib
    n_train = _count_train_images(data_yaml)

    def fraction_for(total):
        # Fixed image-count sample (not iters*batch): a real crash showed a short,
        # batch-scaled sample can miss a dense-box outlier that a bigger fixed
        # sample would have caught. probe_images defaults to a flat 2000 so every
        # candidate batch size gets the same chance at hitting the worst case,
        # instead of smaller batches getting an even thinner sample than large ones.
        imgs = min(n_train, max(256, int(g.get("probe_images", 2000))))
        return max(min(imgs / max(n_train, 1), 1.0), 1e-3)

    # coarse doubling search on per-GPU batch
    best = None
    per_gpu = 2
    while per_gpu <= 256:
        total = per_gpu * n_gpus
        tag, res = _probe(total, imgsz, weights, data_yaml, devices, fraction_for(total))
        if tag == "PROBE_OK":
            peak = res["peak_mib_max"]
            print(f"  per_gpu={per_gpu:<4} total={total:<4} peak={peak} MiB "
                  f"({100*peak/board_mib:.0f}%) target={target_mib:.0f}")
            if peak <= target_mib:
                best = {**res, "peak_pct": round(100 * peak / board_mib, 1)}
                per_gpu *= 2
                continue
            break
        elif tag == "PROBE_OOM":
            print(f"  per_gpu={per_gpu} OOM"); break
        else:
            raise PreflightError(f"probe failed: {res.get('error')}")

    if best is None:
        raise PreflightError(
            f"Even per-GPU batch 2 exceeds {g.get('vram_target_pct',85)}% VRAM on "
            f"{board_mib} MiB boards. Lower imgsz/model_size in config/global.yaml.")

    # linear refine upward by 1 until we cross the target
    per_gpu = best["per_gpu_batch"]
    while True:
        cand = per_gpu + 1
        total = cand * n_gpus
        tag, res = _probe(total, imgsz, weights, data_yaml, devices, fraction_for(total))
        if tag == "PROBE_OK" and res["peak_mib_max"] <= target_mib:
            best = {**res, "peak_pct": round(100 * res["peak_mib_max"] / board_mib, 1)}
            per_gpu = cand
        else:
            break

    payload = {
        "total_batch": best["per_gpu_batch"] * n_gpus,
        "per_gpu_batch": best["per_gpu_batch"], "n_gpus": n_gpus,
        "imgsz": imgsz, "peak_mib": best["peak_mib_max"], "board_mib": board_mib,
        "peak_pct": best["peak_pct"], "target_pct": g.get("vram_target_pct", 85),
        "devices": devices,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(payload, open(cache_path, "w"), indent=2)
    print(f"CHOSEN total={payload['total_batch']} per_gpu={payload['per_gpu_batch']} "
          f"live={payload['peak_pct']}% (<= {payload['target_pct']}%)")
    return payload


if __name__ == "__main__":
    find_optimal_batch(force="--force" in sys.argv)
