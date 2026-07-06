"""Config-driven training with resume + a completion marker for the watchdogs.

Canonical heavy pipeline (multi-stage finetune/eval/conf) lives on the server; this
is the generic core the toolkit ships. It resumes from last.pt automatically and
writes pipeline_state.json {stages:{complete:true}} when training finishes, which is
exactly what recovery/ (in-pod + Windows) polls to know it can stop.
"""
from __future__ import annotations
import json
import os

from .registry import load_global
from .preflight import check_vram, check_ram
from .batch_finder import find_optimal_batch, resolve_init_weights


def _run_dir(g):
    return os.path.join(g["data_root"], "runs", "faces", g["run_name"])


def _pick_cache(g, data_yaml):
    """'ram' fully caches every train image as a raw array — fast, but for a big
    dataset it can need far more than the node's RAM (a 146k-image 640x640 set
    needs ~167 GB, seen exceeding a 48 GB cgroup limit and getting SIGKILLed
    mid-cache). Estimate the RAM a full cache would need and fall back to
    'disk' (memmap, bounded by disk not RAM) if it doesn't comfortably fit."""
    mode = str(g.get("cache", "auto")).lower()
    if mode in ("ram", "disk", "false", "none"):
        return False if mode in ("false", "none") else mode
    try:
        import yaml
        d = yaml.safe_load(open(data_yaml))
        train_dir = os.path.join(d.get("path", ""), d.get("train", "images/train"))
        n_train = sum(1 for f in os.listdir(train_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png")))
    except Exception:
        return "disk"   # can't verify it's safe -> don't risk RAM
    imgsz = int(g["imgsz"])
    need_gb = (n_train * imgsz * imgsz * 3) / (1024 ** 3)
    ram_gb = check_ram(0) or 0
    safe_gb = 0.5 * ram_gb   # leave headroom for the dataloader/model/OS
    if need_gb <= safe_gb:
        return "ram"
    print(f"cache='auto': {n_train} train imgs @ {imgsz} need ~{need_gb:.0f} GB, "
          f"only {safe_gb:.0f} GB of {ram_gb:.0f} GB RAM is safe to use -> 'disk'")
    return "disk"


def train(g=None, devices=None):
    g = g or load_global()
    run_dir = _run_dir(g)
    os.makedirs(run_dir, exist_ok=True)
    state = os.path.join(run_dir, "pipeline_state.json")
    if os.path.exists(state):
        try:
            if json.load(open(state)).get("stages", {}).get("complete"):
                print("pipeline already complete; nothing to do.")
                return
        except Exception:
            pass

    board = check_vram(min_gib_per_gpu=6)             # clear error if no GPU
    if devices is None:
        devices = ",".join(str(i) for i in range(len(board)))
    batch = find_optimal_batch(g, devices=devices)["total_batch"]

    # weights precedence: resume own last.pt  >  init_weights (warm-start)  >  COCO
    last = os.path.join(run_dir, "weights", "last.pt")
    resume = os.path.exists(last)
    if resume:
        weights = last
        print(f"resuming from {last}")
    else:
        weights = resolve_init_weights(g)
        print(f"fresh run starting from {weights}"
              + (" (warm-start)" if g.get("init_weights") else " (COCO pretrained)"))

    data_yaml = os.path.join(g["data_root"], g["out_dataset"], "dataset.yaml")
    cache = _pick_cache(g, data_yaml)
    print(f"cache={cache!r}")

    from ultralytics import YOLO
    YOLO(weights).train(
        data=data_yaml,
        imgsz=int(g["imgsz"]), epochs=int(g["epochs"]), batch=batch,
        device=devices, patience=int(g.get("patience", 50)),
        cache=cache, resume=resume, project=os.path.dirname(run_dir),
        name=g["run_name"], exist_ok=True)

    json.dump({"stages": {"complete": True}}, open(state, "w"))
    print("training complete; wrote", state)


if __name__ == "__main__":
    train()
