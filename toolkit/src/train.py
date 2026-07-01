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
from .preflight import check_vram
from .batch_finder import find_optimal_batch, resolve_init_weights


def _run_dir(g):
    return os.path.join(g["data_root"], "runs", "faces", g["run_name"])


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

    from ultralytics import YOLO
    YOLO(weights).train(
        data=os.path.join(g["data_root"], g["out_dataset"], "dataset.yaml"),
        imgsz=int(g["imgsz"]), epochs=int(g["epochs"]), batch=batch,
        device=devices, patience=int(g.get("patience", 50)),
        cache="ram", resume=resume, project=os.path.dirname(run_dir),
        name=g["run_name"], exist_ok=True)

    json.dump({"stages": {"complete": True}}, open(state, "w"))
    print("training complete; wrote", state)


if __name__ == "__main__":
    train()
