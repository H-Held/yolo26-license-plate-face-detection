#!/usr/bin/env python3
"""ONE entry point. If unsure, this is the only file you run.

    python run.py check        # what datasets are configured + resource preflight
    python run.py build        # split -> merge -> tile -> augment -> write dataset
    python run.py find-batch   # largest batch <= VRAM target (GPU node only)
    python run.py train        # find-batch (cached) then train, resumable
    python run.py all          # build then train

Everything the model does is in config/global.yaml.
Everything about the data is in config/datasets/*.yaml (+ .hiden/datasets/).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.registry import load_global, discover_datasets   # noqa: E402


def cmd_check():
    g = load_global()
    print(f"model: yolo26{g['model_size']} @ {g['imgsz']}  classes={g['classes']}")
    print(f"data_root: {g['data_root']}")
    print("datasets:")
    for c in discover_datasets(include_disabled=True):
        print(f"  - {c['name']:<18} {c.get('visibility','?'):<8} "
              f"format={c.get('format','?'):<9} enabled={c.get('enabled', True)}")
    from src.preflight import check_disk, check_ram, gpu_board_mib
    try:
        print(f"disk free @ data_root: {check_disk(g['data_root'], 1):.0f} GB")
    except Exception as e:
        print("disk:", e)
    ram = check_ram(1)
    print(f"RAM limit: {ram:.0f} GB" if ram else "RAM limit: unknown")
    board = gpu_board_mib()
    print(f"GPUs: {len(board)} x {min(board)//1024 if board else 0} GiB"
          if board else "GPUs: none visible (CPU node) — build here, train on GPU node")


def cmd_build():
    from src.build_dataset import build
    build(load_global())


def cmd_find_batch():
    from src.batch_finder import find_optimal_batch
    find_optimal_batch(load_global(), force="--force" in sys.argv)


def cmd_train():
    from src.train import train
    train(load_global())


def main():
    cmds = {"check": cmd_check, "build": cmd_build, "find-batch": cmd_find_batch,
            "train": cmd_train, "all": lambda: (cmd_build(), cmd_train())}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 2)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()
