#!/usr/bin/env python3
"""ONE entry point. If unsure, this is the only file you run.

    python run.py check        # what datasets are configured + resource preflight
    python run.py build        # split -> merge -> tile -> augment -> write dataset
    python run.py find-batch   # largest batch <= VRAM target (GPU node only)
    python run.py train        # find-batch (cached) then train, resumable
    python run.py all          # build then train
    python run.py status       # one-shot human/agent-readable progress report —
                                # safe to run anytime, from anywhere with a shell
                                # (e.g. the JupyterHub web terminal from a phone
                                # browser), never starts/stops anything itself

Everything the model does is in config/global.yaml.
Everything about the data is in config/datasets/*.yaml (+ .hiden/datasets/).
"""
import os
import subprocess
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


def _pgrep_alive(pattern):
    """pattern must use the bracket-glob trick on its first char (e.g. "[r]un.py
    train", not "run.py train") — otherwise `pgrep -f` can false-positive-match
    the very shell command that's running this check, if that shell's own
    command text happens to contain the plain pattern anywhere (e.g. several
    commands bundled into one shell call, one of which echoes/logs the pattern).
    A bracketed first char is a regex that matches the real target process's
    plain-text cmdline but never matches literal bracket characters in another
    command's text. See recovery/server/start_robust.sh for the same trick."""
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
        return bool(out.strip())
    except Exception:
        return False


def cmd_status():
    """One-shot, read-only status report. Never starts/stops anything — safe to
    run repeatedly from anywhere with a shell (JupyterHub web terminal works from
    a phone browser too). This is the command a sporadic check-in or an agent
    should call FIRST to decide whether recovery/server/start_robust.sh is needed."""
    import json
    g = load_global()
    run_dir = os.path.join(g["data_root"], "runs", "faces", g["run_name"])
    print(f"=== status: {g['run_name']} (yolo26{g['model_size']} @ {g['imgsz']}) ===")

    state_path = os.path.join(run_dir, "pipeline_state.json")
    try:
        complete = bool(json.load(open(state_path)).get("stages", {}).get("complete"))
    except Exception:
        complete = False
    print(f"pipeline complete: {complete}")

    watchdog_alive = _pgrep_alive(r"[r]ecovery/server/watchdog\.py")
    train_alive = _pgrep_alive(r"[r]un\.py train")
    keepalive_alive = _pgrep_alive(r"[h]ub_keepalive\.py")
    print(f"in-pod watchdog running: {watchdog_alive}")
    print(f"train process running:  {train_alive}")
    print(f"keep-alive running:     {keepalive_alive}")

    results_csv = os.path.join(run_dir, "results.csv")
    if os.path.exists(results_csv):
        lines = open(results_csv).read().strip().splitlines()
        if len(lines) > 1:
            header, last = lines[0].split(","), lines[-1].split(",")
            print(f"latest epoch row: {dict(zip(header, last))}")
        else:
            print("results.csv exists but no epoch finished yet")
    else:
        print("results.csv: not written yet (no epoch finished)")

    weights_dir = os.path.join(run_dir, "weights")
    for name in ("last.pt", "best.pt"):
        p = os.path.join(weights_dir, name)
        if os.path.exists(p):
            print(f"{name}: present (mtime {os.path.getmtime(p):.0f})")
        else:
            print(f"{name}: not written yet")

    try:
        from src.batch_finder import current_buffer
        buf, rung, ladder = current_buffer(g)
        print(f"VRAM ladder: rung {rung}/{len(ladder)-1} (buffer={buf})")
    except Exception as e:
        print(f"VRAM ladder: unknown ({e})")

    crash_dir = os.path.join(g["data_root"], "runs", "crashreports")
    if os.path.isdir(crash_dir):
        crashes = sorted(f for f in os.listdir(crash_dir) if f.endswith(".log"))
        print(f"crash reports: {len(crashes)}" + (f" (latest: {crashes[-1]})" if crashes else ""))
    else:
        print("crash reports: none")

    if not complete and not watchdog_alive and not train_alive:
        print("\n-> NOTHING IS RUNNING and the pipeline is not complete. "
              "Run: bash recovery/server/start_robust.sh")
    elif complete:
        print("\n-> Training is DONE. Nothing to resume.")
    else:
        print("\n-> Looks healthy — running normally.")


def main():
    cmds = {"check": cmd_check, "build": cmd_build, "find-batch": cmd_find_batch,
            "train": cmd_train, "status": cmd_status,
            "all": lambda: (cmd_build(), cmd_train())}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 2)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()
