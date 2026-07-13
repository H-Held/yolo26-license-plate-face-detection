"""In-pod watchdog (restart layer #1).

Runs `python run.py train` and, if it dies for any reason other than "finished",
diagnoses why, waits a backoff, and restarts it. Training resumes from last.pt
automatically (train.py handles that). Stops when the run's pipeline_state.json
says stages.complete. Also makes sure the keep-alive is running.

Nothing here is hard-coded to a particular run name — it reads config/global.yaml.

    nohup python recovery/server/watchdog.py >> runs/watchdog.log 2>&1 &

Training's own stdout/stderr (incl. the ultralytics progress bar) is streamed
line-by-line into runs/train_live.log as it happens — not just captured at the
end — so a local tailer always sees the current line, even mid-epoch.
"""
from __future__ import annotations
import collections
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLKIT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TOOLKIT)
from src.registry import load_global           # noqa: E402
from src.batch_finder import lower_buffer_rung  # noqa: E402

MAX_RETRIES = int(os.environ.get("WATCHDOG_MAX_RETRIES", "50"))
BACKOFF = [10, 30, 60, 120, 300]
TRAIN_LIVE_LOG = os.path.join(TOOLKIT, "runs", "train_live.log")
TAIL_LINES = 400


def _state_path(g):
    return os.path.join(g["data_root"], "runs", "faces", g["run_name"],
                        "pipeline_state.json")


def _is_complete(g):
    import json
    p = _state_path(g)
    try:
        return bool(json.load(open(p)).get("stages", {}).get("complete"))
    except Exception:
        return False


def _keepalive_running():
    try:
        out = subprocess.check_output(["pgrep", "-f", "[h]ub_keepalive.py"], text=True)
        return bool(out.strip())
    except Exception:
        return False


def _ensure_keepalive():
    if _keepalive_running():
        return
    log = open(os.path.join(TOOLKIT, "runs", "keepalive.log"), "a")
    os.makedirs(os.path.join(TOOLKIT, "runs"), exist_ok=True)
    subprocess.Popen([sys.executable, os.path.join(HERE, "hub_keepalive.py")],
                     stdout=log, stderr=log, cwd=TOOLKIT)
    print("watchdog: started keepalive", flush=True)


def _diagnose(tail: str) -> tuple[str, bool]:
    """Returns (message, is_vram_oom). is_vram_oom is True only for causes that
    the auto-lowering VRAM ladder can actually fix (a real batch-size-driven OOM,
    including the masked-NVML-assert variant seen on permission-locked MIG
    slices) — NOT for RAM-cgroup kills, NCCL, or other unrelated crashes, where
    lowering VRAM further wouldn't help and would just burn ladder rungs."""
    t = tail.lower()
    if "out of memory" in t or "cuda oom" in t:
        return "CUDA_OOM", True
    if "nvml_success == r" in t or "cudacachingallocator" in t:
        return "CUDA_OOM masked as NVML assert (NVML blocked on this node)", True
    if "nccl" in t:
        return "NCCL (multi-GPU comms; often a transient pod issue)", False
    if "killed" in t or "signal 9" in t or "sigkill" in t:
        return "OOM-KILLED by the pod (RAM cgroup) — check cache mode, not VRAM", False
    if "cuda error" in t or "device-side assert" in t:
        return "CUDA_ERROR", False
    return "UNKNOWN (see crashreport)", False


def _run_train_streaming(attempt: int) -> tuple[int, str]:
    """Runs `python run.py train`, streaming every line it prints (progress bar
    included — ultralytics/tqdm falls back to one line per update, not \\r
    overwrite, once stdout isn't a tty) into TRAIN_LIVE_LOG in real time, so a
    local tailer sees the current line instead of only the final buffered dump
    subprocess.run(capture_output=True) would have given at process exit.
    Returns (returncode, tail_text) for crash diagnosis."""
    os.makedirs(os.path.dirname(TRAIN_LIVE_LOG), exist_ok=True)
    tail = collections.deque(maxlen=TAIL_LINES)
    with open(TRAIN_LIVE_LOG, "a", encoding="utf-8", errors="replace") as logf:
        logf.write(f"\n===== watchdog: launch train (attempt {attempt}) "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        logf.flush()
        proc = subprocess.Popen([sys.executable, "run.py", "train"], cwd=TOOLKIT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            tail.append(line)
        proc.wait()
    return proc.returncode, "".join(tail)[-4000:]


def _write_crashreport(g, attempt, tail, cause):
    d = os.path.join(g["data_root"], "runs", "crashreports")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"crash_{g['run_name']}_{attempt}.log")
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"cause: {cause}\n\n--- tail ---\n{tail}\n")
    print(f"watchdog: crashreport -> {p} ({cause})", flush=True)


def main():
    g = load_global()
    _ensure_keepalive()
    for attempt in range(1, MAX_RETRIES + 1):
        if _is_complete(g):
            print("watchdog: pipeline complete — stopping.", flush=True)
            return
        _ensure_keepalive()
        print(f"watchdog: launch train (attempt {attempt}/{MAX_RETRIES})", flush=True)
        returncode, tail = _run_train_streaming(attempt)
        if _is_complete(g) or returncode == 0:
            print("watchdog: training finished cleanly.", flush=True)
            return
        cause, is_vram_oom = _diagnose(tail)
        _write_crashreport(g, attempt, tail, cause)
        if is_vram_oom:
            new_buf = lower_buffer_rung(g)
            if new_buf is None:
                print(f"watchdog: crash ({cause}); VRAM ladder EXHAUSTED (already at "
                      f"its lowest rung) — giving up, a smaller imgsz/model_size is "
                      f"needed now, not just a smaller batch.", flush=True)
                return
            print(f"watchdog: crash ({cause}); lowering VRAM ladder -> buffer={new_buf} "
                  f"and re-probing batch on next attempt", flush=True)
        else:
            print(f"watchdog: crash ({cause}); retrying without changing VRAM ladder "
                  f"(not a VRAM-fixable cause)", flush=True)
        wait = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
        print(f"watchdog: retrying in {wait}s", flush=True)
        time.sleep(wait)
    print("watchdog: gave up after max retries.", flush=True)


if __name__ == "__main__":
    main()
