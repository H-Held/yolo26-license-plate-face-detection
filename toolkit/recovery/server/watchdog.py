"""In-pod watchdog (restart layer #1).

Runs `python run.py train` and, if it dies for any reason other than "finished",
diagnoses why, waits a backoff, and restarts it. Training resumes from last.pt
automatically (train.py handles that). Stops when the run's pipeline_state.json
says stages.complete. Also makes sure the keep-alive is running.

Nothing here is hard-coded to a particular run name — it reads config/global.yaml.

    nohup python recovery/server/watchdog.py >> runs/watchdog.log 2>&1 &
"""
from __future__ import annotations
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLKIT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TOOLKIT)
from src.registry import load_global   # noqa: E402

MAX_RETRIES = int(os.environ.get("WATCHDOG_MAX_RETRIES", "50"))
BACKOFF = [10, 30, 60, 120, 300]


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


def _diagnose(tail: str) -> str:
    t = tail.lower()
    if "out of memory" in t or "cuda oom" in t:
        return "CUDA_OOM (lower batch / imgsz)"
    if "nvml_success == r" in t or "cudacachingallocator" in t:
        return "CUDA_OOM masked as NVML assert (NVML blocked on this node) — lower batch/imgsz"
    if "nccl" in t:
        return "NCCL (multi-GPU comms; often a transient pod issue)"
    if "killed" in t or "signal 9" in t or "sigkill" in t:
        return "OOM-KILLED by the pod (RAM cgroup) — lower cache/batch"
    if "cuda error" in t or "device-side assert" in t:
        return "CUDA_ERROR"
    return "UNKNOWN (see crashreport)"


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
        proc = subprocess.run([sys.executable, "run.py", "train"], cwd=TOOLKIT,
                              capture_output=True, text=True)
        sys.stdout.write(proc.stdout[-4000:] or "")
        if _is_complete(g) or proc.returncode == 0:
            print("watchdog: training finished cleanly.", flush=True)
            return
        tail = (proc.stderr or proc.stdout)[-4000:]
        cause = _diagnose(tail)
        _write_crashreport(g, attempt, tail, cause)
        wait = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
        print(f"watchdog: crash ({cause}); retrying in {wait}s", flush=True)
        time.sleep(wait)
    print("watchdog: gave up after max retries.", flush=True)


if __name__ == "__main__":
    main()
