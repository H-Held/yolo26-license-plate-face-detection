"""Find the largest batch whose LIVE board usage stays <= vram_target_pct on
EVERY GPU, measured on the REAL dataset, for 1 GPU or N GPUs.

>>> CANONICAL NOTE <<<
The battle-tested version of this logic currently runs on the server
(deepseek/scripts/batch_finder.py + train_pipeline._probe_total). This is a clean
generic reconstruction from the `feedback-real-batch-finder` spec. Before trusting
it on the cluster, DIFF it against the server copy — do not silently replace the
working one. Key invariants it must keep:
  * measure true board peak (`peak_mib_max`), NEVER torch reserved
  * probe on a real, fixed-size dataset sample (dense batches must show up)
  * target = current VRAM-ladder buffer * board (auto-lowers after a real OOM —
    see DEFAULT_VRAM_LADDER / current_buffer / lower_buffer_rung below)

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

# Auto-lowering VRAM ladder: start greedy (80%), and if a REAL training-time OOM
# happens later (not just a probe failure — a probe can say "safe" on its sample
# and still be wrong on a rare dense-box batch, as seen in practice), the watchdog
# calls lower_buffer_rung() to advance to the next, more conservative rung and
# invalidate the cached batch, so the next attempt re-probes lower automatically.
# No human/agent needs to intervene or edit config by hand.
DEFAULT_VRAM_LADDER = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]


def _ladder_state_path(g):
    return os.path.join(g["data_root"], "runs", "faces", g["run_name"], "vram_ladder.json")


def current_buffer(g):
    """(buffer, rung_index, ladder) for this run's current VRAM ladder rung.
    Persisted in the run dir so it survives watchdog restarts / pod culls."""
    ladder = g.get("vram_ladder") or DEFAULT_VRAM_LADDER
    try:
        idx = json.load(open(_ladder_state_path(g)))["rung"]
    except Exception:
        idx = 0
    idx = max(0, min(idx, len(ladder) - 1))
    return ladder[idx], idx, ladder


def lower_buffer_rung(g):
    """Advance to the next, lower rung after a real OOM and delete the now-suspect
    cached batch so the next find_optimal_batch() call re-probes at the new target.
    Returns the new buffer, or None if the ladder is already exhausted (caller
    should stop retrying rather than loop forever at the floor)."""
    _, idx, ladder = current_buffer(g)
    new_idx = idx + 1
    if new_idx >= len(ladder):
        return None
    p = _ladder_state_path(g)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({"rung": new_idx}, open(p, "w"))
    imgsz = int(g["imgsz"])
    cache_path = os.path.join(g["data_root"], "runs", f"optimal_batch_{imgsz}.json")
    if os.path.exists(cache_path):
        os.remove(cache_path)
    return ladder[new_idx]


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


def _label_dir_for(images_dir):
    """Standard ultralytics layout: labels/ mirrors images/ with .txt files."""
    norm = os.path.normpath(images_dir)
    marker = os.sep + "images" + os.sep
    if marker in norm:
        return norm.replace(marker, os.sep + "labels" + os.sep, 1)
    return norm  # unconventional layout -> label lookup will just miss (count 0)


def build_probe_manifest(g, data_yaml):
    """Every probe must have a real shot at the densest-box images in the
    training set, not just whatever a random fraction happens to include —
    a prior real crash (batch=35 @ 81% buffer, mid-epoch) came from exactly
    this gap: the probe's random sample missed the rare dense-box outlier
    that later blew VRAM during actual training. So instead of relying on
    ultralytics' own `fraction=` (uniform random slice), build a fixed probe
    image list that FORCES the top-N densest-box images in on every probe,
    filling the rest with a random sample for general coverage. Returns
    (manifest_data_yaml_path, sample_size, forced_dense_count)."""
    import yaml, random

    total_n = int(g.get("probe_images", 2000))
    dense_n = min(int(g.get("probe_dense_images", 300)), total_n)

    runs_dir = os.path.join(g["data_root"], "runs")
    os.makedirs(runs_dir, exist_ok=True)
    manifest_txt = os.path.join(runs_dir, "probe_manifest.txt")
    manifest_yaml = os.path.join(runs_dir, "probe_manifest.yaml")
    cache_path = os.path.join(runs_dir, "probe_box_counts.json")

    d = yaml.safe_load(open(data_yaml))
    train_dir = os.path.join(d.get("path", ""), d.get("train", "images/train"))
    label_dir = _label_dir_for(train_dir)
    images = sorted(f for f in os.listdir(train_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png")))

    # Box counts are cached (scanning 100k+ label files every find-batch call
    # is wasted work) and invalidated by a simple count change of the train set.
    counts = None
    if os.path.exists(cache_path):
        try:
            cached = json.load(open(cache_path))
            if cached.get("n_images") == len(images):
                counts = cached["counts"]
        except Exception:
            counts = None
    if counts is None:
        counts = {}
        for f in images:
            lp = os.path.join(label_dir, os.path.splitext(f)[0] + ".txt")
            try:
                with open(lp) as fh:
                    counts[f] = sum(1 for line in fh if line.strip())
            except Exception:
                counts[f] = 0
        json.dump({"n_images": len(images), "counts": counts}, open(cache_path, "w"))

    ranked = sorted(images, key=lambda f: counts.get(f, 0), reverse=True)
    dense = ranked[:dense_n]
    dense_set = set(dense)
    remaining = [f for f in images if f not in dense_set]
    random.Random(0).shuffle(remaining)  # fixed seed -> reproducible probe sample
    sample = dense + remaining[:max(total_n - len(dense), 0)]

    with open(manifest_txt, "w") as fh:
        for f in sample:
            fh.write(os.path.join(train_dir, f) + "\n")

    manifest = dict(d)
    manifest.pop("path", None)  # train/val below are already absolute
    manifest["train"] = manifest_txt
    manifest["val"] = manifest_txt  # unused (probes always run with val=False)
    yaml.safe_dump(manifest, open(manifest_yaml, "w"))
    return manifest_yaml, len(sample), len(dense)


def _probe(total, imgsz, weights, data_yaml, devices, fraction,
           idle_timeout=20, max_timeout=45, log_path=None):
    """Runs the probe subprocess, but kills it early if it goes idle
    (no new stdout/stderr output) for `idle_timeout` seconds — much faster
    than waiting out a flat `max_timeout` when something is simply stuck
    (e.g. a hung rendezvous), while still allowing slow-but-progressing
    runs up to `max_timeout` total. If `log_path` is given, every line is
    appended and flushed immediately so progress can be tailed live instead
    of only appearing once the subprocess exits (which looked like a hang)."""
    cmd = [sys.executable, "-m", "src.batch_probe", str(total), str(imgsz),
           weights, data_yaml, devices, f"{fraction:.6f}"]
    proc = subprocess.Popen(cmd, cwd=TOOLKIT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             start_new_session=True)  # own process group -> can kill DDP children too
    import time
    chunks = []
    last_output_time = time.time()
    start_time = last_output_time
    killed_reason = None

    log_f = None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_f = open(log_path, "a")
        log_f.write(f"\n=== probe total={total} imgsz={imgsz} ===\n")
        log_f.flush()

    import select
    try:
        while True:
            rlist, _, _ = select.select([proc.stdout], [], [], 1.0)
            if rlist:
                line = proc.stdout.readline()
                if line:
                    chunks.append(line)
                    last_output_time = time.time()
                    if log_f:
                        log_f.write(line)
                        log_f.flush()
                elif proc.poll() is not None:
                    break
            if proc.poll() is not None:
                break
            now = time.time()
            if now - last_output_time > idle_timeout:
                killed_reason = f"no output for {idle_timeout}s (stuck)"
                break
            if now - start_time > max_timeout:
                killed_reason = f"exceeded max {max_timeout}s total"
                break
    finally:
        if log_f:
            if killed_reason:
                log_f.write(f"[killed: {killed_reason}]\n")
            log_f.flush()
            log_f.close()

    if killed_reason:
        import os as _os, signal
        try:
            _os.killpg(_os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
        combined = "".join(chunks)
        return "PROBE_FAIL", {"error": f"probe killed: {killed_reason}. "
                               f"tail: {combined[-1000:]}"}

    combined = "".join(chunks)
    for line in combined.splitlines():
        for tag in ("PROBE_OK", "PROBE_OOM", "PROBE_FAIL"):
            if line.startswith(tag):
                return tag, json.loads(line[len(tag):].strip())

    # No tagged line found — likely a DDP child crash (e.g. real CUDA OOM)
    # that surfaces only as a non-zero exit, not our own PROBE_OOM line.
    if "CUDA out of memory" in combined or "OutOfMemoryError" in combined:
        return "PROBE_OOM", {"error": "CUDA OOM detected in DDP child output "
                              "(not self-reported)", "batch": total}

    log_path = os.path.join(TOOLKIT, "runs", "last_probe_stderr.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    open(log_path, "w").write(combined)
    return "PROBE_FAIL", {"error": combined[-400:] + f" [full log: {log_path}]"}


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
    buffer, rung, ladder = current_buffer(g)
    target_mib = buffer * board_mib
    print(f"VRAM ladder rung {rung}/{len(ladder)-1}: buffer={buffer} -> target={target_mib:.0f} MiB")
    # A random `fraction=` slice of the train set can simply miss the rare
    # dense-box images that actually drive peak VRAM (this is exactly how a
    # prior probe said "safe" and still crashed mid-epoch later). Build a
    # fixed probe manifest that force-includes the densest-box images on
    # every single probe, topped up with a random sample for general coverage.
    manifest_yaml, sample_n, dense_n = build_probe_manifest(g, data_yaml)
    print(f"Probe manifest: {sample_n} images ({dense_n} forced dense-box "
          f"outliers + {sample_n - dense_n} random) -> {manifest_yaml}")

    log_path = os.path.join(TOOLKIT, "runs", "find_batch_live.log")
    open(log_path, "w").close()  # truncate so each run's log is fresh, but still live-tailable

    def probe(total):
        # fraction=1.0: the manifest IS the exact probe sample already (no
        # further random sub-slicing needed/wanted).
        # idle_timeout stays tight (catches real hangs fast); max_timeout must be
        # generous enough to cover DDP startup (~20-30s) plus the probe's own
        # training loop, or slow-but-progressing probes get killed and
        # misread as crashes -- observed live twice now: per_gpu=3 killed at a
        # too-tight 45s while still advancing normally, then per_gpu=12 killed
        # at 90s once probe_images grew to 8000 (4x the images -> ~4x the loop
        # time). Scale the cap with sample size instead of a fixed guess.
        max_timeout = max(90, int(sample_n / 2000 * 90) + 60)
        return _probe(total, imgsz, weights, manifest_yaml, devices, 1.0,
                      idle_timeout=20, max_timeout=max_timeout, log_path=log_path)

    # start small (per_gpu=12 — a size known to actually fit in practice) and
    # search from there instead of starting at a huge speculative guess like
    # 128, which historically ballooned every probe into the idle/max timeout.
    START_PER_GPU = 12
    best = None
    per_gpu = START_PER_GPU
    tag, res = probe(per_gpu * n_gpus)
    if tag == "PROBE_OK" and res["peak_mib_max"] <= target_mib:
        best = {**res, "peak_pct": round(100 * res["peak_mib_max"] / board_mib, 1)}
        print(f"  per_gpu={per_gpu} OK ({best['peak_pct']}%) -> doubling up")
        # coarse doubling search upward until we overshoot / OOM / crash
        while True:
            cand = per_gpu * 2
            total = cand * n_gpus
            tag, res = probe(total)
            if tag == "PROBE_OK" and res["peak_mib_max"] <= target_mib:
                best = {**res, "peak_pct": round(100 * res["peak_mib_max"] / board_mib, 1)}
                per_gpu = cand
                print(f"  per_gpu={per_gpu} OK ({best['peak_pct']}%) -> doubling up")
            else:
                reason = "over target" if tag == "PROBE_OK" else tag
                print(f"  per_gpu={cand} {reason} -> stop doubling, refine upward by 1")
                break
    else:
        # per_gpu=12 itself didn't fit -> coarse halving search downward
        reason = "over target" if tag == "PROBE_OK" else tag
        print(f"  per_gpu={per_gpu} {reason} -> halving down")
        per_gpu //= 2
        while per_gpu >= 2:
            total = per_gpu * n_gpus
            tag, res = probe(total)
            if tag == "PROBE_OK":
                peak = res["peak_mib_max"]
                print(f"  per_gpu={per_gpu:<4} total={total:<4} peak={peak} MiB "
                      f"({100*peak/board_mib:.0f}%) target={target_mib:.0f}")
                if peak <= target_mib:
                    best = {**res, "peak_pct": round(100 * peak / board_mib, 1)}
                    break
                print(f"  per_gpu={per_gpu} over target -> halving")
            elif tag == "PROBE_OOM":
                print(f"  per_gpu={per_gpu} OOM -> halving")
            else:
                print(f"  per_gpu={per_gpu} CRASH -> halving: {res.get('error')}")
            per_gpu //= 2

    if best is None:
        raise PreflightError(
            f"Even per-GPU batch 2 exceeds {g.get('vram_target_pct',85)}% VRAM on "
            f"{board_mib} MiB boards, or crashes outright. Lower imgsz/model_size "
            f"in config/global.yaml.")

    # linear refine upward by 1 from the last known-good point until we
    # cross the target or hit a crash
    per_gpu = best["per_gpu_batch"]
    while True:
        cand = per_gpu + 1
        total = cand * n_gpus
        tag, res = probe(total)
        if tag == "PROBE_OK" and res["peak_mib_max"] <= target_mib:
            best = {**res, "peak_pct": round(100 * res["peak_mib_max"] / board_mib, 1)}
            per_gpu = cand
        elif tag == "PROBE_FAIL":
            print(f"  per_gpu={cand} CRASH during refine (treated as too large): "
                  f"{res.get('error')}")
            break
        else:
            break

    payload = {
        "total_batch": best["per_gpu_batch"] * n_gpus,
        "per_gpu_batch": best["per_gpu_batch"], "n_gpus": n_gpus,
        "imgsz": imgsz, "peak_mib": best["peak_mib_max"], "board_mib": board_mib,
        "peak_pct": best["peak_pct"], "target_pct": g.get("vram_target_pct", 85),
        "devices": devices, "vram_ladder_rung": rung, "vram_ladder_buffer": buffer,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(payload, open(cache_path, "w"), indent=2)
    print(f"CHOSEN total={payload['total_batch']} per_gpu={payload['per_gpu_batch']} "
          f"live={payload['peak_pct']}% (<= {payload['target_pct']}%)")
    return payload

if __name__ == "__main__":
    find_optimal_batch(force="--force" in sys.argv)
