"""
============================================================================
 train_pipeline.py  —  the training engine (you usually don't edit this)
============================================================================
Reusable building blocks driven by config.py:

  * resolve_devices()      -> use ALL GPUs of the profile (or those in config)
  * find_optimal_batch()   -> real per-GPU VRAM probe, scaled to all GPUs
  * coarse_train()         -> train until early stop
  * finetune()             -> fine-tune the BEST checkpoint (after early stop)
  * evaluate()             -> per-class P/R on the test split + PASS/FAIL report
  * per_class_best_conf()  -> best confidence threshold PER CLASS on test+val,
                              logged to a file so it is easy to extract
  * run_full()             -> the whole thing, resumable via a state file

A "spec" is a small dict describing ONE run; anything it does not set falls
back to config.py. This lets one config drive many runs (e.g. 1280 vs 640).
"""
import os, sys, json, glob, shutil, subprocess, time
from pathlib import Path

# config.py lives next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import piheif_fix  # noqa: F401  (must precede ultralytics import)
import config as C

os.environ.setdefault("YOLO_CONFIG_DIR", str(C.ROOT / "ultralytics_cfg"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

OUTPUT_ROOT = C.ROOT / "runs" / "faces"
SCRIPTS_DIR = Path(__file__).resolve().parent


def log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


# ---------------------------------------------------------------------------
# spec helpers — a spec overrides config defaults for a single run
# ---------------------------------------------------------------------------
def make_spec(name, data=None, imgsz=None, model_size=None, init_from=None,
              cache=None, **extra):
    spec = dict(
        name=name,
        data=str(data) if data else str(C.dataset_yaml()),
        imgsz=imgsz if imgsz is not None else C.IMGSZ,
        model_size=model_size or C.MODEL_SIZE,
        init_from=init_from,            # None -> official pretrained for model_size
        cache=cache if cache is not None else C.CACHE,
    )
    spec.update(extra)
    return spec


# ---------------------------------------------------------------------------
# GPUs & memory
# ---------------------------------------------------------------------------
def resolve_devices():
    devs = C.resolve_gpus()
    if devs is None:
        return "cpu", 0
    return (devs if len(devs) > 1 else devs[0]), len(devs)


def _cgroup_mem_limit_gb():
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v.isdigit():
                return int(v) / 1e9
        except Exception:
            pass
    return 64.0


def auto_cache(spec, n_train, n_val, nproc):
    """Pick ram/disk so cached tiles (x nproc DDP processes) fit the mem limit."""
    if spec["cache"] != "auto":
        return spec["cache"]
    imgsz = spec["imgsz"]
    need_gb = (n_train + n_val) * imgsz * imgsz * 3 / 1e9 * max(1, nproc)
    limit = _cgroup_mem_limit_gb()
    choice = "ram" if need_gb < 0.55 * limit else "disk"
    log(f"auto cache: ~{need_gb:.1f} GB needed (x{nproc} procs) vs {limit:.0f} GB limit -> '{choice}'")
    return choice


def _count_split(data_yaml, split):
    base = Path(data_yaml).parent
    for ext in ("png", "jpg", "jpeg"):
        n = len(glob.glob(str(base / "images" / split / f"*.{ext}")))
        if n:
            return n
    return 0


# ---------------------------------------------------------------------------
# 1. batch finder (real VRAM probe, per GPU, scaled to all GPUs)
# ---------------------------------------------------------------------------
def _build_probe_subset(data_yaml, tag, n=400):
    src = Path(data_yaml).parent
    probe = C.ROOT / f"_probe_{tag}"
    if probe.exists():
        shutil.rmtree(probe)
    for s in ("train", "val"):
        (probe / "images" / s).mkdir(parents=True)
        (probe / "labels" / s).mkdir(parents=True)
    lbl_dir = src / "labels" / "train"
    img_dir = src / "images" / "train"
    pos = sorted(f for f in os.listdir(lbl_dir)
                 if (lbl_dir / f).stat().st_size > 0)[:n]
    # find the matching image extension
    def img_for(stem):
        for ext in ("png", "jpg", "jpeg"):
            p = img_dir / f"{stem}.{ext}"
            if p.exists():
                return p
        return None
    for fn in pos:
        stem = os.path.splitext(fn)[0]
        ip = img_for(stem)
        if not ip:
            continue
        for s in ("train", "val"):
            shutil.copy(lbl_dir / fn, probe / "labels" / s / fn)
            os.symlink(os.path.realpath(ip), probe / "images" / s / ip.name)
    names = "\n".join(f"  {i}: {C.CLASSES[i]}" for i in sorted(C.CLASSES))
    (probe / "dataset.yaml").write_text(
        f"path: {probe}\ntrain: images/train\nval: images/val\ntest: images/val\n"
        f"nc: {C.num_classes()}\nnames:\n{names}\n")
    return probe


def _probe_total(total_batch, imgsz, weights, data, devices_str, nproc):
    """One isolated 1-epoch probe at a TOTAL batch, under real DDP if nproc>1.

    Returns (fit, peak_gb). peak_gb is only available for single-GPU probes."""
    r = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "batch_finder.py"),
         str(total_batch), str(imgsz), weights, data, devices_str],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak, ok = None, False
    for line in r.stdout.splitlines():
        if line.startswith("PROBE_OK"):
            ok = True
            peak = json.loads(line[len("PROBE_OK"):]).get("peak_gb")
    fit = (r.returncode == 0 and ok)
    pg = total_batch // nproc
    tail = "" if fit else "\n   " + "\n   ".join(r.stdout.strip().splitlines()[-3:])
    log(f"  total={total_batch:>3} (per-GPU {pg}) -> {'OK ' if fit else 'no'} "
        f"{'peak=' + str(peak) + ' GB' if peak else ''}{tail}")
    return fit, peak


def find_optimal_batch(spec, nproc, force=False):
    """Find the largest PER-GPU batch that fits **under real DDP across all GPUs**.

    Probes total batch = per_gpu * nproc on every GPU at once, so the result
    already accounts for DDP overhead. Refines in steps of 1 (per GPU) so it can
    land on e.g. per-GPU 5 instead of over-promising 6. The result is profile-
    aware: a cached value is reused only if it was measured for the same model,
    imgsz AND number of GPUs."""
    import torch
    weights = C.init_weights(spec["model_size"], spec.get("init_from"))
    imgsz = spec["imgsz"]
    tag = f"{spec['model_size']}_{imgsz}"
    out = C.ROOT / "runs" / f"optimal_batch_{tag}.json"
    if out.exists() and not force:
        cached = json.loads(out.read_text())
        if cached.get("n_gpus") == nproc:
            log(f"using cached batch probe {out.name}: per_gpu={cached['per_gpu_batch']} "
                f"x {nproc} gpu = total {cached['total_batch']}")
            return cached["total_batch"], cached
        log(f"cached probe is for {cached.get('n_gpus')} GPU(s) but now {nproc} — re-measuring")

    devices_str = ",".join(str(i) for i in range(nproc)) if nproc > 1 else "0"
    probe = _build_probe_subset(spec["data"], tag, n=max(128, 8 * nproc))
    probe_yaml = str(probe / "dataset.yaml")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    mode = f"DDP across {nproc} GPUs" if nproc > 1 else "single GPU"
    log(f"batch probe ({mode}): {Path(weights).name} @ {imgsz}, {total_vram:.1f} GB/GPU")

    results = {}  # per_gpu -> (fit, peak)

    def probe_pg(pg):
        fit, peak = _probe_total(pg * nproc, imgsz, weights, probe_yaml, devices_str, nproc)
        results[pg] = (fit, peak)
        return fit

    log("coarse grid (per-GPU batch):")
    for pg in (2, 4, 6, 8, 12, 16, 24, 32, 48):
        if not probe_pg(pg):
            break
    passed = sorted(pg for pg, (f, _) in results.items() if f)
    failed = sorted(pg for pg, (f, _) in results.items() if not f)
    if not passed:
        shutil.rmtree(probe, ignore_errors=True)
        raise RuntimeError(f"even per-GPU batch 2 fails on {nproc} GPU(s) — "
                           f"model/imgsz too large, or a non-OOM error (check probe output)")
    lo = max(passed)
    hi = min(failed) if failed else None
    if hi is not None and hi - lo > 1:
        log(f"refining per-GPU between {lo} (OK) and {hi} (fail), step 1:")
        for pg in range(lo + 1, hi):
            if not probe_pg(pg):
                break

    per_gpu = max(pg for pg, (f, _) in results.items() if f)
    peak_at = results[per_gpu][1]
    payload = {
        "per_gpu_batch": int(per_gpu),
        "n_gpus": int(nproc),
        "total_batch": int(per_gpu * nproc),
        "imgsz": imgsz, "model": f"{C.BASE_MODEL}{spec['model_size']}",
        "measured_under_ddp": nproc > 1,
        "peak_vram_gb": peak_at, "total_vram_gb": round(total_vram, 2),
        "probes": {str(pg): {"fit": f, "peak_gb": p} for pg, (f, p) in sorted(results.items())},
    }
    out.write_text(json.dumps(payload, indent=2))
    shutil.rmtree(probe, ignore_errors=True)
    log(f"OPTIMAL per-GPU batch = {per_gpu} -> total {per_gpu * nproc} across {nproc} GPU "
        f"(measured under {'DDP' if nproc > 1 else 'single GPU'})")
    return per_gpu * nproc, payload


# ---------------------------------------------------------------------------
# 2. coarse training (until early stop)
# ---------------------------------------------------------------------------
def _resume_if_possible(name, run_dir):
    """If an interrupted run has a last.pt + args.yaml, resume it. Returns the
    best.pt path on success, or None if there was nothing to resume."""
    from ultralytics import YOLO
    last = run_dir / "weights" / "last.pt"
    if not (last.exists() and (run_dir / "args.yaml").exists()):
        return None
    log(f"resuming '{name}' from {last} (interrupted run found)")
    try:
        YOLO(str(last)).train(resume=True)
        log(f"resume of '{name}' completed")
        return run_dir / "weights" / "best.pt"
    except Exception as e:
        # ultralytics refuses resume once the schedule is already finished -> the
        # run is effectively done; use its best.pt.
        best = run_dir / "weights" / "best.pt"
        if best.exists():
            log(f"resume not needed for '{name}' ({e!r}); using existing best.pt")
            return best
        log(f"could not resume '{name}' ({e!r}); will restart it from scratch")
        return None


def coarse_train(spec, device, total_batch, cache):
    from ultralytics import YOLO
    weights = C.init_weights(spec["model_size"], spec.get("init_from"))
    name = spec["name"]
    run_dir = OUTPUT_ROOT / name
    resumed = _resume_if_possible(name, run_dir)
    if resumed is not None:
        return resumed
    log(f"COARSE train '{name}': init={Path(weights).name} imgsz={spec['imgsz']} "
        f"batch={total_batch} device={device} cache={cache}")
    model = YOLO(weights)
    model.train(
        data=spec["data"], imgsz=spec["imgsz"], epochs=C.EPOCHS,
        patience=C.PATIENCE, device=device, batch=total_batch,
        workers=C.WORKERS, cache=cache, save=True, save_period=50,
        project=str(OUTPUT_ROOT), name=name, exist_ok=True,
        optimizer=C.OPTIMIZER, lr0=C.LR0, lrf=C.LRF, momentum=C.MOMENTUM,
        weight_decay=C.WEIGHT_DECAY, warmup_epochs=C.WARMUP_EPOCHS,
        cos_lr=True, amp=True, plots=True, verbose=True,
        conf=0.001, iou=0.5, **C.AUG,
    )
    best = OUTPUT_ROOT / name / "weights" / "best.pt"
    log(f"COARSE done -> {best}")
    return best


# ---------------------------------------------------------------------------
# 3. fine-tune the best checkpoint
# ---------------------------------------------------------------------------
def finetune(spec, best_pt, device, total_batch, cache):
    from ultralytics import YOLO
    name = spec["name"] + "_ft"
    run_dir = OUTPUT_ROOT / name
    resumed = _resume_if_possible(name, run_dir)
    if resumed is not None:
        return resumed
    log(f"FINE-TUNE '{name}' from {Path(best_pt).name}: {C.FINETUNE_EPOCHS} epochs lr0={C.FINETUNE_LR0}")
    aug = dict(C.AUG); aug.update(C.FINETUNE_AUG)
    model = YOLO(str(best_pt))
    model.train(
        data=spec["data"], imgsz=spec["imgsz"], epochs=C.FINETUNE_EPOCHS,
        patience=C.FINETUNE_EPOCHS, device=device, batch=total_batch,
        workers=C.WORKERS, cache=cache, save=True, save_period=25,
        project=str(OUTPUT_ROOT), name=name, exist_ok=True,
        optimizer=C.OPTIMIZER, lr0=C.FINETUNE_LR0, lrf=0.01, momentum=C.MOMENTUM,
        weight_decay=C.WEIGHT_DECAY, warmup_epochs=1, cos_lr=True, amp=True,
        plots=True, verbose=True, conf=0.001, iou=0.5, **aug,
    )
    ft_best = OUTPUT_ROOT / name / "weights" / "best.pt"
    log(f"FINE-TUNE done -> {ft_best}")
    return ft_best


# ---------------------------------------------------------------------------
# 4. evaluation on the test split (per-class P/R + PASS/FAIL)
# ---------------------------------------------------------------------------
def _per_class_from_metrics(metrics, names):
    ap_idx = list(metrics.box.ap_class_index)
    pos = {int(c): i for i, c in enumerate(ap_idx)}
    out = {}
    for ci, nm in names.items():
        if ci in pos:
            p, r, ap50, ap = metrics.box.class_result(pos[ci])
            out[nm] = dict(class_id=int(ci), precision=float(p), recall=float(r),
                           ap50=float(ap50), ap50_95=float(ap), present=True)
        else:
            out[nm] = dict(class_id=int(ci), precision=None, recall=None,
                           ap50=None, ap50_95=None, present=False)
    return out


def evaluate(spec, best_pt, device):
    from ultralytics import YOLO
    name = spec["name"]
    run_dir = OUTPUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(best_pt))
    log(f"EVAL '{name}' on TEST split")
    metrics = model.val(
        data=spec["data"], imgsz=spec["imgsz"], split="test", conf=0.001, iou=0.6,
        plots=True, save_json=True, device=(device if not isinstance(device, list) else device[0]),
        project=str(OUTPUT_ROOT), name=f"{name}_TEST", exist_ok=True,
    )
    names = model.names
    per_class = _per_class_from_metrics(metrics, names)

    # PASS/FAIL vs targets
    focus, overall = {}, True
    for cname, thr in C.TARGETS.items():
        m = per_class.get(cname)
        if not m or not m["present"] or m["recall"] is None:
            focus[cname] = {"present": False}; overall = False; continue
        r_ok = m["recall"] >= thr["recall"]; p_ok = m["precision"] >= thr["precision"]
        mvp = r_ok and p_ok; overall = overall and mvp
        focus[cname] = dict(present=True, recall=m["recall"], precision=m["precision"],
                            recall_threshold=thr["recall"], precision_threshold=thr["precision"],
                            recall_pass=r_ok, precision_pass=p_ok, mvp_pass=mvp)
    report = dict(run=name, model=f"{C.BASE_MODEL}{spec['model_size']}",
                  checkpoint=str(best_pt), split="test", imgsz=spec["imgsz"],
                  overall=dict(map50=float(metrics.box.map50), map50_95=float(metrics.box.map)),
                  per_class=per_class, targets=dict(focus=focus, overall_mvp_pass=overall))
    (run_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    _write_eval_md(run_dir / "eval_report.md", report, metrics)
    log(f"EVAL done: mAP50={metrics.box.map50:.4f}  overall MVP={'PASS' if overall else 'FAIL'}")
    return report


def _write_eval_md(path, report, metrics):
    def f(v): return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"
    L = [f"# Evaluation — {report['run']} (TEST split)", "",
         f"- Checkpoint: `{report['checkpoint']}`",
         f"- mAP50: **{f(report['overall']['map50'])}**  |  mAP50-95: **{f(report['overall']['map50_95'])}**",
         "", "| Class | Precision | Recall | AP50 | AP50-95 |", "|---|---|---|---|---|"]
    for nm, m in report["per_class"].items():
        L.append(f"| {nm} | {f(m['precision'])} | {f(m['recall'])} | {f(m['ap50'])} | {f(m['ap50_95'])} |")
    L += ["", f"**Overall MVP: {'PASS' if report['targets']['overall_mvp_pass'] else 'FAIL'}**", ""]
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# 5. per-class best confidence threshold on TEST + VAL combined
# ---------------------------------------------------------------------------
def _combined_yaml(spec, tag):
    """Build a temp dataset whose 'val' split = val + test images merged."""
    src = Path(spec["data"]).parent
    comb = C.ROOT / f"_combined_{tag}"
    if comb.exists():
        shutil.rmtree(comb)
    (comb / "images" / "all").mkdir(parents=True)
    (comb / "labels" / "all").mkdir(parents=True)
    for split in ("val", "test"):
        for lbl in glob.glob(str(src / "labels" / split / "*.txt")):
            stem = Path(lbl).stem
            # copy label, symlink image
            dst_lbl = comb / "labels" / "all" / f"{split}_{stem}.txt"
            shutil.copy(lbl, dst_lbl)
            for ext in ("png", "jpg", "jpeg"):
                ip = src / "images" / split / f"{stem}.{ext}"
                if ip.exists():
                    os.symlink(os.path.realpath(ip),
                               comb / "images" / "all" / f"{split}_{stem}.{ext}")
                    break
    names = "\n".join(f"  {i}: {C.CLASSES[i]}" for i in sorted(C.CLASSES))
    (comb / "dataset.yaml").write_text(
        f"path: {comb}\ntrain: images/all\nval: images/all\ntest: images/all\n"
        f"nc: {C.num_classes()}\nnames:\n{names}\n")
    return comb / "dataset.yaml"


def _find_curve(metrics, ylabel):
    """Find a (x, y) curve in metrics.box.curves_results by its y-axis label."""
    for entry in getattr(metrics.box, "curves_results", []):
        # entry = [x_array, y_array(nc, N), xlabel, ylabel]
        if len(entry) >= 4 and entry[3] == ylabel and entry[2] == "Confidence":
            return entry[0], entry[1]
    return None, None


def per_class_best_conf(spec, best_pt, device, log_csv):
    """Sweep confidence per class on val+test combined; log best threshold."""
    from ultralytics import YOLO
    import numpy as np
    name = spec["name"]
    tag = f"{spec['model_size']}_{spec['imgsz']}_{name}"
    comb_yaml = _combined_yaml(spec, tag)
    model = YOLO(str(best_pt))
    log(f"CONF sweep '{name}' on val+test combined")
    metrics = model.val(
        data=str(comb_yaml), imgsz=spec["imgsz"], split="val", conf=0.001, iou=0.6,
        plots=False, device=(device if not isinstance(device, list) else device[0]),
        project=str(OUTPUT_ROOT), name=f"{name}_CONF", exist_ok=True,
    )
    x_f1, f1 = _find_curve(metrics, "F1")
    x_p, pc = _find_curve(metrics, "Precision")
    x_r, rc = _find_curve(metrics, "Recall")
    names = model.names
    ap_idx = list(metrics.box.ap_class_index)
    pos = {int(c): i for i, c in enumerate(ap_idx)}

    per_class = {}
    for ci, nm in names.items():
        if ci not in pos or f1 is None:
            per_class[nm] = dict(class_id=int(ci), present=False)
            continue
        row = pos[ci]
        f1_row = np.asarray(f1[row]); x = np.asarray(x_f1)
        if C.CONF_METRIC == "recall" and rc is not None and pc is not None:
            r_row = np.asarray(rc[row]); p_row = np.asarray(pc[row])
            mask = p_row >= C.CONF_MIN_PRECISION
            idx = int(np.argmax(np.where(mask, r_row, -1.0))) if mask.any() else int(np.argmax(f1_row))
        else:
            idx = int(np.argmax(f1_row))
        best_conf = float(x[idx])
        per_class[nm] = dict(
            class_id=int(ci), present=True, best_conf=round(best_conf, 4),
            f1_at_best=round(float(f1_row[idx]), 4),
            precision_at_best=round(float(pc[row][idx]), 4) if pc is not None else None,
            recall_at_best=round(float(rc[row][idx]), 4) if rc is not None else None,
            metric=C.CONF_METRIC,
        )

    run_dir = OUTPUT_ROOT / name
    payload = dict(run=name, model=f"{C.BASE_MODEL}{spec['model_size']}",
                   imgsz=spec["imgsz"], checkpoint=str(best_pt),
                   eval_set="val+test combined", metric=C.CONF_METRIC,
                   per_class=per_class)
    (run_dir / "best_conf.json").write_text(json.dumps(payload, indent=2))

    # append to a global, easy-to-extract CSV log
    header = "run,model,imgsz,class,class_id,best_conf,f1,precision,recall,metric\n"
    rows = []
    for nm, m in per_class.items():
        if not m.get("present"):
            continue
        rows.append(f"{name},{C.BASE_MODEL}{spec['model_size']},{spec['imgsz']},{nm},"
                    f"{m['class_id']},{m['best_conf']},{m['f1_at_best']},"
                    f"{m['precision_at_best']},{m['recall_at_best']},{m['metric']}")
    log_csv = Path(log_csv)
    if not log_csv.exists():
        log_csv.write_text(header)
    with open(log_csv, "a") as fh:
        fh.write("\n".join(rows) + ("\n" if rows else ""))

    shutil.rmtree(comb_yaml.parent, ignore_errors=True)
    log(f"CONF done -> {run_dir/'best_conf.json'} (+ appended {log_csv})")
    for nm, m in per_class.items():
        if m.get("present"):
            log(f"   {nm:16s} best_conf={m['best_conf']:.3f} "
                f"(F1={m['f1_at_best']:.3f} P={m['precision_at_best']} R={m['recall_at_best']})")
    return payload


# ---------------------------------------------------------------------------
# 6. full pipeline with resumable state
# ---------------------------------------------------------------------------
def _state_path(name):
    return OUTPUT_ROOT / name / "pipeline_state.json"


def _load_state(name):
    p = _state_path(name)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"stages": {}}


def _save_state(name, state):
    p = _state_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def run_full(spec):
    """COARSE -> FINE-TUNE -> EVAL -> PER-CLASS CONF, resumable per stage."""
    name = spec["name"]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device, nproc = resolve_devices()
    n_train = _count_split(spec["data"], "train")
    n_val = _count_split(spec["data"], "val")
    cache = auto_cache(spec, n_train, n_val, nproc)
    state = _load_state(name)
    log(f"=== RUN '{name}' === device={device} gpus={nproc} train/val={n_train}/{n_val} cache={cache}")

    # stage: batch
    total_batch, _ = find_optimal_batch(spec, nproc)

    # stage: coarse
    if not state["stages"].get("coarse"):
        best = coarse_train(spec, device, total_batch, cache)
        state["stages"]["coarse"] = {"best": str(best)}
        _save_state(name, state)
    best = Path(state["stages"]["coarse"]["best"])

    # stage: finetune (always run -> "best so far + fine-tune 50-100 epochs")
    if not state["stages"].get("finetune"):
        ft_best = finetune(spec, best, device, total_batch, cache)
        state["stages"]["finetune"] = {"best": str(ft_best)}
        _save_state(name, state)
    final_best = Path(state["stages"]["finetune"]["best"])

    # stage: eval
    if not state["stages"].get("eval"):
        report = evaluate(spec, final_best, device)
        state["stages"]["eval"] = {"map50": report["overall"]["map50"]}
        _save_state(name, state)

    # stage: per-class conf
    if not state["stages"].get("conf"):
        conf_log = C.ROOT / "runs" / "best_conf_log.csv"
        per_class_best_conf(spec, final_best, device, conf_log)
        state["stages"]["conf"] = {"done": True}
        _save_state(name, state)

    state["stages"]["complete"] = True
    _save_state(name, state)
    log(f"=== RUN '{name}' COMPLETE -> {final_best} ===")
    return {"name": name, "final_best": str(final_best), "state": state}
