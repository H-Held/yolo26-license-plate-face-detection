"""Preflight resource checks with human-readable error messages.

Called before the build (disk/RAM) and before training/batch-finding (VRAM).
The point is that a non-expert gets a sentence they can act on, not a stack trace.
"""
from __future__ import annotations
import os
import shutil


class PreflightError(RuntimeError):
    pass


def _gb(bytes_):
    return bytes_ / (1024 ** 3)


def _existing_ancestor(path: str) -> str:
    p = os.path.abspath(path)
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p or os.path.abspath(os.sep)


def check_disk(path: str, need_gb: float):
    free = _gb(shutil.disk_usage(_existing_ancestor(path)).free)
    if free < need_gb:
        raise PreflightError(
            f"Not enough disk at {path}: {free:.1f} GB free, need ~{need_gb:.1f} GB.\n"
            f"  -> free space (delete old runs_*/dataset_* ) or lower tiling "
            f"(max_pos_tiles, scaled_tiles, class_boost) in the dataset config."
        )
    return free


def _cgroup_mem_limit_gb():
    """Container RAM limit (cgroup v1/v2), or physical RAM as a fallback."""
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(p) as f:
                v = f.read().strip()
            if v not in ("max", ""):
                lim = int(v)
                if lim < (1 << 60):
                    return _gb(lim)
        except Exception:
            pass
    try:
        return _gb(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return None


def check_ram(need_gb: float):
    lim = _cgroup_mem_limit_gb()
    if lim is None:
        return None
    if lim < need_gb:
        raise PreflightError(
            f"RAM limit is {lim:.1f} GB, need ~{need_gb:.1f} GB.\n"
            f"  -> lower the dataset RAM cache (train cache='disk' instead of 'ram'), "
            f"reduce batch, or pick a node with more memory."
        )
    return lim


def gpu_board_mib():
    """Per-GPU total board memory in MiB, or [] if no GPU/driver.

    Uses torch's CUDA runtime query (`get_device_properties`), NOT nvidia-smi/pynvml:
    on some shared clusters (MIG slices) both nvidia-smi --query-gpu and pynvml are
    denied with "Insufficient Permissions" / NVMLError_NoPermission, while the CUDA
    runtime API (which every training process needs anyway) still works fine.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return [int(torch.cuda.get_device_properties(i).total_memory / (1024 ** 2))
                for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def check_vram(min_gib_per_gpu: float, n_gpus_expected: int = None):
    board = gpu_board_mib()
    if not board:
        raise PreflightError(
            "No GPU visible (nvidia-smi failed). The batch finder and training must "
            "run on a GPU node. On JupyterHub pick 'GPU large'; the small node has none."
        )
    if n_gpus_expected and len(board) < n_gpus_expected:
        raise PreflightError(
            f"Expected {n_gpus_expected} GPUs but see {len(board)}. "
            f"Re-select the multi-GPU profile or set devices accordingly."
        )
    small = min(board) / 1024.0
    if small < min_gib_per_gpu:
        raise PreflightError(
            f"Smallest GPU has {small:.1f} GiB, need >= {min_gib_per_gpu:.1f} GiB.\n"
            f"  -> reduce imgsz or model_size in config/global.yaml, or use a bigger GPU."
        )
    return board
