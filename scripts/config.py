"""
============================================================================
 config.py  —  THE ONE FILE YOU EDIT
============================================================================
You do NOT need to understand the rest of the code. To change how training
behaves, edit the values in the numbered sections below and re-run the
notebooks (00..06) or `python run_campaign.py`.

Everything else (multi-GPU, batch size, fine-tuning, evaluation, per-class
confidence) is derived automatically from the settings here.
"""
from pathlib import Path

# ===========================================================================
# 1. WHERE EVERYTHING LIVES
# ---------------------------------------------------------------------------
# One working directory that holds raw data, datasets, models and run outputs.
# ===========================================================================
ROOT = Path("/home/jovyan/shared/s0598584")   # <- change to your own path

# ===========================================================================
# 2. WHICH MODEL TO TRAIN  (the most common thing to change)
# ---------------------------------------------------------------------------
# MODEL_SIZE picks the network size. Bigger = more accurate but slower and
# needs more GPU memory:   n < s < m < l < x
#   'n' = nano    'l' = large
#   's' = small   'x' = extra-large
#   'm' = medium
# ===========================================================================
BASE_MODEL = "yolo26"     # architecture family (keep "yolo26")
MODEL_SIZE = "n"          # <-- one of: n, s, m, l, x

# Build on top of an EXISTING model instead of the official pretrained weights?
#   None                 -> start from the official "yolo26<size>.pt"
#   "runs/.../best.pt"   -> continue training / fine-tune YOUR own checkpoint
#                           (path is relative to ROOT or absolute)
INIT_FROM = None

# ===========================================================================
# 3. CLASSES  (add or remove lines to change how many classes are trained)
# ---------------------------------------------------------------------------
# index -> human-readable name. The number of classes (nc) is counted
# automatically. The data-prep notebooks (03b) build the dataset to match.
# ===========================================================================
CLASSES = {
    0: "face",
    1: "license-plate",
    # 2: "person",        # <- just add a line to train another class
}

# ===========================================================================
# 4. IMAGE SIZE fed to the model (pixels). Tiles are 1280; set 640 to train
#    on down-scaled ("compressed") tiles.
# ===========================================================================
IMGSZ = 1280

# ===========================================================================
# 5. GPUs
# ---------------------------------------------------------------------------
#   "all"      -> use EVERY GPU of the current server profile (recommended)
#   [0]        -> only the first GPU
#   [0, 1, 2]  -> these specific GPUs
# ===========================================================================
GPUS = "all"

# ===========================================================================
# 6. TRAINING SCHEDULE
# ---------------------------------------------------------------------------
# The pipeline trains in TWO phases automatically:
#   (a) COARSE: train up to EPOCHS, but stop early after PATIENCE epochs with
#       no validation improvement.
#   (b) FINE-TUNE: take the BEST checkpoint from (a) and fine-tune it for
#       FINETUNE_EPOCHS more epochs at a lower learning rate.
# ===========================================================================
EPOCHS          = 400     # upper bound for the coarse phase
WARMUP_EPOCHS   = 10
PATIENCE        = 50      # early-stop patience (epochs without val improvement)
FINETUNE_EPOCHS = 75      # fine-tune the best checkpoint for this many epochs
FINETUNE_LR0    = 0.005   # lower learning rate for the fine-tune phase

# Optimizer (coarse phase)
OPTIMIZER    = "SGD"
LR0          = 0.01
LRF          = 0.01
MOMENTUM     = 0.937
WEIGHT_DECAY = 0.0005

# Data caching: "ram" is fastest if the dataset fits in memory, "disk" caches
# decoded tiles on disk, False disables caching. "auto" picks ram/disk based on
# the dataset size, image size and number of GPUs vs the memory limit.
CACHE   = "auto"
WORKERS = 4               # dataloader workers PER GPU process

# ===========================================================================
# 7. AUGMENTATION (applied during training) — tweak freely
# ---------------------------------------------------------------------------
# These are passed straight to Ultralytics. Lower values = gentler.
# mosaic/mixup downscale objects and hurt small-object recall, so they are off.
# ===========================================================================
AUG = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,        # colour jitter
    degrees=10.0, translate=0.1, scale=0.3,   # geometric
    shear=2.0, perspective=0.0002,
    flipud=0.0, fliplr=0.5,                    # flips (no upside-down faces)
    mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
)
# Overrides used ONLY in the fine-tune phase (usually gentler):
FINETUNE_AUG = dict(mosaic=0.0, scale=0.1)

# ===========================================================================
# 8. OFFLINE PHOTOMETRIC AUGMENTATION  (data-prep notebook 02 / 03b)
# ===========================================================================
PHOTO_FACTOR_FOCUS  = 20   # crops WITH a target class -> Nx copies
PHOTO_FACTOR_REST   = 10   # other crops -> Nx copies
PHOTO_VARIANTS_KEEP = 4    # how many photometric variants enter the train set
BG_RATIO            = 0.12 # fraction of hard-negative background tiles

# ===========================================================================
# 9. PER-CLASS CONFIDENCE THRESHOLD (found automatically after evaluation)
# ---------------------------------------------------------------------------
# After training, the pipeline finds the best confidence threshold FOR EACH
# CLASS on the test+validation images combined, and logs it to a file.
#   CONF_METRIC = "f1"      -> threshold that maximises F1 (balanced)
#   CONF_METRIC = "recall"  -> highest-recall threshold that still keeps
#                              precision >= CONF_MIN_PRECISION
# ===========================================================================
CONF_METRIC        = "f1"
CONF_MIN_PRECISION = 0.50

# ===========================================================================
# 10. ACCEPTANCE TARGETS (for the PASS/FAIL report). Classes not listed here
#     are reported but not checked.
# ===========================================================================
TARGETS = {
    "face":          {"recall": 0.95, "precision": 0.85},
    "license-plate": {"recall": 0.90, "precision": 0.90},
}

# ===========================================================================
#  DERIVED HELPERS  —  no need to edit below this line
# ===========================================================================
def num_classes() -> int:
    return len(CLASSES)

def class_names() -> list:
    return [CLASSES[i] for i in sorted(CLASSES)]

def init_weights(model_size: str = None, init_from=None) -> str:
    """Path to the weights to start from."""
    src = init_from if init_from is not None else INIT_FROM
    if src:
        p = Path(src)
        return str(p if p.is_absolute() else ROOT / p)
    size = model_size or MODEL_SIZE
    return str(ROOT / f"{BASE_MODEL}{size}.pt")

def resolve_gpus():
    """Return the list of GPU indices to use (or None for CPU)."""
    import torch
    if not torch.cuda.is_available():
        return None
    if GPUS == "all":
        n = torch.cuda.device_count()
        return list(range(n)) if n else None
    return list(GPUS)

def dataset_yaml(name: str = "dataset_face_lp") -> Path:
    return ROOT / name / "dataset.yaml"
