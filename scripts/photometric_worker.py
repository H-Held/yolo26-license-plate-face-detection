"""
photometric_worker.py
----------------------
Picklable top-level function for parallel photometric augmentation
(ProcessPoolExecutor). Writes lossless PNGs with a low compression level
(fast, still lossless).

Each task: one source crop -> 1 original (v0) + N_VARIANTS augmented PNGs.
Every task uses its own deterministic seed (reproducible).
"""
import os
import random
import numpy as np
import cv2
import albumentations as A

# PNG compression level 1 = fast, lossless (vs. default 3)
_PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def build_transform():
    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
        ], p=0.4),
        A.OneOf([
            A.GaussNoise(std_range=(0.05, 0.2), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.4),
        A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.25),
                                   contrast_limit=(-0.3, 0.3), p=0.6),
        A.RandomGamma(gamma_limit=(70, 140), p=0.3),
        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.2),
        A.RandomFog(fog_coef_range=(0.05, 0.3), alpha_coef=0.08, p=0.15),
        A.Downscale(scale_range=(0.5, 0.9), p=0.15),
        A.ImageCompression(quality_range=(35, 80), p=0.25),
    ])


def process_one(task):
    """
    task = dict(src_path, stem, out_dir, n_variants, seed)
    Returns list of output filenames written (v0..vN), or [] on read error.
    Resumable: already-existing PNGs are skipped.
    """
    src_path = task["src_path"]
    stem = task["stem"]
    out_dir = task["out_dir"]
    n_variants = task["n_variants"]

    rng_seed = task["seed"]
    random.seed(rng_seed)
    np.random.seed(rng_seed % (2**32 - 1))

    bgr = cv2.imread(src_path)
    if bgr is None:
        return {"stem": stem, "ok": False, "files": []}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    transform = build_transform()
    written = []

    # v0 = original (lossless copy)
    v0 = f"{stem}_v0.png"
    p0 = os.path.join(out_dir, v0)
    if not os.path.exists(p0):
        cv2.imwrite(p0, bgr, _PNG_PARAMS)
    written.append(v0)

    for v in range(1, n_variants + 1):
        name = f"{stem}_v{v}.png"
        p = os.path.join(out_dir, name)
        if not os.path.exists(p):
            aug = transform(image=rgb)["image"]
            aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            cv2.imwrite(p, aug_bgr, _PNG_PARAMS)
        written.append(name)

    return {"stem": stem, "ok": True, "files": written}
