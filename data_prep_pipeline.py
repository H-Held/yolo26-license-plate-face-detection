#!/usr/bin/env python3
"""
data_prep_pipeline.py  –  YOLO26m face + license-plate @ 640x640

GARANTIERT:
  - Split VOR Augmentation (leakage-frei!)
  - Max 5 Tiles GESAMT pro Originalbild
  - BG <= 5% in ALLEN Splits
  - Milde photometrische Augmentation NUR auf Train-Tiles
"""

import json, os, random, shutil, sys, math
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

ROOT = Path("/home/jovyan/shared/s0598584/deepseek")
CROPS_1280 = ROOT / "crops_1280"
COCO_JSON = CROPS_1280 / "annotations_coco.json"
DATASET_640 = ROOT / "dataset_face_lp_640"

SEED = 42
TARGET_SIZE = 640
MAX_TILES_PER_SOURCE = 5      # Max Tiles GESAMT pro Originalbild (inkl. Augments)
MAX_TRAIN_SOURCE_TILES = 2    # Max Original-Tiles pro Quelle im Train (werden augmentiert)
MAX_BG_RATIO = 0.05           # Max 5% Background

FACE_CAT_ID = None
LP_CAT_ID = None


def extract_source_key(filename):
    stem = Path(filename).stem
    parts = stem.split('_')
    for i, part in enumerate(parts):
        if part.startswith('r') and len(part) > 1:
            rest = part[1:]
            if rest and rest[0].isdigit():
                return '_'.join(parts[:i])
    return stem


def build_transform():
    import albumentations as A
    return A.Compose([
        A.OneOf([A.GaussianBlur(blur_limit=(3,7), p=1.0),
                 A.MotionBlur(blur_limit=(3,9), p=1.0)], p=0.3),
        A.OneOf([A.GaussNoise(std_range=(0.03,0.12), p=1.0),
                 A.ISONoise(color_shift=(0.01,0.05), intensity=(0.1,0.4), p=1.0)], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=(-0.25,0.15), contrast_limit=(-0.2,0.2), p=0.5),
        A.RandomGamma(gamma_limit=(80,130), p=0.2),
        A.CLAHE(clip_limit=(1,4), tile_grid_size=(8,8), p=0.15),
    ])


def process_one(args):
    src, dst, augment, seed = args
    try:
        bgr = cv2.imread(src)
        if bgr is None:
            return (dst, False, "read_error")
        if augment:
            random.seed(seed)
            np.random.seed(seed % (2**32-1))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            aug_img = build_transform()(image=rgb)["image"]
            bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(bgr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return (dst, True, None)
    except Exception as e:
        return (dst, False, str(e))


def yolo_bbox(coco_bbox, img_w, img_h):
    x, y, w, h = coco_bbox
    return [(x + w/2)/img_w, (y + h/2)/img_h, w/img_w, h/img_h]


def main():
    print("=" * 60)
    print("  DATENVORBEREITUNG YOLO26m face + license-plate @ 640")
    print("=" * 60)

    # 1. LADEN
    print("\n[1] COCO laden...")
    coco = json.loads(COCO_JSON.read_text(encoding='utf-8'))
    print(f"    {len(coco['images'])} Tiles, {len(coco['annotations'])} Annotationen")

    id_by_name = {c['name']: c['id'] for c in coco['categories']}
    global FACE_CAT_ID, LP_CAT_ID
    FACE_CAT_ID = id_by_name['face']
    LP_CAT_ID = id_by_name['license-plate']
    TARGET_CATS = {FACE_CAT_ID, LP_CAT_ID}

    anns_by_image = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    tiles_with_target = set()
    for img_id, anns in anns_by_image.items():
        if any(a['category_id'] in TARGET_CATS for a in anns):
            tiles_with_target.add(img_id)

    print(f"    Tiles mit face/lp: {len(tiles_with_target)}")
    print(f"    Tiles ohne Zielobjekt: {len(coco['images']) - len(tiles_with_target)}")

    # 2. QUELLEN
    print("\n[2] Nach Originalbildern gruppieren...")
    source_tiles_all = defaultdict(list)
    for img in coco['images']:
        src_key = extract_source_key(img['file_name'])
        source_tiles_all[src_key].append(img)

    all_sources = sorted(source_tiles_all.keys())
    print(f"    {len(all_sources)} Originalbilder")

    # 3. SPLIT
    print("\n[3] Leakage-freier Split (70/15/15)...")
    rng = random.Random(SEED)
    rng.shuffle(all_sources)
    n_train = int(len(all_sources) * 0.70)
    n_val   = int(len(all_sources) * 0.15)

    train_sources = set(all_sources[:n_train])
    val_sources   = set(all_sources[n_train:n_train+n_val])
    test_sources  = set(all_sources[n_train+n_val:])
    print(f"    Train: {len(train_sources)}  Val: {len(val_sources)}  Test: {len(test_sources)}")

    # 4. TILES AUSWAHLEN
    print(f"\n[4] Tiles auswahlen...")
    print(f"    Train: max {MAX_TRAIN_SOURCE_TILES} Ziel-Tiles/Quelle (werden augmentiert -> {MAX_TRAIN_SOURCE_TILES*2} total)")
    print(f"    Val/Test: max {MAX_TILES_PER_SOURCE} Ziel-Tiles/Quelle, BG <= {MAX_BG_RATIO*100:.0f}%")

    def select_train_tiles(sources):
        """Train: max 2 Ziel-Tiles pro Quelle, keine BG-Tiles."""
        selected = []
        for src in sorted(sources):
            tiles = source_tiles_all.get(src, [])
            target_tiles = [t for t in tiles if t['id'] in tiles_with_target]
            if not target_tiles:
                continue
            local_rng = random.Random(SEED + hash(src) % 10000)
            n = min(len(target_tiles), MAX_TRAIN_SOURCE_TILES)
            chosen = local_rng.sample(target_tiles, n) if len(target_tiles) > n else list(target_tiles)
            for t in chosen:
                selected.append((t, True))
        return selected

    def select_valtest_tiles(sources, split_name):
        """Val/Test: max 5 Ziel-Tiles/Quelle, KEINE BG-Tiles."""
        selected = []
        for src in sorted(sources):
            tiles = source_tiles_all.get(src, [])
            target_tiles = [t for t in tiles if t['id'] in tiles_with_target]
            if not target_tiles:
                continue
            local_rng = random.Random(SEED + hash(src) % 10000)
            n = min(len(target_tiles), MAX_TILES_PER_SOURCE)
            chosen = local_rng.sample(target_tiles, n) if len(target_tiles) > n else list(target_tiles)
            for t in chosen:
                selected.append((t, True))

        print(f"    {split_name}: {len(selected)} Tiles, 0 BG (0.0%)")
        return selected

    train_selected = select_train_tiles(train_sources)
    print(f"    train: {len(train_selected)} Tiles (nur Ziel-Objekte)")
    val_selected   = select_valtest_tiles(val_sources,   "val")
    test_selected  = select_valtest_tiles(test_sources,  "test")

    # 5. AUGMENTATION
    print(f"\n[5] Augmentation (Train: jedes Tile -> Original + 1 Augment)...")
    train_original_count = len(train_selected)
    train_aug_count = train_original_count  # 1:1
    train_total_after_aug = train_original_count + train_aug_count
    print(f"    Train: {train_original_count} Orig + {train_aug_count} Aug = {train_total_after_aug} Tiles")
    print(f"    Val:   {len(val_selected)} Tiles")
    print(f"    Test:  {len(test_selected)} Tiles")

    # 6. RESIZE + AUG
    print(f"\n[6] Resize + Augmentierung auf {TARGET_SIZE}x{TARGET_SIZE}...")

    for sn in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            d = DATASET_640 / sub / sn
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    tasks = []
    dst_to_tile = {}

    for split_name, selected in [("val", val_selected), ("test", test_selected)]:
        for tile, is_target in selected:
            src_path = str(CROPS_1280 / "images" / tile['file_name'])
            if not os.path.exists(src_path):
                continue
            stem = Path(tile['file_name']).stem
            dst_name = f"{stem}_640.png"
            dst_path = str(DATASET_640 / "images" / split_name / dst_name)
            tasks.append((src_path, dst_path, False, SEED + len(tasks)))
            dst_to_tile[dst_path] = (split_name, tile['id'])

    for tile, is_target in train_selected:
        src_path = str(CROPS_1280 / "images" / tile['file_name'])
        if not os.path.exists(src_path):
            continue
        stem = Path(tile['file_name']).stem
        tile_id = tile['id']

        # v0 = Original
        dst_v0 = str(DATASET_640 / "images" / "train" / f"{stem}_v0_640.png")
        tasks.append((src_path, dst_v0, False, SEED + len(tasks)))
        dst_to_tile[dst_v0] = ("train", tile_id)

        # v1 = augmentiert
        dst_v1 = str(DATASET_640 / "images" / "train" / f"{stem}_v1_640.png")
        tasks.append((src_path, dst_v1, True, SEED + len(tasks)))
        dst_to_tile[dst_v1] = ("train", tile_id)

    print(f"    {len(tasks)} Tasks")

    ok = 0
    failed = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs)):
            dst, success, err = fut.result()
            if success:
                ok += 1
            else:
                failed.append((dst, err))
            if (i+1) % 500 == 0:
                print(f"      {i+1}/{len(tasks)}")

    print(f"    Erfolgreich: {ok}/{len(tasks)}")
    if failed:
        print(f"    Fehler: {len(failed)}")

    # 7. YOLO-LABELS
    print("\n[7] YOLO-Labels schreiben...")
    label_stats = Counter()
    empty_count = 0
    label_count = 0

    for dst_path, (split_name, tile_id) in dst_to_tile.items():
        anns = anns_by_image.get(tile_id, [])
        lines = []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in TARGET_CATS:
                continue
            cx, cy, nw, nh = yolo_bbox(ann['bbox'], 1280, 1280)
            yolo_cls = 0 if cat_id == FACE_CAT_ID else 1
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            label_stats[yolo_cls] += 1

        lbl_path = DATASET_640 / "labels" / split_name / f"{Path(dst_path).stem}.txt"
        if lines:
            lbl_path.write_text('\n'.join(lines))
            label_count += 1
        else:
            lbl_path.write_text('')
            empty_count += 1

    print(f"    Labels mit Objekten: {label_count}")
    print(f"    Leere Labels (BG):   {empty_count}")
    print(f"    face: {label_stats[0]}, license-plate: {label_stats[1]}")

    # 8. dataset.yaml
    print("\n[8] dataset.yaml...")
    (DATASET_640 / "dataset.yaml").write_text(
        f"path: {DATASET_640}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "nc: 2\nnames:\n  0: face\n  1: license-plate\n"
    )

    # 9. VERIFIKATION
    print("\n" + "=" * 60)
    print("  VERIFIKATION")
    print("=" * 60)
    checks = []

    for sn in ["train", "val", "test"]:
        imgs = sorted((DATASET_640 / "images" / sn).glob("*.png"))
        lbls = sorted((DATASET_640 / "labels" / sn).glob("*.txt"))
        print(f"\n  [{sn}] {len(imgs)} Bilder, {len(lbls)} Labels")

        i_stems = {p.stem for p in imgs}
        l_stems = {p.stem for p in lbls}
        match = (i_stems == l_stems)
        checks.append(("Match", sn, match))
        print(f"    {'OK' if match else 'FAIL'} Match")

        if imgs:
            s = cv2.imread(str(imgs[0]))
            if s is not None:
                h, w = s.shape[:2]
                sz_ok = (w == 640 and h == 640)
                checks.append(("640x640", sn, sz_ok))
                print(f"    {'OK' if sz_ok else 'FAIL'} {w}x{h}")

        n_empty = sum(1 for l in lbls if l.stat().st_size == 0)
        bg_pct = n_empty / max(1, len(lbls)) * 100
        bg_ok = bg_pct <= 5
        checks.append(("BG<=5%", sn, bg_ok))
        print(f"    {'OK' if bg_ok else 'FAIL'} BG: {n_empty} ({bg_pct:.1f}%)")

    print("\n  Leakage:")
    src_sets = {}
    for sn in ["train", "val", "test"]:
        src_sets[sn] = set()
        for p in (DATASET_640 / "images" / sn).glob("*.png"):
            src_sets[sn].add(extract_source_key(p.name))

    for a, b in [("train","val"), ("train","test"), ("val","test")]:
        overlap = src_sets[a] & src_sets[b]
        no_leak = len(overlap) == 0
        checks.append(("Leakage", f"{a}<->{b}", no_leak))
        tag = "OK" if no_leak else f"FAIL ({len(overlap)})"
        print(f"    {tag} {a} <-> {b}")

    print("\n  Varianten (Train):")
    train_var = defaultdict(int)
    for p in (DATASET_640 / "images" / "train").glob("*.png"):
        train_var[extract_source_key(p.name)] += 1

    max_v = max(train_var.values()) if train_var else 0
    over = sum(1 for v in train_var.values() if v > MAX_TILES_PER_SOURCE)
    var_ok = max_v <= MAX_TILES_PER_SOURCE and over == 0
    checks.append(("MaxTiles", f"<={MAX_TILES_PER_SOURCE}", var_ok))
    print(f"    Max: {max_v} (Limit: {MAX_TILES_PER_SOURCE}), >Limit: {over}  {'OK' if var_ok else 'FAIL'}")

    # DISTRIBUTION
    print(f"\n  Verteilung je Split:")
    print(f"    {'Split':<8} {'Bilder':<8} {'face':<8} {'license-plate':<14}")
    for sn in ["train", "val", "test"]:
        imgs = list((DATASET_640 / "images" / sn).glob("*.png"))
        lbls = list((DATASET_640 / "labels" / sn).glob("*.txt"))
        face_n = 0; lp_n = 0
        for lp in lbls:
            txt = lp.read_text().strip()
            if txt:
                for line in txt.split('\n'):
                    if line.startswith('0 '):
                        face_n += 1
                    elif line.startswith('1 '):
                        lp_n += 1
        print(f"    {sn:<8} {len(imgs):<8} {face_n:<8} {lp_n:<14}")

    print("\n" + "=" * 60)
    failed_checks = [c for c in checks if not c[2]]
    if failed_checks:
        print(f"  {len(failed_checks)} CHECK(S) FEHLGESCHLAGEN:")
        for name, loc, _ in failed_checks:
            print(f"    - {name} ({loc})")
    else:
        print("  ALLE CHECKS BESTANDEN!")
    print("=" * 60)


if __name__ == "__main__":
    main()