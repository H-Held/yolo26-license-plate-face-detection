#!/usr/bin/env python3
"""
image_crop_augment.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Annotation-aware crop & rotation augmentation for COCO JSON datasets.

New in this version
━━━━━━━━━━━━━━━━━━
• Sub-threshold objects are blacked out in the crop image so the model
  is never penalised for detecting unlabelled partial objects.
  Exception: overlap with a valid labelled annotation is preserved.
• Lossless PNG output by default (--output-format jpeg to override).
• Interactive nesting validation: --interactive shows each detected
  parent-child pair and lets you accept or reject it.
• Stricter nesting detection (85 % containment + 1.5× size ratio)
  to eliminate false positives like car ⊂ motorcycle.

Usage
━━━━━
  python image_crop_augment.py \\
      -i export.zip  -o out_640/  --crop-w 640 --crop-h 640 \\
      -n 500  --workers 8  --interactive
"""

import argparse
import json
import math
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MASK_DS      = 8       # validity-mask downsample factor
_ANCHOR_FRACS = [0.10, 0.50, 0.90]
DIV           = '═' * 66


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Geometry helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def rotation_canvas(img_w, img_h, angle):
    cx, cy = img_w / 2.0, img_h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nw = int(math.ceil(img_h * sin_a + img_w * cos_a))
    nh = int(math.ceil(img_h * cos_a + img_w * sin_a))
    M[0, 2] += (nw - img_w) / 2.0
    M[1, 2] += (nh - img_h) / 2.0
    return M, nw, nh


def bbox_via_matrix(bbox, M):
    x, y, w, h = bbox
    pts   = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float64)
    pts_h = np.hstack([pts, np.ones((4,1))])
    n     = (M @ pts_h.T).T
    return [n[:,0].min(), n[:,1].min(), n[:,0].max()-n[:,0].min(), n[:,1].max()-n[:,1].min()]


def inter_area(a, b):
    ix  = max(a[0], b[0]);  iy  = max(a[1], b[1])
    ix2 = min(a[0]+a[2], b[0]+b[2]); iy2 = min(a[1]+a[3], b[1]+b[3])
    return max(0.0, ix2-ix) * max(0.0, iy2-iy)

def box_iou(a, b):
    ia = inter_area(a, b)
    if not ia: return 0.0
    u = a[2]*a[3] + b[2]*b[3] - ia
    return ia/u if u else 0.0

def visibility(ann, region):
    area = ann[2]*ann[3]
    return inter_area(ann, region)/area if area else 0.0

def clip_to_region(ann, region):
    ix  = max(ann[0], region[0]); iy  = max(ann[1], region[1])
    ix2 = min(ann[0]+ann[2], region[0]+region[2])
    iy2 = min(ann[1]+ann[3], region[1]+region[3])
    if ix2 <= ix+0.5 or iy2 <= iy+0.5: return None
    return [ix-region[0], iy-region[1], ix2-ix, iy2-iy]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Annotation acceptance  (small vs large object logic)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def accept_ann_in_region(rbbox, region, crop_w, crop_h, min_vis):
    """
    Small annotation  → standard: inter/ann_area >= min_vis
    Large annotation  → crop size takes priority, but at least one
                        BOUNDARY EDGE must be visible per overflowing
                        dimension (prevents jacket/interior-only crops).
    """
    aw, ah   = rbbox[2], rbbox[3]
    ann_area = aw * ah
    if ann_area <= 0: return False
    ia = inter_area(rbbox, region)
    if ia < 1.0: return False

    ax, ay = rbbox[0], rbbox[1]
    rx, ry, rw, rh = region

    if aw <= crop_w and ah <= crop_h:
        return (ia / ann_area) >= min_vis

    if aw > crop_w:
        if not (rx <= ax <= rx+rw or rx <= ax+aw <= rx+rw):
            return False
    if ah > crop_h:
        if not (ry <= ay <= ry+rh or ry <= ay+ah <= ry+rh):
            return False
    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Validity mask
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_validity_integral(rot_corners, rot_w, rot_h):
    ds  = _MASK_DS
    mw  = max(1, (rot_w+ds-1)//ds); mh = max(1, (rot_h+ds-1)//ds)
    mask = np.zeros((mh, mw), dtype=np.float32)
    cv2.fillPoly(mask, [(rot_corners/ds).astype(np.int32).reshape(-1,1,2)], 1.0)
    return mask.cumsum(0).cumsum(1), ds

def black_ratio(integral, ds, x, y, w, h):
    H, W = integral.shape
    sx=max(0,x//ds); sy=max(0,y//ds)
    ex=min(W-1,(x+w-1)//ds); ey=min(H-1,(y+h-1)//ds)
    if ex<sx or ey<sy: return 1.0
    v=integral[ey,ex]
    if sx>0: v-=integral[ey,sx-1]
    if sy>0: v-=integral[sy-1,ex]
    if sx>0 and sy>0: v+=integral[sy-1,sx-1]
    total=(ey-sy+1)*(ex-sx+1)
    return max(0.0, 1.0-v/total) if total else 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Nested-annotation detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_parent_map(
    annotations:       List[dict],
    cat_by_id:         Optional[Dict] = None,
    allowed_pairs:     Optional[Set[Tuple[str,str]]] = None,
    containment_thresh: float = 0.85,   # was 0.70 – stricter to avoid false positives
    min_size_ratio:    float = 1.5,     # parent must be ≥ 1.5× child area
) -> Dict[int, int]:
    """
    Returns {child_ann_id: parent_ann_id} for annotations where ≥ containment_thresh
    of the child's bbox lies inside a parent bbox that is ≥ min_size_ratio× larger.

    allowed_pairs: if provided (from interactive validation), only pairs whose
    category names appear in this set are accepted.
    """
    pm: Dict[int, int] = {}
    for ann in annotations:
        ax, ay, aw, ah = ann['bbox']
        ann_area = aw * ah
        if ann_area <= 0: continue
        best_id = None; best_area = 0.0
        for other in annotations:
            if other['id'] == ann['id']: continue
            oa = other['bbox'][2] * other['bbox'][3]
            if oa < ann_area * min_size_ratio: continue        # size-ratio filter
            ia = inter_area(ann['bbox'], other['bbox'])
            if ia / ann_area >= containment_thresh and oa > best_area:
                # Optional: check against user-validated allowed pairs
                if allowed_pairs is not None and cat_by_id is not None:
                    cn = cat_by_id[ann['category_id']]['name']
                    pn = cat_by_id[other['category_id']]['name']
                    if (cn, pn) not in allowed_pairs:
                        continue
                best_id = other['id']; best_area = oa
        if best_id is not None:
            pm[ann['id']] = best_id
    return pm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Interactive nesting validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_nesting_interactive(pairs: Set[Tuple[str,str]]) -> Set[Tuple[str,str]]:
    """
    Present each detected (child, parent) category pair and let the user
    accept or reject it.  Returns the validated set of allowed pairs.
    """
    if not pairs:
        return pairs

    print(f'\n{DIV}')
    print('  🔍  Nesting validation  (detected pairs to review)')
    print(DIV)
    print('      Enter  /  y  = accept    n  = reject')
    print('      a            = accept ALL remaining')
    print('      q            = reject ALL remaining\n')

    allowed: Set[Tuple[str,str]] = set()
    accept_rest = False
    reject_rest = False

    for child_cat, parent_cat in sorted(pairs):
        if accept_rest:
            allowed.add((child_cat, parent_cat))
            print(f'  ✅  {child_cat:20s} ⊂ {parent_cat}  (auto-accepted)')
            continue
        if reject_rest:
            print(f'  ❌  {child_cat:20s} ⊂ {parent_cat}  (auto-rejected)')
            continue

        try:
            ans = input(f'  ?   {child_cat:20s} ⊂ {parent_cat:20s}   [Y/n/a/q]: ').strip().lower()
        except EOFError:
            ans = 'y'   # non-interactive fallback

        if ans in ('', 'y', 'yes'):
            allowed.add((child_cat, parent_cat))
            print(f'       → accepted')
        elif ans in ('a', 'all'):
            allowed.add((child_cat, parent_cat))
            accept_rest = True
        elif ans in ('q', 'quit'):
            reject_rest = True
        else:
            print(f'       → rejected')

    n_acc = len(allowed); n_rej = len(pairs) - n_acc
    print(f'\n  Accepted: {n_acc}   Rejected: {n_rej}')
    return allowed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  9-position anchor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def anchor_positions(rbbox, crop_w, crop_h, max_x, max_y):
    ax, ay, aw, ah = rbbox
    ocx, ocy = ax+aw/2.0, ay+ah/2.0
    result: List[Tuple[int,int]] = []; seen: set = set()
    for fy in _ANCHOR_FRACS:
        for fx in _ANCHOR_FRACS:
            x0 = max(0, min(int(round(ocx-fx*crop_w)), max_x))
            y0 = max(0, min(int(round(ocy-fy*crop_h)), max_y))
            if (x0,y0) not in seen: seen.add((x0,y0)); result.append((x0,y0))
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Rotation angle planning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plan_angles(very_under, under, rng):
    base  = [0.0, 90.0, 180.0, 270.0]
    n_rnd = 8 if very_under else (4 if under else 2)
    extra: List[float] = []; tries = 0
    while len(extra) < n_rnd and tries < 800:
        a = round(rng.uniform(1.0, 359.0), 1)
        if all(min(abs(a-e)%360, abs(e-a)%360)>10.0 for e in base+extra):
            extra.append(a)
        tries += 1
    return base + extra


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Crop planning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plan_image_crops(
    img_id, img_w, img_h, annotations,
    crop_w, crop_h, angles, max_black, min_vis, sx, sy,
    parent_map: Optional[Dict[int,int]] = None,
) -> List[dict]:
    """
    Returns crop specs.  Each spec now carries:
      visible_anns  – labelled (passed acceptance test)
      partial_anns  – have some overlap but FAILED acceptance → will be blacked out
    """
    if parent_map is None:
        parent_map = build_parent_map(annotations)
    rot_by_id_cache: Dict[float, Dict[int, List[float]]] = {}
    results: List[dict] = []

    for angle in angles:
        M, rot_w, rot_h = rotation_canvas(img_w, img_h, angle)
        if rot_w < crop_w or rot_h < crop_h: continue

        c_src = np.array([[0,0],[img_w,0],[img_w,img_h],[0,img_h]], np.float64)
        rot_corners = (M @ np.hstack([c_src, np.ones((4,1))]).T).T
        integral, ds = build_validity_integral(rot_corners, rot_w, rot_h)

        rot_anns = [
            {'id': a['id'], 'category_id': a['category_id'],
             'rbbox': bbox_via_matrix(a['bbox'], M)}
            for a in annotations
        ]
        rot_by_id = {ra['id']: ra for ra in rot_anns}

        max_x = rot_w - crop_w; max_y = rot_h - crop_h

        pos_set: set = set(); pos_list: List[Tuple[int,int]] = []
        def _add(x0, y0):
            if (x0,y0) not in pos_set: pos_set.add((x0,y0)); pos_list.append((x0,y0))

        # 1. Regular grid
        xs = list(range(0, max_x+1, sx))
        if xs and xs[-1]!=max_x: xs.append(max_x)
        ys = list(range(0, max_y+1, sy))
        if ys and ys[-1]!=max_y: ys.append(max_y)
        for y0 in ys:
            for x0 in xs: _add(x0, y0)

        # 2. Annotation-anchored (9 positions per annotation)
        for ann in annotations:
            anchor_id    = parent_map.get(ann['id'], ann['id'])
            anchor_rbbox = rot_by_id[anchor_id]['rbbox']
            for pos in anchor_positions(anchor_rbbox, crop_w, crop_h, max_x, max_y):
                _add(*pos)

        for x0, y0 in pos_list:
            br = black_ratio(integral, ds, x0, y0, crop_w, crop_h)
            if br > max_black: continue

            region = [float(x0), float(y0), float(crop_w), float(crop_h)]
            visible: List[dict] = []
            partial: List[dict] = []   # sub-threshold → will be blacked out

            for ra in rot_anns:
                ia = inter_area(ra['rbbox'], region)
                if ia < 1.0: continue
                if accept_ann_in_region(ra['rbbox'], region, crop_w, crop_h, min_vis):
                    clipped = clip_to_region(ra['rbbox'], region)
                    if clipped:
                        visible.append({
                            'ann_id': ra['id'], 'category_id': ra['category_id'],
                            'local_bbox': clipped, 'visibility': ia/(ra['rbbox'][2]*ra['rbbox'][3]),
                        })
                else:
                    # Some overlap but failed acceptance → record for blackout
                    clipped = clip_to_region(ra['rbbox'], region)
                    if clipped:
                        partial.append({'local_bbox': clipped})

            if visible:
                results.append({
                    'image_id': img_id, 'angle': angle,
                    'crop_region': region,
                    'visible_anns': visible,
                    'partial_anns': partial,
                    'black_ratio': br, 'forced': False,
                })

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Deduplication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def dedup_crops(crops, iou_thresh=0.70):
    groups = defaultdict(list)
    for c in crops: groups[(c['image_id'], c['angle'])].append(c)
    kept = []
    for group in groups.values():
        group.sort(key=lambda c: -len(c['visible_anns']))
        accepted = []
        for crop in group:
            c_ids  = frozenset(v['ann_id']      for v in crop['visible_anns'])
            c_cats = frozenset(v['category_id'] for v in crop['visible_anns'])
            dup = False
            for acc in accepted:
                if box_iou(crop['crop_region'], acc['crop_region']) < iou_thresh: continue
                a_ids  = frozenset(v['ann_id']      for v in acc['visible_anns'])
                a_cats = frozenset(v['category_id'] for v in acc['visible_anns'])
                if c_ids.issubset(a_ids) or c_cats == a_cats: dup=True; break
            if not dup: accepted.append(crop)
        kept.extend(accepted)
    return kept


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Forced crops
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def forced_crops_for_missing(
    missing_ids, anns_by_img, images, img_path_map,
    crop_w, crop_h, min_vis,
    cat_by_id=None, allowed_pairs=None,
):
    """Forced crops anchor on the annotation's OWN bbox (9 positions)."""
    result = []
    for img_id, anns in anns_by_img.items():
        targets = [a for a in anns if a['id'] in missing_ids]
        if not targets or img_id not in img_path_map: continue
        info  = images[img_id]
        max_x = info['width']  - crop_w
        max_y = info['height'] - crop_h
        if max_x < 0 or max_y < 0: continue

        for ann in targets:
            ax, ay, aw, ah = ann['bbox']
            generated = False

            for x0, y0 in anchor_positions(ann['bbox'], crop_w, crop_h, max_x, max_y):
                region = [float(x0), float(y0), float(crop_w), float(crop_h)]
                visible: List[dict] = []; partial: List[dict] = []
                for a in anns:
                    ia = inter_area(a['bbox'], region)
                    if ia < 1.0: continue
                    if accept_ann_in_region(a['bbox'], region, crop_w, crop_h, min_vis):
                        cl = clip_to_region(a['bbox'], region)
                        if cl: visible.append({'ann_id': a['id'], 'category_id': a['category_id'],
                                               'local_bbox': cl, 'visibility': ia/(a['bbox'][2]*a['bbox'][3])})
                    else:
                        cl = clip_to_region(a['bbox'], region)
                        if cl: partial.append({'local_bbox': cl})

                if any(v['ann_id']==ann['id'] for v in visible):
                    result.append({'image_id': img_id, 'file_name': info['file_name'],
                                   'img_path': img_path_map[img_id], 'angle': 0.0,
                                   'crop_region': region, 'visible_anns': visible,
                                   'partial_anns': partial, 'black_ratio': 0.0, 'forced': True})
                    generated = True

            if not generated:   # hard fallback: centre crop, ignoring thresholds
                x0 = int(max(0, min(ax+aw/2-crop_w/2, max_x)))
                y0 = int(max(0, min(ay+ah/2-crop_h/2, max_y)))
                region = [float(x0), float(y0), float(crop_w), float(crop_h)]
                visible = []
                for a in anns:
                    cl = clip_to_region(a['bbox'], region)
                    if cl:
                        visible.append({'ann_id': a['id'], 'category_id': a['category_id'],
                                        'local_bbox': cl, 'visibility': visibility(a['bbox'], region)})
                result.append({'image_id': img_id, 'file_name': info['file_name'],
                               'img_path': img_path_map[img_id], 'angle': 0.0,
                               'crop_region': region, 'visible_anns': visible,
                               'partial_anns': [], 'black_ratio': 0.0, 'forced': True})
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Worker  (module-level → picklable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _worker(task_group: dict) -> List[dict]:
    """
    Load & rotate source image once, then for each crop:
      1. Extract patch
      2. Black out sub-threshold partial-annotation areas
         (unless they overlap with a valid labelled annotation)
      3. Save as PNG (lossless) or JPEG depending on task_group['out_ext']
    """
    img_path  = task_group['img_path']
    angle     = task_group['angle']
    crop_w    = task_group['crop_w']
    crop_h    = task_group['crop_h']
    out_ext   = task_group.get('out_ext', '.png')
    crops     = task_group['crops']
    out: List[dict] = []

    img = cv2.imread(img_path)
    if img is None:
        return [{'idx': c['idx'], 'ok': False, 'err': f'Cannot read: {img_path}'} for c in crops]

    h, w = img.shape[:2]
    if angle != 0.0:
        M, nw, nh = rotation_canvas(w, h, angle)
        rotated = cv2.warpAffine(img, M, (nw, nh),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0,0,0))
    else:
        rotated = img

    rh, rw = rotated.shape[:2]

    for c in crops:
        try:
            x0 = max(0, min(int(c['crop_region'][0]), rw-crop_w))
            y0 = max(0, min(int(c['crop_region'][1]), rh-crop_h))
            patch = rotated[y0:y0+crop_h, x0:x0+crop_w].copy()

            if patch.shape[:2] != (crop_h, crop_w):
                out.append({'idx': c['idx'], 'ok': False, 'err': f'Bad patch shape'}); continue

            # ── Blackout sub-threshold partial objects ─────────────────────
            # Build a pixel mask: True = should be blacked out
            partial_anns  = c.get('partial_anns', [])
            visible_anns  = c.get('visible_anns', [])
            if partial_anns:
                blackout = np.zeros((crop_h, crop_w), dtype=bool)

                for p in partial_anns:
                    px,py,pw,ph = p['local_bbox']
                    px,py = int(max(0,px)), int(max(0,py))
                    px2 = min(crop_w, px+int(pw)+1); py2 = min(crop_h, py+int(ph)+1)
                    blackout[py:py2, px:px2] = True

                # Preserve areas covered by valid labelled annotations
                for v in visible_anns:
                    lx,ly,lw,lh = v['local_bbox']
                    lx,ly = int(max(0,lx)), int(max(0,ly))
                    lx2 = min(crop_w, lx+int(lw)+1); ly2 = min(crop_h, ly+int(lh)+1)
                    blackout[ly:ly2, lx:lx2] = False

                patch[blackout] = 0  # apply blackout

            # ── Save ───────────────────────────────────────────────────────
            os.makedirs(os.path.dirname(c['out_path']), exist_ok=True)
            if out_ext.lower() == '.png':
                ok = cv2.imwrite(c['out_path'], patch)   # lossless
            else:
                # q=100 + 4:4:4 subsampling → maximale JPEG-Qualität ohne Farbverlust
                params = [cv2.IMWRITE_JPEG_QUALITY, 100]
                if hasattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR'):
                    params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, 0x111111]  # 4:4:4
                ok = cv2.imwrite(c['out_path'], patch, params)
            out.append({'idx': c['idx'], 'ok': bool(ok), 'err': '' if ok else 'imwrite failed'})

        except Exception as e:
            out.append({'idx': c['idx'], 'ok': False, 'err': str(e)})

    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    out_ext = '.png' if args.output_format == 'png' else '.jpg'

    # ── 0. Resolve input ──────────────────────────────────────────────────
    src = Path(args.input)
    if not src.exists(): raise SystemExit(f'❌  Not found: {src}')
    if src.suffix.lower() == '.zip':
        ext_dir = Path(args.output) / '_extracted'
        ext_dir.mkdir(parents=True, exist_ok=True)
        print(f'📦  Unzip {src.name} → {ext_dir}')
        with zipfile.ZipFile(src) as zf: zf.extractall(ext_dir)
        src = ext_dir

    ann_files = list(src.rglob('*.json'))
    if not ann_files: raise SystemExit('❌  No JSON annotation file found.')
    pref = [f for f in ann_files if 'coco' in f.name.lower() or 'annot' in f.name.lower()]
    ann_file = pref[0] if pref else ann_files[0]
    print(f'📄  Annotation : {ann_file.relative_to(src)}')
    print(f'🖼️   Output fmt : {"JPEG q=100 (kein Qualitätsverlust)" if out_ext==".jpg" else "PNG (lossless)"}')

    img_dirs = [ann_file.parent]
    if (ann_file.parent / 'images').is_dir(): img_dirs.insert(0, ann_file.parent/'images')
    def find_img(name):
        for d in img_dirs:
            p = d/name
            if p.exists(): return p
        for p in src.rglob(name): return p
        return None

    # ── 1. Load COCO ──────────────────────────────────────────────────────
    with open(ann_file, encoding='utf-8') as f: coco = json.load(f)
    images     = {img['id']: img for img in coco['images']}
    categories = coco['categories']
    cat_by_id  = {c['id']: c for c in categories}

    anns_by_img: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_img[ann['image_id']].append(
            {'id': ann['id'], 'category_id': ann['category_id'], 'bbox': ann['bbox']})

    img_path_map: Dict[int, str] = {}
    for img_id, info in images.items():
        p = find_img(info['file_name'])
        if p: img_path_map[img_id] = str(p)
        else: print(f'  ⚠️   Not found: {info["file_name"]}')

    # ── 2. Analysis ───────────────────────────────────────────────────────
    print(f'\n{DIV}'); print('  📊  Label distribution'); print(DIV)
    cat_cnt = Counter(a['category_id'] for anns in anns_by_img.values() for a in anns)
    total_a = sum(cat_cnt.values())
    cat_pct = {k: v/total_a for k,v in cat_cnt.items()}
    avg_pct = 1.0 / max(len(cat_cnt), 1)
    for cid, cnt in sorted(cat_cnt.items(), key=lambda x:-x[1]):
        nm=cat_by_id[cid]['name']; p=cat_pct[cid]; bar='█'*max(1,int(p*36))
        tag=('  ⚠️  very underrepresented → many rotations' if p<avg_pct*0.3
             else '  ⚡ underrepresented → extra rotations' if p<avg_pct*0.7 else '')
        print(f'  {nm:22s}  {cnt:4d}  ({p*100:5.1f}%)  {bar}{tag}')

    # Detect nested pairs (per image, to avoid cross-image false positives)
    raw_pairs: Set[Tuple[str,str]] = set()
    for img_anns in anns_by_img.values():
        pm = build_parent_map(img_anns, cat_by_id=cat_by_id)
        by_id = {a['id']: a for a in img_anns}
        for cid2, pid in pm.items():
            cn = cat_by_id[by_id[cid2]['category_id']]['name']
            pn = cat_by_id[by_id[pid]['category_id']]['name']
            if cn != pn: raw_pairs.add((cn, pn))

    if raw_pairs:
        print(f'\n  🔗  Nested pairs detected (outer bbox → crop anchor):')
        for c2, p2 in sorted(raw_pairs): print(f'       {c2:20s} ⊂ {p2}')

    print(f'\n  Source images : {len(images)}   Annotations : {total_a}   Categories : {len(categories)}')

    # ── 3. Interactive nesting validation ─────────────────────────────────
    if args.interactive and raw_pairs:
        if sys.stdin.isatty():
            allowed_pairs = validate_nesting_interactive(raw_pairs)
        else:
            print('\n  ⚠️   --interactive specified but stdin is not a TTY → accepting all pairs')
            allowed_pairs = raw_pairs
    else:
        allowed_pairs = raw_pairs   # accept all

    # ── 4. Rotation angles ─────────────────────────────────────────────────
    print(f'\n{DIV}'); print('  🔄  Rotation angles per category'); print(DIV)
    cat_angles: Dict[int, List[float]] = {}
    for cid in cat_cnt:
        p = cat_pct[cid]
        ang = plan_angles(p<avg_pct*0.3, p<avg_pct*0.7, rng)
        cat_angles[cid] = ang
        print(f'  {cat_by_id[cid]["name"]:22s}: {len(ang):2d} angles  '
              f'({", ".join(f"{a:.1f}°" for a in sorted(ang)[:4])} …)')

    # ── 5. Plan crops ─────────────────────────────────────────────────────
    print(f'\n{DIV}'); print('  🔍  Planning crop positions …'); print(DIV)
    stride_x = max(16, args.crop_w // args.stride_divisor)
    stride_y = max(16, args.crop_h // args.stride_divisor)
    print(f'  Crop size   : {args.crop_w} × {args.crop_h} px')
    print(f'  Stride      : {stride_x} × {stride_y} px')
    print(f'  +9 anchors  : per annotation × per angle')
    print(f'  Max black   : {args.max_black_pct:.1f} %   Min visible : {args.min_visibility:.1f} %')

    all_crops: List[dict] = []
    for img_id, info in tqdm(images.items(), desc='  Planning', unit='img'):
        anns = anns_by_img.get(img_id, [])
        if not anns or img_id not in img_path_map: continue

        # Per-image parent map (uses validated allowed_pairs)
        pm = build_parent_map(anns, cat_by_id=cat_by_id, allowed_pairs=allowed_pairs)

        ang_set = {0.0}
        for a in anns: ang_set.update(cat_angles.get(a['category_id'], [0.0]))

        crops = plan_image_crops(
            img_id, info['width'], info['height'], anns,
            args.crop_w, args.crop_h, sorted(ang_set),
            args.max_black_pct/100.0, args.min_visibility/100.0,
            stride_x, stride_y, parent_map=pm,
        )
        for c in crops:
            c['file_name'] = info['file_name']
            c['img_path']  = img_path_map[img_id]
        all_crops.extend(crops)

    print(f'\n  Raw candidates : {len(all_crops):,}')

    # ── 6. Deduplicate ────────────────────────────────────────────────────
    print(f'\n  🧹  Removing near-duplicates …')
    deduped = dedup_crops(all_crops)
    print(f'  After dedup   : {len(deduped):,}')

    # ── 7. Coverage guarantee ─────────────────────────────────────────────
    covered = {v['ann_id'] for c in deduped for v in c['visible_anns']}
    all_ids = {a['id'] for anns in anns_by_img.values() for a in anns}
    missing = all_ids - covered
    if missing:
        print(f'\n  📌  {len(missing)} uncovered → forced crops')
        forced = forced_crops_for_missing(
            missing, anns_by_img, images, img_path_map,
            args.crop_w, args.crop_h, args.min_visibility/100.0,
            cat_by_id=cat_by_id, allowed_pairs=allowed_pairs,
        )
        deduped.extend(forced)
        print(f'      +{len(forced)} forced crops')
    else:
        print(f'\n  ✅  All {len(all_ids)} annotations covered')

    # ── 8. Scale to target ────────────────────────────────────────────────
    f_crops = [c for c in deduped if c.get('forced')]
    o_crops = [c for c in deduped if not c.get('forced')]
    rng.shuffle(o_crops)
    print(f'\n  📦  Available : {len(deduped):,}   Target : {args.target_count:,}')
    if len(deduped) > args.target_count:
        take  = max(0, args.target_count - len(f_crops))
        final = f_crops + o_crops[:take]
        print(f'      → {len(final):,} selected  (forced: {len(f_crops)})')
    else:
        final = deduped
        print(f'      → All {len(final):,} crops used')

    # ── 9. Assign filenames & build task groups ───────────────────────────
    out_dir = Path(args.output); img_out = out_dir/'images'
    img_out.mkdir(parents=True, exist_ok=True)

    for idx, c in enumerate(final):
        stem        = Path(c['file_name']).stem
        tag         = f"r{c['angle']:.1f}".replace('.','p')
        c['out_name'] = f'{stem}_{tag}_{idx:06d}{out_ext}'
        c['out_path'] = str(img_out / c['out_name'])
        c['idx']      = idx

    groups_map: Dict[Tuple[str,float], dict] = {}
    for c in final:
        key = (c['img_path'], c['angle'])
        if key not in groups_map:
            groups_map[key] = {'img_path': c['img_path'], 'angle': c['angle'],
                               'crop_w': args.crop_w, 'crop_h': args.crop_h,
                               'out_ext': out_ext, 'crops': []}
        groups_map[key]['crops'].append({
            'idx':          c['idx'],
            'crop_region':  c['crop_region'],
            'out_path':     c['out_path'],
            'visible_anns': c['visible_anns'],   # for blackout keep-mask
            'partial_anns': c.get('partial_anns', []),
        })

    task_groups = list(groups_map.values())

    # ── 10. Parallel execution ────────────────────────────────────────────
    print(f'\n{DIV}')
    print(f'  ✂️   {len(final):,} crops  ·  {len(task_groups)} groups  ·  {args.workers} workers')
    print(DIV)
    ok_idx: set = set()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, tg): tg for tg in task_groups}
        for fut in tqdm(as_completed(futs), total=len(futs), desc='  Groups', unit='grp'):
            try:
                for r in fut.result():
                    if r['ok']: ok_idx.add(r['idx'])
                    elif r.get('err'): tqdm.write(f"  ✗ [{r['idx']}] {r['err']}")
            except Exception as e:
                tqdm.write(f'  ✗ Worker exception: {e}')
    print(f'\n  ✅  Success: {len(ok_idx):,}   ✗ Failed: {len(final)-len(ok_idx)}')

    # ── 11. Output COCO JSON ──────────────────────────────────────────────
    print(f'\n  📝  Writing annotations …')
    out_imgs: List[dict] = []; out_anns: List[dict] = []; nid=aid=0
    for c in final:
        if c['idx'] not in ok_idx: continue
        out_imgs.append({'id': nid, 'file_name': c['out_name'],
                         'width': args.crop_w, 'height': args.crop_h})
        for v in c['visible_anns']:
            lx,ly,lw,lh = v['local_bbox']
            lx,ly = max(0.0,lx), max(0.0,ly)
            lw,lh = min(float(lw),args.crop_w-lx), min(float(lh),args.crop_h-ly)
            if lw<1 or lh<1: continue
            out_anns.append({'id': aid, 'image_id': nid, 'category_id': v['category_id'],
                             'bbox': [round(lx,2),round(ly,2),round(lw,2),round(lh,2)],
                             'area': round(lw*lh,2), 'iscrowd': 0,
                             'attributes': {'source': 'augmented'}})
            aid += 1
        nid += 1

    ann_out = out_dir/'annotations_coco.json'
    with open(ann_out,'w',encoding='utf-8') as f:
        json.dump({'images':out_imgs,'annotations':out_anns,'categories':categories},f,indent=2,ensure_ascii=False)

    # ── 12. Summary ───────────────────────────────────────────────────────
    out_cc = Counter(a['category_id'] for a in out_anns)
    print(f'\n{DIV}'); print('  ✅  DONE'); print(DIV)
    print(f'  Images      : {nid:,}')
    print(f'  Annotations : {aid:,}')
    print(f'  Output      : {out_dir.resolve()}')
    print(f'  Format      : {"PNG lossless" if out_ext==".png" else "JPEG q=100"}')
    print(f'\n  📊  Output label distribution:')
    for cid, cnt in sorted(out_cc.items(), key=lambda x:-x[1]):
        nm=cat_by_id[cid]['name']; p=cnt/aid if aid else 0
        print(f'    {nm:22s}  {cnt:6,}  ({p*100:5.1f}%)  {"█"*max(1,int(p*36))}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parser():
    p = argparse.ArgumentParser(prog='image_crop_augment', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a('-i','--input',  required=True,  help='Folder with images+COCO JSON, or .zip')
    a('-o','--output', required=True,  help='Output folder')
    a('--crop-w',  type=int,   default=640,  metavar='PX', help='Crop width px [640]')
    a('--crop-h',  type=int,   default=640,  metavar='PX', help='Crop height px [640]')
    a('-n','--target-count', type=int, default=200, metavar='N',
      help='Target number of output images [200]')
    a('--max-black-pct',  type=float, default=15.0,  metavar='%',
      help='Max black border %% from rotation [15.0]')
    a('--min-visibility', type=float, default=20.0,  metavar='%',
      help='Min bbox visibility %% to label [20.0]')
    a('--stride-divisor', type=int,   default=4,     metavar='D',
      help='Grid stride = crop/D [4]')
    a('--output-format',  choices=['png','jpeg'], default='jpeg',
      help='Output image format: jpeg q=100 (default) or png (lossless) [jpeg]')
    a('--interactive', action='store_true',
      help='Interactively validate detected nested label pairs before cropping')
    a('--workers', type=int, default=max(1,(os.cpu_count() or 2)-1),
      metavar='N', help='Parallel workers [cpu-1]')
    a('--seed', type=int, default=42, metavar='S', help='Random seed [42]')
    return p

if __name__ == '__main__':
    run(_parser().parse_args())
