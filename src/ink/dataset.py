"""
Builds the pen-labelled ink-unit dataset from IDRBT.

For each cheque, every handwritten unit gets:
    label  1 = written by pen 1 (original filling)
           2 = written by pen 2 (the alteration)
           0 = unknown
    genuine_keep  True if the unit stays in the cheque's *pen-1-only*
                  ("genuine") version used for cheque-level evaluation

Labels come from the metadata (no manual annotation):
  * Amount in figures - the leading k digits were written by pen 2
    (k = number of digits pen 2 wrote), the following digits by pen 1.
  * Amount in words   - the leading k words of the first line were written
    by pen 2; everything after the k-th word gap was written by pen 1.
  * Payee             - pen 2's name is a prefix *or* suffix: left unlabelled.

Run:  python -m src.ink.dataset
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.ink.idrbt import load_metadata
from src.ink.templates import TemplateLibrary
from src.ink.extraction import extract_ink, segment_units
from src.ink.features import unit_features, unit_patch, OD_SCALE

DATASET_PATH = os.path.join(config.OUTPUT_DIR, "ink_dataset.npz")
ALL_FIELDS = ('payee', 'words', 'figures', 'date', 'signature')


def _components(mask, min_px=60):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return lab, st, [i for i in range(1, n) if st[i, 4] >= min_px]


def figures_boundaries(inkmap, k_digits):
    """
    Locate pen-2 / pen-1 boundaries in the amount-in-figures box.

    Returns (pen2_end_x, pen1_start_x, pen1_end_x) in image coords, or None.
    """
    x1, y1, x2, y2 = inkmap.rois['figures']
    lab, st, comps = _components(inkmap.ink[y1:y2, x1:x2])
    if len(comps) < 3:
        return None
    hmed = np.median([st[i, 3] for i in comps])
    top = np.median([st[i, 1] for i in comps if st[i, 3] >= 0.5 * hmed])
    # digits: tall enough, top on the digit line (commas start lower)
    digits = [i for i in comps if st[i, 3] >= 0.5 * hmed
              and st[i, 1] <= top + 0.3 * hmed and st[i, 2] < 3 * hmed]
    digits.sort(key=lambda i: st[i, 0])
    groups = []                         # merge horizontally overlapping parts
    for i in digits:
        xa, xb = st[i, 0], st[i, 0] + st[i, 2]
        if groups and xa < groups[-1][1] - 0.5 * st[i, 2]:
            groups[-1][1] = max(groups[-1][1], xb)
        else:
            groups.append([xa, xb])
    if len(groups) < 4:
        return None
    digit_w = np.median([b - a for a, b in groups[1:-1]])
    pen2_end = groups[0][0] + k_digits * digit_w * 0.85
    gi = 0
    while gi < len(groups) and groups[gi][0] < pen2_end:
        gi += 1
    if gi + 1 >= len(groups):
        return None
    # the last group is the trailing "/" - its pen is unknown
    return x1 + pen2_end, x1 + groups[gi][0], x1 + groups[-1][0]


def words_boundaries(inkmap, k_words):
    """
    Returns (line1_bottom_y, pen1_start_x) for the amount-in-words field:
    pen 2 wrote the first k words of line 1.
    """
    x1, y1, x2, y2 = inkmap.rois['words']
    m = inkmap.ink[y1:y2, x1:x2]
    lab, st, comps = _components(m, min_px=200)
    if not comps:
        return None
    xh = float(np.clip(np.median([st[i, 3] for i in comps]), 35, 120))

    prof = np.convolve(m.sum(1), np.ones(5) / 5, 'same')
    rows = np.where(prof > 0.1 * prof.max())[0]
    start, bottom = rows[0], m.shape[0]
    quiet = prof < 0.03 * prof.max()
    run = 0
    for y in range(int(start + 0.8 * xh), m.shape[0]):
        run = run + 1 if quiet[y] else 0
        if run >= 0.25 * xh:
            bottom = y
            break
    # a second line only counts if it holds a real share of the ink
    if m[bottom:].sum() < 0.15 * m.sum():
        bottom = m.shape[0]
    line1 = m[:bottom]

    merged = segment_words(line1, xh)
    if len(merged) <= k_words:
        return None
    return y1 + bottom, x1 + merged[k_words][0]


def segment_words(mask, xh=None):
    """
    Split one handwritten line into words by column gaps.

    Returns a list of [x_start, x_end] (inclusive) in mask coordinates.
    """
    if xh is None:
        lab, st, comps = _components(mask, min_px=200)
        if not comps:
            return []
        xh = float(np.clip(np.median([st[i, 3] for i in comps]), 35, 120))
    cols = np.where(mask.any(0))[0]
    if cols.size == 0:
        return []
    segs = [[cols[0], cols[0]]]
    for x in cols[1:]:
        if x - segs[-1][1] > 0.6 * xh:
            segs.append([x, x])
        else:
            segs[-1][1] = x
    # merge slivers (detached letter parts) into the nearer neighbour
    merged = []
    for a, b in segs:
        if merged and (b - a < 0.8 * xh or merged[-1][1] - merged[-1][0] < 0.8 * xh):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return merged


def label_units(units, inkmap, row):
    labels = np.zeros(len(units), np.int8)
    keep = np.ones(len(units), bool)          # pen-1-only ("genuine") version
    fb = figures_boundaries(inkmap, int(row.k_digits))
    wb = words_boundaries(inkmap, int(row.k_words))
    for j, u in enumerate(units):
        cx, cy = u['x'] + u['w'] / 2, u['y'] + u['h'] / 2
        if u['field'] == 'figures':
            if fb is None:
                keep[j] = False
                continue
            pen2_end, pen1_start, slash_x = fb
            if cx < pen2_end:
                labels[j] = 2
            elif pen1_start <= cx < slash_x:
                labels[j] = 1
            keep[j] = cx >= pen1_start
        elif u['field'] == 'words':
            if wb is None:
                keep[j] = False
                continue
            line1_bottom, pen1_start = wb
            on_line1 = u['y'] < line1_bottom
            if on_line1 and cx < pen1_start:
                keep[j] = False                 # pen 2's leading words
            elif u['x'] >= pen1_start or not on_line1:
                labels[j] = 1
        else:
            keep[j] = False                     # payee / date / signature
    return labels, keep


def build_dataset(path=DATASET_PATH, limit=None):
    meta = load_metadata()
    if limit:
        meta = meta.head(limit)
    lib = TemplateLibrary()
    if not len(lib):
        raise RuntimeError("No templates found - run `python -m src.ink.templates` first.")

    feats, patches, recs = [], [], []
    t0 = time.time()
    for n, row in enumerate(meta.itertuples()):
        inkmap = extract_ink(row.path, lib, rgb=False)
        if inkmap.mode != 'template':
            print(f"  [WARN] {row.cheque}: template match failed (cc={inkmap.match_cc:.2f})")
            continue
        units = segment_units(inkmap, ALL_FIELDS)
        labels, keep = label_units(units, inkmap, row)
        for u, lab, kp in zip(units, labels, keep):
            feats.append(unit_features(inkmap, u))
            p = unit_patch(inkmap, u) * OD_SCALE
            patches.append(np.clip(p / 3.0 * 255, 0, 255).astype(np.uint8))
            recs.append((row.cheque, u['field'], u['x'], u['y'], u['w'], u['h'],
                         int(lab), bool(kp), int(row.pen1), int(row.pen2), int(u['mask'].sum())))
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(meta)} cheques  ({time.time() - t0:.0f}s)")

    rec = np.array(recs, dtype=[('cheque', 'i8'), ('field', 'U10'), ('x', 'i4'), ('y', 'i4'),
                                ('w', 'i4'), ('h', 'i4'), ('label', 'i1'), ('genuine_keep', '?'),
                                ('pen1', 'i1'), ('pen2', 'i1'), ('n_px', 'i4')])
    np.savez_compressed(path, features=np.stack(feats), patches=np.stack(patches), units=rec)
    print(f"  [OK] {len(rec)} units from {len(set(rec['cheque']))} cheques -> {path}")
    return path


def load_dataset(path=DATASET_PATH):
    """Returns (features, patches_float, units_recarray)."""
    d = np.load(path)
    patches = d['patches'].astype(np.float32) / 255.0 * 3.0 / OD_SCALE
    return d['features'], patches, d['units']


if __name__ == "__main__":
    build_dataset()
