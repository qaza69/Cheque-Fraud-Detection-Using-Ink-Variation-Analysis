"""
Handwritten-ink extraction.

Template mode (cheque matches a known bank layout):
    1. Affine-align the scan to the blank template.
    2. A pixel is handwriting only if it is clearly darker than the *darkest*
       template pixel in its 5x5 neighbourhood.  Printed text, lines, logos,
       security patterns and coloured backgrounds are all present in the
       template, so they can never pass this test (the neighbourhood makes
       it tolerant to 1-2 px alignment error).
    3. Components lying mostly on dark printed content (residue around the
       printed Rupee sign, MICR digits, ...) are discarded.
    4. Only the handwritten *fields* (payee, amount in words, amount in
       figures, date, signature) are kept.

Generic mode (unknown layout): local paper estimate + darkness threshold,
long ruled lines removed, margins / MICR band ignored.  Printed text cannot
be separated reliably in this mode, so results are flagged as lower
confidence.

Ink colour is always measured as *optical density* relative to the local
paper, OD = -log(I / I_paper), which cancels the paper tint (cream, blue,
green security backgrounds) and scanner brightness.
"""
from dataclasses import dataclass, field
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.ink.templates import BANK_ROIS, BANK_NAMES, TemplateLibrary, to_canonical_width

# Fields where tampering is checked by default.  Date and signature are
# commonly written in advance by the account holder (in IDRBT they are
# identical on every cheque), so a different pen there is not evidence of
# alteration; they can be included explicitly.
DEFAULT_FIELDS = ('payee', 'words', 'figures')
FIELD_LABELS = {'payee': 'Payee name', 'words': 'Amount (words)',
                'figures': 'Amount (figures)', 'date': 'Date',
                'signature': 'Signature', 'all': 'Handwriting'}

INK_MARGIN = 0.25        # OD above darkest nearby printed pixel
INK_MIN_OD = 0.6         # summed OD over the 3 channels vs. paper
MIN_COMPONENT_PX = 25
MIN_UNIT_PX = 150


def optical_density(img, paper):
    """Per-channel OD of `img` relative to `paper` (both uint8 BGR)."""
    return -np.log((img.astype(np.float32) + 1.0) / (paper.astype(np.float32) + 1.0))


def paper_background(img_bgr):
    """Local paper colour: max-filter removes thin strokes, median smooths."""
    return cv2.medianBlur(cv2.dilate(img_bgr, np.ones((9, 9), np.uint8)), 31)


@dataclass
class InkMap:
    image: np.ndarray            # aligned / normalised BGR cheque
    paper: np.ndarray            # local paper colour (BGR)
    ink: np.ndarray              # bool handwriting mask (full image)
    rois: dict                   # field -> (x1, y1, x2, y2)
    mode: str                    # 'template' or 'generic'
    bank: str = None             # bank key in template mode
    match_cc: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def bank_name(self):
        return BANK_NAMES.get(self.bank, 'Unknown layout')

    def field_mask(self, fields):
        m = np.zeros_like(self.ink)
        for f in fields:
            if f in self.rois:
                x1, y1, x2, y2 = self.rois[f]
                m[y1:y2, x1:x2] = self.ink[y1:y2, x1:x2]
        return m


def _filter_components(mask, min_px, drop_fn=None):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
    if drop_fn is not None:
        keep[1:] &= ~drop_fn(lab, st, n)[1:]
    return keep[lab]


def _template_ink(aligned, tpl):
    darkest_print = cv2.erode(tpl, np.ones((5, 5), np.uint8))
    paper = paper_background(aligned)
    d_print = optical_density(aligned, darkest_print).max(axis=2)
    od = optical_density(aligned, paper).sum(axis=2)
    ink = (d_print > INK_MARGIN) & (od > INK_MIN_OD)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    tpl_paper = cv2.medianBlur(cv2.dilate(tpl, np.ones((7, 7), np.uint8)), 21)
    printed = optical_density(tpl, tpl_paper).sum(axis=2) > 0.8
    printed = cv2.dilate(printed.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float64)

    def on_print(lab, st, n):
        frac = np.bincount(lab.ravel(), weights=printed.ravel(), minlength=n)
        return frac / np.maximum(st[:, cv2.CC_STAT_AREA], 1) >= 0.5

    return _filter_components(ink, MIN_COMPONENT_PX, on_print), paper


def _generic_ink(img):
    paper = paper_background(img)
    od = optical_density(img, paper).sum(axis=2)
    ink = cv2.morphologyEx((od > INK_MIN_OD).astype(np.uint8), cv2.MORPH_OPEN,
                           np.ones((2, 2), np.uint8))

    def ruled_lines(lab, st, n):
        w, h = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]
        area = st[:, cv2.CC_STAT_AREA]
        long_thin = (w > 15 * np.maximum(h, 1)) | (h > 15 * np.maximum(w, 1))
        sparse_box = (area < 0.08 * w * h) & ((w > 300) | (h > 300))   # rectangles
        return long_thin | sparse_box

    return _filter_components(ink, MIN_COMPONENT_PX, ruled_lines), paper


def extract_ink(image, library=None, rgb=True):
    """
    Isolate handwritten ink in a cheque image.

    Args:
        image: cheque image (H, W, 3) uint8, or a path
        library: TemplateLibrary (loaded lazily if None)
        rgb: True if `image` is RGB (Streamlit/PIL), False for BGR (cv2)

    Returns:
        InkMap
    """
    if isinstance(image, (str, os.PathLike)):
        img = cv2.imread(os.fspath(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        img = np.ascontiguousarray(image)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if rgb else cv2.COLOR_BGRA2BGR)
        elif rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)

    library = library if library is not None else TemplateLibrary()
    bank, tpl, aligned, cc = library.match(img) if len(library) else (None, None, None, -1)

    if bank is not None:
        ink, paper = _template_ink(aligned, tpl)
        return InkMap(aligned, paper, ink, dict(BANK_ROIS[bank]), 'template', bank, cc)

    img = to_canonical_width(img)
    ink, paper = _generic_ink(img)
    h, w = img.shape[:2]
    mx, my = int(0.03 * w), int(0.04 * h)
    rois = {'all': (mx, my, w - mx, int(0.86 * h))}     # skip MICR band
    notes = ["Cheque layout not recognised - printed text could not be removed "
             "with a template, so the verdict is less reliable."]
    return InkMap(img, paper, ink, rois, 'generic', None, max(cc, 0.0), notes)


def segment_units(inkmap, fields=DEFAULT_FIELDS, min_px=MIN_UNIT_PX):
    """
    Split the handwriting of each field into small ink units (~one or two
    letters / one digit).  Each unit is compared against every other unit.

    Returns:
        list of dict(field, x, y, w, h, mask) - mask is a bool crop of the
        unit's bounding box in full-image coordinates.
    """
    if inkmap.mode == 'generic':
        fields = ['all']
    units = []
    for f in fields:
        if f not in inkmap.rois:
            continue
        x1, y1, x2, y2 = inkmap.rois[f]
        m = inkmap.ink[y1:y2, x1:x2].astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        heights = [st[i, cv2.CC_STAT_HEIGHT] for i in range(1, n)
                   if st[i, cv2.CC_STAT_AREA] >= min_px]
        if not heights:
            continue
        xh = float(np.clip(np.median(heights), 30, 120))
        chunk_w = 1.3 * xh
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area < min_px:
                continue
            comp = lab[y:y + h, x:x + w] == i
            k = max(1, int(round(w / chunk_w)))
            edges = np.linspace(0, w, k + 1).astype(int)
            for a, b in zip(edges[:-1], edges[1:]):
                sub = comp[:, a:b]
                rows = np.where(sub.any(1))[0]
                if sub.sum() < min_px or rows.size == 0:
                    continue
                sub = sub[rows[0]:rows[-1] + 1]
                units.append(dict(field=f, x=x1 + x + a, y=y1 + y + rows[0],
                                  w=b - a, h=sub.shape[0], mask=sub))
    return units
