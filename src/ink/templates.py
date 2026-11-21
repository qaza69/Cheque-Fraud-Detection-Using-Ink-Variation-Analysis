"""
Blank cheque templates.

Every cheque of a given bank shares the same printed layout (logo, labels,
lines, security pattern, background tint).  Aligning all scans of one bank
and taking a high per-pixel percentile removes the handwriting (which is
different on every cheque) and leaves a clean "blank cheque".  Subtracting
that template from an aligned scan leaves only what was written by hand.

Templates are built once from the IDRBT dataset:

    python -m src.ink.templates
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Handwritten field boxes (x1, y1, x2, y2) in template pixel coordinates
# (300 dpi scans, ~2365 x 1100 px).  Derived from the ink-occurrence
# heat-map of all cheques of each bank.
BANK_ROIS = {
    'axis': {
        'date': (1790, 80, 2290, 160), 'payee': (100, 190, 1800, 320),
        'words': (100, 320, 1690, 500), 'figures': (1700, 400, 2250, 510),
        'signature': (1880, 640, 2290, 830)},
    'icici': {
        'date': (1800, 60, 2290, 135), 'payee': (150, 180, 1900, 300),
        'words': (150, 300, 1720, 415), 'figures': (1720, 385, 2280, 500),
        'signature': (1880, 660, 2300, 890)},
    'syndicate': {
        'date': (1760, 60, 2290, 155), 'payee': (150, 180, 1800, 300),
        'words': (150, 300, 1780, 430), 'figures': (1790, 390, 2290, 505),
        'signature': (1850, 660, 2300, 890)},
    'canara': {
        'date': (1800, 60, 2290, 160), 'payee': (150, 215, 1700, 330),
        'words': (150, 330, 1600, 430), 'figures': (1790, 395, 2300, 520),
        'signature': (1850, 640, 2300, 890)},
}

BANK_NAMES = {'axis': 'Axis Bank', 'icici': 'ICICI Bank',
              'syndicate': 'Syndicate Bank', 'canara': 'Canara Bank'}

# Width all cheques are normalised to before analysis (300 dpi IDRBT scans)
CANONICAL_WIDTH = 2365

_ECC_SCALE = 0.25
# A correct template gives ECC correlation ~0.72-0.97; other banks ~<0.5
MATCH_MIN_CC = 0.6


def _small_gray(img, scale=_ECC_SCALE):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g.astype(np.float32) / 255.0, (5, 5), 0)


def coarse_cc(template_bgr, img_bgr, scale=0.1, iterations=40):
    """Quick ECC correlation at low resolution (used to rank templates)."""
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, _ = cv2.findTransformECC(
            _small_gray(template_bgr, scale), _small_gray(img_bgr, scale), warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-5), None, 1)
    except cv2.error:
        return -1.0
    return float(cc)


def align_to(template_bgr, img_bgr, iterations=200):
    """
    Affine-align `img_bgr` onto `template_bgr` with ECC.

    Returns:
        (aligned_bgr, correlation). correlation is -1 if ECC failed.
    """
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(
            _small_gray(template_bgr), _small_gray(img_bgr), warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-6),
            None, 1)
    except cv2.error:
        return None, -1.0
    warp[:, 2] /= _ECC_SCALE
    h, w = template_bgr.shape[:2]
    aligned = cv2.warpAffine(img_bgr, warp, (w, h),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned, float(cc)


def to_canonical_width(img_bgr):
    """Rescale a cheque image so its width matches the 300 dpi templates."""
    h, w = img_bgr.shape[:2]
    if w == CANONICAL_WIDTH:
        return img_bgr
    s = CANONICAL_WIDTH / w
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img_bgr, (CANONICAL_WIDTH, int(round(h * s))), interpolation=interp)


class TemplateLibrary:
    """Loads the blank templates and matches an input cheque to one."""

    def __init__(self, template_dir=None):
        self.template_dir = template_dir or config.TEMPLATE_DIR
        self.templates = {}
        if os.path.isdir(self.template_dir):
            for key in BANK_ROIS:
                path = os.path.join(self.template_dir, f"{key}.png")
                if os.path.exists(path):
                    self.templates[key] = cv2.imread(path)

    def __len__(self):
        return len(self.templates)

    def match(self, img_bgr):
        """
        Find the template that best explains `img_bgr`.

        Returns:
            (bank_key, template, aligned_img, cc) or (None, None, None, best_cc)
        """
        img = to_canonical_width(img_bgr)
        h, w = img.shape[:2]
        ranked = []
        for key, tpl in self.templates.items():
            th, tw = tpl.shape[:2]
            # Cheque aspect ratios are close; skip absurd mismatches early
            if abs(h / w - th / tw) > 0.12:
                continue
            ranked.append((coarse_cc(tpl, img), key))
        best_cc = -1.0
        for _, key in sorted(ranked, reverse=True)[:2]:
            aligned, cc = align_to(self.templates[key], img)
            if cc >= MATCH_MIN_CC:
                return key, self.templates[key], aligned, cc
            best_cc = max(best_cc, cc)
        return None, None, None, best_cc


def build_templates(raw_dir=None, out_dir=None, percentile=75):
    """Build one blank template per bank from the IDRBT scans."""
    from src.ink.idrbt import load_metadata

    out_dir = out_dir or config.TEMPLATE_DIR
    os.makedirs(out_dir, exist_ok=True)
    meta = load_metadata(raw_dir)

    for key in BANK_ROIS:
        rows = meta[meta.bank_key == key]
        if rows.empty:
            continue
        ref = cv2.imread(rows.iloc[0].path)
        stack = []
        for path in rows.path:
            aligned, cc = align_to(ref, cv2.imread(path))
            if aligned is not None:
                stack.append(aligned)
        # A high percentile keeps the (lighter) paper/print and drops the
        # darker handwriting, which is at a different place on each cheque.
        tpl = np.percentile(np.stack(stack), percentile, axis=0).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{key}.png"), tpl)
        print(f"  [OK] {BANK_NAMES[key]}: template from {len(stack)} cheques")

    with open(os.path.join(out_dir, "rois.json"), "w") as f:
        json.dump(BANK_ROIS, f, indent=2)


if __name__ == "__main__":
    build_templates()
