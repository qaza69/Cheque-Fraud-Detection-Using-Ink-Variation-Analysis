"""
Create demo cheques for the web app from the IDRBT dataset.

For each selected cheque two images are written to data/samples/:

  fraud_<bank>_<id>.jpg    the real scan - pen 2 added to the payee and
                           prepended to the amount (multi-pen tampering)
  genuine_<bank>_<id>.jpg  the same cheque with pen 2's additions erased,
                           i.e. the cheque as originally filled with one pen

Erased ink is replaced with the bank's blank template (gain-matched to the
scan's paper), so the printed layout underneath is restored.

Which end of the payee line pen 2 wrote on (prefix/suffix) is not in the
metadata, so it is listed here for the curated cheques (checked by eye).

Run:  python make_samples.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.ink.idrbt import load_metadata
from src.ink.templates import TemplateLibrary
from src.ink.extraction import extract_ink, paper_background
from src.ink.dataset import words_boundaries, segment_words

OUT_DIR = os.path.join(config.DATA_DIR, "samples")

# cheque id -> (side of the payee line written by pen 2,
#               x where pen 2's leading digit(s) end in the amount box)
# Both were read off the aligned scans by eye: touching digits ("42", "73")
# cannot be split reliably by connected components.
CURATED = {
    309142: ('prefix', 1836),  # Axis       "Bommakanti | Rahul Singh"   "1|10,000"
    309147: ('suffix', 1888),  # Axis       "Rajib Kumar | Mishra"       "3|5,000"
    309089: ('suffix', 1852),  # Axis       "Sravani | Bhargavi"         "4|2,20,000"
    100835: ('prefix', 1929),  # ICICI      "Kishore | Raj Shukla"       "7|3,00,000"
    120614: ('suffix', 1903),  # Canara     "Kadamanchi Rana Prathap | Reddy"  "6|900000"
}


def pen2_mask(inkmap, row, payee_side, figures_cut):
    """Ink written by pen 2 (amount prefix + payee prefix/suffix)."""
    ink = inkmap.ink
    erase = np.zeros_like(ink)

    x1, y1, x2, y2 = inkmap.rois['figures']
    erase[y1:y2, x1:figures_cut] |= ink[y1:y2, x1:figures_cut]

    wb = words_boundaries(inkmap, int(row.k_words))
    x1, y1, x2, y2 = inkmap.rois['words']
    if wb is None:
        raise RuntimeError("amount words boundary not found")
    erase[y1:int(wb[0]), x1:int(wb[1])] |= ink[y1:int(wb[0]), x1:int(wb[1])]

    x1, y1, x2, y2 = inkmap.rois['payee']
    m = ink[y1:y2, x1:x2]
    segs = segment_words(m)
    n_words = max(1, len(str(row.payee2).split()))
    if len(segs) <= n_words:
        raise RuntimeError("payee words not separable")
    if payee_side == 'prefix':
        xa, xb = segs[0][0], segs[n_words - 1][1] + 1
    else:
        xa, xb = segs[-n_words][0], segs[-1][1] + 1
    erase[y1:y2, x1 + xa:x1 + xb] |= m[:, xa:xb]
    return cv2.dilate(erase.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)


def erase_with_template(image, template, mask):
    """Replace masked pixels by the blank template, matched to the scan's paper."""
    gain = (paper_background(image).astype(np.float32) + 1) / \
           (paper_background(template).astype(np.float32) + 1)
    fill = np.clip(template.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    out = image.copy()
    soft = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)[..., None]
    out = (out * (1 - soft) + fill * soft).astype(np.uint8)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            os.remove(os.path.join(OUT_DIR, f))

    meta = load_metadata().set_index('cheque')
    lib = TemplateLibrary()
    for cid, (side, cut) in CURATED.items():
        row = meta.loc[cid]
        inkmap = extract_ink(row.path, lib, rgb=False)
        tpl = lib.templates[inkmap.bank]
        genuine = erase_with_template(inkmap.image, tpl, pen2_mask(inkmap, row, side, cut))
        tag = f"{inkmap.bank}_{cid}"
        cv2.imwrite(os.path.join(OUT_DIR, f"fraud_{tag}.jpg"), inkmap.image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(OUT_DIR, f"genuine_{tag}.jpg"), genuine,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [OK] {tag}: pen {row.pen1} + pen {row.pen2} ({row.colour})")


if __name__ == "__main__":
    main()
