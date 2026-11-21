"""
Ink descriptors for one handwritten ink unit.

Different pens use different dye/pigment mixtures.  Even when two blue (or
two black) inks look identical to the eye, the *ratio* in which they absorb
red, green and blue light differs.  With OD = -log(I / I_paper) (Beer-Lambert),
those ratios are independent of how much ink was laid down and of the paper
colour, so they are the core features here:

  chr_*      OD chromaticity  (OD_c / sum OD) on the stroke core
  slope_*    how each channel's OD grows with total OD across the stroke
  od_*, t*   absolute ink density (pen flow / pressure)
  width, edge_ratio, core_cv, frac_light
             stroke morphology & texture (ballpoint vs gel vs liquid ink)
  L, a, bb, sat
             perceptual colour of the paper-normalised ink
"""
import cv2
import numpy as np

FEATURE_NAMES = (
    ['chr_b', 'chr_g', 'chr_r', 'chr_iqr_b', 'chr_iqr_g', 'chr_iqr_r',
     'od_b', 'od_g', 'od_r', 'slope_b', 'slope_g', 'slope_r',
     't10', 't25', 't50', 't75', 't90',
     'width', 'edge_ratio', 'core_cv', 'frac_light', 'L', 'a', 'bb', 'sat']
)
PATCH_SIZE = 64
OD_SCALE = 2.0      # OD values are divided by this before feeding CNNs


def _unit_crop(inkmap, unit):
    y, x, h, w = unit['y'], unit['x'], unit['h'], unit['w']
    return inkmap.image[y:y + h, x:x + w], inkmap.paper[y:y + h, x:x + w], unit['mask']


def unit_features(inkmap, unit):
    """Feature vector (len(FEATURE_NAMES),) for one ink unit."""
    img, paper, mask = _unit_crop(inkmap, unit)
    I = img[mask].astype(np.float32) + 1.0
    P = paper[mask].astype(np.float32) + 1.0
    o = np.clip(-np.log(I / P), 0, None)               # (n, 3) BGR OD
    t = o.sum(1) + 1e-6
    core = t >= np.percentile(t, 60)
    oc, tc = o[core], t[core]
    chrom = oc / tc[:, None]

    f = {}
    for i, c in enumerate('bgr'):
        f[f'chr_{c}'] = np.median(chrom[:, i])
        q75, q25 = np.percentile(chrom[:, i], [75, 25])
        f[f'chr_iqr_{c}'] = q75 - q25
        f[f'od_{c}'] = np.median(oc[:, i])
        f[f'slope_{c}'] = (np.polyfit(t, o[:, i], 1)[0] if len(t) > 5 and t.std() > 1e-3
                           else chrom[:, i].mean())
    for p in (10, 25, 50, 75, 90):
        f[f't{p}'] = np.percentile(t, p)

    m8 = mask.astype(np.uint8)
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    f['width'] = 2 * np.percentile(dt[mask], 90)
    inner = cv2.erode(m8, np.ones((3, 3), np.uint8)).astype(bool)
    edge = mask & ~inner
    tm = np.zeros(mask.shape, np.float32)
    tm[mask] = t
    f['edge_ratio'] = tm[edge].mean() / (tm[inner].mean() + 1e-6) if inner.any() else 1.0
    f['core_cv'] = tc.std() / (tc.mean() + 1e-6)
    f['frac_light'] = float((t < 0.5 * np.median(tc)).mean())

    wb = np.clip(I / P * 255, 0, 255).astype(np.uint8)[core].reshape(-1, 1, 3)
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    f['L'], f['a'], f['bb'] = np.median(lab, axis=0)
    f['sat'] = np.median(cv2.cvtColor(wb, cv2.COLOR_BGR2HSV).reshape(-1, 3)[:, 1])
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)


def unit_patch(inkmap, unit, size=PATCH_SIZE, rng=None):
    """
    Fixed-size optical-density patch (3, size, size) of one unit for the
    CNN / Siamese models.  Non-ink pixels are zero, so the network only ever
    sees handwriting ink (never paper or print).  Scale is preserved (no
    resizing) so stroke width stays meaningful; large units are cropped
    (randomly if `rng` is given, centred otherwise).
    """
    img, paper, mask = _unit_crop(inkmap, unit)
    od = np.clip(-np.log((img.astype(np.float32) + 1) / (paper.astype(np.float32) + 1)), 0, None)
    od = od * mask[..., None] / OD_SCALE
    h, w = mask.shape
    out = np.zeros((size, size, 3), np.float32)
    if rng is not None:
        y0 = rng.randint(0, max(h - size, 0) + 1)
        x0 = rng.randint(0, max(w - size, 0) + 1)
    else:
        y0, x0 = max(h - size, 0) // 2, max(w - size, 0) // 2
    crop = od[y0:y0 + size, x0:x0 + size]
    ch, cw = crop.shape[:2]
    oy, ox = (size - ch) // 2, (size - cw) // 2
    out[oy:oy + ch, ox:ox + cw] = crop
    return out.transpose(2, 0, 1)
