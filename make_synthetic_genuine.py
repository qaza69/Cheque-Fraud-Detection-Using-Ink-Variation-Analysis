"""
Synthetic SINGLE-PEN cheques on the real IDRBT bank templates.

IDRBT contains only tampered (two-pen) cheques, so there is no real
genuine data to test false alarms.  This script writes complete cheques
(payee, amount in words, amount in figures, signature) with ONE simulated
pen onto the blank Axis / ICICI / Canara / Syndicate templates.

Realism
  * handwriting fonts with per-letter jitter / rotation / baseline wobble
  * ink laid on with the Beer-Lambert model  out = paper * exp(-OD * coverage)
    using per-channel OD vectors measured from the IDRBT pens (then jittered,
    so each pen is a *new* pen, not one of the 14 training pens)
  * pressure variation along the stroke, soft edges, scanner noise
  * small random shift / rotation of the whole scan (alignment is tested)

The ink is simulated, so this checks false alarms under controlled
single-ink conditions - it is not a substitute for real single-pen scans.

Run:  python make_synthetic_genuine.py          -> data/synthetic_genuine/
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.ink.templates import TemplateLibrary

OUT_DIR = os.path.join(config.DATA_DIR, "synthetic_genuine")
FONT_DIR = r"C:\Windows\Fonts"

# Layout measured from the IDRBT handwriting: left x, right x, baseline y
LAYOUT = {
    'axis':      {'payee': (394, 1446, 296), 'words': (323, 1541, 402), 'figures': (1779, 2203, 481), 'signature': (1950, 2290, 789)},
    'icici':     {'payee': (567, 1501, 288), 'words': (514, 1638, 395), 'figures': (1867, 2261, 480), 'signature': (1998, 2287, 825)},
    'syndicate': {'payee': (570, 1523, 292), 'words': (495, 1660, 393), 'figures': (1852, 2220, 473), 'signature': (1913, 2204, 839)},
    'canara':    {'payee': (384, 1265, 307), 'words': (443, 1288, 407), 'figures': (1862, 2207, 493), 'signature': (1922, 2276, 827)},
}

# Median stroke-core optical density (B, G, R) of IDRBT pens, used as pen "dyes"
BLUE_PENS = [(0.52, 2.91, 4.68), (0.44, 1.81, 1.94), (0.35, 1.83, 1.98), (0.43, 1.47, 1.40),
             (0.42, 2.04, 2.13), (0.52, 2.00, 1.89)]
BLACK_PENS = [(1.66, 2.22, 2.24), (2.10, 2.01, 1.97), (1.82, 2.30, 1.75), (1.40, 1.52, 1.33),
              (0.96, 1.32, 1.10), (2.22, 2.34, 2.31)]

PAYEES = ["Ramesh Chandra", "Sunita Devi", "Mohan Lal Gupta", "Kavita Reddy", "Anil Kumar Jain",
          "Pooja Nair", "Suresh Babu", "Meera Iyer", "Harish Patel", "Lakshmi Prasad",
          "Imran Sheikh", "Deepa Menon", "Vikash Yadav", "Rohit Bansal", "Ananya Das"]
AMOUNTS = [("Twelve Thousand Five Hundred Only", "12,500/-"), ("Forty Five Thousand Only", "45,000/-"),
           ("Seventy Five Thousand Only", "75,000/-"), ("Three Lakh Twenty Thousand Only", "3,20,000/-"),
           ("Eight Thousand Only", "8,000/-"), ("One Lakh Sixty Thousand Only", "1,60,000/-"),
           ("Twenty Five Thousand Only", "25,000/-"), ("Fifty Thousand Only", "50,000/-"),
           ("Six Thousand Two Hundred Only", "6,200/-"), ("Two Lakh Only", "2,00,000/-"),
           ("Ninety Thousand Only", "90,000/-"), ("Thirty Five Thousand Only", "35,000/-")]
SIGNATURES = ["R. Chandra", "S. Devi", "M. Gupta", "K. Reddy", "A. Jain", "P. Nair", "S. Babu", "M. Iyer",
              "H. Patel", "L. Prasad", "I. Sheikh", "D. Menon"]
HAND_FONTS = ["Inkfree.ttf", "segoepr.ttf", "mvboli.ttf", "LHANDW.TTF"]
SIGN_FONTS = ["segoesc.ttf", "FREESCPT.TTF", "Gabriola.ttf", "Inkfree.ttf"]
SS = 3          # supersampling factor for the text mask


def render_text(shape, text, x0, x1, baseline, font_name, size, rng, slant=0.0):
    """Handwriting coverage mask (float 0-1) for one line, fitted into [x0, x1]."""
    H, W = shape
    while True:
        font = ImageFont.truetype(os.path.join(FONT_DIR, font_name), int(size * SS))
        if font.getlength(text) / SS <= (x1 - x0) or size < 30:
            break
        size *= 0.93
    canvas = Image.new('L', (W * SS, H * SS), 0)
    x = x0 * SS + rng.uniform(0, 12) * SS
    phase = rng.uniform(0, 6.28)
    gw, gh = int(size * 2 * SS), int(size * 2.2 * SS)
    ox, oy = int(size * 0.4 * SS), int(size * 1.5 * SS)
    for i, ch in enumerate(text):
        if ch == ' ':
            x += font.getlength(' ') * rng.uniform(0.9, 1.4)
            continue
        dy = (6 * np.sin(phase + i * 0.35) + rng.normal(0, 1.8)) * SS
        glyph = Image.new('L', (gw, gh), 0)
        ImageDraw.Draw(glyph).text((ox, oy), ch, 255, font=font, anchor='ls')
        glyph = glyph.rotate(rng.normal(-slant, 3.5), resample=Image.BICUBIC, center=(ox, oy))
        canvas.paste(255, (int(x - ox), int(baseline * SS + dy - oy)), glyph)
        x += font.getlength(ch) * rng.uniform(0.92, 1.06)
    return np.asarray(canvas.resize((W, H), Image.LANCZOS), np.float32) / 255.0


def stroke_style(mask, thick):
    """Adjust stroke thickness and give the mask pen-like soft edges."""
    m8 = (mask > 0.35).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    if thick > 0:
        m8 = cv2.dilate(m8, k, iterations=thick)
    elif thick < 0:
        m8 = cv2.erode(m8, k, iterations=-thick)
    m = cv2.GaussianBlur(m8.astype(np.float32), (0, 0), 0.9)
    return np.clip(m * 1.15, 0, 1)


def pressure_map(shape, rng):
    h, w = shape
    low = cv2.resize(rng.rand(h // 40 + 2, w // 40 + 2).astype(np.float32), (w, h),
                     interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.rand(h, w).astype(np.float32), (0, 0), 1.5)
    return np.clip(0.78 + 0.28 * (low - 0.5) + 0.35 * (fine - 0.5), 0.45, 1.05)


def make_pen(colour, rng):
    """A new pen: IDRBT-like dye with ~12% per-channel and overall variation."""
    pool = BLUE_PENS if colour == 'blue' else BLACK_PENS
    base = np.array(pool[rng.randint(len(pool))])
    return np.clip(base * rng.uniform(0.9, 1.12) * rng.uniform(0.88, 1.12, 3), 0.2, 5.0)


def scanner_perturb(img, rng):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.normal(0, 0.25), 1.0)
    M[:, 2] += rng.uniform(-6, 6, 2)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    out = out.astype(np.float32) * rng.uniform(0.97, 1.03) + rng.normal(0, 1.2, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_cheque(tpl, bank, idx, colour, rng):
    H, W = tpl.shape[:2]
    lay = LAYOUT[bank]
    od = make_pen(colour, rng)
    font = HAND_FONTS[rng.randint(len(HAND_FONTS))]
    sign_font = SIGN_FONTS[rng.randint(len(SIGN_FONTS))]
    thick = int(rng.choice([-1, 0, 0, 1]))
    amt_words, amt_fig = AMOUNTS[idx % len(AMOUNTS)]
    cov = np.zeros((H, W), np.float32)
    for field, text, fnt, size in [
        ('payee', PAYEES[idx % len(PAYEES)], font, 66),
        ('words', amt_words, font, 62),
        ('figures', amt_fig, font, 64),
        ('signature', SIGNATURES[idx % len(SIGNATURES)], sign_font, 92),
    ]:
        x0, x1, base = lay[field]
        if field == 'figures':
            x0 += rng.randint(0, 25)
        m = render_text((H, W), text, x0, x1, base + rng.randint(-5, 6), fnt, size, rng,
                        slant=rng.uniform(0, 8) if field != 'signature' else 12)
        cov = np.maximum(cov, stroke_style(m, thick if field != 'signature' else 1))
    cov *= pressure_map((H, W), rng)            # same pen, natural pressure variation
    img = tpl.astype(np.float32) * np.exp(-cov[..., None] * od[None, None, :])
    return scanner_perturb(img, rng), od


def main(n_per_bank=3, seed=7):
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, f))
    lib = TemplateLibrary()
    rng = np.random.RandomState(seed)
    idx = 0
    for bank, tpl in lib.templates.items():
        for _ in range(n_per_bank):
            colour = 'blue' if idx % 2 == 0 else 'black'
            img, od = make_cheque(tpl, bank, idx, colour, rng)
            name = f"genuine_{bank}_s{idx + 1:02d}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, name), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
            print(f"  [OK] {name}  ({colour} pen, OD B/G/R = {od[0]:.2f}/{od[1]:.2f}/{od[2]:.2f})")
            idx += 1


if __name__ == "__main__":
    main()
