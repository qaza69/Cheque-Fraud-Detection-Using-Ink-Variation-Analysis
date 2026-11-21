"""Collect everything the deployed API needs into backend/artifacts/.

The blank bank templates, the trained models and the demo cheques are rebuilt from the
IDRBT dataset by `python -m src.train` and `python make_samples.py`; they are not part of
the source tree.  Run this once after training and commit backend/artifacts so that
Render can serve them:

    python backend/prepare_artifacts.py              # templates + models + demo cheques
    python backend/prepare_artifacts.py --no-samples # leave the demo cheque images out
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "backend", "artifacts")

COPY = [
    # (source folder, destination folder, file filter)
    (os.path.join(ROOT, "data", "templates"), os.path.join(OUT, "templates"), lambda f: f.endswith((".png", ".json"))),
    (os.path.join(ROOT, "outputs", "models"), os.path.join(OUT, "models"),
     lambda f: f in ("ink_classical.pkl", "ink_siamese.pth", "ink_cnn.pth", "ink_metrics.json")),
]
SAMPLES = (os.path.join(ROOT, "data", "samples"), os.path.join(OUT, "samples"),
           lambda f: f.lower().endswith((".jpg", ".jpeg", ".png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-samples", action="store_true")
    args = ap.parse_args()

    jobs = COPY + ([] if args.no_samples else [SAMPLES])
    total = 0
    for src, dst, keep in jobs:
        if not os.path.isdir(src):
            sys.exit(f"missing {src} - run `python -m src.train` (and make_samples.py) first")
        os.makedirs(dst, exist_ok=True)
        files = sorted(f for f in os.listdir(src) if keep(f))
        if not files:
            sys.exit(f"nothing to copy from {src}")
        for f in files:
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            total += os.path.getsize(os.path.join(dst, f))
        print(f"  {os.path.relpath(dst, ROOT)}: {len(files)} files")
    print(f"done - {total / 1e6:.1f} MB in {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
