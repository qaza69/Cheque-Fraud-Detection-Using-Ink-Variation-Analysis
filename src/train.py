"""
Full training pipeline on the IDRBT Cheque Image Dataset.

    python -m src.train                    # all three models
    python -m src.train --models classical # just one
    python -m src.train --skip-cv          # retrain final models only

Steps
  1. Blank per-bank templates          (data/templates/, built once)
  2. Pen-labelled ink-unit dataset     (outputs/ink_dataset.npz, built once)
  3. 5-fold cross-validation grouped by cheque -> metrics, thresholds, plots
  4. Final models trained on all cheques -> outputs/models/
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed
from src.ink.templates import TemplateLibrary, build_templates
from src.ink.dataset import DATASET_PATH, build_dataset, load_dataset
from src.ink.models import MODEL_CLASSES, model_path, metrics_path, load_metrics
from src.evaluate import cross_validate, choose_threshold, summarise, plot_results

import numpy as np


def main(models=('classical', 'siamese', 'cnn'), skip_cv=False, rebuild=False):
    set_seed(42)
    t0 = time.time()

    print("\n[1/4] Blank cheque templates")
    if rebuild or not len(TemplateLibrary()):
        build_templates()
    else:
        print("  templates already built")

    print("\n[2/4] Pen-labelled ink-unit dataset")
    if rebuild or not os.path.exists(DATASET_PATH):
        build_dataset()
    F, PT, U = load_dataset()
    print(f"  {len(U)} ink units, {int((U['label'] > 0).sum())} pen-labelled, "
          f"{len(np.unique(U['cheque']))} cheques")

    metrics = load_metrics()
    cvs = {}
    if not skip_cv:
        print("\n[3/4] Cross-validation (5 folds, grouped by cheque)")
        for name in models:
            print(f"  -> {name}")
            cv = cross_validate(name, F, PT, U)
            thr = choose_threshold(cv['fraud'], cv['genuine'])
            metrics[name] = summarise(cv, thr)
            cvs[name] = cv
            m = metrics[name]
            print(f"     pair AUC {m['pair_auc']:.3f} | cheque AUC {m['cheque_auc']:.3f} | "
                  f"acc {m['accuracy']:.3f} | P {m['precision']:.3f} R {m['recall']:.3f} | "
                  f"full-cheque detection {m['full_cheque_detection']:.3f}")
        plot_results(cvs, metrics)

    print("\n[4/4] Final models on all cheques")
    all_idx = np.where(U['label'] > 0)[0]
    for name in models:
        model = MODEL_CLASSES[name]().fit(F, PT, U, all_idx, seed=0)
        model.save(model_path(name))
        metrics.setdefault(name, {}).setdefault('threshold', 0.5)
        print(f"  [OK] {name} -> {model_path(name)}")

    best = max((n for n in metrics if 'cheque_auc' in metrics[n]),
               key=lambda n: metrics[n]['cheque_auc'], default=None)
    for n in metrics:
        metrics[n]['recommended'] = n == best
    with open(metrics_path(), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  metrics -> {metrics_path()}")
    print(f"  [OK] done in {time.time() - t0:.0f}s")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['classical', 'siamese', 'cnn'],
                    choices=list(MODEL_CLASSES))
    ap.add_argument('--skip-cv', action='store_true')
    ap.add_argument('--rebuild', action='store_true', help='rebuild templates + dataset')
    a = ap.parse_args()
    main(a.models, a.skip_cv, a.rebuild)
