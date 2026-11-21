"""Torch-free part of the model code.

The Classical (Random Forest) model, the model registry and the metrics loader live
here so that the detection pipeline - and the web API - can run without importing
PyTorch.  The two neural models are imported lazily, only when they are requested.
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.ink.pairs import within_cheque_pairs

MODEL_FILES = {'classical': 'ink_classical.pkl', 'siamese': 'ink_siamese.pth',
               'cnn': 'ink_cnn.pth'}
MODEL_LABELS = {'classical': 'Classical ML (OD ink features + Random Forest)',
                'siamese': 'Siamese CNN (ink patch similarity)',
                'cnn': 'CNN pen classifier'}
NUM_PENS = 14


def _all_pairs(n):
    return np.triu_indices(n, 1)


def _to_matrix(n, I, J, p):
    P = np.zeros((n, n), np.float32)
    P[I, J] = p
    P[J, I] = p
    return P


# ------------------------------------------------------------
# 1. Classical: hand-crafted optical-density features
# ------------------------------------------------------------
class ClassicalPairModel:
    name = 'classical'

    def __init__(self, method='rf'):
        self.method = method
        self.scale = None
        self.clf = None

    def _pair_x(self, f, n_px, a, b):
        d = np.abs(f[a] - f[b]) / self.scale
        m = (f[a] + f[b]) / 2 / self.scale
        la, lb = np.log(n_px[a]), np.log(n_px[b])
        return np.hstack([d, m, np.minimum(la, lb)[:, None], np.abs(la - lb)[:, None]])

    def _new_clf(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        if self.method == 'svm':
            return make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, random_state=0))
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                      n_jobs=-1, random_state=0)

    def fit(self, features, patches, units, train_idx, seed=0):
        rng = np.random.RandomState(seed)
        self.scale = features[train_idx].std(0) + 1e-6
        a, b, y = within_cheque_pairs(units, train_idx, rng)
        self.clf = self._new_clf().fit(self._pair_x(features, units['n_px'], a, b), y)
        return self

    def pair_matrix(self, features, patches, n_px):
        n = len(features)
        if n < 2:
            return np.zeros((n, n), np.float32)
        I, J = _all_pairs(n)
        p = self.clf.predict_proba(self._pair_x(features, np.asarray(n_px), I, J))[:, 1]
        return _to_matrix(n, I, J, p)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'method': self.method, 'scale': self.scale, 'clf': self.clf}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        m = cls(d['method'])
        m.scale, m.clf = d['scale'], d['clf']
        return m


DEEP_MODELS = ('siamese', 'cnn')


def torch_available():
    """True when PyTorch can be imported (needed only by the two neural models)."""
    import importlib.util
    return importlib.util.find_spec('torch') is not None        # checks without importing it


def available_models(model_dir=None):
    """Models whose weights exist on disk and whose runtime dependencies are installed."""
    found = []
    for name in MODEL_FILES:
        if not os.path.exists(model_path(name, model_dir)):
            continue
        if name in DEEP_MODELS and not torch_available():
            continue
        found.append(name)
    return found


def model_path(name, model_dir=None):
    return os.path.join(model_dir or config.MODEL_DIR, MODEL_FILES[name])


def metrics_path(model_dir=None):
    return os.path.join(model_dir or config.MODEL_DIR, 'ink_metrics.json')


def load_metrics(model_dir=None):
    path = metrics_path(model_dir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_model(name, model_dir=None):
    """Load a trained model and its calibrated cheque threshold."""
    path = model_path(name, model_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained '{name}' model at {path}. Run `python -m src.train` first.")
    if name == 'classical':
        model = ClassicalPairModel.load(path)
    else:
        from src.ink import models          # imports torch
        model = models.MODEL_CLASSES[name].load(path)
    model.threshold = load_metrics(model_dir).get(name, {}).get('threshold', 0.5)
    return model
