"""
Pen-difference models.

All three approaches answer the same question for every pair of ink units
on one cheque:  "were these two strokes written with different pens?"
and expose the same interface:

    model.fit(features, patches, units, train_idx)
    P = model.pair_matrix(features, patches, n_px)   # (n, n) P(different pen)

  ClassicalPairModel  hand-crafted OD ink features -> Random Forest / SVM
                      on the feature difference of a pair
  SiamesePairModel    twin CNN on optical-density ink patches, trained on
                      same-pen / different-pen pairs
  PenClassifierModel  CNN that classifies a patch into one of the 14 pens;
                      two units differ when their pen distributions differ
"""
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ------------------------------------------------------------
# Shared CNN encoder for optical-density ink patches
# ------------------------------------------------------------
class InkEncoder(nn.Module):
    """
    Small CNN over a (3, 64, 64) optical-density patch whose non-ink pixels
    are zero.  Pooling is *masked* to ink pixels so the empty background
    never dilutes the ink signal; masked mean OD / chromaticity are appended
    explicitly.
    """

    def __init__(self, out_dim=32):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True))

        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            blk(24, 32), nn.MaxPool2d(2),
            blk(32, 48), blk(48, 64))
        self.fc = nn.Sequential(nn.Linear(128 + 6, 64), nn.ReLU(), nn.Linear(64, out_dim))

    def forward(self, x):
        m = (x.sum(1, keepdim=True) > 0).float()
        f = self.conv(x)
        m4 = F.adaptive_max_pool2d(m, f.shape[-2:])
        avg = (f * m4).sum((2, 3)) / (m4.sum((2, 3)) + 1e-6)
        mx = (f * m4 - 1e4 * (1 - m4)).amax((2, 3))
        od = (x * m).sum((2, 3)) / (m.sum((2, 3)) + 1e-6)
        chrom = od / (od.sum(1, keepdim=True) + 1e-6)
        return self.fc(torch.cat([avg, mx, od * 5, chrom * 5], 1))

    @property
    def last_conv(self):
        return self.conv[-1][0]


def _augment(x):
    # Ink colour is orientation independent: flips are label preserving
    if torch.rand(1).item() < 0.5:
        x = x.flip(3)
    if torch.rand(1).item() < 0.5:
        x = x.flip(2)
    return x


class _TorchModel:
    epochs = 25
    lr = 2e-3

    def __init__(self, device=None):
        self.device = device or config.DEVICE
        self.net = None

    def _tensor(self, patches, idx=None):
        x = patches if idx is None else patches[idx]
        return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=self.device)

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    @classmethod
    def load(cls, path, device=None):
        m = cls(device)
        m.net = m._build().to(m.device)
        m.net.load_state_dict(torch.load(path, map_location=m.device, weights_only=True))
        m.net.eval()
        return m


# ------------------------------------------------------------
# 2. Siamese network
# ------------------------------------------------------------
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = InkEncoder(32)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def embed(self, x):
        return self.encoder(x)

    def pair_logit(self, ea, eb):
        return self.head(torch.cat([(ea - eb).abs(), ea * eb], 1)).squeeze(1)

    def forward(self, xa, xb):
        return self.pair_logit(self.embed(xa), self.embed(xb))


class SiamesePairModel(_TorchModel):
    name = 'siamese'
    epochs = 40

    def _build(self):
        return SiameseNet()

    def fit(self, features, patches, units, train_idx, seed=0, verbose=False,
            cheques_per_step=8):
        """
        Each step samples a few cheques, embeds all their pen-labelled units
        once, and trains on *every* within-cheque unit pair (same pen = 0,
        different pen = 1, the two classes weighted equally).
        """
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = np.asarray([i for i in train_idx if units['label'][i] > 0])
        by_cheque = [idx[units['cheque'][idx] == c] for c in np.unique(units['cheque'][idx])]
        by_cheque = [g for g in by_cheque if len(g) >= 2]
        self.net = self._build().to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), self.lr, weight_decay=1e-3)
        for ep in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(by_cheque))
            total, steps = 0.0, 0
            for s0 in range(0, len(order), cheques_per_step):
                groups = [by_cheque[g] for g in order[s0:s0 + cheques_per_step]]
                flat = np.concatenate(groups)
                emb = self.net.embed(_augment(self._tensor(patches, flat)))
                A, B, Y, off = [], [], [], 0
                for g in groups:
                    I, J = np.triu_indices(len(g), 1)
                    A.append(I + off)
                    B.append(J + off)
                    Y.append(units['label'][g][I] != units['label'][g][J])
                    off += len(g)
                A, B, Y = np.concatenate(A), np.concatenate(B), np.concatenate(Y)
                pos = max(Y.sum(), 1)
                w = np.where(Y, 0.5 / pos, 0.5 / max(len(Y) - pos, 1)) * len(Y)
                logit = self.net.pair_logit(emb[A], emb[B])
                loss = F.binary_cross_entropy_with_logits(
                    logit, torch.as_tensor(Y, dtype=torch.float32, device=self.device),
                    weight=torch.as_tensor(w, dtype=torch.float32, device=self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                steps += 1
            if verbose:
                print(f"    epoch {ep + 1:2d}/{self.epochs}  loss {total / max(steps, 1):.4f}")
        self.net.eval()
        return self

    @torch.no_grad()
    def pair_matrix(self, features, patches, n_px):
        n = len(patches)
        if n < 2:
            return np.zeros((n, n), np.float32)
        e = self.net.embed(self._tensor(patches))
        I, J = _all_pairs(n)
        p = torch.sigmoid(self.net.pair_logit(e[I], e[J])).cpu().numpy()
        return _to_matrix(n, I, J, p)


# ------------------------------------------------------------
# 3. CNN pen classifier
# ------------------------------------------------------------
class PenClassifierNet(nn.Module):
    def __init__(self, num_pens=NUM_PENS):
        super().__init__()
        self.encoder = InkEncoder(48)
        self.out = nn.Linear(48, num_pens)

    def forward(self, x):
        return self.out(F.relu(self.encoder(x)))


class PenClassifierModel(_TorchModel):
    name = 'cnn'
    epochs = 30

    def _build(self):
        return PenClassifierNet()

    def fit(self, features, patches, units, train_idx, seed=0, verbose=False):
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = np.asarray([i for i in train_idx if units['label'][i] > 0])
        pen = (np.where(units['label'][idx] == 1, units['pen1'][idx], units['pen2'][idx])
               .astype(np.int64) - 1)
        w = 1.0 / np.bincount(pen, minlength=NUM_PENS)[pen]          # balance pens
        self.net = self._build().to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), self.lr, weight_decay=1e-3)
        for ep in range(self.epochs):
            self.net.train()
            draw = rng.choice(len(idx), len(idx), p=w / w.sum())
            total = 0.0
            for i in range(0, len(draw), 128):
                j = draw[i:i + 128]
                loss = F.cross_entropy(self.net(_augment(self._tensor(patches, idx[j]))),
                                       torch.as_tensor(pen[j], device=self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(j)
            if verbose:
                print(f"    epoch {ep + 1:2d}/{self.epochs}  loss {total / len(draw):.4f}")
        self.net.eval()
        return self

    @torch.no_grad()
    def pen_probabilities(self, patches):
        return F.softmax(self.net(self._tensor(patches)), 1).cpu().numpy()

    @torch.no_grad()
    def pair_matrix(self, features, patches, n_px):
        n = len(patches)
        if n < 2:
            return np.zeros((n, n), np.float32)
        p = self.pen_probabilities(patches)
        p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
        return np.clip(1.0 - p @ p.T, 0, 1).astype(np.float32)


MODEL_CLASSES = {'classical': ClassicalPairModel, 'siamese': SiamesePairModel,
                 'cnn': PenClassifierModel}


def model_path(name, model_dir=None):
    return os.path.join(model_dir or config.MODEL_DIR, MODEL_FILES[name])


def load_model(name, model_dir=None):
    """Load a trained model and its calibrated cheque threshold."""
    path = model_path(name, model_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained '{name}' model at {path}. Run `python -m src.train` first.")
    model = MODEL_CLASSES[name].load(path)
    meta = load_metrics(model_dir).get(name, {})
    model.threshold = meta.get('threshold', 0.5)
    return model


def metrics_path(model_dir=None):
    return os.path.join(model_dir or config.MODEL_DIR, 'ink_metrics.json')


def load_metrics(model_dir=None):
    path = metrics_path(model_dir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
