"""
Cross-validated evaluation of the three ink models on IDRBT.

Folds are split *by cheque* (no cheque is ever in both train and test).

Pair level   - within held-out cheques: same pen vs different pen unit pairs
Cheque level - every held-out cheque is scored twice on the amount fields
               (words + figures):
                 * tampered : as scanned (pen 1 + pen 2)          -> FRAUD
                 * genuine  : pen 2's leading words/digits removed -> GENUINE
               This paired design isolates the effect of the second pen.
Full cheque  - the deployed setting (payee + words + figures) on the
               scanned (tampered) cheques: detection rate.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score)
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import save_figure
from src.ink.pairs import within_cheque_pairs, cheque_score
from src.ink.models import MODEL_CLASSES, MODEL_LABELS

EVAL_FIELDS = ('words', 'figures')
DEPLOY_FIELDS = ('payee', 'words', 'figures')
N_FOLDS = 5


def _score_units(model, F, PT, U, sel):
    P = model.pair_matrix(F[sel], PT[sel], U['n_px'][sel])
    return cheque_score(P, U['n_px'][sel])[0]


def cross_validate(name, F, PT, U, n_folds=N_FOLDS, verbose=True):
    cheques = np.unique(U['cheque'])
    labelled = np.where(U['label'] > 0)[0]
    out = dict(pair_p=[], pair_y=[], fraud=[], genuine=[], full=[], cheques=[])
    for fold, (tr, te) in enumerate(GroupKFold(n_folds).split(cheques, groups=cheques)):
        train_c, test_c = set(cheques[tr]), cheques[te]
        tri = labelled[np.isin(U['cheque'][labelled], list(train_c))]
        tei = labelled[np.isin(U['cheque'][labelled], test_c)]
        model = MODEL_CLASSES[name]().fit(F, PT, U, tri, seed=fold)

        a, b, y = within_cheque_pairs(U, tei, np.random.RandomState(1000 + fold))
        for c in np.unique(U['cheque'][a]):
            m = U['cheque'][a] == c
            idx = np.unique(np.r_[a[m], b[m]])
            P = model.pair_matrix(F[idx], PT[idx], U['n_px'][idx])
            pos = {v: k for k, v in enumerate(idx)}
            out['pair_p'].extend(P[[pos[i] for i in a[m]], [pos[i] for i in b[m]]])
            out['pair_y'].extend(y[m])

        for c in test_c:
            amt = np.where((U['cheque'] == c) & np.isin(U['field'], EVAL_FIELDS))[0]
            gen = amt[U['genuine_keep'][amt]]
            full = np.where((U['cheque'] == c) & np.isin(U['field'], DEPLOY_FIELDS))[0]
            out['fraud'].append(_score_units(model, F, PT, U, amt))
            out['genuine'].append(_score_units(model, F, PT, U, gen) if len(gen) >= 4 else 0.0)
            out['full'].append(_score_units(model, F, PT, U, full))
            out['cheques'].append(int(c))
        if verbose:
            print(f"    fold {fold + 1}/{n_folds} done")
    return {k: np.asarray(v) for k, v in out.items()}


def choose_threshold(fraud, genuine):
    """Threshold maximising balanced accuracy on the paired cheque scores."""
    s = np.r_[fraud, genuine]
    y = np.r_[np.ones(len(fraud)), np.zeros(len(genuine))]
    best_t, best = 0.5, -1
    for t in np.unique(s):
        pred = s >= t
        bal = 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())
        if bal > best:
            best_t, best = t, bal
    # place the threshold midway to the next lower score for a margin
    lower = s[s < best_t]
    return float((best_t + lower.max()) / 2) if lower.size else float(best_t)


def summarise(cv, threshold):
    s = np.r_[cv['fraud'], cv['genuine']]
    y = np.r_[np.ones(len(cv['fraud'])), np.zeros(len(cv['genuine']))].astype(int)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'threshold': threshold,
        'pair_auc': float(roc_auc_score(cv['pair_y'], cv['pair_p'])),
        'cheque_auc': float(roc_auc_score(y, s)),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'specificity': float(tn / max(tn + fp, 1)),
        'confusion': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'full_cheque_detection': float((cv['full'] >= threshold).mean()),
        'n_cheques': int(len(cv['fraud'])),
    }


# ─── figures ────────────────────────────────────────────────────────────

def plot_results(cvs, metrics):
    colors = {'classical': '#2563eb', 'siamese': '#dc2626', 'cnn': '#16a34a'}

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, cv in cvs.items():
        s = np.r_[cv['fraud'], cv['genuine']]
        y = np.r_[np.ones(len(cv['fraud'])), np.zeros(len(cv['genuine']))]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=2, color=colors.get(name),
                label=f"{MODEL_LABELS[name]} (AUC {metrics[name]['cheque_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False positive rate (genuine flagged)')
    ax.set_ylabel('True positive rate (tampered detected)')
    ax.set_title('Cheque-level ROC (5-fold, grouped by cheque)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    save_figure(fig, 'roc_curves')

    names = list(cvs)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6))
    for ax, name in zip(np.atleast_1d(axes), names):
        cm = np.array(metrics[name]['confusion'])
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1], ['Genuine', 'Fraud'])
        ax.set_yticks([0, 1], ['Genuine', 'Fraud'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(name.capitalize())
    fig.tight_layout()
    save_figure(fig, 'confusion_matrices')

    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ['pair_auc', 'cheque_auc', 'accuracy', 'precision', 'recall', 'f1']
    w = 0.8 / len(names)
    for k, name in enumerate(names):
        ax.bar(np.arange(len(keys)) + k * w, [metrics[name][m] for m in keys], w,
               label=name.capitalize(), color=colors.get(name))
    ax.set_xticks(np.arange(len(keys)) + w * (len(names) - 1) / 2,
                  ['Pair AUC', 'Cheque AUC', 'Accuracy', 'Precision', 'Recall', 'F1'])
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Model comparison (cross-validated on IDRBT)')
    save_figure(fig, 'model_comparison')

    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        cv = cvs[name]
        bins = np.linspace(0, 1, 26)
        ax.hist(cv['genuine'], bins, alpha=0.6, color='#16a34a', label='Genuine (pen 1 only)')
        ax.hist(cv['fraud'], bins, alpha=0.6, color='#dc2626', label='Tampered (pen 1 + 2)')
        ax.axvline(metrics[name]['threshold'], color='k', ls='--', lw=1.5)
        ax.set_title(name.capitalize())
        ax.set_xlabel('Cheque score')
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, 'score_distributions')
