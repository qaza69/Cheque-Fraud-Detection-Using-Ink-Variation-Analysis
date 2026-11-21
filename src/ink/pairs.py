"""
Pair construction and cheque-level scoring shared by all three models.

Every model ends in the same interface:  given the ink units of one cheque,
return an (n, n) matrix of P(different pen).  The cheque is then judged by
`cheque_score`, so the three approaches are compared on equal terms.
"""
import numpy as np


def within_cheque_pairs(units, idx, rng, max_same=60, max_diff=60):
    """
    Labelled unit pairs inside each cheque (pairs are never formed across
    cheques, which is exactly the situation at inference time).

    Args:
        units: dataset record array (needs 'cheque', 'label')
        idx: indices of units that may be used
    Returns:
        (a, b, y) index arrays; y = 1 for different pens, 0 for same pen
    """
    A, B, Y = [], [], []
    idx = np.asarray(idx)
    for c in np.unique(units['cheque'][idx]):
        ii = idx[units['cheque'][idx] == c]
        p1 = ii[units['label'][ii] == 1]
        p2 = ii[units['label'][ii] == 2]
        same = [(a, b) for k, a in enumerate(p1) for b in p1[k + 1:]]
        same += [(a, b) for k, a in enumerate(p2) for b in p2[k + 1:]]
        diff = [(a, b) for a in p2 for b in p1]
        for pairs, lab, cap in ((same, 0, max_same), (diff, 1, max_diff)):
            if not pairs:
                continue
            sel = rng.choice(len(pairs), min(cap, len(pairs)), replace=False)
            for s in sel:
                A.append(pairs[s][0])
                B.append(pairs[s][1])
                Y.append(lab)
    return np.array(A, int), np.array(B, int), np.array(Y, int)


def cheque_score(P, weights=None, min_units=4):
    """
    Score how strongly a cheque's ink splits into two different pens.

    For each unit, its "distinctness" is the weighted median P(different pen)
    against all other units - high for units whose ink disagrees with most
    of the cheque.  A tampered cheque has a *group* of such units that agree
    with each other (same second pen), so the score is the mean distinctness
    of the top units, discounted when those units disagree among themselves.

    Returns:
        score in [0, 1], per-unit distinctness (n,)
    """
    n = len(P)
    if n < min_units:
        return 0.0, np.zeros(n)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    P = P.copy()
    np.fill_diagonal(P, np.nan)
    distinct = np.zeros(n)
    for i in range(n):
        m = ~np.isnan(P[i])
        order = np.argsort(P[i][m])
        cw = np.cumsum(w[m][order])
        distinct[i] = P[i][m][order][np.searchsorted(cw, cw[-1] / 2)]

    k = max(2, int(round(0.1 * n)))
    top = np.argsort(distinct)[::-1][:k]
    sub = P[np.ix_(top, top)]
    cohesion = 1.0 - np.nanmean(sub) if k > 1 else 1.0
    score = float(np.mean(distinct[top]) * (0.5 + 0.5 * cohesion))
    return score, distinct
