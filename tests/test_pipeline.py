"""
Tests for the ink-variation cheque fraud pipeline.

Run: python -m pytest tests -v

Tests that need the IDRBT templates / trained models are skipped when
those artefacts have not been built yet.
"""
import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.ink.extraction import (InkMap, extract_ink, segment_units, optical_density,
                                paper_background)
from src.ink.features import unit_features, unit_patch, FEATURE_NAMES, PATCH_SIZE
from src.ink.pairs import cheque_score, within_cheque_pairs
from src.ink.models import (ClassicalPairModel, SiamesePairModel, PenClassifierModel,
                            model_path)
from src.ink.templates import TemplateLibrary, BANK_ROIS
from src.ink.idrbt import count_amount_words

SAMPLES = os.path.join(config.DATA_DIR, "samples")
HAS_TEMPLATES = len(TemplateLibrary()) > 0
HAS_MODELS = all(os.path.exists(model_path(m)) for m in ('classical', 'siamese', 'cnn'))


# ─── helpers ──────────────────────────────────────────────────────────

BLUE_A = (170, 60, 20)       # BGR
BLUE_B = (150, 75, 45)       # similar-looking blue with different dye ratios


def synthetic_cheque(inks, paper=(215, 235, 240), seed=0, line=cv2.LINE_AA):
    """Cream paper, a printed dark label and handwriting strokes per ink."""
    rng = np.random.RandomState(seed)
    img = np.full((1100, 2365, 3), paper, np.uint8)
    cv2.putText(img, "PAY", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (40, 40, 40), 4)
    x = 300
    for ink in inks:
        for _ in range(6):
            pts = np.array([[x + rng.randint(0, 20), 300 + rng.randint(-25, 25)]
                            for _ in range(6)]).cumsum(0) // [1, 6] + [0, 250]
            pts[:, 0] = x + np.linspace(0, 110, 6).astype(int)
            cv2.polylines(img, [pts.astype(np.int32)], False, ink, 6, line)
            x += 150
    return img


def make_inkmap(img):
    paper = paper_background(img)
    ink = optical_density(img, paper).sum(2) > 0.6
    h, w = img.shape[:2]
    return InkMap(img, paper, ink, {'all': (0, 0, w, h)}, 'generic')


# ─── templates / layout ──────────────────────────────────────────────

class TestTemplates:
    def test_rois_inside_template(self):
        for bank, rois in BANK_ROIS.items():
            for f, (x1, y1, x2, y2) in rois.items():
                assert 0 <= x1 < x2 <= 2400 and 0 <= y1 < y2 <= 1100, (bank, f)

    def test_amount_word_count(self):
        assert count_amount_words("one lakh") == 2
        assert count_amount_words("onecrore") == 2
        assert count_amount_words("fifty") == 1
        assert count_amount_words("one lakh &") == 3

    @pytest.mark.skipif(not HAS_TEMPLATES, reason="templates not built")
    def test_samples_match_their_bank(self):
        lib = TemplateLibrary()
        for f in os.listdir(SAMPLES):
            bank = f.split('_')[1]
            key, _, aligned, cc = lib.match(cv2.imread(os.path.join(SAMPLES, f)))
            assert key == bank, f
            assert cc > 0.6

    @pytest.mark.skipif(not HAS_TEMPLATES, reason="templates not built")
    def test_printed_content_removed(self):
        """A blank template contains no handwriting at all."""
        lib = TemplateLibrary()
        for key, tpl in lib.templates.items():
            ink = extract_ink(tpl, lib, rgb=False)
            assert ink.mode == 'template'
            assert len(segment_units(ink)) <= 3, key


# ─── extraction & features ───────────────────────────────────────────

class TestExtraction:
    def test_generic_mode_flags_unknown_layout(self):
        img = synthetic_cheque([BLUE_A])
        ink = extract_ink(img, TemplateLibrary(template_dir="__none__"), rgb=False)
        assert ink.mode == 'generic'
        assert ink.notes
        assert ink.ink.any()

    def test_optical_density_cancels_paper_colour(self):
        """Same ink on cream vs blue paper -> same OD chromaticity."""
        chroma = []
        for paper in [(215, 235, 240), (240, 220, 190)]:
            img = synthetic_cheque([BLUE_A], paper=paper, line=cv2.LINE_8)
            # simulate transmissive ink: scale paper by fixed transmittance
            ink_t = np.array(BLUE_A, np.float32) / 255
            mask = np.all(img == BLUE_A, axis=2)
            img = img.astype(np.float32)
            img[mask] = np.array(paper, np.float32) * ink_t
            im = make_inkmap(img.astype(np.uint8))
            u = [u for u in segment_units(im, ['all']) if u['x'] >= 280][0]   # skip "PAY"
            f = dict(zip(FEATURE_NAMES, unit_features(im, u)))
            chroma.append([f['chr_b'], f['chr_g'], f['chr_r']])
        assert np.allclose(chroma[0], chroma[1], atol=0.03)

    def test_units_and_features(self):
        im = make_inkmap(synthetic_cheque([BLUE_A, BLUE_B]))
        units = segment_units(im, ['all'])
        assert len(units) >= 6
        f = unit_features(im, units[0])
        assert f.shape == (len(FEATURE_NAMES),) and np.isfinite(f).all()
        p = unit_patch(im, units[0])
        assert p.shape == (3, PATCH_SIZE, PATCH_SIZE)
        assert (p >= 0).all() and p.max() > 0

    def test_features_separate_similar_blues(self):
        im = make_inkmap(synthetic_cheque([BLUE_A, BLUE_B]))
        units = segment_units(im, ['all'])
        F = np.stack([unit_features(im, u) for u in units])
        units = [u for u in units if u['x'] >= 280]                       # skip "PAY"
        F = np.stack([unit_features(im, u) for u in units])
        left = np.array([u['x'] < 1150 for u in units])        # first ink: x 300-1050
        i = FEATURE_NAMES.index('chr_b')
        assert abs(F[left, i].mean() - F[~left, i].mean()) > 3 * max(F[left, i].std(), 1e-3)


# ─── scoring ─────────────────────────────────────────────────────────

class TestScoring:
    def test_single_pen_scores_low(self):
        P = np.full((20, 20), 0.1)
        assert cheque_score(P)[0] < 0.2

    def test_two_pens_score_high(self):
        P = np.full((20, 20), 0.1)
        P[:4, 4:] = P[4:, :4] = 0.9
        score, distinct = cheque_score(P)
        assert score > 0.5
        assert set(np.argsort(distinct)[::-1][:4]) == {0, 1, 2, 3}

    def test_too_few_units(self):
        assert cheque_score(np.zeros((2, 2)))[0] == 0.0

    def test_pairs_stay_within_cheques(self):
        U = np.array([(1, 1), (1, 1), (1, 2), (2, 1), (2, 2)],
                     dtype=[('cheque', 'i8'), ('label', 'i1')])
        a, b, y = within_cheque_pairs(U, np.arange(5), np.random.RandomState(0))
        assert (U['cheque'][a] == U['cheque'][b]).all()
        assert ((U['label'][a] != U['label'][b]) == y.astype(bool)).all()


# ─── models ──────────────────────────────────────────────────────────

def toy_dataset(n_cheques=6, per=8, seed=0):
    rng = np.random.RandomState(seed)
    F, P, recs = [], [], []
    for c in range(n_cheques):
        for k in range(per):
            lab = 2 if k < 2 else 1
            f = rng.normal(0, 1, len(FEATURE_NAMES)) + (3 if lab == 2 else 0)
            F.append(f)
            p = np.zeros((3, PATCH_SIZE, PATCH_SIZE), np.float32)
            p[:, 20:44, 20:44] = (0.3 if lab == 2 else 0.6) + rng.rand(3, 1, 1) * 0.05
            P.append(p)
            recs.append((c, lab, 1 + c % 7, 8 + c % 7, 500))
    U = np.array(recs, dtype=[('cheque', 'i8'), ('label', 'i1'), ('pen1', 'i1'),
                              ('pen2', 'i1'), ('n_px', 'i4')])
    return np.array(F, np.float32), np.array(P), U


@pytest.mark.parametrize("cls", [ClassicalPairModel, SiamesePairModel, PenClassifierModel])
def test_model_fit_and_pair_matrix(cls, tmp_path):
    F, P, U = toy_dataset()
    m = cls()
    if hasattr(m, 'epochs'):
        m.epochs = 2
    m.fit(F, P, U, np.arange(len(U)))
    M = m.pair_matrix(F[:8], P[:8], U['n_px'][:8])
    assert M.shape == (8, 8)
    assert np.allclose(M, M.T, atol=1e-5)
    assert ((M >= 0) & (M <= 1)).all()
    path = tmp_path / "model.bin"
    m.save(str(path))
    m2 = cls.load(str(path))
    assert np.allclose(m2.pair_matrix(F[:8], P[:8], U['n_px'][:8]), M, atol=1e-5)


def test_gradcam_hooks_are_removed():
    from src.gradcam import gradcam_pen_classifier, gradcam_siamese
    from src.ink.models import PenClassifierNet, SiameseNet
    p = np.zeros((3, PATCH_SIZE, PATCH_SIZE), np.float32)
    p[:, 20:40, 20:40] = 0.5
    net = PenClassifierNet().eval()
    for _ in range(3):
        hm = gradcam_pen_classifier(net, p)
    assert hm.shape == (PATCH_SIZE, PATCH_SIZE)
    assert not net.encoder.last_conv._forward_hooks
    snet = SiameseNet().eval()
    hm = gradcam_siamese(snet, p, p * 0.5)
    assert hm.shape == (PATCH_SIZE, PATCH_SIZE)
    assert not snet.encoder.last_conv._backward_hooks


# ─── end to end ──────────────────────────────────────────────────────

@pytest.mark.skipif(not (HAS_MODELS and HAS_TEMPLATES), reason="models/templates not built")
def test_pipeline_on_samples():
    from src.fraud_pipeline import FraudDetector
    det = FraudDetector('classical')
    f = sorted(os.listdir(SAMPLES))[0]
    r = det.detect(os.path.join(SAMPLES, f), rgb=False)
    assert r['verdict'] in ('FRAUD', 'GENUINE')
    assert r['mode'] == 'template'
    assert 0 <= r['confidence'] <= 1
    assert r['highlighted_image'].shape == r['ink_image'].shape
    assert len(r['units']) >= 4
