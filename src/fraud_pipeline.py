"""
End-to-end cheque fraud detection by ink-variation analysis.

For a cheque image:
  1. Match the bank layout and align to its blank template
  2. Keep only *handwritten* ink in the filled fields (payee, amount in
     words, amount in figures) - printed text, logos, security patterns and
     background colours are subtracted away
  3. Split the handwriting into small ink units and describe each unit's
     ink by its optical density (paper-independent ink colour, density,
     stroke texture)
  4. A trained model scores every unit pair:  P(written with a different pen)
  5. If a coherent group of units disagrees with the rest of the cheque's
     ink, more than one pen was used -> FRAUD
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.ink.extraction import (extract_ink, segment_units, DEFAULT_FIELDS,
                                FIELD_LABELS)
from src.ink.features import unit_features, unit_patch
from src.ink.models import load_model, MODEL_LABELS
from src.ink.pairs import cheque_score
from src.ink.templates import TemplateLibrary

MIN_UNITS = 4


class FraudDetector:
    """
    Usage:
        detector = FraudDetector('classical')
        result = detector.detect("cheque.tif", rgb=False)
        print(result['verdict'], result['confidence'])
    """

    def __init__(self, model_type='classical', fields=DEFAULT_FIELDS, library=None):
        self.model_type = model_type
        self.fields = tuple(fields)
        self.model = load_model(model_type)
        self.threshold = self.model.threshold
        self.library = library or TemplateLibrary()

    def detect(self, image, rgb=True, threshold=None, fields=None):
        """
        Args:
            image: path or (H, W, 3) uint8 array
            rgb: whether an array input is RGB (True) or BGR (False)
            threshold: override the calibrated cheque-score threshold
            fields: override which handwritten fields are analysed

        Returns dict:
            verdict            'FRAUD' / 'GENUINE' / 'INSUFFICIENT INK'
            confidence         [0, 1]
            score, threshold   cheque score and decision threshold
            bank, mode         matched layout; mode 'template' or 'generic'
            units              per-unit dicts (field, bbox, distinct, suspicious)
            pair_matrix        (n, n) P(different pen)
            highlighted_image  RGB cheque with suspicious ink marked
            ink_image          RGB view of only the handwriting analysed
            heatmap            RGB ink-distinctness heat-map overlay
            field_summary      {field: {'units', 'suspicious', 'max_distinct'}}
            notes              list of strings
        """
        thr = self.threshold if threshold is None else threshold
        fields = tuple(fields) if fields else self.fields
        inkmap = extract_ink(image, self.library, rgb=rgb)
        units = segment_units(inkmap, fields)
        notes = list(inkmap.notes)

        if len(units) < MIN_UNITS:
            notes.append("Too little handwriting found in the filled fields to compare inks.")
            n = len(units)
            return self._result('INSUFFICIENT INK', 0.0, 0.0, thr, inkmap, units, fields,
                                np.zeros((n, n)), np.zeros(n), np.zeros(n, bool), notes, None)

        feats = np.stack([unit_features(inkmap, u) for u in units])
        patches = np.stack([unit_patch(inkmap, u) for u in units])
        n_px = np.array([u['mask'].sum() for u in units])
        P = self.model.pair_matrix(feats, patches, n_px)
        score, distinct = cheque_score(P, n_px)

        fraud = score >= thr
        suspicious = np.zeros(len(units), bool)
        if fraud:
            # the coherent minority ink group that drives the score ...
            k = max(2, int(round(0.1 * len(units))))
            core = np.argsort(distinct)[::-1][:k]
            suspicious[core] = True
            # ... plus every unit whose ink agrees with that group
            agree = P[:, core].mean(1) < 0.5
            suspicious |= agree & (distinct >= 0.5 * distinct[core].min())

        if fraud:
            confidence = 0.5 + 0.5 * min(1.0, (score - thr) / max(1e-6, 1 - thr))
        else:
            confidence = 0.5 + 0.5 * min(1.0, (thr - score) / max(1e-6, thr))
        return self._result('FRAUD' if fraud else 'GENUINE', confidence, score, thr,
                            inkmap, units, fields, P, distinct, suspicious, notes, patches)

    # ─── rendering ──────────────────────────────────────────────────────

    def _result(self, verdict, confidence, score, thr, inkmap, units, fields, P,
                distinct, suspicious, notes, patches):
        rgb = cv2.cvtColor(inkmap.image, cv2.COLOR_BGR2RGB)
        unit_info, summary = [], {}
        for u, d, s in zip(units, distinct, suspicious):
            unit_info.append(dict(field=u['field'], x=int(u['x']), y=int(u['y']),
                                  w=int(u['w']), h=int(u['h']), n_px=int(u['mask'].sum()),
                                  distinct=float(d), suspicious=bool(s)))
            fs = summary.setdefault(FIELD_LABELS.get(u['field'], u['field']),
                                    {'units': 0, 'suspicious': 0, 'max_distinct': 0.0})
            fs['units'] += 1
            fs['suspicious'] += int(s)
            fs['max_distinct'] = max(fs['max_distinct'], float(d))
        return {
            'verdict': verdict, 'confidence': float(confidence),
            'score': float(score), 'threshold': float(thr),
            'model': MODEL_LABELS.get(self.model_type, self.model_type),
            'bank': inkmap.bank_name, 'mode': inkmap.mode, 'match_cc': float(inkmap.match_cc),
            'units': unit_info, 'pair_matrix': P, 'patches': patches,
            'highlighted_image': self._highlight(rgb, units, suspicious),
            'ink_image': self._ink_view(rgb, inkmap, fields),
            'heatmap': self._heatmap(rgb, units, distinct),
            'field_summary': summary, 'notes': notes,
        }

    @staticmethod
    def _ink_view(rgb, inkmap, fields):
        """Only the analysed handwriting, on white, with field boxes."""
        shown = list(fields) if inkmap.mode == 'template' else ['all']
        out = np.full_like(rgb, 255)
        m = inkmap.field_mask(shown)
        out[m] = rgb[m]
        for f in shown:
            if f not in inkmap.rois:
                continue
            x1, y1, x2, y2 = inkmap.rois[f]
            cv2.rectangle(out, (x1, y1), (x2, y2), (150, 150, 150), 2)
            cv2.putText(out, FIELD_LABELS.get(f, f), (x1 + 6, y1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
        return out

    @staticmethod
    def _highlight(rgb, units, suspicious):
        out = rgb.copy()
        overlay = rgb.copy()
        for u, s in zip(units, suspicious):
            if not s:
                continue
            y, x, h, w = u['y'], u['x'], u['h'], u['w']
            overlay[y:y + h, x:x + w][u['mask']] = (230, 30, 30)
            cv2.rectangle(out, (x - 4, y - 4), (x + w + 4, y + h + 4), (230, 30, 30), 3)
        return cv2.addWeighted(overlay, 0.6, out, 0.4, 0)

    @staticmethod
    def _heatmap(rgb, units, distinct):
        heat = np.zeros(rgb.shape[:2], np.float32)
        for u, d in zip(units, distinct):
            y, x, h, w = u['y'], u['x'], u['h'], u['w']
            heat[y:y + h, x:x + w] = np.maximum(heat[y:y + h, x:x + w], d)
        heat = cv2.GaussianBlur(cv2.dilate(heat, np.ones((25, 25), np.uint8)), (0, 0), 12)
        col = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        col = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
        alpha = np.clip(heat * 1.5, 0, 0.65)[..., None]
        return (rgb * (1 - alpha) + col * alpha).astype(np.uint8)


def detect_fraud(image_path, model_type='classical'):
    """Convenience wrapper for single-image detection."""
    return FraudDetector(model_type).detect(image_path, rgb=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Detect multi-pen tampering on a cheque")
    ap.add_argument("image")
    ap.add_argument("--model", default="classical", choices=list(MODEL_LABELS))
    args = ap.parse_args()
    r = detect_fraud(args.image, args.model)
    print(f"{r['verdict']}  confidence {r['confidence']:.0%}  "
          f"score {r['score']:.3f} (threshold {r['threshold']:.3f})  layout: {r['bank']}")
    for f, s in r['field_summary'].items():
        print(f"  {f:18s} units={s['units']:3d}  suspicious={s['suspicious']}")
    out = os.path.join(config.SAMPLE_RESULTS_DIR, "detection.png")
    cv2.imwrite(out, cv2.cvtColor(r['highlighted_image'], cv2.COLOR_RGB2BGR))
    print(f"  highlighted image -> {out}")
