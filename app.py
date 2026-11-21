"""Web interface of the cheque tamper detector.

Upload a cheque scan (or load a demo cheque), pick a model and press Detect: the
result shows whether more than one pen wrote the handwriting and which ink units
differ.  Printed parts of the cheque are removed first, so only handwriting counts.
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.fraud_pipeline import FraudDetector
from src.gradcam import GradCAM, gradcam_pen_classifier, gradcam_siamese, patch_to_rgb
from src.ink.extraction import DEFAULT_FIELDS, FIELD_LABELS
from src.ink.models import MODEL_LABELS, model_path, load_metrics
from src.ink.templates import TemplateLibrary, BANK_NAMES


st.set_page_config(page_title="Cheque Fraud Detection", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

APP_CSS = """
<style>
  .main .block-container { padding-top: 1.6rem; max-width: 1280px; }
  .banner {
      background: linear-gradient(120deg, #083344 0%, #0f4c5c 55%, #115e59 100%);
      padding: 1.5rem 2.2rem; border-radius: 14px; margin-bottom: 1.4rem;
      border-left: 6px solid #2dd4bf;
  }
  .banner h1 { color: #ecfeff; font-size: 2rem; margin: 0 0 .25rem 0; font-weight: 700; }
  .banner p  { color: #a5f3fc; font-size: 1rem; margin: 0; }
  .verdict-panel {
      background: #111c2b; border-radius: 12px; padding: 1.2rem 1.6rem; margin: 1rem 0;
      border: 1px solid #1f2f46;
  }
  .verdict { font-size: 2rem; font-weight: 800; letter-spacing: .02em; margin: 0; }
  .verdict-fraud   { color: #fb7185; }
  .verdict-genuine { color: #4ade80; }
  .verdict-unknown { color: #fbbf24; }
  .metric-tile { background: #111c2b; border: 1px solid #1f2f46; border-radius: 10px;
                 padding: .8rem 1rem; text-align: center; height: 100%; }
  .metric-value { font-size: 1.45rem; font-weight: 700; color: #f1f5f9; }
  .metric-label { font-size: .8rem; color: #94a3b8; margin-top: .2rem; }
  .block-title { color: #e2e8f0; font-size: 1.15rem; font-weight: 600; margin: 1.1rem 0 .6rem;
                 padding-bottom: .35rem; border-bottom: 2px solid #2dd4bf55; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="banner"><h1>Cheque Fraud Detection</h1>'
    '<p>Ink-variation analysis of the handwritten fields: finds cheques altered with a '
    'second pen, even when both inks look the same colour.</p></div>',
    unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_library():
    return TemplateLibrary()


@st.cache_resource(show_spinner=False)
def get_detector(model_type):
    return FraudDetector(model_type=model_type, library=get_library())


def stat(col, value, label):
    col.markdown(f'<div class="metric-tile"><div class="metric-value">{value}</div>'
                 f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)


def header(text):
    st.markdown(f'<p class="block-title">{text}</p>', unsafe_allow_html=True)


def sample_label(fname):
    parts = os.path.splitext(fname)[0].split('_', 2)
    if len(parts) != 3:
        return fname
    kind, bank, cid = parts
    tag = '⚠️ Tampered' if kind == 'fraud' else '✅ Genuine'
    return f"{tag} — {BANK_NAMES.get(bank, bank)} #{cid}"


metrics = load_metrics()
available = [m for m in MODEL_LABELS if os.path.exists(model_path(m))]

# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Detection settings")
    if not available:
        st.error("No trained models found. Run `python -m src.train` first.")
        st.stop()
    default = next((m for m in available if metrics.get(m, {}).get('recommended')), available[0])
    model_type = st.selectbox(
        "Detection model", available, index=available.index(default),
        format_func=lambda m: MODEL_LABELS[m] + ("  ★" if metrics.get(m, {}).get('recommended') else ""))
    calibrated = float(metrics.get(model_type, {}).get('threshold', 0.5))
    threshold = st.slider("Decision threshold", 0.05, 0.95, round(calibrated, 2), 0.01,
                          key=f"thr_{model_type}",
                          help=f"Cheque score above which the cheque is flagged. "
                               f"Cross-validated value: {calibrated:.2f}")
    fields = st.multiselect(
        "Handwritten fields to compare",
        ['payee', 'words', 'figures', 'date', 'signature'], default=list(DEFAULT_FIELDS),
        format_func=lambda f: FIELD_LABELS[f],
        help="Date and signature are often written in advance by the account holder, "
             "so they are excluded by default.")

    st.markdown("---")
    st.markdown("### 📈 Cross-validated performance")
    m = metrics.get(model_type, {})
    if 'cheque_auc' in m:
        st.markdown(f"""
| Metric | Value |
|---|---|
| Cheque ROC-AUC | **{m['cheque_auc']:.3f}** |
| Accuracy | {m['accuracy']:.1%} |
| Precision / Recall | {m['precision']:.1%} / {m['recall']:.1%} |
| Ink-pair ROC-AUC | {m['pair_auc']:.3f} |
""")
        st.caption(f"5 folds grouped by cheque · {m['n_cheques']} IDRBT cheques")
    st.markdown("---")
    st.markdown("### 🏦 Known cheque layouts")
    st.markdown("\n".join(f"- {BANK_NAMES[k]}" for k in get_library().templates) or "_none_")
    st.caption("Other layouts are analysed in generic mode, where printed text cannot be "
               "fully removed.")


tab_detect, tab_models, tab_method = st.tabs(["🔍 Detect", "📊 Model comparison", "📖 How it works"])

# ─── Detection ───────────────────────────────────────────────────────
with tab_detect:
    left_col, right_col = st.columns([5, 3])
    with left_col:
        header("📤 Upload a cheque scan")
        uploaded = st.file_uploader("Cheque image", type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
                                    label_visibility="collapsed")
    with right_col:
        header("🖼️ Or try a sample")
        sample_dir = os.path.join(config.DATA_DIR, "samples")
        samples = sorted(f for f in os.listdir(sample_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))) \
            if os.path.isdir(sample_dir) else []
        choice = st.selectbox("Sample", samples, format_func=sample_label,
                              label_visibility="collapsed") if samples else None
        if st.button("Load sample", use_container_width=True, disabled=not samples):
            st.session_state.sample = choice
            st.session_state.pop('result', None)

    image, source = None, None
    if uploaded is not None:
        bgr = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not decode the uploaded file as an image.")
        else:
            image, source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), uploaded.name
    elif st.session_state.get('sample'):
        bgr = cv2.imread(os.path.join(sample_dir, st.session_state.sample))
        if bgr is not None:
            image, source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), st.session_state.sample

    if image is None:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        for col, icon, title, text in [
            (c1, "✂️", "1. Isolate handwriting",
             "Printed text, logos, security patterns and paper colour are subtracted "
             "using a blank template of the bank's cheque"),
            (c2, "🧪", "2. Measure the ink",
             "Each stroke's optical density in R, G and B — a fingerprint of the ink's dye "
             "that separates pens of the same colour"),
            (c3, "⚖️", "3. Compare all strokes",
             "If a consistent group of strokes was written with a different pen, "
             "the cheque is flagged")]:
            col.markdown(f'<div class="metric-tile"><div style="font-size:2rem">{icon}</div>'
                         f'<div class="metric-label" style="font-size:1rem"><b>{title}</b></div>'
                         f'<div class="metric-label">{text}</div></div>', unsafe_allow_html=True)
        st.info("👆 Upload a cheque scan or load one of the samples to start.")
    else:
        st.markdown("---")
        c_img, c_act = st.columns([3, 1])
        with c_img:
            header("📄 Input cheque")
            st.image(image, width="stretch", caption=source)
        with c_act:
            header("🎯 Actions")
            run = st.button("🔍 Detect Fraud", use_container_width=True, type="primary")
            show_gradcam = st.checkbox("Grad-CAM (CNN models)", value=True)

        if run:
            if not fields:
                st.warning("Select at least one handwritten field.")
            else:
                with st.spinner("Isolating handwriting and comparing inks..."):
                    try:
                        st.session_state.result = get_detector(model_type).detect(
                            image, rgb=True, threshold=threshold, fields=fields)
                        st.session_state.result_model = model_type
                        st.session_state.result_source = source
                    except Exception as e:
                        st.session_state.pop('result', None)
                        st.error(f"Detection error: {e}")

        result = st.session_state.get('result')
        if result and st.session_state.get('result_source') == source:
            v = result['verdict']
            cls = {'FRAUD': 'verdict-fraud', 'GENUINE': 'verdict-genuine'}.get(v, 'verdict-unknown')
            label = {'FRAUD': '⚠️ FRAUD — multiple pens detected',
                     'GENUINE': '✅ GENUINE — single pen'}.get(v, '❔ INSUFFICIENT HANDWRITING')
            border = {'FRAUD': '#ef4444', 'GENUINE': '#22c55e'}.get(v, '#f59e0b')
            n_susp = sum(u['suspicious'] for u in result['units'])
            msg = (f"{n_susp} handwritten ink units disagree with the rest of the cheque's ink "
                   f"— they were written with a different pen." if v == 'FRAUD' else
                   "All handwritten strokes are consistent with a single pen." if v == 'GENUINE'
                   else "Not enough handwriting in the selected fields to compare inks.")
            st.markdown(f'<div class="verdict-panel" style="border-left: 5px solid {border}">'
                        f'<p class="verdict {cls}">{label}</p>'
                        f'<p style="color:#94a3b8;margin:.3rem 0 0">{msg}</p></div>',
                        unsafe_allow_html=True)
            for note in result['notes']:
                st.warning(note)

            c1, c2, c3, c4, c5 = st.columns(5)
            stat(c1, f"{result['confidence']:.0%}", "Confidence")
            stat(c2, f"{result['score']:.2f}", f"Cheque score · threshold {result['threshold']:.2f}")
            stat(c3, len(result['units']), "Ink units compared")
            stat(c4, n_susp, "Suspicious ink units")
            stat(c5, result['bank'], "Cheque layout")

            t1, t2, t3, t4, t5, t6 = st.tabs(["🔴 Suspicious ink", "✍️ Handwriting analysed",
                                               "🌡️ Heat-map", "🧪 Ink fingerprint",
                                               "🧮 Pen comparison", "🧠 Grad-CAM"])
            with t1:
                st.caption("Ink written with a different pen is filled red and boxed."
                           if v == 'FRAUD' else "No ink was attributed to a second pen.")
                st.image(result['highlighted_image'], width="stretch")
            with t2:
                st.caption("Only these pixels were analysed — all printed and coloured parts of the "
                           "cheque were removed using the blank template of its layout.")
                st.image(result['ink_image'], width="stretch")
            with t3:
                st.caption("How strongly each stroke's ink disagrees with the rest of the cheque.")
                st.image(result['heatmap'], width="stretch")
            with t4:
                units = result['units']
                if units and result.get('patches') is not None:
                    od = result['patches'].reshape(len(units), 3, -1)
                    chrom = []
                    for o in od:
                        t = o.sum(0)
                        chrom.append((o[:, t > 0] / t[t > 0]).mean(1) if (t > 0).any() else [0, 0, 0])
                    chrom = np.array(chrom)
                    susp = np.array([u['suspicious'] for u in units])
                    size = np.array([u['n_px'] for u in units]) / 20
                    fig, ax = plt.subplots(figsize=(7, 4.2))
                    ax.scatter(chrom[~susp, 0], chrom[~susp, 2], s=size[~susp], c='#3b82f6',
                               alpha=.7, label='Main ink', edgecolor='k', lw=.3)
                    if susp.any():
                        ax.scatter(chrom[susp, 0], chrom[susp, 2], s=size[susp], c='#ef4444',
                                   alpha=.85, label='Suspicious ink', edgecolor='k', lw=.3)
                    ax.set_xlabel('Blue-channel share of optical density')
                    ax.set_ylabel('Red-channel share of optical density')
                    ax.set_title('Ink fingerprint of every handwritten unit')
                    ax.legend()
                    ax.grid(alpha=.3)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption("Each dot is one ink unit (size = amount of ink). Inks of different "
                               "pens absorb red, green and blue light in different proportions even "
                               "when they look alike. The models use these and further ink "
                               "features (density, stroke width, texture).")
            with t5:
                P = result['pair_matrix']
                if len(P):
                    order = np.argsort([u['distinct'] for u in result['units']])
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(P[np.ix_(order, order)], cmap='RdYlGn_r', vmin=0, vmax=1)
                    fig.colorbar(im, ax=ax, label='P(different pen)')
                    ax.set_title('Every pair of ink units (sorted by distinctness)')
                    ax.set_xlabel('ink unit')
                    ax.set_ylabel('ink unit')
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                rows = [{'Field': f, 'Ink units': s['units'], 'Suspicious': s['suspicious'],
                         'Max distinctness': round(s['max_distinct'], 3)}
                        for f, s in result['field_summary'].items()]
                st.dataframe(pd.DataFrame(rows), hide_index=True)
            with t6:
                mt = st.session_state.result_model
                if mt == 'classical':
                    st.info("Grad-CAM applies to the CNN models. The classical model uses "
                            "hand-crafted optical-density features — see the ink fingerprint tab.")
                elif not show_gradcam:
                    st.info("Enable Grad-CAM in the actions panel.")
                elif result.get('patches') is not None and len(result['units']):
                    net = get_detector(mt).model.net
                    dist = np.array([u['distinct'] for u in result['units']])
                    top = np.argsort(dist)[::-1][:4]
                    ref = int(np.argmin(dist))
                    cols = st.columns(len(top))
                    for col, i in zip(cols, top):
                        p = result['patches'][i]
                        hm = (gradcam_siamese(net, p, result['patches'][ref]) if mt == 'siamese'
                              else gradcam_pen_classifier(net, p))
                        col.image(GradCAM.overlay(patch_to_rgb(p), hm), width="stretch",
                                  caption=f"{FIELD_LABELS[result['units'][i]['field']]} · "
                                          f"distinctness {dist[i]:.2f}")
                    st.caption("Model attention on the four most distinct ink units.")

# ─── Model comparison ────────────────────────────────────────────────
with tab_models:
    header("📊 Cross-validated comparison on the IDRBT cheque dataset")
    rows = []
    for name in MODEL_LABELS:
        mm = metrics.get(name, {})
        if 'cheque_auc' in mm:
            rows.append({'Model': MODEL_LABELS[name], 'Cheque AUC': mm['cheque_auc'],
                         'Accuracy': mm['accuracy'], 'Precision': mm['precision'],
                         'Recall': mm['recall'], 'F1': mm['f1'], 'Specificity': mm['specificity'],
                         'Ink-pair AUC': mm['pair_auc'], 'Threshold': mm['threshold']})
    if rows:
        st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True)
        st.caption("Each held-out cheque is scored twice: as scanned (tampered with a second pen) "
                   "and with the second pen's additions removed (genuine). Folds never share a "
                   "cheque, so every score comes from a model that never saw that cheque.")
    for a, b in [("roc_curves", "model_comparison"), ("score_distributions", "confusion_matrices")]:
        c1, c2 = st.columns(2)
        for col, name in ((c1, a), (c2, b)):
            path = os.path.join(config.FIGURES_DIR, f"{name}.png")
            if os.path.exists(path):
                col.image(path, width="stretch")

# ─── Method ──────────────────────────────────────────────────────────
with tab_method:
    st.markdown("""
#### Why ink variation?
A fraudster alters a cheque (payee, amount) after it was filled in, usually with a pen that
*looks* the same. Different pens use different dyes, so their inks absorb red, green and blue
light in different proportions — a difference a scanner can measure even when the eye cannot.

#### Pipeline
1. **Layout matching & alignment** — the scan is matched to a blank template of its bank's
   cheque (built from the IDRBT scans) and aligned to it.
2. **Handwriting isolation** — a pixel counts as ink only if it is darker than *any* printed
   content of the blank cheque nearby. Logos, labels, ruled lines, the security pattern and the
   coloured background are never analysed. Only the filled fields (payee, amount in words,
   amount in figures) are compared.
3. **Ink measurement** — handwriting is split into small units (about one letter or digit).
   For each unit the optical density `OD = −log(I / I_paper)` per colour channel cancels the
   paper colour; features are OD chromaticity, ink density, stroke width and texture.
4. **Pen comparison** — a model scores every pair of units: *P(different pen)*.
5. **Verdict** — the cheque is flagged when a coherent group of units disagrees with the rest.

#### Models
| | Input | Learns |
|---|---|---|
| Classical ML | 25 optical-density ink features | Random Forest on the feature difference of a unit pair |
| Siamese CNN | 64×64 optical-density ink patches | whether two patches share a pen |
| CNN classifier | 64×64 optical-density ink patches | which of the 14 pens wrote a patch |

#### Training data
IDRBT Cheque Image Dataset — 112 cheques, 9 writers, 14 pens (7 blue, 7 black). Every cheque
was filled with one pen and altered with a second pen **of the same colour**. The metadata
records what the second pen wrote (always a prefix of the amount), which provides pen labels
for the leading digits and words without manual annotation.
""")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#64748b;font-size:.85rem">'
            'Cheque Fraud Detection using Ink Variation Analysis · IDRBT Cheque Image Dataset · '
            'Dansena, Bag &amp; Pal, ICPRMI 2017</div>', unsafe_allow_html=True)
