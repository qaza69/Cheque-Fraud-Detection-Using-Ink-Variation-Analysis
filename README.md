# Cheque Fraud Detection using Ink Variation Analysis

Detects bank cheques that were altered with a **second pen**, even when both pens
are the same colour, by comparing the ink of the handwritten fields only. Everything
that is printed on the cheque (logo, labels, ruled lines, security pattern,
coloured background) is removed before the analysis.

## How it works

1. **Template matching** - the scan is matched to a blank template of its bank
   layout (Axis, ICICI, Canara, Syndicate) and aligned with ECC registration.
2. **Handwriting isolation** - a pixel counts as ink only when it is darker than
   anything printed on the blank template next to it.
3. **Ink units** - the payee, amount in words and amount in figures are cut into
   letter-sized units.
4. **Optical-density features** - `OD = -log(I / I_paper)` per colour channel removes
   the paper colour; 25 features describe the dye ratio, density, stroke width and texture.
5. **Pair model** - for every pair of units a model gives P(different pen).
6. **Decision** - the cheque is flagged when a coherent group of units disagrees with
   the rest; those units are highlighted.

Three models share one interface (`pair_matrix`): a Random Forest on the ink features,
a Siamese CNN and a 14-pen CNN classifier.

## Results (IDRBT, 5-fold cross-validation grouped by cheque)

| Model | Cheque AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Random Forest on ink features (recommended) | 0.966 | 92.9% | 97.1% | 88.4% |
| CNN pen classifier | 0.912 | 83.9% | 91.3% | 75.0% |
| Siamese CNN | 0.884 | 81.7% | 87.4% | 74.1% |

Every IDRBT cheque is tampered, so each held-out cheque is scored twice: as scanned
(tampered) and with the second pen's additions removed (genuine). These numbers are
therefore an estimate, not a measurement on real single-pen cheques.

## Run it

```bash
pip install -r requirements.txt
# put IDRBT_Cheque_Image_Dataset.zip's contents (DATA.xlsx and the 300/ folder) into data/raw/
python -m src.train          # templates, dataset, cross-validation, final models (~15-25 min on CPU)
python make_samples.py       # optional: demo cheque pairs in data/samples/
streamlit run app.py         # http://localhost:8501
python -m pytest tests -v    # 17 tests
python test_detection.py     # smoke test on the demo cheques
```

## Code map

| Path | Purpose |
|---|---|
| `src/ink/templates.py` | blank templates, layout matching, alignment, field boxes |
| `src/ink/extraction.py` | handwriting isolation and ink units |
| `src/ink/features.py` | optical-density features and patches |
| `src/ink/idrbt.py`, `dataset.py` | dataset metadata and the pen-labelled ink-unit dataset |
| `src/ink/pairs.py` | within-cheque pairs and the cheque score |
| `src/ink/models.py` | the three pen-difference models |
| `src/fraud_pipeline.py` | end-to-end `FraudDetector` |
| `src/train.py`, `src/evaluate.py` | training and cross-validation |
| `src/gradcam.py` | Grad-CAM explanations |
| `app.py` | Streamlit interface |
| `make_samples.py`, `make_synthetic_genuine.py` | demo cheques and simulated single-pen cheques |

## Limitations

- No real single-pen cheques exist in the dataset; genuine examples are derived or simulated.
- The same 14 pens occur in the training and test folds.
- Only four bank layouts have templates; other layouts use a weaker generic mode.
- Flatbed scans only; phone photographs would need perspective and colour correction.
- The output is a flag for a human reviewer, never an automatic rejection.

## Data reference

IDRBT Cheque Image Dataset: https://www.idrbt.ac.in/idrbt-cheque-image-dataset/
