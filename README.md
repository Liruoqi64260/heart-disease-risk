# Heart Disease Risk Prediction

A reproducible binary-classification case study built for **CISC7204 — Data
Visualisation & Communication** (University of Macau). The project applies the same
end-to-end classification workflow developed in the semester *e-commerce customer
churn prediction* assignment to an independent clinical screening problem.

> **Ethical note.** This model is a **risk-stratification aid** only. It is not a
> diagnostic device and must never be used in place of clinical judgement.

---

## 1. Project background

Cardiovascular disease is a leading cause of mortality, and early identification of
at-risk patients supports timely referral and intervention. Routine clinical
measurements — age, blood pressure, cholesterol, resting ECG and exercise-test
results — are inexpensive to collect and may carry substantial predictive signal.

This study asks: *can a simple, fully interpretable machine-learning pipeline
distinguish patients with and without heart disease from these measurements?* Two
classifiers are compared, and evaluation emphasises **recall** (missing a true case
is costly in screening) alongside ROC-AUC and precision.

## 2. Dataset description

| Property | Value |
| --- | --- |
| Source | [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), Cleveland subset (`processed.cleveland.data`) |
| Raw URL | `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data` |
| Records | 303 patients (after cleaning) |
| Input features | 13 |
| Target | `num` recoded to binary: `0` = no disease (164), `1` = disease (139) |
| Missing data | 4 `ca` values and 2 `thal` values marked `?`; median-imputed |
| Class balance | 45.9% positive |

**Feature dictionary**

| Feature | Meaning | Type |
| --- | --- | --- |
| `age` | Age in years | Numeric |
| `sex` | 1 = male, 0 = female | Categorical |
| `cp` | Chest-pain type (1–4) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| `chol` | Serum cholesterol (mg/dl) | Numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl | Categorical |
| `restecg` | Resting ECG result (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numeric |
| `exang` | Exercise-induced angina | Categorical |
| `oldpeak` | ST depression induced by exercise | Numeric |
| `slope` | Slope of peak exercise ST segment (1–3) | Categorical |
| `ca` | Number of major vessels coloured by fluoroscopy | Categorical |
| `thal` | Thalassemia category (3/6/7) | Categorical |

The raw UCI file is cached at `data/processed.cleveland.raw` so that reproduction
does not depend on network availability. The cleaned, analysis-ready file is
`data/heart.csv`.

## 3. Environment dependencies

Tested on Python 3.14.6 with the following packages (see `requirements.txt`):

```text
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
```

Create an isolated environment and install:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No other dataset download, API key or manual step is required.

## 4. Reproducing the analysis

From the project root, run the three numbered stages **in order**, or run the wrapper:

```bash
# One-command reproduction
python src/run_all.py

# Equivalent staged execution
python src/01_prepare_data.py       # download -> clean -> data/heart.csv
python src/02_eda.py                # EDA figures + outputs/eda_summary.json
python src/03_model_evaluation.py   # train/evaluate + model figures + metrics
```

All randomness is fixed (`RANDOM_STATE = 42`) and the train/test split is stratified
(80/20). Expected console output includes:

```text
Train size = 242, test size = 61
Logistic Regression: {'accuracy': 0.8852, 'precision': 0.8387, 'recall': 0.9286, 'f1': 0.8814, 'roc_auc': 0.9665}
Random Forest:       {'accuracy': 0.8852, 'precision': 0.8182, 'recall': 0.9643, 'f1': 0.8852, 'roc_auc': 0.9464}
```

Figures are written to `outputs/figures/`; machine-readable results to
`outputs/metrics.json`, `outputs/eda_summary.json` and
`outputs/classification_report.txt`. These exact figures are copied into the
portfolio site at `../portfolio-site/assets/img/`.

## 5. Methodology

1. **Cleaning** — `?` markers → missing, median imputation for `ca`/`thal`, binary target.
2. **Preprocessing** — pipelines only: median imputation + `StandardScaler` for
   numerical features; mode imputation + one-hot encoding for categorical features.
   Fitting happens on the training fold exclusively, preventing data leakage.
3. **Models** — regularised Logistic Regression (linear baseline) and Random Forest
   (300 trees, `class_weight="balanced"`, captures non-linear interactions).
4. **Evaluation** — accuracy, precision, recall, F1 and ROC-AUC on 61 unseen cases;
   confusion matrices and ROC curves for diagnostic inspection; impurity-based
   feature importance for interpretation.

## 6. Results interpretation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.885 | 0.839 | 0.929 | 0.881 | **0.967** |
| Random Forest | 0.885 | 0.818 | **0.964** | **0.885** | 0.946 |

Both models reach ~89% accuracy and clearly outperform chance. The Logistic
Regression ranks cases slightly better globally (higher ROC-AUC), whereas the Random
Forest is more sensitive: it correctly identifies 27 of 28 diseased patients
(recall 0.964) at the cost of a few more false positives. For screening, the higher
recall is usually preferable because a missed case is more harmful than an extra
confirmatory test.

EDA and model interpretation agree on the dominant predictors: **thalassemia**,
**number of major vessels**, **maximum heart rate** and **exercise-induced
angina**. Notably, serum cholesterol is only weakly related to the target in this
cohort — a useful reminder that domain expectations should be checked against data.

### Figures

| Figure | File |
| --- | --- |
| 1. Target distribution | `outputs/figures/fig1_target_distribution.png` |
| 2. Correlation heatmap | `outputs/figures/fig2_correlation_heatmap.png` |
| 3. Measurements by diagnosis | `outputs/figures/fig3_numeric_by_target.png` |
| 4. Categorical risk rates | `outputs/figures/fig4_categorical_risk.png` |
| 5. Confusion matrices | `outputs/figures/fig5_confusion_matrix.png` |
| 6. ROC curves | `outputs/figures/fig6_roc_curve.png` |
| 7. Feature importance | `outputs/figures/fig7_feature_importance.png` |
| 8. Metric comparison | `outputs/figures/fig8_model_comparison.png` |

## 7. Limitations and future work

* **Small, single-site cohort** (n = 303; test n = 61). Confidence intervals are
  wide; results should not be extrapolated to other populations.
* **No external validation** — performance on a hospital from a different region
  is unknown.
* **Fairness and calibration not assessed.** subgroup performance by sex and age,
  and probability calibration, must be checked before any operational use.
* **Planned improvements**: interactive Plotly dashboards, a lightweight deployed
  demo, a fuller discussion of class imbalance and threshold selection, and a
  tighter visual narrative linking every chart to a decision.

## 8. Repository structure

```text
heart-disease-risk/
├── README.md                        # this file
├── requirements.txt                 # pinned-compatible dependencies
├── data/
│   ├── processed.cleveland.raw      # cached UCI raw file
│   └── heart.csv                    # cleaned analysis-ready data
├── src/
│   ├── 01_prepare_data.py           # download, clean, derive target
│   ├── 02_eda.py                    # EDA figures and summary
│   ├── 03_model_evaluation.py       # pipeline, models, evaluation, figures
│   └── run_all.py                   # one-command reproduction
└── outputs/
    ├── eda_summary.json
    ├── metrics.json
    ├── classification_report.txt
    └── figures/fig1…fig8.png
```

## 9. Data source and attribution

Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J.-J., Sandhu, S.,
Guppy, J. S., Lee, S., & Froelicher, V. (1989). *Heart Disease* [Dataset]. UCI
Machine Learning Repository. https://doi.org/10.24432/C52P4X

---

**Student:** Ruoqi Li (mc64260) · University of Macau · CISC7204
