"""
03_model_evaluation.py
----------------------
Train two classifiers (Logistic Regression + Random Forest) on the UCI Heart
Disease dataset, evaluate them on a held-out stratified test set, and export
reproducible figures and metrics.

Outputs
-------
outputs/figures/fig5_confusion_matrix.png
outputs/figures/fig6_roc_curve.png
outputs/figures/fig7_feature_importance.png
outputs/figures/fig8_model_comparison.png
outputs/metrics.json
outputs/classification_report.txt

Run from the project root:
    python src/03_model_evaluation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

BLUE = "#2563a8"
ORANGE = "#c9662b"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "heart.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[info] Train size = {len(X_train)}, test size = {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000,
                                                   random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1
        ),
    }

    fitted: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}

    for name, estimator in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        predictions[name] = pred
        probabilities[name] = proba
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        }
        print(f"[info] {name}: {metrics[name]}")

    # ---- Figure 5: confusion matrices -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for ax, (name, _) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, predictions[name])
        image = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else "#1e293b")
        ax.set_title(f"{name}\nACC={metrics[name]['accuracy']:.3f}  "
                     f"Recall={metrics[name]['recall']:.3f}")
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion matrices on the held-out test set (n=%d)" % len(y_test), y=1.03)
    fig.savefig(FIG_DIR / "fig5_confusion_matrix.png")
    plt.close(fig)

    # ---- Figure 6: ROC curves ---------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    for name, color in [("Logistic Regression", BLUE), ("Random Forest", ORANGE)]:
        fpr, tpr, _ = roc_curve(y_test, probabilities[name])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={metrics[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Receiver operating characteristic (test set)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.savefig(FIG_DIR / "fig6_roc_curve.png")
    plt.close(fig)

    # ---- Figure 7: Random Forest feature importance -----------------------
    rf_pipe = fitted["Random Forest"]
    feature_names = rf_pipe.named_steps["prep"].get_feature_names_out()
    importances = rf_pipe.named_steps["model"].feature_importances_
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .assign(feature=lambda d: d["feature"].str.replace("cat__", "", regex=False)
                                                     .str.replace("num__", "", regex=False))
        .sort_values("importance", ascending=False)
        .head(12)
        .sort_values("importance")
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(importance["feature"], importance["importance"], color=BLUE)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest: top 12 predictive features")
    ax.grid(axis="y", alpha=0)
    fig.savefig(FIG_DIR / "fig7_feature_importance.png")
    plt.close(fig)

    # ---- Figure 8: metric comparison --------------------------------------
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metric_keys))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for offset, (name, color) in zip([-width / 2, width / 2],
                                     [("Logistic Regression", BLUE), ("Random Forest", ORANGE)]):
        values = [metrics[name][key] for key in metric_keys]
        bars = ax.bar(x + offset, values, width, label=name, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                    f"{value:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [k.upper() if k != "roc_auc" else "ROC-AUC" for k in metric_keys])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Test-set performance comparison")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.savefig(FIG_DIR / "fig8_model_comparison.png")
    plt.close(fig)

    # ---- Export metrics and report ----------------------------------------
    results = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(y_test)),
        "models": metrics,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = (
        f"Classification report — Random Forest (test set, seed={RANDOM_STATE})\n"
        + "=" * 64 + "\n"
        + classification_report(y_test, predictions["Random Forest"],
                                target_names=["No disease", "Disease"], digits=4)
    )
    (OUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    print("[info] Saved figures to", FIG_DIR)
    print("[info] Saved metrics.json and classification_report.txt")
    print(report)


if __name__ == "__main__":
    main()
