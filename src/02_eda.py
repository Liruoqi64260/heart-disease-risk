"""
02_eda.py
---------
Exploratory data analysis for the UCI Heart Disease (Cleveland) dataset.

Outputs
-------
outputs/figures/fig1_target_distribution.png
outputs/figures/fig2_correlation_heatmap.png
outputs/figures/fig3_numeric_by_target.png
outputs/figures/fig4_categorical_risk.png
outputs/eda_summary.json

Run from the project root:
    python src/02_eda.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / reproducible rendering
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = ROOT / "outputs" / "eda_summary.json"

# Clean, consistent visual style (no animation, print-friendly).
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
GREY = "#64748b"


def style_axes(ax: plt.Axes) -> None:
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#334155")


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "heart.csv")
    n = len(df)
    pos = int(df["target"].sum())
    print(f"[info] Loaded heart.csv: rows={n}, disease={pos}, no_disease={n - pos}")

    # ---- Figure 1: target balance -----------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    counts = df["target"].value_counts().sort_index()
    bars = ax.bar(["No disease (0)", "Disease (1)"], counts.values,
                  color=[BLUE, ORANGE], width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 4,
                f"{value}\n({value / n:.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of patients")
    ax.set_title("Target class distribution (Cleveland cohort)")
    ax.set_ylim(0, max(counts.values) * 1.25)
    style_axes(ax)
    fig.savefig(FIG_DIR / "fig1_target_distribution.png")
    plt.close(fig)

    # ---- Figure 2: correlation heatmap ------------------------------------
    corr = df.corr(numeric_only=True)
    labels = list(corr.columns)
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(value) > 0.55 else "#1e293b")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Pearson correlation among numerical features and target")
    ax.grid(False)
    fig.savefig(FIG_DIR / "fig2_correlation_heatmap.png")
    plt.close(fig)

    # ---- Figure 3: clinical measurements by target ------------------------
    measurements = [
        ("age", "Age (years)"),
        ("trestbps", "Resting blood pressure (mm Hg)"),
        ("chol", "Serum cholesterol (mg/dl)"),
        ("thalach", "Max heart rate achieved (bpm)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8))
    for ax, (col, title) in zip(axes.ravel(), measurements):
        groups = [df.loc[df["target"] == t, col].dropna().values for t in (0, 1)]
        ax.boxplot(groups, tick_labels=["No disease", "Disease"],
                   patch_artist=True, widths=0.55,
                   boxprops=dict(facecolor="#dbeafe", edgecolor=BLUE),
                   medianprops=dict(color=ORANGE, linewidth=1.6))
        ax.set_title(title, fontsize=10)
        style_axes(ax)
    fig.suptitle("Distribution of key clinical measurements by diagnosis", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_numeric_by_target.png")
    plt.close(fig)

    # ---- Figure 4: categorical risk rates ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (col, title) in zip(axes, [("cp", "Chest pain type (cp)"),
                                       ("thal", "Thalassemia (thal)")]):
        rates = df.groupby(col, observed=True)["target"].mean().sort_index()
        bars = ax.bar([str(x) for x in rates.index], rates.values,
                      color=BLUE, width=0.6)
        for bar, value in zip(bars, rates.values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                    f"{value:.0%}", ha="center", fontsize=8)
        ax.set_ylabel("Observed disease rate")
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        style_axes(ax)
    fig.suptitle("Empirical disease rate by categorical clinical feature", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_categorical_risk.png")
    plt.close(fig)

    # ---- Machine-readable EDA summary -------------------------------------
    summary = {
        "n_rows": n,
        "n_features": df.shape[1] - 1,
        "n_disease": pos,
        "n_no_disease": n - pos,
        "disease_rate": round(pos / n, 4),
        "top_absolute_correlations_with_target": [
            {"feature": f, "pearson_r": round(float(corr.loc[f, "target"]), 4)}
            for f in corr["target"].drop("target").abs().sort_values(ascending=False).head(6).index
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[info] Saved {OUT_JSON}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
