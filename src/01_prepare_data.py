"""
01_prepare_data.py
------------------
Download the UCI Heart Disease (Cleveland) dataset, clean its missing-data
markers, derive a binary target, and save a single reproducible CSV.

Run from the project root:
    python src/01_prepare_data.py

Notes
-----
* The raw UCI file uses '?' as the missing-value marker in columns `ca`/`thal`.
* The original target `num` is ordinal (0 = no disease, 1-4 = disease severity).
  We collapse it into a binary target: 0 -> 0 (no disease), 1-4 -> 1 (disease).
* Columns `ca` and `thal` are median-imputed so that the shipped data/heart.csv
  is complete and can be used directly by the modelling script.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RAW_PATH = DATA_DIR / "processed.cleveland.raw"
OUT_PATH = DATA_DIR / "heart.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def download_raw() -> str:
    """Return raw CSV text, using a local cache when available."""
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 1000:
        print(f"[info] Using cached raw file: {RAW_PATH}")
        return RAW_PATH.read_text(encoding="utf-8")
    print(f"[info] Downloading dataset from {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        text = response.read().decode("utf-8")
    RAW_PATH.write_text(text, encoding="utf-8")
    print(f"[info] Cached raw file at {RAW_PATH}")
    return text


def main() -> None:
    text = download_raw()
    df = pd.read_csv(StringIO(text), header=None, names=COLUMNS, na_values="?")

    print(f"[info] Raw shape: {df.shape}")
    missing = df.isna().sum()
    print("[info] Missing values (non-zero only):")
    print(missing[missing > 0].to_string())

    # Binary target: any disease severity (1-4) becomes class 1.
    df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)
    df = df.drop(columns=["num"])

    # Enforce numeric dtypes and impute the two columns that contain '?'.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["ca", "thal"]:
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna().reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"[info] Saved clean dataset: {OUT_PATH} (shape={df.shape})")
    print("[info] Target distribution (0=no disease, 1=disease):")
    print(df["target"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
