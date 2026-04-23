"""
scripts/train_upstream_model.py
================================
Downloads UCI Adult dataset, trains XGBoost binary classifier,
saves model + feature metadata to disk.

Protected: sex, race, age, native-country
Merit: education, hours-per-week, workclass, occupation
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import json
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

PROTECTED_COLS = ["sex", "race", "age", "native-country"]
MERIT_COLS = ["education", "hours-per-week", "workclass", "occupation"]
ALL_COLS = MERIT_COLS + PROTECTED_COLS


def train():
    print("Fetching UCI Adult dataset...")
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets

    # Clean target
    y = y.iloc[:, 0].str.strip().str.replace(".", "", regex=False)
    y = (y == ">50K").astype(int)

    # Select only our columns
    X = X[ALL_COLS].copy()

    # Drop rows with missing values
    mask = ~X.isin(["?"]).any(axis=1) & X.notna().all(axis=1)
    X = X[mask]
    y = y[X.index]

    # Encode categoricals
    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le.classes_.tolist()

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.4f}")

    joblib.dump(model, MODEL_DIR / "xgboost_upstream.joblib")
    joblib.dump(encoders, MODEL_DIR / "encoders.joblib")

    # Save column metadata
    metadata = {
        "feature_columns": ALL_COLS,
        "merit_columns": MERIT_COLS,
        "protected_columns": PROTECTED_COLS,
        "categorical_columns": list(encoders.keys()),
        "numeric_columns": [c for c in ALL_COLS if c not in encoders],
        "train_size": len(X_train),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_auc": float(roc_auc_score(y_test, proba)),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save a sample of 500 real rows for the data stream
    sample_idx = np.random.RandomState(42).choice(len(X), size=min(500, len(X)), replace=False)
    sample = X.iloc[sample_idx].copy()
    # Re-attach original string values using encoders for display
    sample_display = sample.copy()
    for col, classes in encoders.items():
        sample_display[col] = sample[col].map(
            lambda i, c=classes: c[int(i)] if int(i) < len(c) else str(i)
        )
    sample_display["true_label"] = y.iloc[sample_idx].values
    sample_display.to_json(MODEL_DIR / "sample_rows.json", orient="records")

    print(f"Model saved to {MODEL_DIR}/")
    print(f"Sample rows saved: {len(sample_display)}")


if __name__ == "__main__":
    train()
