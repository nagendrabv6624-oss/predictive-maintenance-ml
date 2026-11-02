"""Predictive-maintenance classification demo.

Synthetic data is generated when --data is omitted. It is only for testing
the pipeline, not for claiming real-world predictive performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "air_temperature", "process_temperature", "rotational_speed",
    "torque", "tool_wear",
]
TARGET = "failure"


def make_demo_data(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Generate illustrative sensor readings with an artificial failure label."""
    rng = np.random.default_rng(seed)
    air = rng.normal(300, 2, n)
    process = air + rng.normal(10, 1.5, n)
    speed = rng.normal(1500, 180, n).clip(500, 2600)
    torque = rng.normal(40, 10, n).clip(5, 90)
    wear = rng.uniform(0, 250, n)
    risk = (-8 + 0.018 * wear + 0.035 * torque
            + 0.0015 * np.abs(speed - 1500)
            + 0.3 * (process - air - 10))
    probability = 1 / (1 + np.exp(-risk))
    failure = rng.binomial(1, probability)
    return pd.DataFrame({
        "air_temperature": air, "process_temperature": process,
        "rotational_speed": speed, "torque": torque,
        "tool_wear": wear, "failure": failure,
    })


def load_data(path: str | None) -> tuple[pd.DataFrame, bool]:
    frame = pd.read_csv(path) if path else make_demo_data()
    synthetic = path is None
    if synthetic:
        print("Using SYNTHETIC demonstration data; metrics are not factory evidence.")
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    frame = frame[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("failure target must contain only 0 and 1.")
    if frame[TARGET].nunique() != 2 or len(frame) < 50:
        raise ValueError("Need at least 50 rows and both target classes.")
    return frame, synthetic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    frame, synthetic = load_data(args.data)
    X, y = frame[FEATURES], frame[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, min_samples_leaf=2, n_jobs=-1
        )),
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "data_type": "synthetic_demo" if synthetic else "user_provided",
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "classification_report": classification_report(
            y_test, pred, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    joblib.dump(pipeline, out / "maintenance_model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No failure", "Failure"], cmap="Blues"
    )
    plt.title("Predictive maintenance - confusion matrix")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    importances = pipeline.named_steps["classifier"].feature_importances_
    order = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(FEATURES)[order], importances[order])
    plt.xlabel("Feature importance")
    plt.title("Random Forest feature importance")
    plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=150)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
