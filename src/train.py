"""Predict machine failure with the UCI AI4I 2020 benchmark dataset.

The UCI AI4I 2020 dataset is explicitly synthetic (10,000 simulated records),
not a collection of real factory measurements. This project reports results
on that benchmark only and must not be described as production validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    classification_report, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
TARGET = "Machine failure"
NUMERIC = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
CATEGORICAL = ["Type"]
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_data(path: str | None):
    source = path or UCI_URL
    frame = pd.read_csv(source)
    required = set(NUMERIC + CATEGORICAL + [TARGET])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing expected AI4I columns: {sorted(missing)}")
    frame = frame.dropna(subset=list(required)).copy()
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Machine failure must be a binary 0/1 target.")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Both failure classes must be present.")
    # Exclude IDs and failure-mode labels: those labels define the composite target
    # and would leak target information into the classifier.
    X = frame[NUMERIC + CATEGORICAL].copy()
    y = frame[TARGET]
    return X, y


def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])


def build_models():
    return {
        "dummy_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline([
            ("preprocess", make_preprocessor()),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocess", make_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2,
                class_weight="balanced_subsample",
                max_features="sqrt", random_state=42, n_jobs=-1,
            )),
        ]),
    }


def evaluate(y_true, probability, threshold=0.5):
    prediction = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
        "classification_report": classification_report(
            y_true, prediction, zero_division=0, output_dict=True
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Local ai4i2020.csv; defaults to official UCI download")
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary, fitted = {}, {}
    for name, model in models.items():
        if name == "dummy_baseline":
            # Baseline is reported on the holdout set, not used to select a model.
            continue
        scores = cross_validate(
            model, X_train, y_train, cv=cv, n_jobs=-1,
            scoring={"average_precision": "average_precision", "recall": "recall",
                     "f1": "f1", "roc_auc": "roc_auc"},
        )
        cv_summary[name] = {
            f"cv_{metric}_mean": float(values.mean())
            for metric, values in [
                ("average_precision", scores["test_average_precision"]),
                ("recall", scores["test_recall"]),
                ("f1", scores["test_f1"]),
                ("roc_auc", scores["test_roc_auc"]),
            ]
        }
        model.fit(X_train, y_train)
        fitted[name] = model

    best_name = max(cv_summary, key=lambda name: cv_summary[name]["cv_average_precision_mean"])
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    all_metrics = {
        "dataset": {
            "name": "AI4I 2020 Predictive Maintenance Dataset",
            "source": "UCI Machine Learning Repository, DOI 10.24432/C5HS5C",
            "dataset_type": "synthetic simulated industrial data",
            "rows": int(len(X)), "failure_rate": float(y.mean()),
            "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "features": NUMERIC + CATEGORICAL,
            "excluded_columns": ["UDI", "Product ID"] + LEAKAGE_COLUMNS,
            "split": "stratified 80/20 holdout, random_state=42",
        },
        "cross_validation": cv_summary,
        "holdout": {},
        "selected_model": best_name,
        "interpretation": "Benchmark performance on simulated AI4I records only; not proof of performance on real factory data.",
    }

    for name, model in models.items():
        if name == "dummy_baseline":
            model.fit(X_train, y_train)
        else:
            model = fitted[name]
        probability = model.predict_proba(X_test)[:, 1]
        all_metrics["holdout"][name] = evaluate(y_test, probability)
        if name == best_name:
            best_probability = probability
            best_model = model

    joblib.dump(best_model, out / "maintenance_model.joblib")
    (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    predictions = pd.DataFrame({
        "actual_failure": y_test.to_numpy(),
        "failure_probability": best_probability,
        "predicted_failure": (best_probability >= 0.5).astype(int),
    })
    predictions.to_csv(out / "holdout_predictions.csv", index=False)

    ConfusionMatrixDisplay.from_predictions(
        y_test, (best_probability >= 0.5).astype(int),
        display_labels=["No failure", "Failure"], cmap="Blues",
    )
    plt.title(f"AI4I holdout confusion matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, best_probability)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–recall curve (holdout)")
    plt.tight_layout()
    plt.savefig(out / "precision_recall_curve.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, best_probability)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, best_probability):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (holdout)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=160)
    plt.close()

    if best_name == "random_forest":
        names = best_model.named_steps["preprocess"].get_feature_names_out()
        importance = best_model.named_steps["classifier"].feature_importances_
        order = np.argsort(importance)[-10:]
        plt.figure(figsize=(8, 5))
        plt.barh(names[order], importance[order])
        plt.xlabel("Feature importance")
        plt.title("Random Forest — top 10 features")
        plt.tight_layout()
        plt.savefig(out / "feature_importance.png", dpi=160)
        plt.close()

    print(json.dumps({
        "selected_model": best_name,
        "cross_validation": cv_summary,
        "holdout": {k: {m: v for m, v in val.items()
                        if m not in ["classification_report", "confusion_matrix"]}
                    for k, val in all_metrics["holdout"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
