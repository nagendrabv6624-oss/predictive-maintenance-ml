"""Predict machine-failure probability for new AI4I-format rows."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Type",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV containing the six model input columns")
    parser.add_argument("--output", default="outputs/new_predictions.csv")
    args = parser.parse_args()
    model_path = ROOT / "outputs" / "maintenance_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Run python src/train.py first.")
    frame = pd.read_csv(args.input)
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    model = joblib.load(model_path)
    probability = model.predict_proba(frame[FEATURES])[:, 1]
    result = pd.DataFrame({
        "failure_probability": probability,
        "predicted_failure_at_0_5": (probability >= 0.5).astype(int),
    })
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.head())
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
