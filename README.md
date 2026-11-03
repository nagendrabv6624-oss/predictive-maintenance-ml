# Predictive Maintenance Using Machine Learning

A reproducible machine-failure classification project using the **AI4I 2020 Predictive Maintenance Dataset** from UCI.

> **Dataset transparency:** UCI explicitly describes AI4I 2020 as synthetic data designed to reflect industrial predictive-maintenance patterns. It is not real factory telemetry or data from a previous employer. This is an independent benchmark project.

## Problem
Estimate machine-failure probability from operating-condition measurements so maintenance teams can prioritize inspection.

## Dataset
- 10,000 simulated machine-operation records
- Air temperature, process temperature, rotational speed, torque and tool wear
- Product type (categorical)
- Target: Machine failure (0/1)
- Source: [UCI AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)
- Licence: CC BY 4.0; cite UCI

The training script fetches the dataset using `ucimlrepo` by default. Source data are not copied into this repository.

## Method
1. Validate columns, target labels and missing values.
2. Stratified 80/20 train/test split.
3. Preprocess numeric and categorical columns in a scikit-learn pipeline.
4. Compare Logistic Regression, Random Forest and a prior-probability baseline.
5. Select the model using five-fold average-precision cross-validation on training data.
6. Evaluate on the holdout set using precision, recall, F1, ROC-AUC, average precision and confusion matrix.

## Run
```bash
pip install -r requirements.txt
python src/train.py
```
Optional local CSV: `python src/train.py --data path/to/ai4i2020.csv`.

Inference: `python src/predict.py --input path/to/new_machine_rows.csv`.

## Results
GitHub Actions trains on a hosted runner and commits metrics, plots, holdout predictions and the selected fitted model into `outputs/`. Results are valid only after a successful workflow run; inspect `outputs/metrics.json`.

## Outputs
- `outputs/metrics.json` — dataset provenance and metrics
- `outputs/maintenance_model.joblib` — selected fitted pipeline
- `outputs/holdout_predictions.csv`
- `outputs/confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, `feature_importance.png`

## Limitations
AI4I is simulated; a random split is not future-time or unseen-machine validation. Failure labels are imbalanced. Real deployment needs plant-specific temporal validation, calibrated alert thresholds and a maintenance-cost analysis.
