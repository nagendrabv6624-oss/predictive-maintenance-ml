# Predictive Maintenance Using Machine Learning

A binary classification workflow that estimates machine-failure risk from operating-condition and sensor features.

## Dataset: AI4I 2020

This project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository:
- 10,000 records
- Sensor inputs: air temperature, process temperature, rotational speed, torque and tool wear
- Categorical product type: L, M or H
- Target: `Machine failure` (0/1)

**Important data limitation:** UCI explicitly describes AI4I 2020 as a *synthetic dataset designed to reflect predictive-maintenance data*. It is not a collection of real factory sensor records. Therefore, model metrics from this dataset are benchmark results on simulated industrial data, not proof of deployment performance. [UCI dataset documentation](https://archive.ics.uci.edu/dataset/601/ai4i)

Dataset citation: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C), CC BY 4.0.

## Objective

Predict whether a machine failure is recorded for a given operating point. The model uses five numeric sensor variables and product type.

## Leakage prevention

The pipeline intentionally excludes:
- `UDI` and `Product ID`: identifiers, not operational predictors.
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF`: component failure-mode labels used to construct the composite `Machine failure` target. Including them would leak target information.

## Methodology

1. Load and validate the official UCI CSV (or a local copy).
2. Stratified 80/20 train/test split.
3. Median imputation and scaling for numeric columns; one-hot encoding for product type.
4. Compare Logistic Regression and Random Forest using five-fold stratified cross-validation.
5. Include a prior-probability dummy baseline.
6. Select the model by mean cross-validation average precision.
7. Report precision, recall, F1, ROC-AUC, average precision and confusion matrix on the held-out test set.

Because failures are uncommon, accuracy alone is not an adequate measure. The precision–recall curve and recall are important for understanding missed failures and false alarms.

## Run

Python 3.10–3.12 is recommended.

```bash
pip install -r requirements.txt
python src/train.py
```

The training script downloads the CSV from the UCI public dataset endpoint by default. To use a local copy:

```bash
python src/train.py --data data/ai4i2020.csv
```

For inference on new rows with the same six input columns:

```bash
python src/predict.py --input data/new_machine_readings.csv
```

## Outputs

After training, `outputs/` contains the selected model, metrics JSON, holdout predictions, confusion matrix, precision–recall curve, ROC curve and feature-importance chart (when Random Forest is selected).

No model metrics are pre-filled in this repository: run the pipeline to generate them from the downloaded dataset. This avoids presenting unverified numbers as experimental results.

## Limitations

- AI4I is synthetic and cannot establish performance on real machines.
- The dataset does not provide a true time-series maintenance history for every machine; random stratified validation is a benchmark split, not a deployment simulation.
- Threshold 0.5 is a default, not a maintenance policy. Thresholds should be selected using operational costs and validated on real, time-ordered plant data.
- A real deployment needs machine-level and temporal validation, monitoring, calibration, and human review.

## Skills demonstrated

Python, Pandas, scikit-learn pipelines, class imbalance, feature preprocessing, cross-validation, Random Forest, Logistic Regression, classification metrics and data visualization.
