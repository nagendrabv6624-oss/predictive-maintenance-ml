# Project 2 — Predictive Maintenance ML

## Industrial problem
Estimate machine-failure risk from operating-condition measurements so maintenance teams can prioritize inspection.

## Public benchmark
AI4I 2020 Predictive Maintenance Dataset — 10,000 simulated machine-operation records from UCI. UCI explicitly states the data are synthetic and reflect industrial predictive-maintenance patterns. This is an independent benchmark project, not a claim of previous-employer data.

Source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset  
DOI: 10.24432/C5HS5C

## Inputs and target
Air temperature (K), process temperature (K), rotational speed (rpm), torque (Nm), tool wear (min), product type; target is machine failure (0/1).

## Models and evaluation
Logistic Regression, Random Forest and prior-probability baseline; stratified 80/20 holdout; five-fold cross-validation; precision, recall, F1, ROC-AUC, average precision and confusion matrix.

## Reproducible results
GitHub Actions downloads UCI data, trains/evaluates models, then commits metrics, plots, holdout predictions and selected model. Report only metrics in outputs/metrics.json after a successful workflow.

## Limitations
The benchmark is synthetic, and random splitting is not temporal or machine-level validation. Deployment requires plant-specific data, a time-based test and false-alarm cost analysis.
