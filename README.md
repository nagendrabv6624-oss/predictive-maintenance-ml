# Predictive Maintenance Using Machine Learning

A classification workflow that estimates whether an industrial machine is approaching failure from sensor and operating-condition features.

> **Data transparency:** No real factory or sensor records are included. The default script generates synthetic demonstration data so the workflow can be run immediately. Synthetic results are not evidence of production performance.

## Features
- Data validation and exploratory summary
- Train/test split and feature standardization
- Random Forest classification with class weighting
- Accuracy, precision, recall, F1-score and confusion matrix
- Feature-importance visualization
- Saved model and evaluation artifacts

## Expected CSV
Put a CSV at `data/maintenance.csv`. It must contain numeric columns:

`air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`, `failure`.

The target `failure` must be binary: 0 = no failure, 1 = failure. Use a documented dataset and verify its label definitions before training.

## Run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```
Use your own data: `python src/train.py --data path/to/maintenance.csv`.

## Outputs
`outputs/maintenance_model.joblib`, `outputs/metrics.json`, `outputs/confusion_matrix.png`, `outputs/feature_importance.png`.

## Interpretation
In maintenance settings, false negatives can be costly. Review recall, precision, class balance, threshold choice and the cost of each error—not accuracy alone. Avoid random splitting if timestamps or machines create leakage; consider time- or machine-based validation.

## Skills
Python, Pandas, scikit-learn, classification, feature engineering, model evaluation and data visualization.
