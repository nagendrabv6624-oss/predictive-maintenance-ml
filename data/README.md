# Predictive maintenance dataset

The training script downloads the AI4I 2020 CSV from the UCI Machine Learning Repository by default. For offline use, download `ai4i2020.csv` from the official [UCI dataset page](https://archive.ics.uci.edu/dataset/601/ai4i) and place it here.

**Dataset facts:** 10,000 records. UCI explicitly describes the dataset as synthetic simulated data designed to reflect industrial predictive-maintenance scenarios. It is not real factory telemetry.

Citation: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, DOI 10.24432/C5HS5C, CC BY 4.0.

The model excludes the ID columns and the component failure-mode target columns (TWF, HDF, PWF, OSF, RNF) to prevent leakage into the composite Machine failure label.
