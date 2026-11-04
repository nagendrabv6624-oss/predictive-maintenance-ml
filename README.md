# Predictive Maintenance — Bearing Fault Classification

A presentation-ready bearing fault classification demo with a reproducible signal-simulation pipeline and Random Forest/SVM comparison.

## Dataset reference
The project is designed around the [JNU Bearing Dataset](https://github.com/ClarkGableWang/JNU-Bearing-Dataset), described as 50 kHz vertical acceleration recordings at 600, 800 and 1,000 RPM. It includes normal, inner-race, outer-race and rolling-element bearing conditions.

**Data provenance:** The binary CSV recordings could not be retrieved into the training environment during this build. The included trained demo model and metrics use generated illustrative vibration signals, NOT measured JNU recordings. Do not describe these metrics as JNU benchmark performance or industrial validation.

## Demo workflow
1. Generate illustrative vibration waveforms for four bearing states.
2. Extract time-domain statistics and FFT band-energy features.
3. Train Random Forest and RBF-SVM classifiers.
4. Compare models on a stratified hold-out split.
5. Save model, metrics and predictions.

## Reproduce
Install requirements, then run:
    
    python src/train_demo.py

Outputs are written to outputs/. Results are simulation-only and demonstrate the pipeline, not measured performance on JNU data.

## Next step for real benchmark validation
Retrieve the CSV recordings from the JNU repository, replace the signal generator with a loader for measured files, split by recording/file to avoid leakage, then retrain and report separate JNU test metrics.

## Limitations
- Synthetic signals are not measured bearing vibration.
- Frequencies and amplitudes are illustrative, not calibrated fault characteristic frequencies.
- No remaining-useful-life estimation is performed.
- This is a presentation/demo classifier, not a safety-critical maintenance system.
