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

## Measured JNU benchmark pipeline

A separate real-data pipeline is available at `src/train_jnu.py`. It downloads the 12 CSV recordings from the original JNU repository, validates numeric samples, segments each recording into 2,048-sample windows (1,024-sample hop), and evaluates Random Forest and RBF-SVM using leave-one-rotational-speed-out tests. Each fold holds out all recordings at one speed (600, 800, or 1,000 RPM), avoiding random window-level leakage across the same recording.

Run in a Python environment with internet access:

    pip install numpy scikit-learn joblib
    python src/train_jnu.py

Outputs are saved under `outputs/jnu/`. **No measured JNU metrics are claimed until this script has actually run and its outputs have been reviewed.** The earlier `outputs/demo_metrics.json` remains simulation-only and must not be confused with measured-data results.

Dataset source: [JNU Bearing Dataset](https://github.com/ClarkGableWang/JNU-Bearing-Dataset). The source describes 50 kHz vertical acceleration measurements at 600/800/1,000 rpm and the four conditions used here. A [browser-based JNU toolkit/Colab workflow](https://github.com/kavianpour/jnu-bearing-toolkit) is also available.
