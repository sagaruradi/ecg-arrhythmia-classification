# Real-Time ECG Arrhythmia Signal Classification Using 1D-CNN

## Abstract

This educational project uses the locally provided MIT-BIH Arrhythmia Database to classify ECG heartbeat windows. WFDB reads signals and expert beat annotations, SciPy applies a bandpass filter, and a small PyTorch 1D-CNN learns four annotation-derived heartbeat groups when they are represented in the selected records. Results are measured on separate recordings and displayed in a Streamlit dashboard. The current demonstration processes recorded data; it is not a clinical diagnostic tool or a live wearable-device system.

## Problem statement

ECG recordings contain repeated heartbeats and may include different beat patterns. Manually inspecting a long recording is time-consuming. This project demonstrates how annotated beats can be segmented and automatically classified with a simple machine-learning model.

## Objectives

- Read genuine local ECG records and their reference annotations.
- Filter signals and extract consistently sized heartbeat windows.
- Train a small 1D-CNN using recording-level training and validation splits.
- Evaluate the model on recordings excluded from training.
- Display records, predictions, metrics, and plots in a simple dashboard.

## Technology stack

Python, WFDB, NumPy, SciPy, PyTorch, scikit-learn, Matplotlib, and Streamlit. The model runs on CPU.

## Dataset source and labels

The workspace includes a local copy of the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). The project does not download data. WFDB reads the `.dat` signal, `.hea` header, and `.atr` expert annotations.

The annotation symbols use a documented AAMI-style grouping:

| Project class | MIT-BIH symbols | Simple meaning |
|---|---|---|
| Normal | N, L, R, e, j | Normal beats and similar conduction patterns |
| Supraventricular ectopic | A, a, J, S | Early beats originating above the ventricles |
| Ventricular ectopic | V, E | Early beats originating in the ventricles |
| Fusion | F | Beat with a blend of normal and ventricular activation |

Unknown symbols and windows too close to signal boundaries are excluded and counted in `data/processed/dataset_manifest.json`; they are never silently mapped to a class. This is heartbeat-pattern classification, not diagnosis of all arrhythmias or atrial fibrillation.

## System architecture

```text
MIT-BIH .dat + .hea + .atr
        ↓ WFDB
single ECG channel + annotation samples
        ↓ SciPy bandpass filter (0.5–40 Hz)
fixed window (-0.2 s to +0.4 s), z-score normalized
        ↓ PyTorch 1D-CNN
class scores → softmax probabilities
        ↓ scikit-learn
held-out metrics, classification report, confusion matrix
        ↓ Streamlit
record viewer, example prediction, results dashboard
```

## Methodology

1. Discover records that contain all three required files.
2. Select up to 10 configured records and assign whole records to curated training, validation, and test groups. These groups were selected to include the supported classes in each split.
3. Read the first signal channel and reference annotations with WFDB.
4. Apply a third-order Butterworth bandpass filter. Extract a 0.6-second window around each supported annotated beat (0.2 seconds before and 0.4 seconds after the annotated sample), excluding unsafe boundaries.
5. Normalize each window with its own mean and standard deviation. Save segments, labels, recording names, symbols, and a manifest for repeatable runs.
6. Train using class-weighted CrossEntropyLoss and Adam for five epochs. Keep the epoch with the lowest validation loss.
7. Evaluate once on test recordings that are not used for model fitting.

Records are split before their beat windows are formed, so windows from one recording cannot appear in both training and test sets. This small demonstration split is not a clinical validation protocol.

## Model architecture

Input window `(1, samples)` → Conv1D (1 to 16 filters, kernel 7) → ReLU → MaxPool → Conv1D (16 to 32 filters, kernel 5) → ReLU → Global Average Pool → Fully Connected class output. Cross-entropy consumes raw logits during training; softmax is applied only for prediction confidence.

## Project files

- `main.py`: prepare, train, and evaluate command-line entry point.
- `config.py`: dataset location and simple model/preprocessing settings.
- `app.py`: Streamlit dashboard; loads the saved model without retraining.
- `src/data_loader.py`: complete-record discovery and WFDB reads.
- `src/preprocessing.py`: filtering, annotation mapping, and heartbeat windows.
- `src/model.py`: the small 1D-CNN.
- `src/train.py`: class-weighted model training and checkpoint saving.
- `src/evaluate.py`: inference, metrics, and result figures.
- `src/predict.py`: one-segment dashboard inference.
- `data/processed/`: cached beat windows and the actual record/count manifest.
- `models/`: saved PyTorch checkpoint and class mapping.
- `results/`: generated metrics, report, and genuine plots.
- `VIVA_NOTES.md`: student-focused explanation and viva questions.

## Setup (Windows / Anaconda Prompt)

In PowerShell, activate the existing project environment and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks script activation, run the project interpreter directly with
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`. Anaconda users can
instead activate their chosen Conda environment first and use the same `python` commands.

The local dataset should be at `Project\mit-bih-arrhythmia-database-1.0.0` and contain matching `.dat`, `.hea`, and `.atr` files. The current workspace already contains a local dataset copy. The project does not include or redownload it.

## Prepare, train, and evaluate

```powershell
python main.py --prepare
python main.py --train
python main.py --evaluate
```

Run the complete sequence in one command:

```powershell
python main.py --all
```

To regenerate the cache after changing the record set or preprocessing settings:

```powershell
python main.py --prepare --force-prepare
```

The record split, cap, epochs, and other small settings are in `config.py`. The default cap is 10 complete records and training uses five epochs. To expand or change the experiment, add complete record names to `RECORD_SPLITS` and raise `MAX_RECORDS` as needed, keeping each record in only one group. The actual records and counts are written to `data/processed/dataset_manifest.json`.

## Launch the dashboard

```powershell
python -m streamlit run app.py
```

If you are not activating the project `.venv`, use its interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed in the terminal (normally `http://localhost:8501`). The dashboard does not train a model. Prepare data, train, and evaluate first. The signal viewer can show any discovered record with its reference annotation positions. The prediction tab uses a saved training-record heartbeat and displays its reference label separately from the model output.

## Deploy on Streamlit Community Cloud

1. Push this source folder to a GitHub repository.
2. Open [share.streamlit.io](https://share.streamlit.io/), sign in with GitHub, and choose **Create app**.
3. Select the repository, branch, and `app.py`, then deploy.

The repository includes the small trained CPU model and generated result plots/metrics, but excludes the MIT-BIH dataset and processed cache. When the app runs without local records, the signal viewer and prediction tab fetch a 15-second preview of the selected public record from PhysioNet instead of downloading the full recording. This requires outbound internet access in the hosting environment. The app does not retrain the model.

## Experimental results

The current run used records 100, 106, 114, 200, 213, and 233 for training; 201 and 205 for validation; and 208 and 209 for testing. It processed 15,105 training, 4,618 validation, and 5,955 held-out test segments. The five-epoch CPU run produced:

| Test measure | Actual result |
|---|---:|
| Accuracy | 0.8821 |
| Weighted precision | 0.8250 |
| Weighted recall | 0.8821 |
| Weighted F1-score | 0.8381 |
| Macro F1-score | 0.56 |

The class report is important: on the held-out records, recall was 0.21 for Fusion and 0.00 for Supraventricular ectopic, despite the overall accuracy. The high accuracy is influenced by the majority Normal class; the model is not reliably identifying all four groups. These figures are a genuine run of this small configured split, not a claim of clinical performance. Re-run the commands to produce results for your own data/configuration. The project writes metrics and the detailed report to `results/metrics.json` and `results/classification_report.txt`; the confusion matrix, class distribution, training-loss curve, and example beat waveform are saved as PNG files in `results/`.

## Limitations

- This is a small educational experiment using at most 10 records by default.
- Performance depends on which records and classes are available locally.
- Class counts are imbalanced; class-weighted loss helps but did not resolve the missed Supraventricular ectopic class in the current test. More representative training records and further validated model development are needed.
- Evaluation on a small set of held-out recordings is not clinical validation.
- The current system classifies pre-recorded, annotation-centred windows. It does not perform continuous live monitoring, detect every arrhythmia, or diagnose atrial fibrillation.

## Future scope

Use a carefully specified patient-independent benchmark split, include more records, report per-class confidence intervals, test a live data acquisition pipeline, and seek clinical expertise and formal validation before any medical use.

## References

1. Moody, G. B. and Mark, R. G., “The impact of the MIT-BIH Arrhythmia Database,” *IEEE Engineering in Medicine and Biology Magazine*, 2001.
2. PhysioNet, [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).
3. WFDB Python documentation, [wfdb.readthedocs.io](https://wfdb.readthedocs.io/).
4. PyTorch documentation, [pytorch.org/docs](https://pytorch.org/docs/).

**Disclaimer:** For educational purposes only. Not intended for clinical diagnosis or treatment.
