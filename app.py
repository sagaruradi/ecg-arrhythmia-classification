"""Streamlit demonstration dashboard for the trained ECG model."""

import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import wfdb

from config import DATASET_DIR, PROCESSED_DIR, RESULTS_DIR, MODEL_DIR
from src.data_loader import discover_records
from src.evaluate import load_saved_model, predict_segments
from src.preprocessing import make_heartbeat_segments

REMOTE_DEMO_RECORDS = ("100", "114", "200", "208", "209", "213", "233")
REMOTE_PREVIEW_SECONDS = 15


st.set_page_config(page_title="ECG Arrhythmia Classification System", layout="wide")


@st.cache_data
def get_processed_data():
    path = PROCESSED_DIR / "heartbeat_data.npz"
    manifest_path = PROCESSED_DIR / "dataset_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        return None, None
    with np.load(path, allow_pickle=False) as cached:
        payload = {
            f"{split}_{key}": cached[f"{split}_{key}"]
            for split in ("train", "validation", "test")
            for key in ("segments", "labels", "records", "symbols")
        }
    return payload, json.loads(manifest_path.read_text(encoding="utf-8"))


@st.cache_resource
def get_model(model_path: str):
    return load_saved_model(model_path)


@st.cache_data(show_spinner="Loading ECG record...")
def get_record_data(record_name: str):
    """Read local data when available; fetch one public PhysioNet record otherwise."""
    local_record = DATASET_DIR / record_name
    if local_record.with_suffix(".hea").is_file():
        waveform, fields = wfdb.rdsamp(str(local_record), channels=[0])
        annotation = wfdb.rdann(str(local_record), "atr")
    else:
        header = wfdb.rdheader(record_name, pn_dir="mitdb")
        sample_count = int(header.fs * REMOTE_PREVIEW_SECONDS)
        waveform, fields = wfdb.rdsamp(
            record_name,
            pn_dir="mitdb",
            channels=[0],
            sampfrom=0,
            sampto=sample_count,
        )
        annotation = wfdb.rdann(
            record_name,
            "atr",
            pn_dir="mitdb",
            sampfrom=0,
            sampto=sample_count,
        )
    return (
        waveform,
        float(fields["fs"]),
        annotation.sample,
        list(annotation.symbol),
    )


@st.cache_data(show_spinner="Preparing annotated beats...")
def get_record_segments(record_name: str):
    """Filter and segment one selected local or PhysioNet record for prediction."""
    waveform, sampling_frequency, annotation_samples, annotation_symbols = get_record_data(
        record_name
    )
    return make_heartbeat_segments(
        waveform[:, 0],
        sampling_frequency,
        annotation_samples,
        annotation_symbols,
    )


st.title("ECG Arrhythmia Classification System")
st.caption("Educational project: recorded MIT-BIH ECG heartbeat classification with a 1D-CNN.")

home, viewer, prediction, results = st.tabs(
    ["Home", "ECG Signal Viewer", "Heartbeat Prediction", "Model Results"]
)

with home:
    st.subheader("Project overview")
    st.write(
        "This project filters recorded ECG signals, extracts heartbeat windows around "
        "expert annotations, and uses a small 1D convolutional neural network to classify "
        "supported heartbeat patterns."
    )
    st.markdown("**Workflow:** ECG record → bandpass filter → annotated beat windows → 1D-CNN → evaluation")
    st.markdown("**Technology:** Python, WFDB, NumPy, SciPy, PyTorch, scikit-learn, Matplotlib, Streamlit")
    st.info(
        "This is for educational purposes only. It is not clinically validated and must "
        "not be used for diagnosis or treatment."
    )

with viewer:
    st.subheader("Recorded ECG signal")
    try:
        try:
            records = discover_records(DATASET_DIR)
        except FileNotFoundError:
            records = list(REMOTE_DEMO_RECORDS)
        selected_record = st.selectbox("Choose an available MIT-BIH record", records)
        if selected_record:
            waveform, sampling_frequency, annotation_samples, _ = get_record_data(
                selected_record
            )
            seconds = st.slider("View duration (seconds)", 2, 15, 5)
            length = min(int(seconds * sampling_frequency), len(waveform))
            figure, axis = plt.subplots(figsize=(12, 4))
            axis.plot(np.arange(length) / sampling_frequency, waveform[:length, 0], linewidth=0.8)
            visible = annotation_samples < length
            axis.scatter(
                annotation_samples[visible] / sampling_frequency,
                waveform[annotation_samples[visible], 0],
                s=12,
                color="#d1495b",
                label="Reference beat annotation",
            )
            axis.set(xlabel="Time (seconds)", ylabel="ECG amplitude", title=f"Record {selected_record}")
            axis.legend()
            st.pyplot(figure)
            plt.close(figure)
            st.caption(
                f"Record {selected_record} | Sampling frequency: {sampling_frequency:g} Hz | "
                f"Reference annotations shown: {int(visible.sum())}"
            )
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Could not load a local ECG record: {error}")

with prediction:
    st.subheader("Classify an annotated heartbeat")
    data, manifest = get_processed_data()
    model_path = MODEL_DIR / "ecg_cnn.pt"
    if not model_path.is_file():
        st.warning("A trained model is not available. Run `python main.py --train` first.")
    else:
        if data is None:
            record_options = list(REMOTE_DEMO_RECORDS)
            selected_record = st.selectbox("Choose a MIT-BIH record", record_options)
            segments, labels, _, _ = get_record_segments(selected_record)
            if not len(labels):
                st.warning("No supported annotated heartbeat segments in this record.")
                st.stop()
            selected_index = st.selectbox(
                "Heartbeat index in this record",
                range(len(labels)),
                format_func=lambda index: f"Beat {int(index)} — reference: {labels[index]}",
            )
            segment = segments[selected_index]
            reference_label = labels[selected_index]
        else:
            source_records = data["train_records"]
            record_options = sorted(set(source_records.tolist()))
            selected_record = st.selectbox("Choose a training-record heartbeat", record_options)
            candidates = np.flatnonzero(source_records == selected_record)
            selected_index = st.selectbox(
                "Heartbeat index in this record",
                candidates.tolist(),
                format_func=lambda index: f"Beat {int(index)} — reference: {data['train_labels'][index]}",
            )
            segment = data["train_segments"][selected_index]
            reference_label = data["train_labels"][selected_index]
        figure, axis = plt.subplots(figsize=(10, 3))
        axis.plot(segment, color="#146c94")
        axis.set(title=f"Selected heartbeat from record {selected_record}", xlabel="Sample", ylabel="Normalized amplitude")
        st.pyplot(figure)
        plt.close(figure)
        model, class_names, _ = get_model(str(model_path))
        predicted_indices, probabilities = predict_segments(
            model, np.asarray(segment, dtype=np.float32)[None, :]
        )
        class_index = int(predicted_indices[0])
        predicted = class_names[class_index]
        confidence = float(probabilities[0, class_index])
        st.metric("Model prediction", predicted, f"{confidence:.1%} confidence")
        st.write(f"Reference annotation-derived class: **{reference_label}**")
        st.caption("Reference label and model prediction are shown separately.")

with results:
    st.subheader("Evaluation on held-out records")
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.caption(
            "Accuracy is the overall fraction correct. Precision measures how often "
            "predictions for a class are correct; recall measures how many actual beats "
            "of that class are found; F1 balances precision and recall."
        )
        columns = st.columns(4)
        for column, label, key in zip(
            columns,
            ("Accuracy", "Weighted precision", "Weighted recall", "Weighted F1"),
            ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted"),
        ):
            column.metric(label, f"{metrics[key]:.3f}")
        st.write(f"Held-out records: {', '.join(metrics['test_records'])}")
        st.write(f"Test heartbeat segments: {metrics['test_segments']}")
        report_path = RESULTS_DIR / "classification_report.txt"
        if report_path.is_file():
            st.text(report_path.read_text(encoding="utf-8"))
    else:
        st.info("No evaluation results yet. Run `python main.py --evaluate` after training.")

    for image_name, caption in (
        ("confusion_matrix.png", "Confusion matrix: actual classes versus predicted classes."),
        ("class_distribution.png", "Class counts in the prepared data."),
        ("training_loss.png", "Training and validation loss by epoch."),
        ("example_ecg_waveform.png", "Example normalized ECG heartbeat window."),
    ):
        image_path = RESULTS_DIR / image_name
        if image_path.is_file():
            st.image(str(image_path), caption=caption, use_container_width=True)

    st.caption(
        "Normal = N/L/R/e/j; supraventricular ectopic = A/a/J/S; "
        "ventricular ectopic = V/E; fusion = F. Unsupported symbols are excluded."
    )
