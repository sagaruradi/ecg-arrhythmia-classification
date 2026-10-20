"""Held-out evaluation and genuine project result plots."""

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import RESULTS_DIR
from src.model import ECGConvNet


def load_saved_model(model_path):
    """Load a saved CPU model and its class names without training."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = ECGConvNet(len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["class_names"], checkpoint["input_length"]


def predict_segments(model, segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted class indices and softmax probabilities."""
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(np.asarray(segments, dtype=np.float32)).unsqueeze(1)
        probabilities = torch.softmax(model(inputs), dim=1)
        predictions = probabilities.argmax(dim=1)
    return predictions.numpy(), probabilities.numpy()


def save_class_distribution(label_arrays: list[np.ndarray]) -> None:
    counts = Counter(label for labels in label_arrays for label in labels.tolist())
    names = list(counts)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(names, [counts[name] for name in names], color="#2878b5")
    axis.set_title("Prepared heartbeat class distribution")
    axis.set_ylabel("Heartbeat segments")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
    plt.close(figure)


def save_example_waveform(segment: np.ndarray, record_name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 3.5))
    axis.plot(np.arange(len(segment)), segment, color="#146c94", linewidth=1)
    axis.set_title(f"Example normalized ECG heartbeat ({record_name})")
    axis.set_xlabel("Sample within heartbeat window")
    axis.set_ylabel("Normalized amplitude")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "example_ecg_waveform.png", dpi=150)
    plt.close(figure)


def evaluate_model(model, class_names: list[str], data: dict) -> dict:
    """Evaluate only on held-out records and save reports and the confusion matrix."""
    segments = data["test"]["segments"]
    true_names = data["test"]["labels"].tolist()
    if len(true_names) == 0:
        raise ValueError("The held-out test split contains no heartbeat segments.")
    unknown = sorted(set(true_names) - set(class_names))
    if unknown:
        raise ValueError("Test set contains classes absent from training: " + ", ".join(unknown))

    predicted_indices, probabilities = predict_segments(model, segments)
    predicted_names = [class_names[index] for index in predicted_indices]
    labels = list(range(len(class_names)))
    true_indices = [class_names.index(name) for name in true_names]
    accuracy = float(accuracy_score(true_names, predicted_names))
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_names,
        predicted_names,
        labels=class_names,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(true_indices, predicted_indices, labels=labels)
    report_text = classification_report(
        true_names,
        predicted_names,
        labels=class_names,
        target_names=class_names,
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy,
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "test_segments": len(true_names),
        "class_names": class_names,
        "test_records": data["record_splits"]["test"],
        "classification_report": report_text,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    np.savetxt(RESULTS_DIR / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=labels,
        yticks=labels,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Held-out recording confusion matrix",
    )
    axis.tick_params(axis="x", rotation=25)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall: {recall:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    print(report_text)
    return metrics


def save_training_loss(history: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    axis.plot(epochs, history["train_loss"], marker="o", label="Training loss")
    axis.plot(epochs, history["validation_loss"], marker="o", label="Validation loss")
    axis.set(title="Training progress", xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "training_loss.png", dpi=150)
    plt.close(figure)
