"""Small inference helpers used by the dashboard."""

import numpy as np

from src.evaluate import load_saved_model, predict_segments


def predict_one(model_path, segment: np.ndarray) -> tuple[str, float, list[float]]:
    model, class_names, input_length = load_saved_model(model_path)
    if len(segment) != input_length:
        raise ValueError(
            f"Expected a heartbeat with {input_length} samples, got {len(segment)}."
        )
    predictions, probabilities = predict_segments(
        model, np.asarray(segment, dtype=np.float32)[None, :]
    )
    scores = probabilities[0].tolist()
    index = int(predictions[0])
    return class_names[index], scores[index], scores
