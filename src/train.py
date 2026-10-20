"""Training loop and saved-model creation."""

import json
import random
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_DIR,
    RANDOM_SEED,
)
from src.model import ECGConvNet


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def _make_loader(
    segments: np.ndarray, labels: np.ndarray, label_to_index: dict[str, int], shuffle: bool
) -> DataLoader:
    inputs = torch.from_numpy(segments).unsqueeze(1)
    targets = torch.tensor(
        [label_to_index[label] for label in labels], dtype=torch.long
    )
    return DataLoader(TensorDataset(inputs, targets), batch_size=BATCH_SIZE, shuffle=shuffle)


def train_model(data: dict, epochs: int = EPOCHS) -> dict:
    """Train on training records, select by validation loss, and save the best model."""
    set_seed()
    train_segments = data["train"]["segments"]
    train_labels = data["train"]["labels"]
    validation_segments = data["validation"]["segments"]
    validation_labels = data["validation"]["labels"]

    if len(train_labels) == 0 or len(validation_labels) == 0:
        raise ValueError("Training and validation splits must both contain heartbeat segments.")

    class_names = sorted(set(train_labels.tolist()))
    missing_validation = sorted(set(validation_labels.tolist()) - set(class_names))
    if missing_validation:
        raise ValueError(
            "Validation records contain classes absent from training: "
            + ", ".join(missing_validation)
        )
    label_to_index = {name: index for index, name in enumerate(class_names)}
    train_loader = _make_loader(train_segments, train_labels, label_to_index, True)
    validation_loader = _make_loader(
        validation_segments, validation_labels, label_to_index, False
    )

    counts = Counter(train_labels.tolist())
    class_weights = torch.tensor(
        [len(train_labels) / (len(class_names) * counts[name]) for name in class_names],
        dtype=torch.float32,
    )
    model = ECGConvNet(len(class_names))
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "validation_loss": []}
    best_validation_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_items = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * len(targets)
            train_items += len(targets)

        model.eval()
        validation_loss_total = 0.0
        validation_items = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                loss = loss_function(model(inputs), targets)
                validation_loss_total += float(loss.item()) * len(targets)
                validation_items += len(targets)

        train_loss = train_loss_total / train_items
        validation_loss = validation_loss_total / validation_items
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train loss: {train_loss:.4f}, validation loss: {validation_loss:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "ecg_cnn.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "input_length": int(train_segments.shape[1]),
            "architecture": "ECGConvNet",
        },
        model_path,
    )
    (MODEL_DIR / "class_mapping.json").write_text(
        json.dumps(
            {"class_to_index": label_to_index, "index_to_class": class_names},
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"model": model, "class_names": class_names, "history": history}
