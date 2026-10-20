"""Command-line entry point for preparing data, training, and evaluation."""

import argparse
import json

import numpy as np

from config import (
    CLASS_SYMBOLS,
    DATASET_DIR,
    EPOCHS,
    MAX_RECORDS,
    PROCESSED_DIR,
    RECORD_SPLITS,
    RESULTS_DIR,
)
from src.data_loader import discover_records, read_record
from src.evaluate import evaluate_model, load_saved_model, save_class_distribution
from src.preprocessing import make_heartbeat_segments
from src.train import set_seed, train_model
from src.evaluate import save_example_waveform, save_training_loss


def split_records(records: list[str]) -> dict[str, list[str]]:
    """Use curated whole-record groups, preserving class coverage between splits."""
    assigned = {
        name: list(RECORD_SPLITS[name])
        for name in ("train", "validation", "test")
    }
    flattened = [record for group in assigned.values() for record in group]
    duplicates = sorted({record for record in flattened if flattened.count(record) > 1})
    if duplicates:
        raise ValueError("Records must appear in only one split: " + ", ".join(duplicates))
    if len(flattened) > MAX_RECORDS:
        raise ValueError(
            f"Configured split contains {len(flattened)} records, above MAX_RECORDS={MAX_RECORDS}."
        )
    missing = sorted(set(flattened) - set(records))
    if missing:
        missing_files = [
            str(DATASET_DIR / f"{record}{extension}")
            for record in missing
            for extension in (".hea", ".dat", ".atr")
            if not (DATASET_DIR / f"{record}{extension}").is_file()
        ]
        raise FileNotFoundError(
            "The configured record split is missing required files: "
            + ", ".join(missing_files)
        )
    if any(not assigned[name] for name in assigned):
        raise ValueError("Training, validation, and test record groups must all be non-empty.")
    return assigned


def prepare_data(force: bool = False) -> dict:
    """Read the local records, split by recording, and cache annotated beat windows."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PROCESSED_DIR / "heartbeat_data.npz"
    manifest_path = PROCESSED_DIR / "dataset_manifest.json"
    if cache_path.is_file() and manifest_path.is_file() and not force:
        print(f"Using processed data cache: {cache_path}")
        return load_cache(cache_path, manifest_path)

    available = discover_records(DATASET_DIR)
    record_splits = split_records(available)
    chosen = [record for group in record_splits.values() for record in group]
    if len(chosen) < 3:
        raise ValueError("At least three complete records are needed for leakage-safe splits.")
    all_records = set(chosen)
    assert all_records == set(record_splits["train"] + record_splits["validation"] + record_splits["test"])

    split_data = {
        name: {"segments": [], "labels": [], "records": [], "symbols": []}
        for name in record_splits
    }
    record_details = {}
    example = None
    for split_name, split_records_for_name in record_splits.items():
        for record_name in split_records_for_name:
            print(f"Reading {record_name} ({split_name})...")
            record, annotation = read_record(record_name, DATASET_DIR)
            signal = record.p_signal[:, 0]
            segments, labels, symbols, excluded = make_heartbeat_segments(
                signal,
                float(record.fs),
                annotation.sample,
                annotation.symbol,
            )
            target = split_data[split_name]
            target["segments"].append(segments)
            target["labels"].append(labels)
            target["records"].extend([record_name] * len(labels))
            target["symbols"].extend(symbols)
            if example is None and len(segments):
                example = (segments[0], record_name)
            record_details[record_name] = {
                "sampling_frequency_hz": float(record.fs),
                "signal_samples": int(len(signal)),
                "processed_segments": int(len(segments)),
                "class_counts": {
                    label: int(count)
                    for label, count in zip(*np.unique(labels, return_counts=True))
                },
                "excluded_symbols_and_reasons": dict(excluded),
            }

    for target in split_data.values():
        target["segments"] = (
            np.concatenate(target["segments"])
            if target["segments"]
            else np.empty((0, 0), dtype=np.float32)
        )
        target["labels"] = (
            np.concatenate(target["labels"])
            if target["labels"]
            else np.empty((0,), dtype=str)
        )
        target["records"] = np.asarray(target["records"], dtype=str)
        target["symbols"] = np.asarray(target["symbols"], dtype=str)
    if any(len(split_data[name]["labels"]) == 0 for name in split_data):
        empty_splits = [name for name, content in split_data.items() if len(content["labels"]) == 0]
        raise ValueError(
            "No supported heartbeat segments were found in: "
            + ", ".join(empty_splits)
            + ". Choose different records in config.py."
        )

    arrays = {
        f"{name}_{key}": value
        for name, content in split_data.items()
        for key, value in content.items()
    }
    np.savez_compressed(cache_path, **arrays)
    manifest = {
        "dataset_directory": str(DATASET_DIR),
        "available_complete_records": available,
        "processed_records": chosen,
        "record_splits": record_splits,
        "record_details": record_details,
        "segment_samples": int(split_data["train"]["segments"].shape[1]),
        "segment_window_seconds": [0.2, 0.4],
        "annotation_mapping": {
            name: list(symbols) for name, symbols in CLASS_SYMBOLS.items()
        },
        "cache_file": str(cache_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Cached processed beats: {cache_path}")
    print("Record split (kept separate to avoid recording-level leakage):")
    for name, names in record_splits.items():
        count = sum(record_details[item]["processed_segments"] for item in names)
        print(f"  {name}: {', '.join(names)} — {count} segments")
    save_class_distribution([content["labels"] for content in split_data.values()])
    if example is not None:
        save_example_waveform(*example)
    return {**split_data, "record_splits": record_splits, "manifest": manifest}


def load_cache(cache_path, manifest_path) -> dict:
    with np.load(cache_path, allow_pickle=False) as cached:
        result = {
            name: {
                "segments": cached[f"{name}_segments"],
                "labels": cached[f"{name}_labels"],
                "records": cached[f"{name}_records"],
                "symbols": cached[f"{name}_symbols"],
            }
            for name in ("train", "validation", "test")
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result["record_splits"] = manifest["record_splits"]
    result["manifest"] = manifest
    return result


def train(data: dict) -> dict:
    result = train_model(data, epochs=EPOCHS)
    save_training_loss(result["history"])
    return result


def evaluate(data: dict) -> dict:
    from config import MODEL_DIR

    model, class_names, _ = load_saved_model(MODEL_DIR / "ecg_cnn.pt")
    return evaluate_model(model, class_names, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="MIT-BIH ECG heartbeat classification")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="prepare and cache local ECG data")
    action.add_argument("--train", action="store_true", help="train the 1D-CNN")
    action.add_argument("--evaluate", action="store_true", help="evaluate on held-out records")
    action.add_argument("--all", action="store_true", help="prepare, train, and evaluate")
    parser.add_argument("--force-prepare", action="store_true", help="rebuild the processed-data cache")
    args = parser.parse_args()
    set_seed()
    data = prepare_data(force=args.force_prepare) if args.prepare or args.all else prepare_data()
    if args.train or args.all:
        train(data)
    if args.evaluate or args.all:
        metrics = evaluate(data)
        print(f"Saved metrics to {RESULTS_DIR / 'metrics.json'}")
        print(f"Evaluated {metrics['test_segments']} held-out beats.")


if __name__ == "__main__":
    main()
