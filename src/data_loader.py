"""Discovery and reading helpers for local MIT-BIH records."""

from pathlib import Path

import wfdb

from config import DATASET_DIR


def discover_records(dataset_dir: Path = DATASET_DIR) -> list[str]:
    """Return record names that have signal, header, and ATR annotation files."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"MIT-BIH dataset directory was not found: {dataset_dir}"
        )

    records = []
    for header in sorted(dataset_dir.glob("*.hea")):
        record_name = header.stem
        required = [
            dataset_dir / f"{record_name}.dat",
            dataset_dir / f"{record_name}.atr",
        ]
        if all(path.is_file() for path in required):
            records.append(record_name)

    if not records:
        raise FileNotFoundError(
            "No complete MIT-BIH records found. Each record needs .hea, .dat, "
            f"and .atr files in {dataset_dir}."
        )
    return records


def read_record(record_name: str, dataset_dir: Path = DATASET_DIR):
    """Read a record and its expert reference annotations with WFDB."""
    base = dataset_dir / record_name
    missing = [
        f"{base}{extension}"
        for extension in (".hea", ".dat", ".atr")
        if not Path(f"{base}{extension}").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required record files: " + ", ".join(missing))

    record = wfdb.rdrecord(str(base), channels=[0], physical=True)
    annotation = wfdb.rdann(str(base), "atr")
    return record, annotation
