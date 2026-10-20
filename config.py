"""Project settings shared by the command-line workflow and dashboard."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# The dataset already supplied with this workspace is kept in place.
DATASET_DIR = ROOT_DIR / "Project" / "mit-bih-arrhythmia-database-1.0.0"

MAX_RECORDS = 10
RANDOM_SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEGMENT_BEFORE_SECONDS = 0.2
SEGMENT_AFTER_SECONDS = 0.4
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 40.0

# Curated record-level split: these recordings cover all four supported groups.
# Keep every record in exactly one split to prevent recording-level leakage.
RECORD_SPLITS = {
    "train": ("100", "106", "114", "200", "213", "233"),
    "validation": ("201", "205"),
    "test": ("208", "209"),
}

# MIT-BIH symbols mapped to the four AAMI heartbeat groups.
CLASS_SYMBOLS = {
    "Normal": ("N", "L", "R", "e", "j"),
    "Supraventricular ectopic": ("A", "a", "J", "S"),
    "Ventricular ectopic": ("V", "E"),
    "Fusion": ("F",),
}
CLASS_DESCRIPTIONS = {
    "Normal": "Normal beats and beats with similar conduction patterns.",
    "Supraventricular ectopic": "Early beats originating above the ventricles.",
    "Ventricular ectopic": "Early beats originating in the ventricles.",
    "Fusion": "Beats formed by a blend of normal and ventricular activation.",
}
