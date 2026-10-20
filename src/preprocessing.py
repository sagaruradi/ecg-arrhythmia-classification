"""Filtering and annotation-centred ECG heartbeat preparation."""

from collections import Counter

import numpy as np
from scipy.signal import butter, sosfiltfilt

from config import (
    CLASS_SYMBOLS,
    FILTER_HIGH_HZ,
    FILTER_LOW_HZ,
    SEGMENT_AFTER_SECONDS,
    SEGMENT_BEFORE_SECONDS,
)


def annotation_class(symbol: str) -> str | None:
    """Map known MIT-BIH beat symbols; return None for unsupported symbols."""
    for class_name, symbols in CLASS_SYMBOLS.items():
        if symbol in symbols:
            return class_name
    return None


def filter_signal(signal: np.ndarray, sampling_frequency: float) -> np.ndarray:
    """Remove baseline drift and high-frequency noise with a Butterworth filter."""
    nyquist = sampling_frequency / 2.0
    high_hz = min(FILTER_HIGH_HZ, nyquist * 0.95)
    if FILTER_LOW_HZ >= high_hz:
        raise ValueError(
            f"Sampling frequency {sampling_frequency} Hz is too low for the "
            "configured ECG bandpass filter."
        )
    sections = butter(
        3,
        (FILTER_LOW_HZ, high_hz),
        btype="bandpass",
        fs=sampling_frequency,
        output="sos",
    )
    return sosfiltfilt(sections, np.asarray(signal, dtype=np.float64))


def make_heartbeat_segments(
    signal: np.ndarray,
    sampling_frequency: float,
    annotation_samples: np.ndarray,
    annotation_symbols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], Counter]:
    """Filter and normalize fixed-length windows around supported annotated beats."""
    filtered = filter_signal(signal, sampling_frequency)
    before = round(SEGMENT_BEFORE_SECONDS * sampling_frequency)
    after = round(SEGMENT_AFTER_SECONDS * sampling_frequency)
    expected_length = before + after
    segments: list[np.ndarray] = []
    labels: list[str] = []
    symbols: list[str] = []
    excluded: Counter = Counter()

    for sample, symbol in zip(annotation_samples, annotation_symbols):
        class_name = annotation_class(symbol)
        if class_name is None:
            excluded[symbol] += 1
            continue

        start = int(sample) - before
        stop = int(sample) + after
        if start < 0 or stop > len(filtered):
            excluded["boundary"] += 1
            continue

        segment = filtered[start:stop]
        if len(segment) != expected_length:
            excluded["boundary"] += 1
            continue

        # Per-beat z-score normalization keeps amplitude scale consistent.
        standard_deviation = float(np.std(segment))
        if standard_deviation < 1e-8:
            excluded["flat"] += 1
            continue
        normalized = (segment - float(np.mean(segment))) / standard_deviation
        segments.append(normalized.astype(np.float32))
        labels.append(class_name)
        symbols.append(symbol)

    if not segments:
        return (
            np.empty((0, expected_length), dtype=np.float32),
            np.empty((0,), dtype=str),
            [],
            excluded,
        )
    return np.stack(segments), np.asarray(labels), symbols, excluded
