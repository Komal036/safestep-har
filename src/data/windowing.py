"""Sliding-window segmentation for continuous sensor streams.

UCI HAR ships pre-windowed, but PAMAP2, SisFall and live streams do not,
so every later loader goes through this function.
"""
import numpy as np


def sliding_windows(signal, labels, subjects, window: int, step: int):
    """Segment a continuous recording into fixed-length windows.

    Args:
        signal: (n_samples, n_channels) array.
        labels: (n_samples,) int array, per-sample activity label.
        subjects: (n_samples,) int array, per-sample subject ID.
        window: window length in samples.
        step: hop size in samples (window // 2 gives 50% overlap).

    Returns:
        X (n_windows, window, n_channels), y (n_windows,), subj (n_windows,).
        Windows that cross a subject boundary are dropped; the label is the
        majority label inside the window.
    """
    signal = np.asarray(signal, dtype=np.float32)
    labels = np.asarray(labels)
    subjects = np.asarray(subjects)
    if signal.ndim != 2:
        raise ValueError("signal must be 2D (n_samples, n_channels)")
    if not (len(signal) == len(labels) == len(subjects)):
        raise ValueError("signal, labels and subjects must have equal length")
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive")

    X, y, s = [], [], []
    for start in range(0, len(signal) - window + 1, step):
        end = start + window
        win_subj = subjects[start:end]
        if win_subj[0] != win_subj[-1] or np.any(win_subj != win_subj[0]):
            continue
        vals, counts = np.unique(labels[start:end], return_counts=True)
        X.append(signal[start:end])
        y.append(vals[np.argmax(counts)])
        s.append(win_subj[0])

    if not X:
        n_ch = signal.shape[1]
        return (np.empty((0, window, n_ch), np.float32),
                np.empty(0, labels.dtype), np.empty(0, subjects.dtype))
    return np.stack(X), np.array(y), np.array(s)
