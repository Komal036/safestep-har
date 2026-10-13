import numpy as np
import pytest

from src.data.windowing import sliding_windows


def test_shapes_and_overlap():
    sig = np.arange(200, dtype=np.float32).reshape(100, 2)
    X, y, s = sliding_windows(sig, np.zeros(100, int), np.ones(100, int), window=20, step=10)
    assert X.shape == (9, 20, 2)
    assert X[1, 0, 0] == sig[10, 0]  # 50% overlap


def test_drops_windows_across_subjects():
    sig = np.zeros((40, 1))
    subj = np.array([1] * 20 + [2] * 20)
    _, _, s = sliding_windows(sig, np.zeros(40, int), subj, window=20, step=10)
    assert list(s) == [1, 2]


def test_majority_label():
    labels = np.array([0] * 12 + [3] * 8)
    _, y, _ = sliding_windows(np.zeros((20, 1)), labels, np.ones(20, int), 20, 20)
    assert y[0] == 0


def test_bad_input():
    with pytest.raises(ValueError):
        sliding_windows(np.zeros(10), np.zeros(10), np.zeros(10), 5, 5)
