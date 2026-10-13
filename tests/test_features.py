import numpy as np

from src.data.features import STAT_NAMES, extract_features
from src.data.uci_har import CHANNELS


def test_feature_shape_and_names():
    X = np.random.default_rng(0).normal(size=(10, 128, 9)).astype(np.float32)
    F, names = extract_features(X, CHANNELS)
    assert F.shape == (10, len(names))
    assert len(names) == len(set(names))
    assert np.isfinite(F).all()


def test_dominant_frequency():
    fs, T = 50, 128
    t = np.arange(T) / fs
    freq = 50 / 128 * 10  # exactly on an FFT bin
    X = np.sin(2 * np.pi * freq * t)[None, :, None]
    F, names = extract_features(X, ["x"], fs=fs)
    assert abs(F[0, names.index("x_dom_freq")] - freq) < 1e-4


def test_constant_signal_is_safe():
    F, _ = extract_features(np.ones((2, 64, 3)), ["a", "b", "c"])
    assert np.isfinite(F).all()
