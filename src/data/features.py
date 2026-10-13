"""Hand-crafted time and frequency domain features for HAR windows.

All functions are vectorized over windows: input (n, T, C), output (n, F).
"""
import warnings

import numpy as np
from scipy.stats import kurtosis, skew

STAT_NAMES = ["mean", "std", "min", "max", "median", "iqr", "rms",
              "skew", "kurtosis", "energy", "zcr", "dom_freq", "spec_entropy"]


def _per_channel(X, fs):
    n, T, C = X.shape
    centered = X - X.mean(axis=1, keepdims=True)
    q75, q25 = np.percentile(X, [75, 25], axis=1)
    sign = np.signbit(centered)
    zcr = (sign[:, 1:] != sign[:, :-1]).mean(axis=1)

    spec = np.abs(np.fft.rfft(centered, axis=1)) ** 2  # (n, F, C)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    spec_nodc = spec[:, 1:]
    dom_freq = freqs[1:][np.argmax(spec_nodc, axis=1)]
    p = spec_nodc / (spec_nodc.sum(axis=1, keepdims=True) + 1e-12)
    spec_entropy = -(p * np.log2(p + 1e-12)).sum(axis=1) / np.log2(p.shape[1])

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sk = np.nan_to_num(skew(X, axis=1))
        ku = np.nan_to_num(kurtosis(X, axis=1))

    feats = [
        X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1),
        np.median(X, axis=1), q75 - q25, np.sqrt((X ** 2).mean(axis=1)),
        sk, ku, (X ** 2).sum(axis=1) / T, zcr, dom_freq, spec_entropy,
    ]
    return np.stack(feats, axis=-1).reshape(n, -1)  # (n, C * len(STAT_NAMES))


def _magnitude(X, idx):
    return np.linalg.norm(X[:, :, idx], axis=-1)


def extract_features(X, channels, fs=50):
    """Compute the feature matrix and matching feature names."""
    X = np.asarray(X, dtype=np.float64)
    parts = [_per_channel(X, fs)]
    names = [f"{ch}_{s}" for ch in channels for s in STAT_NAMES]

    # Magnitude signals are orientation independent, which helps
    # when the device is worn loosely (common for wrist wearables).
    groups = {}
    for i, c in enumerate(channels):
        if c[-2:] in ("_x", "_y", "_z"):
            groups.setdefault(c[:-2], {})[c[-1]] = i
    groups = {k: [v["x"], v["y"], v["z"]] for k, v in groups.items() if len(v) == 3}
    for prefix, idx in groups.items():
        mag = _magnitude(X, idx)[..., None]
        parts.append(_per_channel(mag, fs))
        names += [f"{prefix}_mag_{s}" for s in STAT_NAMES]
        # Signal magnitude area
        parts.append(np.abs(X[:, :, idx]).sum(axis=-1).mean(axis=1, keepdims=True))
        names.append(f"{prefix}_sma")
        # Pairwise axis correlations
        a = X[:, :, idx]
        a = a - a.mean(axis=1, keepdims=True)
        sd = a.std(axis=1) + 1e-12
        for i, j in ((0, 1), (0, 2), (1, 2)):
            corr = (a[:, :, i] * a[:, :, j]).mean(axis=1) / (sd[:, i] * sd[:, j])
            parts.append(corr[:, None])
            names.append(f"{prefix}_corr_{'xyz'[i]}{'xyz'[j]}")

    F = np.concatenate(parts, axis=1).astype(np.float32)
    return np.nan_to_num(F), names
