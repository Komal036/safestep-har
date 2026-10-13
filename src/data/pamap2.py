"""Loader for PAMAP2 (Reiss and Stricker, 2012), protocol recordings.

Each Protocol/subject10X.dat row has 54 space-separated columns at 100 Hz:
  0 timestamp | 1 activity ID | 2 heart rate (bpm, ~9 Hz, NaN between)
  3-19 hand (wrist) IMU | 20-36 chest IMU | 37-53 ankle IMU
Per IMU: temp, acc +-16 g (3, m/s^2), acc +-6 g (3), gyro (3, rad/s), mag (3), orientation (4).

We use the wrist IMU (the realistic wearable placement), optionally heart
rate, convert acceleration to g, interpolate dropped samples, downsample to
50 Hz and window each continuous activity run separately, so no window
straddles two activities or the transient (ID 0) gaps between them.
"""
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from src.data.windowing import sliding_windows

RAW_HZ = 100
G = 9.80665
ACTIVITY_IDS = {1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
                6: "cycling", 7: "nordic_walking", 12: "ascending_stairs",
                13: "descending_stairs", 16: "vacuum_cleaning", 17: "ironing",
                24: "rope_jumping"}
WRIST_COLS = {"wrist_acc_x": 4, "wrist_acc_y": 5, "wrist_acc_z": 6,
              "wrist_gyro_x": 10, "wrist_gyro_y": 11, "wrist_gyro_z": 12}


def _interp_nan(x):
    """Linear interpolation over NaNs per column (edges held constant)."""
    x = x.copy()
    idx = np.arange(len(x))
    for c in range(x.shape[1]):
        bad = np.isnan(x[:, c])
        if bad.all():
            x[:, c] = 0.0
        elif bad.any():
            x[bad, c] = np.interp(idx[bad], idx[~bad], x[~bad, c])
    return x


def _runs(labels):
    """(start, end) of each run of identical consecutive labels."""
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    return list(zip(starts, ends))


def load_pamap2(data_dir="data/raw", target_hz=50, window=128, step=64,
                include_hr=False, min_run_s=5.0):
    root = Path(data_dir) / "PAMAP2_Dataset" / "Protocol"
    files = sorted(root.glob("subject1*.dat"))
    if not files:
        raise FileNotFoundError(f"No PAMAP2 files in {root}. Run: python data/download.py --dataset pamap2")
    if RAW_HZ % target_hz:
        raise ValueError("target_hz must divide 100")
    q = RAW_HZ // target_hz
    ids = sorted(ACTIVITY_IDS)
    id_to_idx = {a: i for i, a in enumerate(ids)}
    channels = list(WRIST_COLS) + (["heart_rate"] if include_hr else [])
    cols = list(WRIST_COLS.values())

    Xs, ys, ss = [], [], []
    for f in files:
        subj = int(f.stem.replace("subject", ""))
        raw = np.loadtxt(f)
        act = raw[:, 1].astype(int)
        sig = raw[:, cols].copy()
        sig[:, :3] /= G                                  # m/s^2 -> g
        if include_hr:
            sig = np.concatenate([sig, raw[:, [2]]], axis=1)
        for a, b in _runs(act):
            if act[a] not in id_to_idx or (b - a) < min_run_s * RAW_HZ:
                continue                                 # transient (0) or too short
            seg = _interp_nan(sig[a:b])
            if q > 1:
                seg = resample_poly(seg, 1, q, axis=0)
            n = len(seg)
            X, y, s = sliding_windows(seg, np.full(n, id_to_idx[act[a]]), np.full(n, subj), window, step)
            if len(X):
                Xs.append(X), ys.append(y), ss.append(s)
    if not Xs:
        raise ValueError("PAMAP2 produced no windows")
    return {"X": np.concatenate(Xs).astype(np.float32), "y": np.concatenate(ys),
            "subjects": np.concatenate(ss), "channels": channels,
            "classes": [ACTIVITY_IDS[a] for a in ids], "dataset": "pamap2"}
