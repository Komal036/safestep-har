"""Loader for the SisFall dataset (Sucerquia et al., 2017).

Layout expected under data_dir:
    SisFall_dataset/SA01/D01_SA01_R01.txt   (young adults SA01..SA23)
    SisFall_dataset/SE01/D01_SE01_R01.txt   (elderly SE01..SE15)

Each line holds 9 raw ADC counts at 200 Hz, ending with ';':
    ADXL345 acc (3), ITG3200 gyro (3), MMA8451Q acc (3)
Codes D01..D19 are activities of daily living (ADL), F01..F15 are falls.

We keep the ADXL345 accelerometer (g) and the gyroscope (deg/s),
downsample to 50 Hz to match the HAR models, and expose two views:
  * load_sisfall_trials: whole recordings, for stream-level evaluation
  * trials_to_windows:  fixed windows, for training a window classifier
"""
import re
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

RAW_HZ = 200
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CLASSES = ["adl", "fall"]

# ADC count -> physical unit, from the dataset readme
ACC_SCALE = (2 * 16) / (2 ** 13)      # ADXL345, +-16 g, 13 bit
GYRO_SCALE = (2 * 2000) / (2 ** 16)   # ITG3200, +-2000 deg/s, 16 bit

_NAME = re.compile(r"^([DF])(\d{2})_(S[AE])(\d{2})_R(\d{2})\.txt$")


def subject_id(group, num):
    """SA01 -> 1, SE01 -> 101, so young and elderly IDs never collide."""
    return int(num) + (100 if group == "SE" else 0)


def parse_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            parts = [p for p in line.replace(";", ",").split(",") if p.strip()]
            if len(parts) >= 6:
                rows.append([float(p) for p in parts[:6]])
    raw = np.asarray(rows, dtype=np.float64)
    if raw.ndim != 2 or len(raw) == 0:
        raise ValueError(f"No samples in {path}")
    return np.concatenate([raw[:, :3] * ACC_SCALE, raw[:, 3:6] * GYRO_SCALE], axis=1)


def load_sisfall_trials(data_dir="data/raw", target_hz=50):
    root = Path(data_dir) / "SisFall_dataset"
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. See `python data/download.py --dataset sisfall` for instructions."
        )
    if RAW_HZ % target_hz:
        raise ValueError("target_hz must divide 200")
    q = RAW_HZ // target_hz
    trials = []
    for path in sorted(root.glob("S[AE][0-9][0-9]/*.txt")):
        m = _NAME.match(path.name)
        if not m:
            continue
        kind, code, group, num, rep = m.groups()
        sig = parse_file(path)
        if q > 1:
            sig = resample_poly(sig, 1, q, axis=0)
        trials.append({
            "signal": sig.astype(np.float32),
            "subject": subject_id(group, num),
            "elderly": group == "SE",
            "code": f"{kind}{code}",
            "is_fall": kind == "F",
            "fs": target_hz,
            "file": path.name,
        })
    if not trials:
        raise FileNotFoundError(f"No SisFall trial files found under {root}")
    return trials


def magnitude(sig):
    return np.linalg.norm(sig[:, :3], axis=1)


def impact_index(sig):
    """Sample index of the acceleration-magnitude peak (the impact)."""
    return int(np.argmax(magnitude(sig)))


def _fall_window(sig, peak, window, peak_pos):
    start = peak - peak_pos
    start = min(max(start, 0), max(len(sig) - window, 0))
    win = sig[start:start + window]
    if len(win) < window:  # very short trial: edge-pad
        win = np.pad(win, ((0, window - len(win)), (0, 0)), mode="edge")
    return win


def trials_to_windows(trials, window=128, adl_step=64, peak_positions=(48,)):
    """Turn trials into labelled windows.

    Falls: one window per entry in peak_positions, with the impact at that
    sample index inside the window (several positions = shift augmentation).
    ADL:   sliding windows over the whole recording.
    """
    X, y, subj, elderly = [], [], [], []
    for t in trials:
        sig = t["signal"]
        if t["is_fall"]:
            peak = impact_index(sig)
            for pos in peak_positions:
                X.append(_fall_window(sig, peak, window, pos))
                y.append(1)
                subj.append(t["subject"])
                elderly.append(t["elderly"])
        else:
            if len(sig) < window:
                continue
            for s in range(0, len(sig) - window + 1, adl_step):
                X.append(sig[s:s + window])
                y.append(0)
                subj.append(t["subject"])
                elderly.append(t["elderly"])
    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=int),
        "subjects": np.asarray(subj, dtype=int),
        "elderly": np.asarray(elderly, dtype=bool),
        "channels": list(CHANNELS),
        "classes": list(CLASSES),
    }


def load_sisfall_windows(data_dir="data/raw", window=128, adl_step=64, peak_positions=(48,)):
    return trials_to_windows(load_sisfall_trials(data_dir), window, adl_step, peak_positions)
