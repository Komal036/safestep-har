"""Loader for the UCI HAR dataset (Anguita et al., 2013).

Returns raw inertial windows (128 samples at 50 Hz, 9 channels) with
train and test merged, so evaluation can use our own subject-wise CV
instead of the fixed split.
"""
from pathlib import Path

import numpy as np

CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

ACTIVITIES = [
    "walking", "walking_upstairs", "walking_downstairs",
    "sitting", "standing", "lying",
]


def _load_split(root: Path, split: str):
    sig_dir = root / split / "Inertial Signals"
    X = np.stack(
        [np.loadtxt(sig_dir / f"{ch}_{split}.txt", dtype=np.float32) for ch in CHANNELS],
        axis=-1,
    )  # (n, 128, 9)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=int) - 1  # 0-indexed
    subj = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)
    return X, y, subj


def load_uci_har(data_dir="data/raw"):
    root = Path(data_dir) / "UCI HAR Dataset"
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Run: python data/download.py --dataset uci_har"
        )
    parts = [_load_split(root, s) for s in ("train", "test")]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    subjects = np.concatenate([p[2] for p in parts])
    return {"X": X, "y": y, "subjects": subjects,
            "channels": list(CHANNELS), "classes": list(ACTIVITIES), "dataset": "uci_har"}
