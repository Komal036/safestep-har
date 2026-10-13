"""Loader for the SisFall dataset (Sucerquia et al., 2017).

Everything under data_dir/SisFall_dataset/ is searched recursively, so both
the original layout and re-packaged mirrors work:
    SisFall_dataset/SA01/D01_SA01_R01.txt              (original)
    SisFall_dataset/Training/SA01/D01_SA01_R01.csv     (e.g. Kaggle re-uploads)

Two file formats are read automatically, at 200 Hz:
  * original: 9 raw ADC counts per line ending with ';'
      ADXL345 acc (3), ITG3200 gyro (3), MMA8451Q acc (3)
  * CSV with a header row, already converted to g and deg/s
      (columns ADXL345_Acc_X..ITG3200_Gyro_Z first, extra columns ignored)
The trial code (D01..D19 = activities of daily living, F01..F15 = falls) and
the subject (SA01..SA23 young, SE01..SE15 elderly) are taken from the file
name, or from its parent folders if the name does not contain them.

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

_CODE = re.compile(r"(?<![A-Za-z0-9])([DF])(\d{2})(?!\d)", re.IGNORECASE)
_SUBJ = re.compile(r"(?<![A-Za-z0-9])(S[AE])(\d{2})(?!\d)", re.IGNORECASE)
_REP = re.compile(r"(?<![A-Za-z0-9])R(\d{2})(?!\d)", re.IGNORECASE)


def parse_name(path, root):
    """(kind, code, group, num, rep) from the file name, falling back to parent folders."""
    parts = [path.stem] + [p for p in reversed(path.relative_to(root).parent.parts)]
    found = {}
    for part in parts:
        for key, rx in (("code", _CODE), ("subj", _SUBJ), ("rep", _REP)):
            if key not in found:
                m = rx.search(part)
                if m:
                    found[key] = m.groups()
    if "code" not in found or "subj" not in found:
        return None
    kind, code = found["code"]
    group, num = found["subj"]
    rep = found.get("rep", ("01",))[0]
    return kind.upper(), code, group.upper(), num, rep


def subject_id(group, num):
    """SA01 -> 1, SE01 -> 101, so young and elderly IDs never collide."""
    return int(num) + (100 if group == "SE" else 0)


def parse_file(path):
    """Return (n, 6) array: acc in g, gyro in deg/s."""
    with open(path) as f:
        first = f.readline()
    if any(c.isalpha() for c in first):          # CSV with header, physical units
        header = [h.strip().lower() for h in first.strip().split(",")]
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        want = ["adxl345_acc_x", "adxl345_acc_y", "adxl345_acc_z",
                "itg3200_gyro_x", "itg3200_gyro_y", "itg3200_gyro_z"]
        idx = [header.index(w) for w in want] if all(w in header for w in want) else list(range(6))
        out = data[:, idx]
        if len(out) == 0:
            raise ValueError(f"No samples in {path}")
        return out.astype(np.float64)

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
    trials, seen, skipped = [], set(), []
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".txt", ".csv"))
    for path in files:
        parsed = parse_name(path, root)
        if parsed is None:
            skipped.append(path)
            continue
        kind, code, group, num, rep = parsed
        key = (group, num, kind, code, rep)
        if key in seen:                      # same trial present twice in a mirror
            continue
        seen.add(key)
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
        examples = ", ".join(str(p.relative_to(root)) for p in skipped[:5]) or "none"
        raise FileNotFoundError(
            f"No SisFall trials found under {root}. Expected names containing a trial code "
            f"(D01/F01) and subject (SA01/SE01). Example files seen: {examples}")
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