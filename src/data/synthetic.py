"""Synthetic recordings written in the *exact* on-disk formats of
UCI HAR, SisFall and PAMAP2.

Purpose: let every script run end to end with no download
(`python run_all.py --smoke`) and power the test suite. The signals are
simple physics-flavoured toys, so any score on them means nothing.
"""
from pathlib import Path

import numpy as np

UCI_FS = 50
UCI_ACTIVITIES = ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "lying"]


# ---------------------------------------------------------------- shared toys
def _gravity(activity):
    """Unit gravity direction in a waist-worn device frame (x = up when upright)."""
    return {"sitting": [0.85, 0.0, 0.53], "lying": [0.1, 0.2, 0.97]}.get(activity, [1.0, 0.0, 0.0])


def _activity_signal(rng, activity, n, fs, subj_scale=1.0):
    """Return (total_acc[g], gyro[rad/s]) for one activity, n samples."""
    t = np.arange(n) / fs
    g = np.array(_gravity(activity), dtype=float)
    acc = np.tile(g, (n, 1))
    gyro = np.zeros((n, 3))
    if activity.startswith("walking") or activity in ("running", "nordic_walking",
                                                      "ascending_stairs", "descending_stairs",
                                                      "rope_jumping", "cycling"):
        f = {"walking_upstairs": 1.6, "ascending_stairs": 1.6, "walking_downstairs": 2.1,
             "descending_stairs": 2.1, "running": 2.8, "rope_jumping": 2.4, "cycling": 1.3}.get(activity, 1.9)
        amp = {"walking_upstairs": 0.25, "walking_downstairs": 0.45, "running": 0.9,
               "rope_jumping": 1.2, "cycling": 0.15}.get(activity, 0.3) * subj_scale
        ph = rng.uniform(0, 2 * np.pi)
        acc[:, 0] += amp * np.sin(2 * np.pi * f * t + ph)
        acc[:, 1] += 0.4 * amp * np.sin(2 * np.pi * f / 2 * t + ph)
        acc[:, 2] += 0.3 * amp * np.sin(2 * np.pi * f * t + ph + 1.0)
        gyro += 0.8 * amp * np.stack([np.sin(2 * np.pi * f / 2 * t + ph + k) for k in range(3)], 1)
    elif activity in ("vacuum_cleaning", "ironing"):
        f = 0.8 if activity == "vacuum_cleaning" else 1.2
        acc[:, 1] += 0.15 * np.sin(2 * np.pi * f * t)
        gyro[:, 2] += 0.4 * np.sin(2 * np.pi * f * t)
    noise = {"lying": 0.004, "sitting": 0.006, "standing": 0.01}.get(activity, 0.03)
    acc += rng.normal(0, noise, acc.shape)
    gyro += rng.normal(0, noise * 3, gyro.shape)
    return acc, gyro


def _lowpass(x, fs, cutoff=0.3):
    from scipy.signal import butter, filtfilt
    b, a = butter(3, cutoff / (fs / 2))
    return filtfilt(b, a, x, axis=0)


# ------------------------------------------------------------------- UCI HAR
UCI_CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
                "total_acc_x", "total_acc_y", "total_acc_z"]


def write_fake_uci_har(root, n_subjects=10, seconds_per_activity=40, seed=0):
    """Write 'UCI HAR Dataset/' with 128-sample windows at 50% overlap."""
    rng = np.random.default_rng(seed)
    base = Path(root) / "UCI HAR Dataset"
    win, step = 128, 64
    rows = {"train": {c: [] for c in UCI_CHANNELS}, "test": {c: [] for c in UCI_CHANNELS}}
    labels = {"train": [], "test": []}
    subjects = {"train": [], "test": []}
    for s in range(1, n_subjects + 1):
        split = "test" if s % 3 == 0 else "train"
        scale = rng.uniform(0.8, 1.2)
        for a_idx, act in enumerate(UCI_ACTIVITIES):
            n = seconds_per_activity * UCI_FS
            acc, gyro = _activity_signal(rng, act, n, UCI_FS, scale)
            body = acc - _lowpass(acc, UCI_FS)
            sig = {"body_acc": body, "body_gyro": gyro, "total_acc": acc}
            for start in range(0, n - win + 1, step):
                for c in UCI_CHANNELS:
                    prefix, axis = c[:-2], "xyz".index(c[-1])
                    rows[split][c].append(sig[prefix][start:start + win, axis])
                labels[split].append(a_idx + 1)
                subjects[split].append(s)
    for split in ("train", "test"):
        d = base / split / "Inertial Signals"
        d.mkdir(parents=True, exist_ok=True)
        for c in UCI_CHANNELS:
            np.savetxt(d / f"{c}_{split}.txt", np.array(rows[split][c]), fmt="%.7e")
        np.savetxt(base / split / f"y_{split}.txt", labels[split], fmt="%d")
        np.savetxt(base / split / f"subject_{split}.txt", subjects[split], fmt="%d")
    return Path(root)


# ------------------------------------------------------------------- SisFall
SIS_FS = 200


def _sis_walk(rng, seconds):
    acc, gyro = _activity_signal(rng, "walking", int(seconds * SIS_FS), SIS_FS)
    return acc[:, [1, 0, 2]] * [1, -1, 1], np.rad2deg(gyro)     # SisFall: -y is up


def _sis_still(rng, seconds, axis=1, sign=-1.0):
    n = int(seconds * SIS_FS)
    acc = np.zeros((n, 3))
    acc[:, axis] = sign
    return acc + rng.normal(0, 0.01, acc.shape), rng.normal(0, 1, (n, 3))


def _sis_lie_down(rng, seconds):
    """Smoothly rotate gravity from -y (upright) to +z (lying)."""
    n = int(seconds * SIS_FS)
    a = np.linspace(0, np.pi / 2, n)
    acc = np.stack([np.zeros(n), -np.cos(a), np.sin(a)], 1)
    return acc + rng.normal(0, 0.02, acc.shape), rng.normal(0, 10, (n, 3))


def _sis_impact(rng):
    n = int(0.15 * SIS_FS)
    acc = np.zeros((n, 3))
    acc[:, 0] = np.hanning(n) * 4.5
    return acc + rng.normal(0, 0.05, acc.shape), rng.normal(0, 150, (n, 3))


def _sis_write(path, acc, gyro):
    from src.data.sisfall import ACC_SCALE, GYRO_SCALE
    raw = np.concatenate([acc / ACC_SCALE, gyro / GYRO_SCALE, acc / ACC_SCALE], 1)
    with open(path, "w") as f:
        for row in np.round(raw).astype(int):
            f.write(",".join(f"{v:4d}" for v in row) + ";\n")


def write_fake_sisfall(root, young=6, elderly=2, reps=2, seed=0):
    """Write 'SisFall_dataset/' with ADL (D01) and fall (F01) trials."""
    rng = np.random.default_rng(seed)
    base = Path(root) / "SisFall_dataset"
    for group, n in (("SA", young), ("SE", elderly)):
        for s in range(1, n + 1):
            d = base / f"{group}{s:02d}"
            d.mkdir(parents=True, exist_ok=True)
            for r in range(1, reps + 1):
                parts = [_sis_still(rng, 2), _sis_walk(rng, 10), _sis_still(rng, 4)]
                _sis_write(d / f"D01_{group}{s:02d}_R{r:02d}.txt",
                           np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts]))
                # D02: lie down slowly (same final posture as a fall, no impact)
                down = _sis_lie_down(rng, 3)
                parts = [_sis_still(rng, 2), _sis_walk(rng, 3), down, _sis_still(rng, 6, axis=2, sign=1.0)]
                _sis_write(d / f"D02_{group}{s:02d}_R{r:02d}.txt",
                           np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts]))
                if group == "SA":
                    parts = [_sis_still(rng, 2), _sis_walk(rng, 4), _sis_impact(rng),
                             _sis_still(rng, 7, axis=2, sign=1.0)]
                    _sis_write(d / f"F01_{group}{s:02d}_R{r:02d}.txt",
                               np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts]))
    return Path(root)


# -------------------------------------------------------------------- PAMAP2
PAMAP_FS = 100


def write_fake_pamap2(root, n_subjects=4, seconds_per_activity=30, seed=0):
    """Write 'PAMAP2_Dataset/Protocol/subject10X.dat' (54 space-separated columns)."""
    from src.data.pamap2 import ACTIVITY_IDS
    rng = np.random.default_rng(seed)
    d = Path(root) / "PAMAP2_Dataset" / "Protocol"
    d.mkdir(parents=True, exist_ok=True)
    for s in range(1, n_subjects + 1):
        blocks, t0 = [], 0.0
        for act_id, name in list(ACTIVITY_IDS.items())[:8]:
            for aid, secs, act in ((0, 3, "standing"), (act_id, seconds_per_activity, name)):
                n = int(secs * PAMAP_FS)
                acc, gyro = _activity_signal(rng, act, n, PAMAP_FS)
                block = np.full((n, 54), np.nan)
                block[:, 0] = t0 + np.arange(n) / PAMAP_FS
                block[:, 1] = aid
                block[::11, 2] = 80 + 20 * (act in ("running", "rope_jumping")) + rng.normal(0, 2, len(block[::11]))
                for base in (3, 20, 37):                  # hand, chest, ankle IMUs
                    block[:, base] = 32.0
                    block[:, base + 1:base + 4] = acc * 9.80665
                    block[:, base + 4:base + 7] = acc * 9.80665
                    block[:, base + 7:base + 10] = gyro
                    block[:, base + 10:base + 13] = rng.normal(0, 1, (n, 3))
                    block[:, base + 13:base + 17] = [1, 0, 0, 0]
                drop = rng.random(n) < 0.01               # dropped packets, like the real files
                block[drop, 3:20] = np.nan
                blocks.append(block)
                t0 += secs
        np.savetxt(d / f"subject{100 + s}.dat", np.vstack(blocks), fmt="%.6g")
    return Path(root)


def write_all(root, seed=0):
    write_fake_uci_har(root, seed=seed)
    write_fake_sisfall(root, seed=seed)
    write_fake_pamap2(root, seed=seed)
    return Path(root)
