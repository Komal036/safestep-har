"""Build a replayable "day" for the dashboard from real public recordings.

Activity segments come from one UCI HAR subject (by default a subject the
deployed HAR model never saw). The fall is a real SisFall fall trial from a
held-out fall subject, rotated into the UCI frame by gravity alignment.
Everything is the 3-axis accelerometer in g at 50 Hz, the input both
deployed models share.

Usage:
    python -m src.demo.session --data-dir data/raw --out outputs/demo/session.npz \\
        --har-meta outputs/export/deploy_har-cnn1d/meta.json \\
        --fall-meta outputs/export/fall_acc_only-fall/meta.json
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.data.sisfall import load_sisfall_trials
from src.data.uci_har import ACTIVITIES, load_uci_har
from src.demo.frames import estimate_up, rotation_between, to_frame

FS = 50
TOTAL_ACC = slice(6, 9)

# (clock time, [(activity, seconds), ...]); "fall" inserts a SisFall trial
DEFAULT_SCENARIO = [
    ("08:00", "Morning routine", [("standing", 30), ("walking", 90), ("sitting", 150),
                                   ("standing", 30), ("walking", 60)]),
    ("10:30", "Fall in the hallway", [("walking", 40), ("fall", None), ("lying", 75)]),
    ("14:00", "Afternoon restlessness", [("walking", 25), ("standing", 90)] * 7),
    ("23:30", "Night wandering", [("lying", 60), ("standing", 15), ("walking", 380), ("standing", 30)]),
]


def activity_stream(uci, subject, activity):
    """Continuous signal for one subject/activity: UCI windows overlap 50%,
    so every other window tiles the recording without repeats."""
    a = ACTIVITIES.index(activity)
    idx = np.flatnonzero((uci["subjects"] == subject) & (uci["y"] == a))
    if len(idx) == 0:
        raise ValueError(f"subject {subject} has no '{activity}' windows")
    return np.concatenate(uci["X"][idx[::2], :, TOTAL_ACC])


def take(sig, n, rng):
    """n samples from sig starting at a random offset, looping if needed."""
    start = int(rng.integers(0, len(sig)))
    reps = int(np.ceil((start + n) / len(sig)))
    return np.tile(sig, (reps, 1))[start:start + n]


def build_session(data_dir="data/raw", har_subject=None, fall_subject=None,
                  scenario=None, date="2026-01-05", seed=0):
    rng = np.random.default_rng(seed)
    scenario = scenario or DEFAULT_SCENARIO
    uci = load_uci_har(data_dir)
    trials = load_sisfall_trials(data_dir, FS)

    subjects = sorted(set(uci["subjects"].tolist()))
    har_subject = har_subject if har_subject is not None else subjects[-1]
    falls = [t for t in trials if t["is_fall"] and (fall_subject is None or t["subject"] == fall_subject)]
    if not falls:
        raise ValueError(f"no SisFall fall trials for subject {fall_subject}")
    fall_trial = falls[int(rng.integers(len(falls)))]

    # gravity alignment SisFall -> UCI frame
    upright = ACTIVITIES.index("standing")
    up_uci = estimate_up(uci["X"][uci["y"] == upright][:, :, TOTAL_ACC])
    adl_starts = [t["signal"][:FS, :3] for t in trials if not t["is_fall"]]
    up_sis = estimate_up(np.concatenate(adl_starts))
    R = rotation_between(up_sis, up_uci)                 # sisfall -> uci

    cache, segments = {}, []
    for clock, title, plan in scenario:
        parts, truth = [], []
        for act, secs in plan:
            if act == "fall":
                sig = to_frame(fall_trial["signal"][:, :3], R)
                parts.append(sig)
                truth += ["fall"] * len(sig)
                continue
            if act not in cache:
                cache[act] = activity_stream(uci, har_subject, act)
            n = int(secs * FS)
            parts.append(take(cache[act], n, rng))
            truth += [act] * n
        start = datetime.fromisoformat(f"{date}T{clock}:00")
        segments.append({"title": title, "start": start,
                         "acc": np.concatenate(parts).astype(np.float32), "truth": truth})

    meta = {"fs": FS, "channels": ["total_acc_x", "total_acc_y", "total_acc_z"],
            "har_subject": int(har_subject), "fall_subject": int(fall_trial["subject"]),
            "fall_file": fall_trial["file"], "fall_frame_rotation": R.tolist(),
            "up_uci": up_uci.tolist(), "up_sisfall": up_sis.tolist(), "seed": seed}
    return {"segments": segments, "meta": meta}


def save_session(session, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted({l for s in session["segments"] for l in s["truth"]})
    arrays = {}
    for i, s in enumerate(session["segments"]):
        arrays[f"acc_{i}"] = s["acc"]
        arrays[f"truth_{i}"] = np.array([labels.index(l) for l in s["truth"]], dtype=np.int16)
    header = {**session["meta"], "labels": labels,
              "segments": [{"title": s["title"], "start": s["start"].isoformat()} for s in session["segments"]]}
    np.savez_compressed(path, header=json.dumps(header), **arrays)
    # device calibration consumed by the API
    (path.parent / "frame_rotation.json").write_text(json.dumps(
        {"fall_frame_rotation": session["meta"]["fall_frame_rotation"]}, indent=2))
    return path


def load_session(path):
    z = np.load(path, allow_pickle=False)
    header = json.loads(str(z["header"]))
    labels = header.pop("labels")
    segs = header.pop("segments")
    segments = [{"title": s["title"], "start": datetime.fromisoformat(s["start"]),
                 "acc": z[f"acc_{i}"], "truth": [labels[k] for k in z[f"truth_{i}"]]}
                for i, s in enumerate(segs)]
    return {"segments": segments, "meta": header}


def _subjects_from_meta(path):
    if path and Path(path).exists():
        subs = json.loads(Path(path).read_text()).get("test_subjects") or []
        return subs[0] if subs else None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out", default="outputs/demo/session.npz")
    p.add_argument("--har-meta", default="outputs/export/deploy_har-cnn1d/meta.json",
                   help="use one of the HAR model's held-out subjects")
    p.add_argument("--fall-meta", default="outputs/export/fall_acc_only-fall/meta.json",
                   help="use one of the fall model's held-out subjects")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    fall_subj = None
    if args.fall_meta and Path(args.fall_meta).exists():
        test = json.loads(Path(args.fall_meta).read_text()).get("test_subjects") or []
        fall_subj = next((s for s in test if s < 100), None)   # young subjects did the falls
    s = build_session(args.data_dir, _subjects_from_meta(args.har_meta), fall_subj, seed=args.seed)
    out = save_session(s, args.out)
    mins = sum(len(x["acc"]) for x in s["segments"]) / FS / 60
    print(f"Saved {out}: {len(s['segments'])} segments, {mins:.1f} min of signal, "
          f"HAR subject {s['meta']['har_subject']}, fall from {s['meta']['fall_file']}")


if __name__ == "__main__":
    main()
