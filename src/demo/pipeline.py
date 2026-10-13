"""Run a sensor stream through both deployed models and the safety monitor.

One 3-axis accelerometer stream (g, 50 Hz) feeds:
  * the HAR model, in its own (UCI) frame
  * the fall model, after rotating into its (SisFall) frame
Every `step` samples a window is classified, turned into an Observation,
and passed to SafetyMonitor. Needs only numpy + onnxruntime.
"""
import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from src.export.runtime import OnnxClassifier
from src.safety import Observation, SafetyConfig, SafetyMonitor, summarize_day


def resolve_model(model_dir):
    """Pick the recommended ONNX variant written by onnx_export."""
    d = Path(model_dir)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    variant = meta.get("recommended_variant", "onnx_int8")
    path = d / ("model_int8.onnx" if variant == "onnx_int8" else "model_fp32.onnx")
    if not path.exists():
        path = next((p for p in (d / "model_int8.onnx", d / "model_fp32.onnx") if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No ONNX model in {d}. Run the export step first (see README).")
    return path


def load_classifier(model_dir, threads=1):
    clf = OnnxClassifier(resolve_model(model_dir), Path(model_dir) / "meta.json", threads)
    n_ch = len(clf.meta.get("channels", []))
    if n_ch != 3:
        raise ValueError(f"{model_dir} expects {n_ch} channels; the demo needs 3-axis "
                         "accelerometer models (configs/deploy_har.yaml, configs/fall_acc_only.yaml)")
    return clf


def motion_level(windows):
    """Std of acceleration magnitude per window (g): ~0 when still."""
    return np.linalg.norm(windows, axis=-1).std(axis=1)


class StreamProcessor:
    def __init__(self, har, fall, fall_rotation=None, step=32, fs=50, safety_config=None):
        self.har, self.fall, self.step, self.fs = har, fall, step, fs
        self.window = int(har.meta.get("seq_len", 128))
        if int(fall.meta.get("seq_len", 128)) != self.window:
            raise ValueError("HAR and fall models must share the window length")
        self.R = np.eye(3) if fall_rotation is None else np.asarray(fall_rotation, dtype=np.float32)
        thr = fall.threshold if fall.threshold is not None else 0.5
        self.safety_config = safety_config or SafetyConfig(fall_threshold=float(thr))

    def windows(self, acc):
        starts = np.arange(0, len(acc) - self.window + 1, self.step)
        return starts, np.stack([acc[s:s + self.window] for s in starts]).astype(np.float32)

    def infer(self, wins):
        har_p = self.har.predict_proba(wins)
        fall_p = self.fall.predict_proba((wins @ self.R).astype(np.float32))[:, 1]
        return har_p, fall_p

    def process_segment(self, start, acc, truth=None):
        starts, wins = self.windows(acc)
        har_p, fall_p = self.infer(wins)
        motion = motion_level(wins)
        classes = self.har.classes
        rows = []
        for i, s in enumerate(starts):
            row = {"time": start + timedelta(seconds=(s + self.window) / self.fs),
                   "activity": classes[int(har_p[i].argmax())],
                   "confidence": float(har_p[i].max()),
                   "fall_prob": float(fall_p[i]), "motion": float(motion[i])}
            if truth is not None:
                vals, counts = np.unique(truth[s:s + self.window], return_counts=True)
                row["truth"] = str(vals[counts.argmax()])
            rows.append(row)
        return rows

    def run(self, session):
        monitor = SafetyMonitor(self.safety_config)
        rows, alerts = [], []
        for seg_id, seg in enumerate(session["segments"]):
            seg_rows = self.process_segment(seg["start"], seg["acc"], seg.get("truth"))
            for r in seg_rows:
                r["segment"] = seg_id
                new = monitor.update(Observation(r["time"], r["activity"], r["fall_prob"], r["motion"]))
                for a in new:
                    a.details["segment"] = seg_id
                alerts.extend(new)
            rows.extend(seg_rows)
        obs = [Observation(r["time"], r["activity"], r["fall_prob"], r["motion"]) for r in rows]
        return {"rows": rows, "alerts": alerts, "summary": summarize_day(obs, alerts)}


def har_agreement(rows, classes):
    """Share of windows where the HAR prediction matches the replay's ground truth."""
    scored = [r for r in rows if r.get("truth") in classes]
    if not scored:
        return float("nan")
    return float(np.mean([r["activity"] == r["truth"] for r in scored]))
