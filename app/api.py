"""SafeStep inference API.

Run:  uvicorn app.api:app --reload
Docs: http://localhost:8000/docs

Environment variables (defaults match the README pipeline):
  SAFESTEP_OUTPUTS outputs  (run_all.py --smoke uses outputs_smoke)
  HAR_MODEL_DIR   outputs/export/deploy_har-cnn1d
  FALL_MODEL_DIR  outputs/export/fall_acc_only-fall
  FRAME_ROTATION  outputs/demo/frame_rotation.json   (optional device calibration)
"""
import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.demo.pipeline import StreamProcessor, load_classifier, motion_level
from src.safety import Observation, SafetyMonitor, summarize_day

ROOT = os.getenv("SAFESTEP_OUTPUTS", "outputs")
HAR_DIR = os.getenv("HAR_MODEL_DIR", f"{ROOT}/export/deploy_har-cnn1d")
FALL_DIR = os.getenv("FALL_MODEL_DIR", f"{ROOT}/export/fall_acc_only-fall")
ROT_PATH = os.getenv("FRAME_ROTATION", f"{ROOT}/demo/frame_rotation.json")
MAX_HISTORY = 50_000

app = FastAPI(title="SafeStep HAR API",
              description="Activity recognition, fall detection and caregiver alerts "
                          "from a 3-axis accelerometer window (g, 50 Hz).",
              version="1.0.0")
state = {"processor": None, "error": None, "devices": {}}


def get_processor():
    if state["processor"] is None:
        try:
            rot = None
            if Path(ROT_PATH).exists():
                rot = json.loads(Path(ROT_PATH).read_text())["fall_frame_rotation"]
            state["processor"] = StreamProcessor(load_classifier(HAR_DIR), load_classifier(FALL_DIR), rot)
            state["error"] = None
        except Exception as e:  # noqa: BLE001 - surfaced via /health and 503s
            state["error"] = str(e)
            raise HTTPException(503, f"Models not loaded: {e}")
    return state["processor"]


class WindowIn(BaseModel):
    window: list[list[float]] = Field(..., description="seq_len rows of [x, y, z] acceleration in g")


class IngestIn(WindowIn):
    timestamp: datetime = Field(..., description="time at the END of the window (ISO 8601)")


def to_array(p, window):
    x = np.asarray(window, dtype=np.float32)
    if x.shape != (p.window, 3):
        raise HTTPException(422, f"window must have shape [{p.window}, 3], got {list(x.shape)}")
    if not np.isfinite(x).all():
        raise HTTPException(422, "window contains NaN or inf")
    return x


def classify(p, x):
    har_p, fall_p = p.infer(x[None])
    probs = {c: round(float(v), 4) for c, v in zip(p.har.classes, har_p[0])}
    return {"activity": max(probs, key=probs.get), "activity_probs": probs,
            "fall_prob": round(float(fall_p[0]), 4),
            "fall_threshold": p.safety_config.fall_threshold,
            "motion": round(float(motion_level(x[None])[0]), 4)}


@app.get("/health")
def health():
    try:
        p = get_processor()
    except HTTPException:
        return {"status": "models_missing", "error": state["error"],
                "har_model_dir": HAR_DIR, "fall_model_dir": FALL_DIR}
    return {"status": "ok", "window": p.window, "classes": p.har.classes,
            "fall_threshold": p.safety_config.fall_threshold,
            "har_model": p.har.model_path, "fall_model": p.fall.model_path,
            "devices": len(state["devices"])}


@app.post("/predict")
def predict(body: WindowIn):
    """Stateless: classify one window."""
    p = get_processor()
    return classify(p, to_array(p, body.window))


@app.post("/devices/{device_id}/ingest")
def ingest(device_id: str, body: IngestIn):
    """Stateful: classify the window and update this device's safety monitor."""
    p = get_processor()
    out = classify(p, to_array(p, body.window))
    dev = state["devices"].setdefault(device_id, {"monitor": SafetyMonitor(p.safety_config),
                                                  "history": deque(maxlen=MAX_HISTORY), "alerts": []})
    obs = Observation(body.timestamp, out["activity"], out["fall_prob"], out["motion"])
    try:
        alerts = dev["monitor"].update(obs)
    except ValueError as e:
        raise HTTPException(409, str(e))
    dev["history"].append(obs)
    dev["alerts"].extend(alerts)
    return {**out, "alerts": [a.to_dict() for a in alerts]}


@app.get("/devices/{device_id}/summary")
def summary(device_id: str):
    dev = state["devices"].get(device_id)
    if dev is None:
        raise HTTPException(404, "unknown device")
    return {**summarize_day(dev["history"], dev["alerts"]),
            "recent_alerts": [a.to_dict() for a in dev["alerts"][-20:]]}


@app.delete("/devices/{device_id}")
def reset(device_id: str):
    state["devices"].pop(device_id, None)
    return {"reset": device_id}
