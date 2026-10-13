"""Week 6: replay session, stream pipeline, API and dashboard, built from
tiny models trained on synthetic data (module-scoped, ~30 s)."""
import json
import os
from datetime import timedelta

import numpy as np
import pytest

from src.data.synthetic import write_fake_sisfall, write_fake_uci_har
from src.demo.frames import estimate_up, rotation_between, to_frame


def test_rotation_between():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = rotation_between(a, b)
        assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-8)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
    R = rotation_between([0, -1, 0], [0, 1, 0])           # antiparallel case
    assert np.allclose(R @ [0, -1, 0], [0, 1, 0])
    assert np.allclose(estimate_up([[0, 0, 2], [0, 0, 4]]), [0, 0, 1])
    assert np.allclose(to_frame([[1, 0, 0]], rotation_between([1, 0, 0], [0, 1, 0])), [[0, 1, 0]])


@pytest.fixture(scope="module")
def deployed(tmp_path_factory):
    """Synthetic data -> tiny HAR + fall models -> ONNX export -> replay session."""
    from src.demo.session import build_session, save_session
    from src.export.onnx_export import export_and_benchmark
    from src.data.registry import load_windows
    from src.train_deep import run as run_deep
    from src.train_fall import run as run_fall

    root = tmp_path_factory.mktemp("w6")
    data = root / "data"
    write_fake_uci_har(data, n_subjects=6, seconds_per_activity=30)
    write_fake_sisfall(data, young=5, elderly=1, reps=2)
    train = {"epochs": 8, "batch_size": 32, "lr": 0.003, "weight_decay": 0.0, "patience": 8,
             "val_subject_frac": 0.3, "seed": 0, "augment": None, "label_smoothing": 0.0}
    run_deep({"dataset": "uci_har", "data_dir": str(data), "channels": ["total_acc"], "use_gpu": False,
              "cv": {"n_splits": 2}, "train": train, "models": {"cnn1d": {"filters": [32, 64]}},
              "output_dir": str(root / "deploy_har")})
    run_fall({"data_dir": str(data), "sampling_rate_hz": 50, "use_gpu": False,
              "channels": ["acc_x", "acc_y", "acc_z"],
              "window": {"length": 128, "adl_step": 64, "train_peak_positions": [32, 48, 64],
                         "eval_peak_position": 48, "stream_step": 25},
              "cv": {"n_splits": 2}, "train": train,
              "model": {"name": "cnn1d", "params": {"filters": [32, 64]}},
              "events": {"target_recall": 0.9, "refractory_s": 10, "tolerance_s": 1.0},
              "threshold_baseline": {"impact_g": 2.5, "still_std_g": 0.15, "settle_s": 0.5},
              "output_dir": str(root / "fall_acc_only")})
    export_and_benchmark(root / "deploy_har" / "cnn1d_last_fold.pt", root / "export",
                         data=load_windows("uci_har", str(data)), n_calib=64, runs=5)
    export_and_benchmark(root / "fall_acc_only" / "fall_last_fold.pt", root / "export",
                         data=load_windows("sisfall", str(data), peak_positions=(48,)), n_calib=64, runs=5)
    session = build_session(str(data), seed=0)
    save_session(session, root / "demo" / "session.npz")
    return root


def test_session_roundtrip(deployed):
    from src.demo.session import load_session
    s = load_session(deployed / "demo" / "session.npz")
    assert len(s["segments"]) == 4
    seg = s["segments"][1]
    assert len(seg["acc"]) == len(seg["truth"]) and "fall" in seg["truth"]
    R = np.array(s["meta"]["fall_frame_rotation"])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert (deployed / "demo" / "frame_rotation.json").exists()


def test_stream_pipeline(deployed):
    from src.demo.pipeline import StreamProcessor, har_agreement, load_classifier
    from src.demo.session import load_session
    s = load_session(deployed / "demo" / "session.npz")
    p = StreamProcessor(load_classifier(deployed / "export" / "deploy_har-cnn1d"),
                        load_classifier(deployed / "export" / "fall_acc_only-fall"),
                        s["meta"]["fall_frame_rotation"])
    res = p.run(s)
    assert len(res["rows"]) > 1000
    assert {"time", "activity", "fall_prob", "motion", "truth", "segment"} <= set(res["rows"][0])
    assert 0 <= har_agreement(res["rows"], p.har.classes) <= 1
    times = [r["time"] for r in res["rows"]]
    assert times == sorted(times)
    # no data gap is credited to an activity
    assert sum(res["summary"]["minutes_per_activity"].values()) < 40


def test_rejects_multichannel_model(tmp_path, deployed):
    from src.demo.pipeline import load_classifier
    d = tmp_path / "m"
    d.mkdir()
    for f in ("model_int8.onnx", "model_fp32.onnx"):
        (d / f).write_bytes((deployed / "export" / "deploy_har-cnn1d" / f).read_bytes())
    meta = json.loads((deployed / "export" / "deploy_har-cnn1d" / "meta.json").read_text())
    meta["channels"] = ["a"] * 9
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="3-axis"):
        load_classifier(d)


@pytest.fixture()
def api_client(deployed, monkeypatch):
    import importlib
    monkeypatch.setenv("SAFESTEP_OUTPUTS", str(deployed))
    import app.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    return TestClient(api.app)


def test_api(api_client, deployed):
    from src.demo.session import load_session
    h = api_client.get("/health").json()
    assert h["status"] == "ok" and h["window"] == 128
    seg = load_session(deployed / "demo" / "session.npz")["segments"][0]
    w = seg["acc"][:128].tolist()
    r = api_client.post("/predict", json={"window": w}).json()
    assert r["activity"] in h["classes"] and 0 <= r["fall_prob"] <= 1
    assert abs(sum(r["activity_probs"].values()) - 1) < 0.02
    assert api_client.post("/predict", json={"window": w[:5]}).status_code == 422
    t = seg["start"]
    for i in range(3):
        ok = api_client.post("/devices/p1/ingest",
                             json={"timestamp": (t + timedelta(seconds=i)).isoformat(), "window": w})
        assert ok.status_code == 200
    assert api_client.post("/devices/p1/ingest", json={"timestamp": t.isoformat(), "window": w}).status_code == 409
    assert api_client.get("/devices/p1/summary").status_code == 200
    assert api_client.get("/devices/nobody/summary").status_code == 404
    assert api_client.delete("/devices/p1").json() == {"reset": "p1"}


def test_api_reports_missing_models(tmp_path, monkeypatch):
    import importlib
    monkeypatch.setenv("SAFESTEP_OUTPUTS", str(tmp_path))
    import app.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "models_missing"
    assert c.post("/predict", json={"window": [[0, 0, 1]] * 128}).status_code == 503


def test_dashboard_renders(deployed, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("SAFESTEP_OUTPUTS", str(deployed))
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app", "dashboard.py"),
                           default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Critical alerts" in labels and "Detected activity" in labels
    for i in range(len(at.selectbox[0].options)):
        at.selectbox[0].set_value(i).run()
        assert not at.exception


def test_dashboard_without_outputs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("SAFESTEP_OUTPUTS", str(tmp_path))
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app", "dashboard.py")).run()
    assert not at.exception
    assert at.warning, "should explain what is missing"
