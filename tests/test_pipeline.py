"""End-to-end smoke test on synthetic data (no download needed)."""
import numpy as np

from src.train_baseline import run


def _synthetic(n_subj=6, per=30, n_cls=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y, s = [], [], []
    t = np.arange(128) / 50
    for subj in range(n_subj):
        for c in range(n_cls):
            for _ in range(per // n_cls):
                base = np.sin(2 * np.pi * (1 + c) * t)[:, None] * (c + 1)
                X.append(base + rng.normal(0, 0.3, (128, 9)))
                y.append(c)
                s.append(subj)
    from src.data.uci_har import CHANNELS
    return {"X": np.array(X, np.float32), "y": np.array(y), "subjects": np.array(s),
            "channels": CHANNELS, "classes": [f"c{i}" for i in range(n_cls)]}


def test_run_end_to_end(tmp_path):
    cfg = {"data_dir": "", "sampling_rate_hz": 50, "cv": {"n_splits": 3, "seed": 0},
           "models": {"random_forest": {"n_estimators": 20},
                      "xgboost": {"n_estimators": 20, "max_depth": 3}},
           "output_dir": str(tmp_path)}
    res = run(cfg, data=_synthetic())
    assert res["random_forest"]["macro_f1_mean"] > 0.9
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "cm_xgboost.png").exists()
