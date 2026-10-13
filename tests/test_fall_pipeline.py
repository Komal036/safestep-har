import json

import torch

from src.train_fall import run
from tests.sisfall_synth import make_sisfall


def test_fall_run_end_to_end(tmp_path):
    data = make_sisfall(tmp_path / "raw", young=6, elderly=2, reps=2)
    cfg = {
        "data_dir": str(data), "sampling_rate_hz": 50, "use_gpu": False,
        "window": {"length": 128, "adl_step": 64, "train_peak_positions": [32, 48, 64],
                   "eval_peak_position": 48, "stream_step": 25},
        "cv": {"n_splits": 3},
        "train": {"epochs": 10, "batch_size": 32, "lr": 0.003, "weight_decay": 0.0, "patience": 10,
                  "val_subject_frac": 0.3, "seed": 0, "augment": None},
        "model": {"name": "cnn1d", "params": {"filters": [16, 16]}},
        "events": {"target_recall": 0.9, "refractory_s": 10, "tolerance_s": 1.0},
        "threshold_baseline": {"impact_g": 2.5, "still_std_g": 0.15, "settle_s": 0.5},
        "output_dir": str(tmp_path / "out"),
    }
    res = run(cfg)
    assert res["cnn_tuned"]["n_falls"] == 12
    assert res["cnn_tuned"]["recall"] >= 0.75
    assert res["threshold_baseline"]["recall"] == 1.0
    ck = torch.load(tmp_path / "out" / "fall_last_fold.pt", weights_only=False)
    assert ck["dataset"] == "sisfall" and 0 < ck["threshold"] < 1
    assert not set(ck["train_subjects"]) & set(ck["test_subjects"])
    assert json.loads((tmp_path / "out" / "results.json").read_text())["tradeoff_curve"]
    assert (tmp_path / "out" / "fall_tradeoff.png").exists()


def test_fall_acc_only_and_export(tmp_path):
    from src.export.onnx_export import export_and_benchmark
    from src.data.sisfall import load_sisfall_windows
    data = make_sisfall(tmp_path / "raw", young=5, elderly=1, reps=2)
    cfg = {
        "data_dir": str(data), "sampling_rate_hz": 50, "use_gpu": False,
        "channels": ["acc_x", "acc_y", "acc_z"],
        "window": {"length": 128, "adl_step": 64, "train_peak_positions": [40, 56],
                   "eval_peak_position": 48, "stream_step": 25},
        "cv": {"n_splits": 2},
        "train": {"epochs": 6, "batch_size": 32, "lr": 0.003, "weight_decay": 0.0, "patience": 6,
                  "val_subject_frac": 0.3, "seed": 0, "augment": None},
        # realistic width: on ~1k-param toy models INT8 metadata outweighs the savings
        "model": {"name": "cnn1d", "params": {"filters": [32, 64, 64]}},
        "events": {"target_recall": 0.9, "refractory_s": 10, "tolerance_s": 1.0},
        "threshold_baseline": {"impact_g": 2.5, "still_std_g": 0.15, "settle_s": 0.5},
        "output_dir": str(tmp_path / "out"),
    }
    res = run(cfg)
    assert res["channels"] == ["acc_x", "acc_y", "acc_z"]
    # the checkpoint exports straight to ONNX on its held-out subjects
    windows = load_sisfall_windows(str(data), peak_positions=(48,))
    r = export_and_benchmark(tmp_path / "out" / "fall_last_fold.pt", tmp_path / "export",
                             data=windows, n_calib=64, runs=10)
    assert r["variants"]["onnx_int8"]["size_kb"] < r["variants"]["onnx_fp32"]["size_kb"]
    assert "recall@thr" in r["variants"]["onnx_int8"]
