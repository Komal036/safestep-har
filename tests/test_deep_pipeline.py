"""Smoke test: full deep CV loop on synthetic data, CPU, a few epochs."""
from src.train_deep import run
from tests.test_pipeline import _synthetic


def test_deep_run_end_to_end(tmp_path):
    cfg = {
        "data_dir": "", "channels": None, "use_gpu": False,
        "cv": {"n_splits": 3},
        "train": {"epochs": 12, "batch_size": 32, "lr": 0.003, "weight_decay": 0.0,
                  "patience": 12, "val_subject_frac": 0.25, "seed": 0,
                  "augment": {"jitter_std": 0.02, "scale_std": 0.05, "rotate_deg": 10, "p": 0.5}},
        "models": {"cnn1d": {"filters": [16, 16]},
                   "cnn_lstm": {"filters": [16], "hidden": 16, "lstm_layers": 1},
                   "transformer": {"patch": 16, "d_model": 16, "heads": 2, "layers": 1}},
        "output_dir": str(tmp_path),
    }
    res = run(cfg, data=_synthetic(n_subj=6, per=30))
    for name, r in res.items():  # 3 classes, chance is ~0.33
        assert r["macro_f1_mean"] > 0.6, name
    assert (tmp_path / "cnn1d_last_fold.pt").exists()
    assert (tmp_path / "results.md").exists()


def test_channel_ablation(tmp_path):
    from src.train_deep import select_channels
    d = select_channels(_synthetic(n_subj=2, per=6), ["total_acc"])
    assert d["X"].shape[-1] == 3
    assert all(c.startswith("total_acc") for c in d["channels"])
