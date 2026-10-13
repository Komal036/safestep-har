import numpy as np
import pytest
import torch

from src.export.onnx_export import export_and_benchmark
from src.export.runtime import OnnxClassifier
from src.models import build_model
from src.train_deep import Normalizer, train_one
from tests.test_pipeline import _synthetic


def _make_checkpoint(tmp_path, name, params):
    data = _synthetic(n_subj=6, per=30)
    data["X"] = data["X"] * 3 + 1          # non-trivial normalization to bake in
    tr = np.isin(data["subjects"], [0, 1, 2, 3])
    norm = Normalizer().fit(data["X"][tr])
    torch.manual_seed(0)
    model = build_model(name, 9, 3, 128, **params)
    tcfg = {"epochs": 8, "batch_size": 32, "lr": 0.003, "weight_decay": 0.0,
            "patience": 8, "seed": 0}
    model, _, _ = train_one(model, data["X"][tr], data["y"][tr], data["X"][~tr], data["y"][~tr],
                            norm, None, tcfg, 3, "cpu")
    path = tmp_path / f"{name}_last_fold.pt"
    torch.save({"model_name": name, "state_dict": model.state_dict(), "config": params,
                "dataset": "synthetic", "channels": data["channels"], "classes": data["classes"],
                "seq_len": 128, "norm_mean": norm.mean, "norm_std": norm.std,
                "train_subjects": [0, 1, 2, 3], "test_subjects": [4, 5]}, path)
    return path, data


@pytest.mark.parametrize("name,params", [
    ("cnn1d", {"filters": [16, 32]}),
    ("cnn_lstm", {"filters": [16], "hidden": 16, "lstm_layers": 1}),
    ("transformer", {"patch": 16, "d_model": 16, "heads": 2, "layers": 1}),
])
def test_export_quantize_benchmark(tmp_path, name, params):
    ckpt, data = _make_checkpoint(tmp_path, name, params)
    r = export_and_benchmark(ckpt, tmp_path / "export", data=data, n_calib=64, runs=10)
    assert r["onnx_vs_pytorch_max_abs_diff"] < 1e-3
    assert r["int8_vs_fp32_agreement"] >= 0.9
    v = r["variants"]
    assert v["onnx_int8"]["size_kb"] < v["onnx_fp32"]["size_kb"]
    out = tmp_path / "export" / f"{tmp_path.name}-{name}"
    assert (out / "benchmark.md").exists() and (out / "meta.json").exists()

    # deployable runtime: raw windows in, probabilities out, no torch needed
    clf = OnnxClassifier(out / "model_int8.onnx")
    probs = clf.predict_proba(data["X"][:5])
    assert probs.shape == (5, 3)
    assert np.allclose(probs.sum(1), 1, atol=1e-2)
    assert clf.classes == data["classes"]
