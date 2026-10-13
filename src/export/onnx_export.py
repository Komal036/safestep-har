"""Export trained checkpoints to ONNX, quantize to INT8, and benchmark.

For each checkpoint:
  1. wrap the model so raw windows go in and probabilities come out
     (normalization + softmax inside the graph)
  2. export FP32 ONNX and check it matches PyTorch
  3. INT8 post-training quantization with ONNX Runtime: static (QDQ,
     per-channel, calibrated on *training* subjects) and dynamic are both
     tried, and the smallest one that stays faithful to FP32 is kept
  4. benchmark on the checkpoint's held-out *test* subjects:
     size, single-thread batch-1 latency, macro-F1, INT8 vs FP32 agreement

Usage:
    python -m src.export.onnx_export --checkpoint outputs/deploy_har/cnn1d_last_fold.pt \
                                                  outputs/fall_acc_only/fall_last_fold.pt
Each model lands in outputs/export/<run>-<model>/ (e.g. deploy_har-cnn1d).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data.registry import load_windows
from src.export.runtime import make_session
from src.models import build_model, count_params


class DeployModel(nn.Module):
    """Raw window in, class probabilities out."""

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.as_tensor(np.asarray(mean).reshape(1, 1, -1), dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.asarray(std).reshape(1, 1, -1), dtype=torch.float32))

    def forward(self, x):
        return torch.softmax(self.model((x - self.mean) / self.std), dim=-1)


class WindowReader(CalibrationDataReader):
    def __init__(self, X, input_name, batch=1):
        self.it = iter([{input_name: X[i:i + batch]} for i in range(0, len(X), batch)])

    def get_next(self):
        return next(self.it, None)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["model_name"], len(ck["channels"]), len(ck["classes"]),
                        ck["seq_len"], **(ck.get("config") or {}))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return ck, DeployModel(model, ck["norm_mean"], ck["norm_std"]).eval()


def select(data, channels, subjects):
    idx = [data["channels"].index(c) for c in channels]
    mask = np.isin(data["subjects"], list(subjects))
    return data["X"][mask][:, :, idx].astype(np.float32), data["y"][mask]


def export_fp32(deploy, seq_len, n_ch, path):
    dummy = torch.randn(1, seq_len, n_ch)
    torch.onnx.export(deploy, dummy, str(path), input_names=["imu"], output_names=["probs"],
                      dynamic_axes={"imu": {0: "batch"}, "probs": {0: "batch"}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))


def _static(fp32_path, out_path, calib_X):
    pre = fp32_path.with_name(fp32_path.stem + "_pre.onnx")
    try:
        quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
        quantize_static(str(pre), str(out_path), WindowReader(calib_X, "imu"),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    finally:
        pre.unlink(missing_ok=True)


def _dynamic(fp32_path, out_path, calib_X):
    quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8)


def quantize(fp32_path, int8_path, calib_X, min_agreement=0.98):
    """Try static (QDQ, per-channel) and dynamic INT8; keep the best one.

    Selection uses only calibration windows from training subjects:
    the smallest model whose predictions agree with FP32 on at least
    min_agreement of them, otherwise the most faithful one.
    """
    ref = ort_predict(make_session(fp32_path), calib_X).argmax(1)
    candidates, notes = [], {}
    for method, fn in (("static_qdq", _static), ("dynamic", _dynamic)):
        path = int8_path.with_name(f"{int8_path.stem}_{method}.onnx")
        try:
            fn(fp32_path, path, calib_X)
            agree = float((ort_predict(make_session(path), calib_X).argmax(1) == ref).mean())
            size = path.stat().st_size
            candidates.append((method, path, agree, size))
            notes[method] = {"size_kb": size / 1024, "calib_agreement": agree}
        except Exception as e:  # noqa: BLE001 - report and try the other method
            notes[method] = {"error": str(e)[:200]}
    if not candidates:
        raise RuntimeError(f"INT8 quantization failed: {notes}")
    good = [c for c in candidates if c[2] >= min_agreement]
    best = min(good, key=lambda c: c[3]) if good else max(candidates, key=lambda c: c[2])
    for c in candidates:
        if c is best:
            c[1].replace(int8_path)
        else:
            c[1].unlink(missing_ok=True)
    return best[0], notes


def ort_predict(session, X, batch=256):
    return np.concatenate([session.run(None, {"imu": X[i:i + batch]})[0]
                           for i in range(0, len(X), batch)])


def latency_ms(fn, x, warmup=20, runs=300):
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def classification_metrics(y, probs, threshold=None):
    pred = probs.argmax(1)
    out = {"macro_f1": float(f1_score(y, pred, average="macro")),
           "accuracy": float((pred == y).mean())}
    if threshold is not None and probs.shape[1] == 2:
        pos = (probs[:, 1] >= threshold).astype(int)
        out["recall@thr"] = float(recall_score(y, pos, zero_division=0))
        out["precision@thr"] = float(precision_score(y, pos, zero_division=0))
    return out


def default_tag(ckpt_path):
    """outputs/deploy_har/cnn1d_last_fold.pt -> 'deploy_har-cnn1d'."""
    p = Path(ckpt_path)
    return f"{p.parent.name}-{p.stem.replace('_last_fold', '')}"


def export_and_benchmark(ckpt_path, out_root, data_dir="data/raw", data=None,
                         n_calib=512, runs=300, seed=0, tag=None):
    ckpt_path = Path(ckpt_path)
    ck, deploy = load_checkpoint(ckpt_path)
    tag = tag or default_tag(ckpt_path)
    out = Path(out_root) / tag
    out.mkdir(parents=True, exist_ok=True)

    data = data or load_windows(ck["dataset"], data_dir, **(ck.get("dataset_kwargs") or {}))
    Xtr, _ = select(data, ck["channels"], ck["train_subjects"])
    Xte, yte = select(data, ck["channels"], ck["test_subjects"])
    rng = np.random.default_rng(seed)
    calib = Xtr[rng.permutation(len(Xtr))[:n_calib]]
    T, C = ck["seq_len"], len(ck["channels"])

    fp32_path, int8_path = out / "model_fp32.onnx", out / "model_int8.onnx"
    export_fp32(deploy, T, C, fp32_path)
    method, quant_notes = quantize(fp32_path, int8_path, calib)

    torch.set_num_threads(1)
    with torch.no_grad():
        p_torch = torch.cat([deploy(torch.from_numpy(Xte[i:i + 256]))
                             for i in range(0, len(Xte), 256)]).numpy()
    s32, s8 = make_session(fp32_path), make_session(int8_path)
    p32, p8 = ort_predict(s32, Xte), ort_predict(s8, Xte)
    export_err = float(np.abs(p_torch - p32).max())
    if export_err > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {export_err:.2e}")

    thr = ck.get("threshold")
    x1 = Xte[:1]
    with torch.no_grad():
        torch_lat = latency_ms(lambda x: deploy(torch.from_numpy(x)), x1, runs=runs)
    lat32 = latency_ms(lambda x: s32.run(None, {"imu": x}), x1, runs=runs)
    lat8 = latency_ms(lambda x: s8.run(None, {"imu": x}), x1, runs=runs)

    rows = {
        "pytorch_fp32": {"size_kb": count_params(deploy) * 4 / 1024,
                         "latency_ms_p50": torch_lat[0], "latency_ms_p95": torch_lat[1],
                         **classification_metrics(yte, p_torch, thr)},
        "onnx_fp32": {"size_kb": fp32_path.stat().st_size / 1024,
                      "latency_ms_p50": lat32[0], "latency_ms_p95": lat32[1],
                      **classification_metrics(yte, p32, thr)},
        "onnx_int8": {"size_kb": int8_path.stat().st_size / 1024,
                      "latency_ms_p50": lat8[0], "latency_ms_p95": lat8[1],
                      **classification_metrics(yte, p8, thr)},
    }
    result = {
        "checkpoint": str(ckpt_path), "model": ck["model_name"], "dataset": ck["dataset"],
        "params": count_params(deploy), "quantization": method,
        "quantization_candidates": quant_notes,
        "n_test_windows": int(len(Xte)), "n_test_subjects": len(ck["test_subjects"]),
        "n_calibration_windows": int(len(calib)),
        "onnx_vs_pytorch_max_abs_diff": export_err,
        "int8_vs_fp32_agreement": float((p8.argmax(1) == p32.argmax(1)).mean()),
        "int8_macro_f1_drop": rows["onnx_fp32"]["macro_f1"] - rows["onnx_int8"]["macro_f1"],
        "size_reduction_x": rows["onnx_fp32"]["size_kb"] / rows["onnx_int8"]["size_kb"],
        # INT8 adds scale/zero-point metadata; for very small models FP32 can be smaller
        "recommended_variant": ("onnx_int8" if rows["onnx_int8"]["size_kb"] < rows["onnx_fp32"]["size_kb"]
                                else "onnx_fp32"),
        "variants": rows,
    }
    meta = {"classes": ck["classes"], "channels": ck["channels"], "seq_len": T,
            "threshold": thr, "model": ck["model_name"], "dataset": ck["dataset"],
            "input": "raw window, float32, shape (batch, seq_len, n_channels)",
            "test_subjects": [int(x) for x in ck["test_subjects"]],
            "recommended_variant": result["recommended_variant"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "benchmark.json").write_text(json.dumps(result, indent=2))
    (out / "benchmark.md").write_text(benchmark_table({tag: result}))
    return result


def benchmark_table(results):
    lines = ["| Model | Variant | Size (KB) | Latency p50 (ms) | Macro-F1 | Agreement w/ FP32 |",
             "|---|---|---|---|---|---|"]
    for tag, r in results.items():
        for v, m in r["variants"].items():
            agree = f"{r['int8_vs_fp32_agreement']:.4f}" if v == "onnx_int8" else ""
            lines.append(f"| {tag} | {v} | {m['size_kb']:.1f} | {m['latency_ms_p50']:.3f} "
                         f"| {m['macro_f1']:.4f} | {agree} |")
    notes = [f"- {tag}: INT8 via {r['quantization']}, {r['size_reduction_x']:.1f}x smaller, "
             f"macro-F1 change {-r['int8_macro_f1_drop']:+.4f}, "
             f"{r['n_test_windows']} windows from {r['n_test_subjects']} unseen subjects"
             for tag, r in results.items()]
    return "\n".join(lines) + "\n\n" + "\n".join(notes) + \
        "\n\nLatency: single thread, batch 1, ONNX Runtime CPU, median of repeated runs.\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out", default="outputs/export")
    p.add_argument("--runs", type=int, default=300)
    args = p.parse_args()
    results = {}
    for ck in args.checkpoint:
        print(f"Exporting {ck} ...")
        r = export_and_benchmark(ck, args.out, args.data_dir, runs=args.runs)
        results[default_tag(ck)] = r
    table = benchmark_table(results)
    Path(args.out, "benchmark.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
