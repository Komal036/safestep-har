"""Train deep HAR models with subject-wise cross-validation.

For each outer fold:
  * test subjects are held out completely
  * a further set of training subjects becomes the validation set
    (early stopping and checkpoint selection never see test subjects)
  * normalization stats come from the inner training subjects only

Usage:
    python -m src.train_deep --config configs/deep.yaml
    python -m src.train_deep --config configs/deep.yaml --models cnn1d
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.augment import Augmenter
from src.data.registry import load_windows
from src.evaluation import per_class_f1, plot_cm, results_table
from src.models import build_model, count_params


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_channels(data, prefixes):
    """Keep only channels whose name starts with one of the prefixes (ablations)."""
    if not prefixes:
        return data
    idx = [i for i, c in enumerate(data["channels"]) if c.startswith(tuple(prefixes))]
    if not idx:
        raise ValueError(f"No channels match {prefixes}")
    out = dict(data)
    out["X"] = data["X"][:, :, idx]
    out["channels"] = [data["channels"][i] for i in idx]
    return out


class Normalizer:
    def fit(self, X):
        self.mean = X.mean(axis=(0, 1), keepdims=True)
        self.std = X.std(axis=(0, 1), keepdims=True) + 1e-6
        return self

    def __call__(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)


def predict(model, X, device, batch_size=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty(0, int)


def train_one(model, Xtr, ytr, Xva, yva, norm, aug, tcfg, n_classes, device):
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                  label_smoothing=tcfg.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])
    Xva_n = norm(Xva)

    best_f1, best_state, wait, history = -1.0, None, 0, []
    rng = np.random.default_rng(tcfg["seed"])
    for epoch in range(tcfg["epochs"]):
        model.train()
        order = rng.permutation(len(Xtr))
        total = 0.0
        for i in range(0, len(order), tcfg["batch_size"]):
            idx = order[i:i + tcfg["batch_size"]]
            xb = Xtr[idx]
            if aug is not None:
                xb = aug(xb)               # augment raw signal, then normalize
            xb = torch.from_numpy(norm(xb)).to(device)
            yb = torch.from_numpy(ytr[idx]).long().to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        val_f1 = f1_score(yva, predict(model, Xva_n, device), average="macro")
        history.append({"epoch": epoch + 1, "loss": total / len(Xtr), "val_f1": float(val_f1)})
        if val_f1 > best_f1:
            best_f1, best_state, wait = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= tcfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best_f1, history


def cpu_latency_ms(model, seq_len, n_channels, runs=50):
    """Single-window CPU latency, a rough proxy for on-device cost."""
    model = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, seq_len, n_channels)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
    return (time.perf_counter() - t0) / runs * 1000


def subject_cv(data, name, mcfg, tcfg, n_splits, out_dir, device):
    X, y, subjects = data["X"], data["y"], data["subjects"]
    n_classes, seq_len, n_ch = len(data["classes"]), X.shape[1], X.shape[2]
    gkf = GroupKFold(n_splits=n_splits)
    fold_f1, fold_acc, histories = [], [], []
    cm = np.zeros((n_classes, n_classes), dtype=int)
    model = None

    for k, (tr, te) in enumerate(gkf.split(X, y, groups=subjects)):
        set_seed(tcfg["seed"] + k)
        gss = GroupShuffleSplit(n_splits=1, test_size=tcfg["val_subject_frac"],
                                random_state=tcfg["seed"] + k)
        itr, iva = next(gss.split(X[tr], y[tr], groups=subjects[tr]))
        itr, iva = tr[itr], tr[iva]
        assert not set(subjects[te]) & set(subjects[tr]), "test subject leak"
        assert not set(subjects[iva]) & set(subjects[itr]), "val subject leak"

        norm = Normalizer().fit(X[itr])
        aug = Augmenter(data["channels"], seed=tcfg["seed"] + k, **tcfg["augment"]) \
            if tcfg.get("augment") else None
        model = build_model(name, n_ch, n_classes, seq_len, **(mcfg or {})).to(device)
        model, val_f1, hist = train_one(model, X[itr], y[itr], X[iva], y[iva],
                                        norm, aug, tcfg, n_classes, device)
        pred = predict(model, norm(X[te]), device)
        fold_f1.append(f1_score(y[te], pred, average="macro"))
        fold_acc.append(accuracy_score(y[te], pred))
        cm += confusion_matrix(y[te], pred, labels=range(n_classes))
        histories.append(hist)
        print(f"  [{name}] fold {k + 1}/{n_splits}: epochs={len(hist)} "
              f"val-F1={val_f1:.4f} test macro-F1={fold_f1[-1]:.4f}")

    # Last fold's model + the subjects it never saw, so export/benchmarks
    # can evaluate on genuinely held-out people.
    torch.save({"model_name": name, "state_dict": model.state_dict(), "config": mcfg or {},
                "dataset": data.get("dataset", "uci_har"), "dataset_kwargs": data.get("dataset_kwargs", {}),
                "channels": data["channels"], "classes": data["classes"], "seq_len": seq_len,
                "norm_mean": norm.mean, "norm_std": norm.std,
                "train_subjects": sorted(set(subjects[itr].tolist())),
                "test_subjects": sorted(set(subjects[te].tolist()))},
               out_dir / f"{name}_last_fold.pt")
    return {
        "macro_f1_mean": float(np.mean(fold_f1)),
        "macro_f1_std": float(np.std(fold_f1)),
        "accuracy_mean": float(np.mean(fold_acc)),
        "fold_macro_f1": [float(v) for v in fold_f1],
        "per_class_f1": [float(v) for v in per_class_f1(cm)],
        "confusion_matrix": cm.tolist(),
        "params": count_params(model),
        "cpu_latency_ms": float(cpu_latency_ms(model, seq_len, n_ch)),
        "history": histories,
    }


def run(cfg, data=None, only=None):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_gpu", True) else "cpu")
    data = data or load_windows(cfg.get("dataset", "uci_har"), cfg["data_dir"],
                                **(cfg.get("dataset_kwargs") or {}))
    data = {**data, "dataset_kwargs": cfg.get("dataset_kwargs") or {}}
    data = select_channels(data, cfg.get("channels"))
    print(f"{len(data['X'])} windows | {len(np.unique(data['subjects']))} subjects | "
          f"channels={data['channels']} | device={device}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, mcfg in cfg["models"].items():
        if only and name not in only:
            continue
        res = subject_cv(data, name, mcfg, cfg["train"], cfg["cv"]["n_splits"], out, device)
        results[name] = res
        plot_cm(res["confusion_matrix"], data["classes"], f"{name} (subject-wise CV)",
                out / f"cm_{name}.png")
        print(f"{name}: macro-F1 {res['macro_f1_mean']:.4f} ± {res['macro_f1_std']:.4f} | "
              f"{res['params']:,} params | {res['cpu_latency_ms']:.2f} ms/window")

    (out / "results.json").write_text(json.dumps(
        {"classes": data["classes"], "channels": data["channels"], "results": results}, indent=2))
    table = results_table(results, extra_cols=("params", "cpu_latency_ms"))
    (out / "results.md").write_text(table)
    print(table)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/deep.yaml")
    p.add_argument("--models", nargs="*", help="subset of models to run")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(cfg, only=args.models)


if __name__ == "__main__":
    main()
