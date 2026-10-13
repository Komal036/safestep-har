"""Train feature-based baselines with subject-wise cross-validation.

Usage:
    python -m src.train_baseline --config configs/default.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from src.data.features import extract_features
from src.data.registry import load_windows
from src.evaluation import per_class_f1, plot_cm, results_table


def build_model(name, params, n_classes, seed):
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, class_weight="balanced", **params)
    if name == "xgboost":
        return XGBClassifier(objective="multi:softprob", num_class=n_classes,
                             eval_metric="mlogloss", random_state=seed,
                             tree_method="hist", **params)
    raise ValueError(f"Unknown model: {name}")


def subject_cv(F, y, subjects, name, params, n_classes, n_splits, seed):
    gkf = GroupKFold(n_splits=n_splits)
    fold_f1, fold_acc = [], []
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for k, (tr, te) in enumerate(gkf.split(F, y, groups=subjects)):
        assert not set(subjects[tr]) & set(subjects[te]), "subject leak"
        model = build_model(name, params, n_classes, seed)
        model.fit(F[tr], y[tr])
        pred = model.predict(F[te])
        fold_f1.append(f1_score(y[te], pred, average="macro"))
        fold_acc.append(accuracy_score(y[te], pred))
        cm += confusion_matrix(y[te], pred, labels=range(n_classes))
        print(f"  [{name}] fold {k + 1}/{n_splits}: macro-F1={fold_f1[-1]:.4f}")
    per_class = per_class_f1(cm)
    return {
        "macro_f1_mean": float(np.mean(fold_f1)),
        "macro_f1_std": float(np.std(fold_f1)),
        "accuracy_mean": float(np.mean(fold_acc)),
        "fold_macro_f1": [float(v) for v in fold_f1],
        "per_class_f1": [float(v) for v in per_class],
        "confusion_matrix": cm.tolist(),
    }


def run(cfg, data=None):
    data = data or load_windows(cfg.get("dataset", "uci_har"), cfg["data_dir"],
                                **(cfg.get("dataset_kwargs") or {}))
    X, y, subjects = data["X"], data["y"], data["subjects"]
    classes = data["classes"]
    print(f"Loaded {len(X)} windows, {len(np.unique(subjects))} subjects, {X.shape[-1]} channels")

    F, names = extract_features(X, data["channels"], fs=cfg["sampling_rate_hz"])
    print(f"Extracted {F.shape[1]} features")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, params in cfg["models"].items():
        res = subject_cv(F, y, subjects, name, params or {}, len(classes),
                         cfg["cv"]["n_splits"], cfg["cv"]["seed"])
        results[name] = res
        plot_cm(res["confusion_matrix"], classes, f"{name} (subject-wise CV)",
                out / f"cm_{name}.png")
        print(f"{name}: macro-F1 {res['macro_f1_mean']:.4f} ± {res['macro_f1_std']:.4f}")

    (out / "results.json").write_text(json.dumps(
        {"classes": classes, "n_features": F.shape[1], "results": results}, indent=2))
    table = results_table(results)
    (out / "results.md").write_text(table)
    print(table)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(cfg)


if __name__ == "__main__":
    main()
