"""Fall detector on SisFall with subject-wise CV and event-level metrics.

Per outer fold:
  1. train a window classifier (fall vs ADL) on training subjects,
     with impact-shift augmentation for fall windows
  2. choose the alarm threshold on held-out *validation subjects* to reach
     a target fall recall with the fewest false alarms
  3. report event recall and false alarms/hour on unseen test subjects,
     next to a classic impact+stillness threshold detector

Usage:
    python -m src.train_fall --config configs/fall.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from src.augment import Augmenter
from src.data.sisfall import CHANNELS, CLASSES, load_sisfall_trials, trials_to_windows
from src.fall.stream import (event_metrics, model_score_fn, pick_threshold, score_trials,
                             threshold_score_fn)
from src.models import build_model, count_params
from src.train_deep import Normalizer, cpu_latency_ms, predict, set_seed, train_one


def split_val_subjects(subjects, fall_subjects, frac, seed):
    """Pick validation subjects, guaranteeing at least one subject with falls."""
    rng = np.random.default_rng(seed)
    subs = np.array(sorted(subjects))
    n_val = max(2, int(round(frac * len(subs))))
    with_falls = [s for s in subs if s in fall_subjects]
    first = rng.choice(with_falls)
    rest = rng.permutation([s for s in subs if s != first])[: n_val - 1]
    return set([int(first)] + [int(s) for s in rest])


def sum_metrics(rows):
    keys = ["n_falls", "detected", "false_alarms", "false_alarms_elderly",
            "adl_hours", "adl_hours_elderly"]
    tot = {k: sum(r[k] for r in rows) for k in keys}
    tot["recall"] = tot["detected"] / tot["n_falls"] if tot["n_falls"] else float("nan")
    tot["fa_per_hour"] = tot["false_alarms"] / tot["adl_hours"] if tot["adl_hours"] else float("nan")
    tot["fa_per_hour_elderly"] = (tot["false_alarms_elderly"] / tot["adl_hours_elderly"]
                                  if tot["adl_hours_elderly"] else float("nan"))
    return tot


def run(cfg, trials=None):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_gpu", True) else "cpu")
    trials = trials or load_sisfall_trials(cfg["data_dir"], cfg["sampling_rate_hz"])
    channels = cfg.get("channels") or list(CHANNELS)
    if channels[:3] != CHANNELS[:3]:
        raise ValueError("channels must start with acc_x, acc_y, acc_z (impact detection uses them)")
    idx = [CHANNELS.index(c) for c in channels]
    trials = [{**t, "signal": t["signal"][:, idx]} for t in trials]
    w, tcfg, ev = cfg["window"], cfg["train"], cfg["events"]
    window, step = w["length"], w["stream_step"]
    subjects = np.array([t["subject"] for t in trials])
    fall_subjects = {t["subject"] for t in trials if t["is_fall"]}
    print(f"{len(trials)} trials | {len(set(subjects))} subjects | "
          f"{sum(t['is_fall'] for t in trials)} falls | channels={channels} | device={device}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ev_kw = {"refractory_s": ev["refractory_s"], "tolerance_s": ev["tolerance_s"]}
    gkf = GroupKFold(n_splits=cfg["cv"]["n_splits"])
    tuned_rows, fixed_rows, base_rows, win_f1, thresholds = [], [], [], [], []
    test_caches = []
    model = norm = None

    for k, (tr_i, te_i) in enumerate(gkf.split(subjects, groups=subjects)):
        set_seed(tcfg["seed"] + k)
        tr_subj = set(subjects[tr_i].tolist())
        te_subj = set(subjects[te_i].tolist())
        va_subj = split_val_subjects(tr_subj, fall_subjects, tcfg["val_subject_frac"], tcfg["seed"] + k)
        in_subj = tr_subj - va_subj
        assert not (te_subj & tr_subj) and not (va_subj & in_subj), "subject leak"

        tr_trials = [t for t in trials if t["subject"] in in_subj]
        va_trials = [t for t in trials if t["subject"] in va_subj]
        te_trials = [t for t in trials if t["subject"] in te_subj]

        tr_w = trials_to_windows(tr_trials, window, w["adl_step"], tuple(w["train_peak_positions"]))
        va_w = trials_to_windows(va_trials, window, w["adl_step"], (w["eval_peak_position"],))
        te_w = trials_to_windows(te_trials, window, w["adl_step"], (w["eval_peak_position"],))

        norm = Normalizer().fit(tr_w["X"])
        aug = Augmenter(channels, seed=tcfg["seed"] + k, **tcfg["augment"]) if tcfg.get("augment") else None
        model = build_model(cfg["model"]["name"], len(channels), 2, window,
                            **(cfg["model"].get("params") or {})).to(device)
        model, _, _ = train_one(model, tr_w["X"], tr_w["y"], va_w["X"], va_w["y"],
                                norm, aug, tcfg, 2, device)

        score = model_score_fn(model, norm, window, step, device)
        chosen = pick_threshold(score_trials(va_trials, score), window,
                                ev["target_recall"], **ev_kw)
        thr = chosen["threshold"]
        thresholds.append(thr)

        te_cache = score_trials(te_trials, score)
        test_caches.extend(te_cache)
        tuned_rows.append(event_metrics(te_cache, thr, window, **ev_kw))
        fixed_rows.append(event_metrics(te_cache, 0.5, window, **ev_kw))
        base_rows.append(event_metrics(
            score_trials(te_trials, threshold_score_fn(window, step, cfg["sampling_rate_hz"],
                                                       **cfg["threshold_baseline"])),
            0.5, window, **ev_kw))
        win_f1.append(f1_score(te_w["y"], predict(model, norm(te_w["X"]), device), average="macro"))
        r = tuned_rows[-1]
        print(f"  fold {k + 1}: thr={thr:.2f} recall={r['recall']:.3f} "
              f"FA/h={r['fa_per_hour']:.2f} window-F1={win_f1[-1]:.4f}")

    results = {
        "cnn_tuned": {**sum_metrics(tuned_rows), "thresholds": thresholds},
        "cnn_thr_0.5": sum_metrics(fixed_rows),
        "threshold_baseline": sum_metrics(base_rows),
        "window_macro_f1_mean": float(np.mean(win_f1)),
        "window_macro_f1_std": float(np.std(win_f1)),
        "params": count_params(model),
        "cpu_latency_ms": float(cpu_latency_ms(model, window, len(channels))),
        "channels": channels,
        "target_recall": ev["target_recall"],
    }

    # Trade-off curve on pooled test predictions (common thresholds across folds)
    grid = np.round(np.linspace(0.05, 0.99, 48), 3)
    curve = [event_metrics(test_caches, t, window, **ev_kw) for t in grid]
    results["tradeoff_curve"] = [{k: c[k] for k in ("threshold", "recall", "fa_per_hour")} for c in curve]
    plot_tradeoff(curve, results["threshold_baseline"], out / "fall_tradeoff.png")

    torch.save({"model_name": cfg["model"]["name"], "state_dict": model.state_dict(),
                "config": cfg["model"].get("params") or {}, "dataset": "sisfall",
                "dataset_kwargs": {"window": window, "adl_step": w["adl_step"],
                                   "peak_positions": [w["eval_peak_position"]]},
                "channels": channels, "classes": CLASSES, "seq_len": window,
                "norm_mean": norm.mean, "norm_std": norm.std, "threshold": thresholds[-1],
                "train_subjects": sorted(in_subj), "test_subjects": sorted(te_subj)},
               out / "fall_last_fold.pt")

    (out / "results.json").write_text(json.dumps(results, indent=2))
    table = fall_table(results)
    (out / "results.md").write_text(table)
    print(table)
    return results


def plot_tradeoff(curve, baseline, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([c["fa_per_hour"] for c in curve], [c["recall"] for c in curve], marker=".", label="CNN (threshold sweep)")
    ax.scatter([baseline["fa_per_hour"]], [baseline["recall"]], color="crimson", zorder=3,
               label="Impact + stillness rule")
    ax.set_xlabel("False alarms per hour of ADL")
    ax.set_ylabel("Fall recall (event level)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Fall detection trade-off (unseen subjects)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fall_table(res):
    lines = ["| Detector | Fall recall | False alarms / h (all ADL) | False alarms / h (elderly ADL) |",
             "|---|---|---|---|"]
    names = {"cnn_tuned": f"CNN, threshold tuned for {res['target_recall']:.0%} recall on val subjects",
             "cnn_thr_0.5": "CNN, threshold 0.5",
             "threshold_baseline": "Impact + stillness rule"}
    fmt = lambda v: "n/a" if v != v else f"{v:.2f}"
    for key, label in names.items():
        r = res[key]
        lines.append(f"| {label} | {r['recall']:.3f} ({r['detected']}/{r['n_falls']}) "
                     f"| {fmt(r['fa_per_hour'])} | {fmt(r['fa_per_hour_elderly'])} |")
    lines.append(f"\nWindow-level macro-F1: {res['window_macro_f1_mean']:.4f} ± {res['window_macro_f1_std']:.4f} "
                 f"| {res['params']:,} params | {res['cpu_latency_ms']:.2f} ms/window (CPU)")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fall.yaml")
    args = p.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
