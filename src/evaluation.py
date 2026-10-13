"""Shared metrics and plotting used by baseline and deep training."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def per_class_f1(cm):
    cm = np.asarray(cm, dtype=float)
    return np.diag(cm) * 2 / (cm.sum(0) + cm.sum(1) + 1e-12)


def plot_cm(cm, classes, title, path):
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def results_table(results, extra_cols=()):
    header = "| Model | Macro-F1 | Accuracy |" + "".join(f" {c} |" for c in extra_cols)
    sep = "|---|---|---|" + "---|" * len(extra_cols)
    lines = [header, sep]
    for name, r in results.items():
        row = (f"| {name} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} "
               f"| {r['accuracy_mean']:.4f} |")
        for c in extra_cols:
            v = r.get(c)
            row += f" {v:.2f} |" if isinstance(v, float) else f" {v} |"
        lines.append(row)
    return "\n".join(lines) + "\n"
