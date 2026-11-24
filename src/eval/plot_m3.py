"""M3 figures: training curves and seed variance of the Part B baselines.

Run from repo root:  uv run python src/eval/plot_m3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.plots import MODEL_COLORS, MODEL_LABELS, plt, save, setup

DATASETS = ["adult", "bank-marketing", "amazon_employee_access"]


def main() -> None:
    setup()
    curves = pd.read_csv(ROOT / "results" / "m3_training_curves.csv")
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex="col")
    for j, ds in enumerate(DATASETS):
        c = curves[curves.dataset == ds]
        for model, g in c.groupby("model"):
            color = MODEL_COLORS[model]
            for seed, s in g.groupby("seed"):
                axes[0, j].plot(s.epoch, s.train_loss, color=color, lw=1, alpha=0.35)
                axes[0, j].plot(s.epoch, s.val_loss, color=color, lw=1, ls="--", alpha=0.35)
                axes[1, j].plot(s.epoch, s.val_auc, color=color, lw=1, alpha=0.5,
                                label=MODEL_LABELS[model] if seed == 0 else None)
        axes[0, j].set_title(ds)
        axes[1, j].set_xlabel("epoch")
    axes[0, 0].set_ylabel("log-loss (solid train, dashed val)")
    axes[1, 0].set_ylabel("validation ROC-AUC")
    axes[1, 0].legend(loc="lower right")
    save(fig, "m3_training_curves")

    runs = pd.read_csv(ROOT / "results" / "m3_part_b_baseline.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for j, ds in enumerate(DATASETS):
        r = runs[runs.dataset == ds]
        models = sorted(r.model.unique())
        for i, model in enumerate(models):
            g = r[r.model == model].sort_values("seed")
            jitter = (g.seed.to_numpy() - g.seed.mean()) * 0.07  # spread seeds horizontally so ties stay visible
            axes[j].scatter(i + jitter, g.roc_auc, s=40, color=MODEL_COLORS[model], edgecolor="white", lw=1.5, zorder=3)
            axes[j].hlines(g.roc_auc.mean(), i - 0.3, i + 0.3, color=MODEL_COLORS[model], lw=2)
        short = {"ft_transformer": "FT-Transformer", "mlp": "MLP"}
        axes[j].set_xticks(range(len(models)), [short[m] for m in models])
        axes[j].set_xlim(-0.6, len(models) - 0.4)
        axes[j].set_title(ds)
    axes[0].set_ylabel("validation ROC-AUC")
    save(fig, "m3_seed_variance")
    print("saved figures to report/figures/")


if __name__ == "__main__":
    main()
