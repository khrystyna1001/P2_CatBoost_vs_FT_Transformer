"""Shared figure style. Colours follow the model (fixed order), never the rank."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent.parent / "report" / "figures"

# Categorical palette (validated, fixed order): blue, orange, aqua, yellow, magenta, green, violet
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MODEL_COLORS = {
    "catboost": PALETTE[0],
    "ft_transformer": PALETTE[1],
    "mlp": PALETTE[2],
    "te_lgbm": PALETTE[3],
    "ordinal_hgb": PALETTE[4],
    "onehot_lr": PALETTE[5],
    "naive_te_lgbm": PALETTE[6],
}
MODEL_LABELS = {
    "catboost": "CatBoost",
    "ft_transformer": "FT-Transformer",
    "ft_transformer_mean_pool": "FT-T (mean pool)",
    "ft_transformer_no_rare_bucket": "FT-T (no rare bucket)",
    "mlp": "MLP + embeddings",
    "te_lgbm": "TargetEnc + LightGBM",
    "ordinal_hgb": "Ordinal + HistGB",
    "onehot_lr": "One-hot + LogReg",
    "naive_te_lgbm": "Naive TE + LightGBM (leaky)",
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 2, "legend.frameon": False,
    })


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    return path
