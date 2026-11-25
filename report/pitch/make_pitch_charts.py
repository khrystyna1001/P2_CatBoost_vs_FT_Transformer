"""Charts for the pitch deck (PNG, so every viewer shows them). Run: python report/pitch/make_pitch_charts.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
CAT, FTT, DARK, MUTE = "#1F7A6D", "#D9822B", "#17323A", "#5B6B70"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": "#C9D3D1"})


def bars(name, labels, a, b, ylim, title, size):
    fig, ax = plt.subplots(figsize=size, dpi=200)
    x = np.arange(len(labels)); w = 0.36
    for off, v, c, lab in ((-w / 2, a, CAT, "CatBoost"), (w / 2, b, FTT, "FT-Transformer")):
        r = ax.bar(x + off, v, w, color=c, label=lab)
        for rect, val in zip(r, v):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=11, color=DARK)
    ax.set_xticks(x, labels, color=DARK, fontsize=13); ax.set_ylim(*ylim)
    ax.set_title(title, color=MUTE, fontsize=13, loc="left")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="y", colors=MUTE)
    ax.yaxis.grid(True, color="#E3E8E7"); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / name); plt.close(fig)


bars("pitch_test_auc.png", ["adult", "bank-marketing", "Amazon"], [0.9306, 0.9392, 0.8702], [0.9182, 0.9376, 0.8074],
     (0.75, 0.98), "Test ROC-AUC (mean over seeds), axis starts at 0.75", (6.2, 4.0))
bars("pitch_seen_unseen.png", ["seen levels", "unseen levels"], [0.883, 0.825], [0.821, 0.780],
     (0.70, 0.93), "Amazon test ROC-AUC by RESOURCE level", (5.0, 4.0))
