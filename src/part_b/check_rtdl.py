"""M3 — check our FT-Transformer against rtdl_revisiting_models.FTTransformer.

Both implementations are trained to convergence (early stopping on validation ROC-AUC) with the same
architecture size, optimiser, data and seeds. With --ours-from-baseline our rows are taken from the M3
baseline run (src/part_b/baseline.py), whose configuration is identical, so only rtdl is trained here.
The check passes if the difference of mean validation ROC-AUC is within the seed noise
(|diff| <= 2 * pooled seed sd, with a floor of 0.002 AUC).

Run from repo root:  uv run python src/part_b/check_rtdl.py [--config configs/m3_rtdl_check.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from rtdl_revisiting_models import FTTransformer as RTDLFTTransformer

from data.datasets import load_dataset
from src.part_b.models import FTTransformer, count_parameters
from src.part_b.train import prepare, set_seed, train


class RTDLWrapper(nn.Module):
    def __init__(self, n_num, cardinalities, d, n_blocks, n_heads, attn_dropout, ffn_dropout, residual_dropout):
        super().__init__()
        self.net = RTDLFTTransformer(
            n_cont_features=n_num, cat_cardinalities=cardinalities, d_out=1, n_blocks=n_blocks, d_block=d,
            attention_n_heads=n_heads, attention_dropout=attn_dropout, ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3, ffn_dropout=ffn_dropout, residual_dropout=residual_dropout,
        )

    def forward(self, x_num, x_cat):
        return self.net(x_num if x_num.shape[1] else None, x_cat if x_cat.shape[1] else None).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "m3_rtdl_check.yaml"))
    ap.add_argument("--ours-from-baseline", action="store_true",
                    help="take our rows from results/m3_part_b_baseline.csv instead of retraining")
    ap.add_argument("--reuse-rtdl", action="store_true",
                    help="keep the rtdl rows already in the output CSV and only rerun our implementation")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    arch = cfg["architecture"]
    rows = []
    out = ROOT / cfg["output"]
    if args.reuse_rtdl and out.exists():
        prev = pd.read_csv(out)
        rows = prev[prev.impl == "rtdl"].to_dict("records")
    if args.ours_from_baseline:
        base = pd.read_csv(ROOT / "results" / "m3_part_b_baseline.csv")
        base = base[(base.model == "ft_transformer") & base.dataset.isin(cfg["datasets"]) & base.seed.isin(cfg["seeds"])]
        rows += [dict(dataset=r.dataset, impl="ours", seed=int(r.seed), val_auc=r.roc_auc, best_epoch=int(r.best_epoch),
                      n_params=int(r.n_params), train_time_s=r.train_time_s) for r in base.itertuples()]
    for ds_name in cfg["datasets"]:
        data = prepare(load_dataset(ds_name))
        n_num, cards = data.train.x_num.shape[1], data.pre.cardinalities
        for impl in ("ours", "rtdl"):
            if (impl == "rtdl" and args.reuse_rtdl) or (impl == "ours" and args.ours_from_baseline):
                continue
            for seed in cfg["seeds"]:
                set_seed(seed)
                if impl == "ours":
                    model = FTTransformer(n_num, cards, **arch)
                else:
                    model = RTDLWrapper(n_num, cards, **arch)
                r = train("ft_transformer", cfg["optim"], data, seed, max_epochs=cfg["max_epochs"],
                          patience=cfg["patience"], model=model)
                rows.append(dict(dataset=ds_name, impl=impl, seed=seed, val_auc=r.best_val_auc,
                                 best_epoch=r.best_epoch, n_params=count_parameters(model),
                                 train_time_s=round(r.train_time_s, 1)))
                print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    for ds_name, g in df.groupby("dataset"):
        a = g[g.impl == "ours"].val_auc
        b = g[g.impl == "rtdl"].val_auc
        diff = a.mean() - b.mean()
        tol = max(2 * np.sqrt((a.var() + b.var()) / 2), 0.002)
        status = "PASS" if abs(diff) <= tol else "FAIL"
        print(f"{ds_name}: ours {a.mean():.4f}±{a.std():.4f}  rtdl {b.mean():.4f}±{b.std():.4f}  "
              f"diff={diff:+.4f} tol={tol:.4f} -> {status}")


if __name__ == "__main__":
    main()
