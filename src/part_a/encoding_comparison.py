"""M2 — Part A baseline: encoding comparison + leakage demonstration (AML S2, S7).

Four encoding pipelines with default hyper-parameters (see pipelines.py) plus
the deliberately leaky ``naive_te_lgbm``. Only TRAIN and VALIDATION scores are
computed here: the test split stays locked until M4 (common protocol).

Randomised pipelines are run with 3 seeds (common protocol, rule 5).

Run from repo root:  uv run python src/part_a/encoding_comparison.py [--config configs/m2_part_a_baseline.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.datasets import load_dataset
from data.splits import load_split
from src.eval.metrics import avg_precision, roc_auc
from src.part_a.pipelines import LEAKY_PIPELINES, PIPELINES, RANDOMIZED, build_pipeline, fit_pipeline, predict_proba


def run(config: dict) -> pd.DataFrame:
    rows = []
    for ds_name in config["datasets"]:
        ds = load_dataset(ds_name)
        sp = load_split(ds_name)
        X, y = ds.X, ds.y
        X_tr, X_va = X.iloc[sp["train"]], X.iloc[sp["val"]]
        y_tr, y_va = y[sp["train"]], y[sp["val"]]
        for name in list(PIPELINES) + list(LEAKY_PIPELINES):
            seeds = config["seeds"] if name in RANDOMIZED else config["seeds"][:1]
            for seed in seeds:
                model = build_pipeline(name, ds.cat_cols, ds.num_cols, seed=seed)
                fr = fit_pipeline(name, model, X_tr, y_tr, X_va, y_va, ds.cat_cols, ds.num_cols)
                for split, Xs, ys in (("train", X_tr, y_tr), ("val", X_va, y_va)):
                    p = predict_proba(name, fr.model, Xs, ds.cat_cols, ds.num_cols)
                    rows.append(
                        dict(dataset=ds_name, model=name, seed=seed, split=split,
                             roc_auc=roc_auc(ys, p), avg_precision=avg_precision(ys, p),
                             fit_time_s=round(fr.fit_time_s, 2))
                    )
                last = rows[-1]
                print(f"{ds_name:24s} {name:14s} seed={seed} val AUC={last['roc_auc']:.4f} AP={last['avg_precision']:.4f} ({fr.fit_time_s:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "m2_part_a_baseline.yaml"))
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    df = run(config)
    out = ROOT / config["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["dataset", "model", "split"])[["roc_auc", "avg_precision"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print(summary.to_string())
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
