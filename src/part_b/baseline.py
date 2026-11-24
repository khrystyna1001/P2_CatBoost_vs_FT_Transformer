"""M3 — Part B baseline: MLP with embeddings and FT-Transformer, starting configs.

Trains both networks on the fixed splits with 5 seeds each, logs per-epoch
training curves and reports VALIDATION scores only (test stays locked until M4).

Run from repo root:  uv run python src/part_b/baseline.py [--config configs/m3_part_b_baseline.yaml]
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
from src.eval.metrics import avg_precision, roc_auc
from src.part_b.models import count_parameters
from src.part_b.train import predict, prepare, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "m3_part_b_baseline.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    out = ROOT / cfg["output"]
    curves_out = ROOT / cfg["curves_output"]
    rows, curves, done = [], [], set()
    if out.exists() and curves_out.exists():  # resume: runs are long on CPU
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        curves = [pd.read_csv(curves_out)]
        done = set(zip(prev.dataset, prev.model, prev.seed))
    for ds_name in cfg["datasets"]:
        data = prepare(load_dataset(ds_name), min_count=cfg["min_count"])
        for kind, params in cfg["models"].items():
            for seed in cfg["seeds"][kind]:
                if (ds_name, kind, seed) in done:
                    continue
                r = train(kind, params, data, seed, max_epochs=cfg["max_epochs"], patience=cfg["patience"])
                p_va = predict(r.model, data.val)
                yv = data.val.y.numpy()
                rows.append(dict(dataset=ds_name, model=kind, seed=seed, split="val",
                                 roc_auc=roc_auc(yv, p_va), avg_precision=avg_precision(yv, p_va),
                                 best_epoch=r.best_epoch, epochs_run=len(r.history),
                                 train_time_s=round(r.train_time_s, 1), n_params=count_parameters(r.model)))
                h = r.history.assign(dataset=ds_name, model=kind, seed=seed)
                curves.append(h)
                print(f"{ds_name:24s} {kind:15s} seed={seed} val AUC={rows[-1]['roc_auc']:.4f} "
                      f"AP={rows[-1]['avg_precision']:.4f} best_epoch={r.best_epoch} ({r.train_time_s:.0f}s)", flush=True)

                # write after every run: runs are long on CPU
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(out, index=False)
                pd.concat(curves).to_csv(curves_out, index=False)

    df = pd.DataFrame(rows)
    print(df.groupby(["dataset", "model"])[["roc_auc", "avg_precision", "best_epoch", "train_time_s"]]
          .agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
