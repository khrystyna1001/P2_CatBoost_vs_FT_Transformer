"""Numbers for docs/leakage_checklist.md: split sizes, class balance, duplicate
feature rows shared across splits (L1.4) and unseen-category rates.

Run from repo root:  uv run python src/eval/leakage_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.datasets import load_dataset
from data.splits import load_split
from src.eval.slices import category_slices, highest_cardinality_feature

DATASETS = ["adult", "bank-marketing", "amazon_employee_access"]


def main() -> None:
    rows = []
    for name in DATASETS:
        ds = load_dataset(name)
        sp = load_split(name)
        X = ds.X
        key = pd.util.hash_pandas_object(X, index=False)
        tr_keys = set(key.iloc[sp["train"]])
        tr = X.iloc[sp["train"]]
        hc = highest_cardinality_feature(tr, ds.cat_cols)
        row = dict(dataset=name, n_rows=len(X), n_num=len(ds.num_cols), n_cat=len(ds.cat_cols),
                   max_cardinality_feature=hc, max_cardinality=int(X[hc].nunique()),
                   positive_rate=round(float(ds.y.mean()), 4))
        assert not (set(sp["train"]) & set(sp["val"])) and not (set(sp["train"]) & set(sp["test"]))
        assert not (set(sp["val"]) & set(sp["test"]))
        for s in ("train", "val", "test"):
            row[f"n_{s}"] = len(sp[s])
            row[f"pos_rate_{s}"] = round(float(ds.y[sp[s]].mean()), 4)
        for s in ("val", "test"):
            row[f"{s}_rows_duplicating_a_train_row"] = int(key.iloc[sp[s]].isin(tr_keys).sum())
            sl = category_slices(tr, X.iloc[sp[s]], hc)
            row[f"{s}_rows_unseen_level_of_max_card_feature"] = int((sl == "unseen").sum())
            row[f"{s}_rows_rare_level_of_max_card_feature"] = int((sl == "rare").sum())
        rows.append(row)
    df = pd.DataFrame(rows).set_index("dataset").T
    out = ROOT / "results" / "leakage_audit.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out)
    print(df.to_string())


if __name__ == "__main__":
    main()
