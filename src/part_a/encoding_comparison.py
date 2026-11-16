"""Part A step 1 — encoding comparison (AML S2, S7).

Three pipelines on the fixed train/val/test splits:
  1. one-hot + logistic regression
  2. ordinal encoding + HistGradientBoosting
  3. CatBoost with native categorical features

(LightGBM omitted pending OpenMP install.)

Run from repo root:  uv run python src/part_a/encoding_comparison.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from catboost import CatBoostClassifier

from data.datasets import Dataset, load_dataset
from data.splits import load_split
from src.eval.metrics import avg_precision, roc_auc

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def build_pipelines(ds: Dataset) -> dict[str, tuple[object, str]]:
    cat, num = ds.cat_cols, ds.num_cols

    onehot_lr = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat),
                        ("num", StandardScaler(), num),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    ordinal_hgb = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                            cat,
                        ),
                        ("num", "passthrough", num),
                    ]
                ),
            ),
            ("clf", HistGradientBoostingClassifier(random_state=0)),
        ]
    )

    catboost = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2000,
        early_stopping_rounds=100,
        random_seed=0,
        allow_writing_files=False,
        verbose=0,
    )

    return {
        "onehot_lr": (onehot_lr, "pipeline"),
        "ordinal_hgb": (ordinal_hgb, "pipeline"),
        "catboost": (catboost, "catboost"),
    }


def evaluate_model(name: str, model, kind: str, ds: Dataset, splits: dict) -> list[dict]:
    X, y = ds.X, ds.y
    tr, va, te = splits["train"], splits["val"], splits["test"]

    t0 = time.time()
    if kind == "catboost":
        frames = [X.iloc[idx].copy() for idx in (tr, va, te)]
        for frame in frames:
            for col in ds.cat_cols:
                frame[col] = frame[col].astype("category")
        model.set_params(cat_features=[frames[0].columns.get_loc(c) for c in ds.cat_cols])
        model.fit(frames[0], y[tr], eval_set=(frames[1], y[va]))
        p_val = model.predict_proba(frames[1])[:, 1]
        p_test = model.predict_proba(frames[2])[:, 1]
    else:
        cols = ds.num_cols + ds.cat_cols
        model.fit(X.iloc[tr][cols], y[tr])
        p_val = model.predict_proba(X.iloc[va][cols])[:, 1]
        p_test = model.predict_proba(X.iloc[te][cols])[:, 1]
    fit_time = time.time() - t0

    rows = []
    for split_name, y_true, p in (("val", y[va], p_val), ("test", y[te], p_test)):
        rows.append(
            {
                "dataset": ds.name,
                "model": name,
                "split": split_name,
                "roc_auc": roc_auc(y_true, p),
                "avg_precision": avg_precision(y_true, p),
                "fit_time_s": round(fit_time, 1),
            }
        )
    return rows


def main() -> None:
    all_rows: list[dict] = []

    for ds_name in ("adult", "bank-marketing", "amazon_employee_access"):
        ds = load_dataset(ds_name)
        splits = load_split(ds_name)
        ds_rows: list[dict] = []

        for pname, (model, kind) in build_pipelines(ds).items():
            rows = evaluate_model(pname, model, kind, ds, splits)
            all_rows.extend(rows)
            ds_rows.extend(rows)
            print(
                f"{ds_name} {pname}: "
                + ", ".join(f"{r['split']} AUC={r['roc_auc']:.4f} AP={r['avg_precision']:.4f}" for r in rows)
            )

        table = pd.DataFrame(ds_rows).pivot_table(index="model", columns="split", values="roc_auc")
        print(f"\n{ds_name} — validation/test ROC-AUC\n{table.round(4)}\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "encoding_comparison.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
