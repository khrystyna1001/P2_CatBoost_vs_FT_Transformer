"""Cost of every final model (common protocol, rule 8).

Training time (from the M4 run logs), in-process inference latency for batch 1
and batch 1 000 (median and p95 over repeated calls, single thread), parameter
count / model size on disk.

Run from repo root after M4 (and again after M5 to include ONNX):
    uv run python src/eval/cost.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from catboost import CatBoostClassifier

from data.datasets import load_dataset
from data.splits import load_split
from src.part_a.pipelines import to_catboost_frame
from src.part_b.models import build_model, count_parameters
from src.part_b.preprocessing import TabularPreprocessor

torch.set_num_threads(1)
DATASETS = ["adult", "bank-marketing", "amazon_employee_access"]
REPEATS = {1: 200, 1000: 30}


def timeit(fn, bs):
    for _ in range(3):
        fn()
    lat = []
    for _ in range(REPEATS[bs]):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000)
    return float(np.median(lat)), float(np.percentile(lat, 95))


def main() -> None:
    rows = []
    for ds_name in DATASETS:
        mdir = ROOT / "models" / ds_name
        ds = load_dataset(ds_name)
        X_te = ds.X.iloc[load_split(ds_name)["test"][:1000]]
        pre = TabularPreprocessor.load(mdir / "preprocessor.json")
        runs = pd.concat([pd.read_csv(f) for f in (ROOT / "results").glob(f"m4_part_*_runs_{ds_name}*.csv")])
        train_time = runs.groupby("model").fit_time_s.mean()

        cb = CatBoostClassifier(thread_count=1)
        cb.load_model(str(mdir / "catboost.cbm"))
        cols = ds.num_cols + ds.cat_cols
        Xc = to_catboost_frame(X_te[cols], ds.cat_cols)
        entries = {"catboost": dict(size_mb=(mdir / "catboost.cbm").stat().st_size / 1e6,
                                    n_params=int(cb.tree_count_), param_note="trees",
                                    fn=lambda X: cb.predict_proba(X)[:, 1], data=Xc, runtime="catboost")}
        for kind in ("mlp", "ft_transformer"):
            spec = json.loads((mdir / f"{kind}.json").read_text())
            m = build_model(kind, spec["n_num"], spec["cardinalities"], spec["params"])
            m.load_state_dict(torch.load(mdir / f"{kind}.pt"))
            m.eval()

            def f(X, m=m):  # includes pre-processing, as in serving
                xn, xc = pre.transform(X)
                with torch.no_grad():
                    return torch.sigmoid(m(torch.from_numpy(xn), torch.from_numpy(xc))).numpy()

            entries[kind] = dict(size_mb=(mdir / f"{kind}.pt").stat().st_size / 1e6, n_params=count_parameters(m),
                                 param_note="parameters", fn=f, data=X_te, runtime="pytorch")
        onnx_path = mdir / "ft_transformer.onnx"
        if onnx_path.exists():
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
            names = {i.name for i in sess.get_inputs()}

            def g(X):
                xn, xc = pre.transform(X)
                return sess.run(["logit"], {k: v for k, v in (("x_num", xn), ("x_cat", xc)) if k in names})[0]

            entries["ft_transformer_onnx"] = dict(size_mb=onnx_path.stat().st_size / 1e6,
                                                  n_params=entries["ft_transformer"]["n_params"],
                                                  param_note="parameters", fn=g, data=X_te, runtime="onnxruntime")

        for name, e in entries.items():
            row = dict(dataset=ds_name, model=name, runtime=e["runtime"],
                       train_time_s=float(train_time.get(name.replace("_onnx", ""), np.nan)),
                       n_params=e["n_params"], param_note=e["param_note"], size_mb=round(e["size_mb"], 3))
            for bs in (1, 1000):
                X = e["data"].iloc[:bs]
                p50, p95 = timeit(lambda: e["fn"](X), bs)
                row[f"latency_b{bs}_p50_ms"], row[f"latency_b{bs}_p95_ms"] = p50, p95
            rows.append(row)
            print(row, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "cost.csv", index=False)
    (ROOT / "results" / "tables").mkdir(exist_ok=True)
    (ROOT / "results" / "tables" / "cost.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
