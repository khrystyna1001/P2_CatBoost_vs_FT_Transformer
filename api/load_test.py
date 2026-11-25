"""Load test for POST /predict: p50 / p95 latency per dataset, model and batch size.

Start the API first (locally or in Docker), then:

    uv run python api/load_test.py --url http://localhost:8000 --n 200

Rows are sampled from each data set's TEST split (raw feature values, exactly
what a client would send). Results go to results/m5_latency_http.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.datasets import load_dataset
from data.splits import load_split


def records(ds_name: str, n: int) -> list[dict]:
    ds = load_dataset(ds_name)
    idx = load_split(ds_name)["test"][:n]
    X = ds.X.iloc[idx]
    X = X.astype(object).where(X.notna(), None)
    return X.to_dict("records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200, help="requests per (dataset, model, batch size)")
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 1000])
    ap.add_argument("--out", default=str(ROOT / "results" / "m5_latency_http.csv"))
    args = ap.parse_args()

    rows = []
    with httpx.Client(base_url=args.url, timeout=60) as client:
        datasets = client.get("/health").json()["datasets_loaded"]
        for ds_name in datasets:
            pool = records(ds_name, max(args.batch_sizes))
            for model in ("part_a", "part_b"):
                for bs in args.batch_sizes:
                    n = args.n if bs == 1 else max(20, args.n // 10)
                    payload_rows = [pool[i % len(pool)] for i in range(bs)]
                    body = {"dataset": ds_name, "model": model, "records": payload_rows}
                    for _ in range(5):  # warm-up
                        client.post("/predict", json=body).raise_for_status()
                    lat = []
                    for i in range(n):
                        if bs == 1:
                            body["records"] = [pool[i % len(pool)]]
                        t0 = time.perf_counter()
                        client.post("/predict", json=body).raise_for_status()
                        lat.append((time.perf_counter() - t0) * 1000)
                    lat = np.array(lat)
                    rows.append(dict(dataset=ds_name, model=model, batch_size=bs, n_requests=n,
                                     p50_ms=np.percentile(lat, 50), p95_ms=np.percentile(lat, 95),
                                     mean_ms=lat.mean(), rows_per_s=bs * 1000 / lat.mean()))
                    print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.round(2).to_string())


if __name__ == "__main__":
    main()
