"""Export the OpenML datasets (from the local sklearn cache) to data/raw/.

Used when OpenML is unreachable from the machine that runs the experiments.
data/raw/ is gitignored: data is never committed. Run from repo root:

    uv run python data/export_raw.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import OPENML_SPECS, RAW_DIR, load_dataset


def main() -> None:
    RAW_DIR.mkdir(exist_ok=True)
    for name in OPENML_SPECS:
        ds = load_dataset(name, use_local=False)
        pd.to_pickle(ds, RAW_DIR / f"{name}.pkl")
        print(f"{name}: {ds.X.shape} -> {RAW_DIR / (name + '.pkl')}")


if __name__ == "__main__":
    main()
