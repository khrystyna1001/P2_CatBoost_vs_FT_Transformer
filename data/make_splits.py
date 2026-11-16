"""Generate and save all stratified splits. Run from repo root:

    uv run python data/make_splits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import OPENML_SPECS, load_dataset
from data.splits import make_split_indices, save_split


def main() -> None:
    for name in OPENML_SPECS:
        ds = load_dataset(name)
        indices = make_split_indices(ds.y)
        path = save_split(name, indices)
        sizes = {k: len(v) for k, v in indices.items()}
        print(f"{name}: {sizes} -> {path}")


if __name__ == "__main__":
    main()
