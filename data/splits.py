"""Stratified train/validation/test split generation.

Protocol (see project brief): one 60/20/20 stratified split per dataset,
seed 2026, indices saved to data/splits/<dataset>.npz. Indices refer to row
positions in the frame returned by data.datasets.load_dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

SPLIT_SEED = 2026
FRACTIONS = (0.6, 0.2, 0.2)  # train / validation / test
SPLIT_DIR = Path(os.environ.get("P2_SPLIT_DIR", Path(__file__).resolve().parent / "splits"))


def make_split_indices(y: np.ndarray, seed: int = SPLIT_SEED) -> dict[str, np.ndarray]:
    """Return stratified train/val/test index arrays for labels y."""
    tr_frac, va_frac, _ = FRACTIONS
    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        idx, test_size=1 - tr_frac - va_frac, random_state=seed, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=va_frac / (tr_frac + va_frac),
        random_state=seed,
        stratify=y[train_val_idx],
    )
    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}


def save_split(dataset_name: str, indices: dict[str, np.ndarray]) -> Path:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLIT_DIR / f"{dataset_name}.npz"
    np.savez(path, **indices)
    return path


def load_split(dataset_name: str) -> dict[str, np.ndarray]:
    path = SPLIT_DIR / f"{dataset_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Split file {path} not found; run `uv run python data/make_splits.py`")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
