"""Dataset loading for the P2 capstone (CatBoost vs FT-Transformer).

Datasets are fetched from OpenML via sklearn.datasets.fetch_openml and cached
locally by sklearn (~/scikit_learn_data). Data itself is never committed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.datasets import fetch_openml

# Exact OpenML versions are recorded for reproducibility.
OPENML_SPECS = {
    "adult": {"name": "adult", "version": 2},
    "bank-marketing": {"name": "bank-marketing", "version": 1},
    "amazon_employee_access": {"name": "Amazon_employee_access", "version": 1},
}

# Target column and the mapping of raw labels to binary 0/1.
# Positive class (1) is the minority/interesting class per dataset.
TARGET_SPECS = {
    "adult": {"column": "class", "positive": ">50K"},
    "bank-marketing": {"column": "Class", "positive": "2"},  # '2' = subscription 'yes'
    "amazon_employee_access": {"column": "target", "positive": "1"},
}

MISSING_TOKEN = "<MISSING>"


@dataclass(frozen=True)
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    num_cols: list[str]
    cat_cols: list[str]
    target_name: str


def load_dataset(name: str) -> Dataset:
    """Fetch an OpenML dataset and return a normalized Dataset object."""
    if name not in OPENML_SPECS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(OPENML_SPECS)}")

    spec = OPENML_SPECS[name]
    fetched = fetch_openml(name=spec["name"], version=spec["version"], as_frame=True)
    df = fetched.frame if fetched.frame is not None else fetched.data
    df = df.copy()

    tspec = TARGET_SPECS[name]
    target_col = tspec["column"]
    y_raw = df.pop(target_col)
    if not isinstance(y_raw, pd.Series):
        y_raw = pd.Series(np.asarray(y_raw), index=df.index)
    y_raw = y_raw.astype(str)
    y = (y_raw == tspec["positive"]).astype(np.int64).to_numpy()

    cat_cols = [c for c in df.columns if not is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

    # Normalize categorical columns to plain strings with an explicit missing token.
    # 'unknown'/'?' style levels are kept as-is; only true NaN gets MISSING_TOKEN.
    for col in cat_cols:
        df[col] = df[col].astype("string").fillna(MISSING_TOKEN)

    return Dataset(
        name=name,
        X=df,
        y=y,
        num_cols=num_cols,
        cat_cols=cat_cols,
        target_name=target_col,
    )
