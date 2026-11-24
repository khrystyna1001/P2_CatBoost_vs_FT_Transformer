"""Seen / rare / unseen category slices for the highest-cardinality feature."""

from __future__ import annotations

import numpy as np
import pandas as pd

UNSEEN_TOKEN = "<UNSEEN_STRESS_TEST>"
RARE_THRESHOLD = 10  # same threshold as the Part B vocabulary (index 0)


def highest_cardinality_feature(X_train: pd.DataFrame, cat_cols: list[str]) -> str:
    """Categorical column with the most distinct levels in the TRAINING rows."""
    return max(cat_cols, key=lambda c: X_train[c].nunique())


def category_slices(X_train: pd.DataFrame, X_eval: pd.DataFrame, feature: str) -> np.ndarray:
    """Label every eval row 'seen' (>= 10 training rows), 'rare' (1-9) or 'unseen' (0)."""
    counts = X_train[feature].astype(str).value_counts()
    c = X_eval[feature].astype(str).map(counts).fillna(0).to_numpy()
    return np.where(c == 0, "unseen", np.where(c < RARE_THRESHOLD, "rare", "seen"))
