"""Shared evaluation metrics and statistical comparison helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def avg_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def paired_bootstrap_diff(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn=roc_auc,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap CI for metric(score_a) - metric(score_b).

    Resamples test rows with replacement; both models are scored on the same
    resample so the comparison is paired.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        diffs[i] = metric_fn(y_true[idx], score_a[idx]) - metric_fn(y_true[idx], score_b[idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_greater": float((diffs > 0).mean()),
    }
