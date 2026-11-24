"""Part A pipelines: four encoding strategies + one deliberately leaky pipeline.

Every learned transformation (scaler, encoder, target statistics) lives inside
a scikit-learn Pipeline (or inside CatBoost), so it is fitted on training rows
only. The only exception is the ``naive_te_lgbm`` pipeline, which exists to
demonstrate target leakage (AML S2 / brief Part A step 2).

Pipelines
---------
onehot_lr       one-hot encoding + standardised numericals + logistic regression
ordinal_hgb     ordinal encoding + HistGradientBoosting
te_lgbm         cross-fitted sklearn TargetEncoder + LightGBM
catboost        CatBoost with native categorical features (ordered target stats)
naive_te_lgbm   target mean per category on *all* training rows (no cross-fit) + LightGBM
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

PIPELINES = ("onehot_lr", "ordinal_hgb", "te_lgbm", "catboost")
LEAKY_PIPELINES = ("naive_te_lgbm",)
# Pipelines whose fit is random (seeded); onehot_lr (lbfgs) is deterministic.
RANDOMIZED = {"ordinal_hgb", "te_lgbm", "catboost", "naive_te_lgbm"}


class NaiveTargetEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by the mean target of *all* training rows with it.

    No cross-fitting and no smoothing: a training row's own label contributes
    to its encoding, so rare categories leak their label. Unseen categories get
    the global prior. Used only for the leakage demonstration.
    """

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float)
        self.prior_ = float(y.mean())
        self.maps_ = {}
        for col in X.columns:
            self.maps_[col] = pd.Series(y, index=X[col].astype(str).values).groupby(level=0).mean()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.empty(X.shape, dtype=float)
        for j, col in enumerate(X.columns):
            out[:, j] = X[col].astype(str).map(self.maps_[col]).fillna(self.prior_).to_numpy()
        return out


def _lgbm(params: dict, seed: int) -> LGBMClassifier:
    base = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=2)
    base.update(params)
    return LGBMClassifier(random_state=seed, **base)


def build_pipeline(name: str, cat: list[str], num: list[str], params: dict | None = None, seed: int = 0):
    """Return an unfitted estimator for pipeline ``name`` with hyper-parameters ``params``."""
    params = dict(params or {})

    if name == "onehot_lr":
        min_freq = params.pop("min_frequency", None)
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq), cat),
                ("num", StandardScaler(), num),
            ]
        )
        clf = LogisticRegression(max_iter=2000, C=params.pop("C", 1.0))
        return Pipeline([("preprocess", pre), ("clf", clf)])

    if name == "ordinal_hgb":
        pre = ColumnTransformer(
            [
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
                ("num", "passthrough", num),
            ]
        )
        clf = HistGradientBoostingClassifier(random_state=seed, **params)
        return Pipeline([("preprocess", pre), ("clf", clf)])

    if name == "te_lgbm":
        # TargetEncoder.fit_transform cross-fits (5 folds) on the training rows;
        # transform() at prediction time uses the encoding learned on all training rows.
        smooth = params.pop("te_smooth", "auto")
        pre = ColumnTransformer(
            [
                ("cat", TargetEncoder(target_type="binary", smooth=smooth, cv=5, random_state=seed), cat),
                ("num", "passthrough", num),
            ]
        )
        return Pipeline([("preprocess", pre), ("clf", _lgbm(params, seed))])

    if name == "naive_te_lgbm":
        pre = ColumnTransformer([("cat", NaiveTargetEncoder(), cat), ("num", "passthrough", num)])
        return Pipeline([("preprocess", pre), ("clf", _lgbm(params, seed))])

    if name == "catboost":
        base = dict(iterations=2000, learning_rate=0.08, early_stopping_rounds=100)
        base.update(params)
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            allow_writing_files=False,
            verbose=0,
            thread_count=2,
            **base,
        )

    raise ValueError(f"unknown pipeline {name!r}")


def to_catboost_frame(X: pd.DataFrame, cat: list[str]) -> pd.DataFrame:
    X = X.copy()
    for col in cat:
        X[col] = X[col].astype(str)
    return X


@dataclass
class FitResult:
    model: object
    fit_time_s: float
    best_iteration: int | None = None


def fit_pipeline(name, model, X_tr, y_tr, X_va, y_va, cat, num) -> FitResult:
    """Fit a pipeline. CatBoost uses the validation split for early stopping only."""
    cols = num + cat
    t0 = time.perf_counter()
    if name == "catboost":
        model.set_params(cat_features=[cols.index(c) for c in cat])
        model.fit(to_catboost_frame(X_tr[cols], cat), y_tr, eval_set=(to_catboost_frame(X_va[cols], cat), y_va))
        best = int(model.get_best_iteration())
    else:
        model.fit(X_tr[cols], y_tr)
        best = None
    return FitResult(model, time.perf_counter() - t0, best)


def predict_proba(name, model, X, cat, num) -> np.ndarray:
    cols = num + cat
    if name == "catboost":
        return model.predict_proba(to_catboost_frame(X[cols], cat))[:, 1]
    return model.predict_proba(X[cols])[:, 1]
