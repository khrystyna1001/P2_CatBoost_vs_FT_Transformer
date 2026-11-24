"""The naive target encoder leaks; the cross-fitted pipeline does not."""

import numpy as np
import pandas as pd

from src.part_a.pipelines import NaiveTargetEncoder, build_pipeline


def _data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    # pure-noise high-cardinality feature: no real signal
    X = pd.DataFrame({"c": rng.integers(0, 1500, n).astype(str), "x": rng.normal(size=n)})
    y = rng.integers(0, 2, n)
    return X, y


def test_naive_encoder_memorises_labels_of_rare_levels():
    X, y = _data()
    enc = NaiveTargetEncoder().fit(X[["c"]], y)
    z = enc.transform(X[["c"]])[:, 0]
    singletons = X.c.map(X.c.value_counts()) == 1
    np.testing.assert_array_equal(z[singletons.to_numpy()], y[singletons.to_numpy()])


def test_crossfitted_pipeline_does_not_overfit_noise_as_much():
    from sklearn.metrics import roc_auc_score

    X, y = _data()
    good = build_pipeline("te_lgbm", ["c"], ["x"], dict(n_estimators=100)).fit(X, y)
    bad = build_pipeline("naive_te_lgbm", ["c"], ["x"], dict(n_estimators=100)).fit(X, y)
    auc_good = roc_auc_score(y, good.predict_proba(X)[:, 1])
    auc_bad = roc_auc_score(y, bad.predict_proba(X)[:, 1])
    assert auc_bad > 0.95 and auc_bad > auc_good
