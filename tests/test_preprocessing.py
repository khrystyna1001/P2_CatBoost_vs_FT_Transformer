"""Leakage / correctness checks for the Part B pre-processing (no data download needed)."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.part_b.models import FTTransformer, MLPWithEmbeddings, embedding_dim
from src.part_b.preprocessing import UNK, TabularPreprocessor


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"x": rng.normal(size=500), "c": rng.choice(["a"] * 20 + ["b"] * 20 + ["rare"], 500)})
    tr.loc[:2, "c"] = ["r1", "r2", "r3"]
    te = pd.DataFrame({"x": [0.0, 100.0, -100.0], "c": ["a", "never_seen", "r1"]})
    return tr, te


def test_vocab_built_on_train_only_and_unseen_maps_to_zero(frames):
    tr, te = frames
    pre = TabularPreprocessor(["x"], ["c"], min_count=10).fit(tr)
    assert "never_seen" not in pre.vocabs["c"]
    assert "r1" not in pre.vocabs["c"]  # 1 training row < min_count -> shares index 0
    codes = pre.transform_cat(te)[:, 0]
    assert codes[0] != UNK and codes[1] == UNK and codes[2] == UNK
    assert pre.cardinalities == [len(pre.vocabs["c"]) + 1]


def test_quantile_transform_is_fitted_on_train_and_bounded(frames):
    tr, te = frames
    pre = TabularPreprocessor(["x"], ["c"]).fit(tr)
    z = pre.transform_num(te)[:, 0]
    assert abs(z[0]) < 0.5 and z[1] > 4 and z[2] < -4 and np.isfinite(z).all()
    ztr = pre.transform_num(tr)[:, 0]
    assert abs(ztr.mean()) < 0.1 and abs(ztr.std() - 1) < 0.1


def test_serialisation_roundtrip(frames, tmp_path):
    tr, te = frames
    pre = TabularPreprocessor(["x"], ["c"]).fit(tr)
    pre.save(tmp_path / "p.json")
    pre2 = TabularPreprocessor.load(tmp_path / "p.json")
    np.testing.assert_array_equal(pre.transform_cat(te), pre2.transform_cat(te))
    np.testing.assert_allclose(pre.transform_num(te), pre2.transform_num(te))


def test_models_forward_shapes():
    xn, xc = torch.randn(4, 2), torch.randint(0, 5, (4, 3))
    assert MLPWithEmbeddings(2, [5, 5, 5])(xn, xc).shape == (4,)
    ftt = FTTransformer(2, [5, 5, 5], d=32, n_blocks=2, n_heads=4).eval()
    logit, att = ftt(xn, xc, return_attention=True)
    assert logit.shape == (4,) and att.shape == (4, 2, 4, 6, 6)
    torch.testing.assert_close(att.sum(-1), torch.ones_like(att.sum(-1)))
    assert embedding_dim(3) == 2 and embedding_dim(7518) == 50


def test_attention_path_matches_plain_forward_and_uses_nn_transformer_encoder():
    torch.manual_seed(0)
    xn, xc = torch.randn(6, 2), torch.randint(0, 5, (6, 3))
    ftt = FTTransformer(2, [5, 5, 5], d=32, n_blocks=3, n_heads=4).eval()
    assert isinstance(ftt.encoder, torch.nn.TransformerEncoder) and len(ftt.encoder.layers) == 3
    with torch.no_grad():
        plain = ftt(xn, xc)
        logit, att = ftt(xn, xc, return_attention=True)
    torch.testing.assert_close(plain, logit, atol=1e-5, rtol=1e-4)
    assert att.shape == (6, 3, 4, 6, 6)
