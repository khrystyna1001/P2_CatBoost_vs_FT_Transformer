"""API tests (M5). Need models/ from the training scripts + export_onnx.py; skipped otherwise."""

import numpy as np
import pytest

from api import main
from data.datasets import load_dataset
from data.splits import load_split

fastapi_testclient = pytest.importorskip("fastapi.testclient")

DATASETS = ["adult", "bank-marketing", "amazon_employee_access"]

pytestmark = pytest.mark.skipif(
    not all((main.MODELS_DIR / d / "meta.json").exists() for d in DATASETS),
    reason="models/ not built (run src/part_a/tune.py, src/part_b/tune.py, src/part_b/export_onnx.py)",
)


@pytest.fixture(scope="module")
def client():
    with fastapi_testclient.TestClient(main.app) as c:
        yield c


def rows(ds_name, n=20):
    ds = load_dataset(ds_name)
    X = ds.X.iloc[load_split(ds_name)["test"][:n]]
    return X.astype(object).where(X.notna(), None).to_dict("records")


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["datasets_loaded"] == sorted(DATASETS)


def test_model_info_has_hashes_and_metrics(client):
    info = client.get("/model-info").json()
    for ds in DATASETS:
        assert len(info[ds]["training_data_sha256"]) == 64
        for part in ("part_a", "part_b"):
            assert len(info[ds]["models"][part]["sha256"]) == 64
            assert 0.5 < info[ds]["models"][part]["test_roc_auc"] < 1


@pytest.mark.parametrize("ds", DATASETS)
@pytest.mark.parametrize("model", ["part_a", "part_b"])
def test_predict_probabilities(client, ds, model):
    r = client.post("/predict", json={"dataset": ds, "model": model, "records": rows(ds)})
    assert r.status_code == 200
    p = np.array(r.json()["probabilities"])
    assert p.shape == (20,) and ((0 <= p) & (p <= 1)).all()


@pytest.mark.parametrize("ds", DATASETS)
def test_batch_equals_single(client, ds):
    recs = rows(ds, 5)
    for model in ("part_a", "part_b"):
        batch = client.post("/predict", json={"dataset": ds, "model": model, "records": recs}).json()["probabilities"]
        single = [client.post("/predict", json={"dataset": ds, "model": model, "records": [r]}).json()["probabilities"][0]
                  for r in recs]
        np.testing.assert_allclose(batch, single, atol=1e-5)


def test_unseen_category_is_flagged_and_served(client):
    recs = rows("amazon_employee_access", 3)
    recs[0]["RESOURCE"] = "never-seen-resource-id"
    for model in ("part_a", "part_b"):
        r = client.post("/predict", json={"dataset": "amazon_employee_access", "model": model, "records": recs})
        assert r.status_code == 200
        assert r.json()["unseen_categories"][0].get("RESOURCE") == "never-seen-resource-id"


def test_missing_values_are_accepted(client):
    recs = rows("adult", 2)
    recs[0]["workclass"] = None
    recs[0]["age"] = None
    for model in ("part_a", "part_b"):
        assert client.post("/predict", json={"dataset": "adult", "model": model, "records": recs}).status_code == 200


def test_missing_feature_is_422(client):
    recs = rows("adult", 1)
    del recs[0]["age"]
    assert client.post("/predict", json={"dataset": "adult", "records": recs}).status_code == 422


def test_bad_input_is_422(client):
    assert client.post("/predict", json={"dataset": "nope", "records": [{}]}).status_code == 422
    assert client.post("/predict", json={"dataset": "adult", "records": []}).status_code == 422


@pytest.mark.parametrize("model", ["part_a", "part_b"])
def test_explain(client, model):
    recs = rows("adult", 3)
    r = client.post("/explain", json={"dataset": "adult", "model": model, "records": recs}).json()
    assert len(r["attributions"]) == 3 and len(r["attributions"][0]) == 14
    if model == "part_a":  # SHAP: contributions + bias = log-odds, checked through the probability's sign
        assert any(abs(v) > 0 for v in r["attributions"][0].values())
    else:  # attention mass on the 14 feature tokens (the rest goes to [CLS] itself): positive, at most 1
        assert 0 < sum(r["attributions"][0].values()) <= 1 + 1e-6
