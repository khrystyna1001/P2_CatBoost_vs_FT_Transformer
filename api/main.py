"""FastAPI service serving the best Part A model (CatBoost) and the best Part B
model (FT-Transformer, ONNX) for each of the three data sets.

Endpoints
  GET  /health       liveness + which models are loaded
  GET  /model-info   model versions (file hashes), training-data hash, test metrics
  POST /predict      {"dataset", "model": "part_a" | "part_b", "records": [...]} -> probabilities
  POST /explain      part_a: CatBoost SHAP values per feature; part_b: [CLS] attention per feature

Models are read from $MODELS_DIR (default: <repo>/models), produced by
src/part_a/tune.py, src/part_b/tune.py and src/part_b/export_onnx.py.

Run locally:  uv run uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.part_b.preprocessing import TabularPreprocessor  # noqa: E402

MISSING_TOKEN = "<MISSING>"  # same token as data/datasets.py

MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
MAX_RECORDS = 10_000


class ModelChoice(str, Enum):
    part_a = "part_a"  # CatBoost (Applied ML)
    part_b = "part_b"  # FT-Transformer (Deep Learning)


class DatasetName(str, Enum):
    adult = "adult"
    bank_marketing = "bank-marketing"
    amazon_employee_access = "amazon_employee_access"


class PredictRequest(BaseModel):
    dataset: DatasetName
    model: ModelChoice = ModelChoice.part_a
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=MAX_RECORDS,
                                          description="One dict per row: feature name -> raw value")


class PredictResponse(BaseModel):
    dataset: DatasetName
    model: ModelChoice
    model_name: str
    probabilities: list[float]
    unseen_categories: list[dict[str, str]] = Field(
        description="Per record, categorical features whose value is out of the training vocabulary "
                    "(unseen, or seen in fewer than 10 training rows). Part B maps them to embedding "
                    "index 0; CatBoost falls back towards its prior target statistic.")


class ExplainResponse(BaseModel):
    dataset: DatasetName
    model: ModelChoice
    method: str
    probabilities: list[float]
    attributions: list[dict[str, float]]


class Bundle:
    def __init__(self, path: Path):
        self.meta = json.loads((path / "meta.json").read_text())
        self.pre = TabularPreprocessor.load(path / "preprocessor.json")
        self.cols = self.pre.num_cols + self.pre.cat_cols
        self.catboost = CatBoostClassifier()
        self.catboost.load_model(str(path / "catboost.cbm"))
        self.cb_cols = list(self.catboost.feature_names_)
        self.cb_cat_idx = [self.cb_cols.index(c) for c in self.pre.cat_cols]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
        self.onnx = ort.InferenceSession(str(path / "ft_transformer.onnx"), opts, providers=["CPUExecutionProvider"])
        self.onnx_inputs = {i.name for i in self.onnx.get_inputs()}  # no x_num when there are no numericals

    def frame(self, records: list[dict]) -> pd.DataFrame:
        missing = sorted({c for r in records for c in self.cols if c not in r})
        if missing:
            raise HTTPException(422, f"missing features: {missing}")
        df = pd.DataFrame.from_records(records)[self.cols]
        for c in self.pre.num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        for c in self.pre.cat_cols:
            df[c] = df[c].where(df[c].notna(), MISSING_TOKEN).astype(str)
        return df

    def unseen(self, df: pd.DataFrame) -> list[dict[str, str]]:
        out = []
        for _, row in df[self.pre.cat_cols].iterrows():
            out.append({c: row[c] for c in self.pre.cat_cols if row[c] not in self.pre.vocabs[c]})
        return out

    def predict_a(self, df: pd.DataFrame, shap: bool = False):
        pool = Pool(df[self.cb_cols], cat_features=self.cb_cat_idx)
        p = self.catboost.predict_proba(pool)[:, 1]
        if not shap:
            return p, None
        sv = self.catboost.get_feature_importance(pool, type="ShapValues")[:, :-1]
        return p, [dict(zip(self.cb_cols, map(float, r))) for r in sv]

    def predict_b(self, df: pd.DataFrame):
        xn, xc = self.pre.transform(df)
        feed = {k: v for k, v in (("x_num", xn), ("x_cat", xc)) if k in self.onnx_inputs}
        logit, att = self.onnx.run(None, feed)
        p = 1 / (1 + np.exp(-logit))
        return p, [dict(zip(self.cols, map(float, r))) for r in att]


BUNDLES: dict[str, Bundle] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for ds in DatasetName:
        path = MODELS_DIR / ds.value
        if (path / "meta.json").exists():
            BUNDLES[ds.value] = Bundle(path)
    yield


app = FastAPI(title="P2 — CatBoost vs FT-Transformer", version="1.0.0", lifespan=lifespan)


def get_bundle(ds: DatasetName) -> Bundle:
    if ds.value not in BUNDLES:
        raise HTTPException(503, f"models for {ds.value} not loaded from {MODELS_DIR}")
    return BUNDLES[ds.value]


@app.get("/health")
def health():
    return {"status": "ok" if BUNDLES else "no_models", "datasets_loaded": sorted(BUNDLES)}


@app.get("/model-info")
def model_info():
    return {ds: b.meta for ds, b in BUNDLES.items()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    b = get_bundle(req.dataset)
    df = b.frame(req.records)
    if req.model == ModelChoice.part_a:
        p, _ = b.predict_a(df)
    else:
        p, _ = b.predict_b(df)
    return PredictResponse(dataset=req.dataset, model=req.model, model_name=b.meta["models"][req.model.value]["name"],
                           probabilities=[float(x) for x in p], unseen_categories=b.unseen(df))


@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest):
    b = get_bundle(req.dataset)
    if len(req.records) > 100:
        raise HTTPException(422, "/explain accepts at most 100 records")
    df = b.frame(req.records)
    if req.model == ModelChoice.part_a:
        p, attr = b.predict_a(df, shap=True)
        method = "CatBoost SHAP values (log-odds contribution per feature)"
    else:
        p, attr = b.predict_b(df)
        method = "FT-Transformer [CLS]-to-feature attention, mean over heads and layers (not an attribution in the SHAP sense)"
    return ExplainResponse(dataset=req.dataset, model=req.model, method=method,
                           probabilities=[float(x) for x in p], attributions=attr)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
