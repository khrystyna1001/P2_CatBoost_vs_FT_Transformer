# M5 — Deployment: ONNX export, FastAPI service, Docker, latency

## What is served

For each of the three data sets the service holds the best model of each part: **CatBoost** (Part A, native `.cbm`) and
the tuned **FT-Transformer, seed 0** (Part B, ONNX, opset 17, run with ONNX Runtime on CPU, no PyTorch in the image).
The ONNX graph has two outputs: the logit and the [CLS]-to-feature attention (mean over heads and layers), used by
`/explain`. Categorical pre-processing (vocabulary with index 0 for unseen/rare levels, quantile transform) is the same
code as in training (`src/part_b/preprocessing.py`, fitted on the train split only, shipped as `preprocessor.json`).

Build the artefacts (models are release assets, not in git: `models/` is ignored):

```bash
uv run python src/part_a/tune.py && uv run python src/part_b/tune.py   # trains CatBoost / FT-T (hours on CPU)
uv run python src/part_b/export_onnx.py                                # ONNX + models/<ds>/meta.json
```

`export_onnx.py` checks ONNX Runtime against PyTorch on 1 000 test rows: **max |Δlogit| = 1.9e-6 (adult, bank-marketing),
2.9e-6 (Amazon)**, far inside the 1e-4 tolerance.

## API (`api/main.py`)

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness, which data sets are loaded |
| `GET /model-info` | per data set and part: model file SHA-256, training-data SHA-256, #train rows, test ROC-AUC / AP, ONNX-vs-torch difference |
| `POST /predict` | `{"dataset", "model": "part_a"\|"part_b", "records": [{feature: raw value}, ...]}` → probabilities, plus per record the categorical values outside the training vocabulary (`unseen_categories`) |
| `POST /explain` | part_a: CatBoost SHAP values (log-odds) per feature; part_b: [CLS] attention per feature (≤ 100 records) |

Missing values (`null`) are accepted; a missing feature, unknown data set or an empty batch gives HTTP 422. Unseen
categories do not fail: Part B maps them to embedding index 0 (as in training), CatBoost falls back to its prior statistic;
the response flags them, which is the serving-side counterpart of the seen/unseen analysis in M4.
Versioning: the response of `/model-info` pins which model and which training data produced a prediction.

```bash
uv run uvicorn api.main:app --port 8000
curl -s localhost:8000/model-info | head
uv run python api/load_test.py --url http://localhost:8000 --n 200      # → results/m5_latency_http.csv
```

Tests: `tests/test_api.py` (17 tests: both parts × three data sets, batch == single-record results, unseen category flagged
and served, `null` handling, 422 cases, `/explain`). All 24 tests of the repository pass; the API tests are skipped when
`models/` has not been built.

## Docker

```bash
docker build -f api/Dockerfile -t p2-api .     # from repo root, models/ must exist
docker run --rm -p 8000:8000 p2-api
```

`python:3.12-slim`, pinned runtime requirements (`api/requirements.txt`: FastAPI, ONNX Runtime, CatBoost, pandas; no PyTorch),
one worker, `ORT_THREADS=1`, `HEALTHCHECK` on `/health`. Only `api/`, the pre-processing module and `models/` are copied.

**Not built in the authoring environment**: Docker Hub was unreachable there, so the image itself was not built or run.
What was checked instead: the exact set of files the Dockerfile copies (`api/`, `src/part_b/preprocessing.py`, `models/`
without the PyTorch checkpoints) was assembled in a clean directory and the API started from it in isolated mode
(`python -I`); `/health` and a `/predict` call with an unseen category succeeded. The versions in `requirements.txt`
are those the tests ran with. **Please run the two commands above once on your machine** and put the result in the report.

## Latency

Measured on the 2-vCPU sandbox, 1 thread, no GPU. The load test sends real test-split rows through HTTP on localhost
(JSON parsing + pre-processing + inference + JSON response), 200 requests at batch 1, 20 at batch 1 000
(`results/m5_latency_http.csv`).

| Data set | Model | HTTP p50 / p95, batch 1 (ms) | HTTP p50, batch 1 000 (ms) | rows/s at batch 1 000 |
|---|---|---|---|---|
| adult | CatBoost | 6.4 / 8.1 | 45.8 | 21 953 |
| adult | FT-T (ONNX) | 9.2 / 12.8 | 152.0 | 6 678 |
| bank-marketing | CatBoost | 6.8 / 9.7 | 50.0 | 19 518 |
| bank-marketing | FT-T (ONNX) | 9.6 / 10.2 | 162.3 | 6 152 |
| Amazon | CatBoost | 5.8 / 9.1 | 44.5 | 23 067 |
| Amazon | FT-T (ONNX) | 8.2 / 11.8 | 68.1 | 14 430 |

Model-only latency (in-process, `results/cost.csv`, re-run in M5 with the ONNX rows added; timings differ by a few
percent from the M4 table because of run-to-run noise):

| Data set | CatBoost b=1 p50 (ms) | FT-T PyTorch b=1 | FT-T ONNX b=1 | CatBoost b=1000 | FT-T PyTorch b=1000 | FT-T ONNX b=1000 |
|---|---|---|---|---|---|---|
| adult | 0.31 | 3.47 | 2.60 | 4.0 | 154 | 96 |
| bank-marketing | 0.29 | 3.08 | 3.17 | 3.9 | 97 | 110 |
| Amazon | 0.17 | 3.47 | 2.77 | 2.8 | 42.7 | 33.4 |

Reading: through HTTP both models answer a single record in under 13 ms at p95, dominated by the web stack (JSON,
pandas frame construction), not by the model. The FT-Transformer is about 10–20× slower than CatBoost in the model itself at
batch 1 and 15–40× at batch 1 000 (PyTorch; HTTP: 3× at batch 1 000 on adult, since the web stack adds a fixed cost to both). ONNX Runtime
reduces the FT-Transformer's latency by up to 38 % (adult, batch 1 000), but not consistently: on bank-marketing the
ONNX is not faster (110 vs 97 ms at batch 1 000). For an accuracy gain of ≤ 0.012 AUC on adult and none on bank-marketing,
CatBoost is the cheaper choice to serve; the FT-Transformer is only justified where its extras matter.

## Limitations

* Docker image not built here (see above). No authentication, rate limiting or TLS: this is a coursework service.
* The service serves one seed per model; the 5-seed variability reported in M4 is not reflected in `/predict`.
* Latencies are from a shared 2-vCPU machine over localhost; absolute numbers on other hardware will differ, ratios are the
  point.
* `/explain` for Part B returns attention, which M4 showed is only moderately related to SHAP (Kendall τ 0.36–0.61) and
  unstable across seeds. It is labelled as attention in the response and must not be read as feature attribution.
