# P2 — Categorical Data: CatBoost vs. FT-Transformer

High-cardinality categorical features are where tabular models differ most. CatBoost encodes them with ordered target statistics; FT-Transformer learns an embedding per category and lets attention combine them. Which approach generalises better, handles unseen categories more gracefully, and gives explanations that agree?

Joint capstone, Applied Machine Learning (Part A) × Deep Learning (Part B), UEK Kraków 2026/27.
Papers: Gorishniy et al. (2021), *Revisiting Deep Learning Models for Tabular Data*; Prokhorenkova et al. (2018), *CatBoost*.

## Setup

```bash
uv sync
uv run python data/make_splits.py      # splits are already committed in data/splits/ (seed 2026, 60/20/20)
# no OpenML access? export the cached data once on a machine that has it:
uv run python data/export_raw.py       # -> data/raw/*.pkl (gitignored)
```

Data sets (OpenML via `sklearn.datasets.fetch_openml`, never committed): `adult` v2, `bank-marketing` v1,
`Amazon_employee_access` v1.

## Part A — Applied ML (status: complete)

| Step | Command | Output |
|---|---|---|
| Encoding comparison + leakage demo, validation only (M2) | `uv run python src/part_a/encoding_comparison.py` | `results/m2_part_a_baseline.csv` |
| Tuning (Optuna, 12 trials/pipeline), 3-seed final fits, SHAP | `uv run python src/part_a/tune.py` | `results/m4_part_a_runs_*.csv`, `results/best_params/`, `results/optuna/` |
| Test results, leakage gap, seen/unseen, SHAP agreement | `uv run python src/part_a/analysis.py` | `results/part_a_*.csv`, `report/figures/part_a_*` |
| Leakage audit | `uv run python src/eval/leakage_audit.py` | `results/leakage_audit.csv`, `docs/leakage_checklist.md` |

Test ROC-AUC (mean over 3 seeds):

| | One-hot + LogReg | Ordinal + HistGB | TE + LightGBM | CatBoost |
|---|---|---|---|---|
| adult | 0.909 | 0.929 | 0.929 | **0.931** |
| bank-marketing | 0.910 | 0.933 | 0.935 | **0.939** |
| Amazon | 0.832 | 0.807 | 0.834 | **0.870** |

Write-up with the leakage explanation, rare/unseen categories and SHAP: **[docs/part_a_results.md](docs/part_a_results.md)**.

## Part B — Deep Learning (status: complete)

Same splits and preprocessing protocol as Part A. Models: an MLP with categorical embeddings and an FT-Transformer
(`nn.TransformerEncoder`, pre-norm, ReGLU, [CLS] token; rare/unseen levels map to a reserved embedding index 0).

| Step | Command | Output |
|---|---|---|
| Baseline (M3): MLP 5 seeds, FT-Transformer 3 seeds, validation only | `uv run python src/part_b/baseline.py` | `results/m3_part_b_baseline.csv`, `results/m3_training_curves.csv` |
| Check against the reference FT-Transformer implementation | `uv run python src/part_b/check_rtdl.py` | `results/m3_rtdl_check.csv` |
| Tuning (Optuna, 12 trials/family), 5 seeds, attention, embeddings, ablation (M4) | `uv run python src/part_b/tune.py` | `results/m4_part_b_runs_*`, `models/<dataset>/` |
| ONNX export (M5) | `uv run python src/part_b/export_onnx.py` | `models/<dataset>/ft_transformer.onnx` |

Test ROC-AUC (mean over 5 seeds): MLP 0.911 / 0.934 / 0.818 and FT-Transformer 0.918 / 0.938 / 0.807 on adult /
bank-marketing / Amazon. Details: [docs/part_b_baseline.md](docs/part_b_baseline.md) (M3) and
[docs/m4_results.md](docs/m4_results.md). Deviations from the brief: [docs/deviations.md](docs/deviations.md).

## M4 — Full results (Part A + Part B)

| Step | Command | Output |
|---|---|---|
| Part B tuning (12 trials/family), 5-seed final fits, attention, embeddings, ablation | `uv run python src/part_b/tune.py` | `results/m4_part_b_runs_*`, `results/m4_attention_importance_*` |
| Joint comparison, paired bootstrap, seen/unseen, SHAP vs attention | `uv run python src/eval/joint_analysis.py` | `results/m4_*.csv`, `report/figures/m4_*` |
| Cost (training time, latency, size) | `uv run python src/eval/cost.py` | `results/cost.csv` |

Test ROC-AUC, CatBoost vs FT-Transformer: adult 0.931 vs 0.918, bank-marketing 0.939 vs 0.938 (equivalent within ROPE),
Amazon 0.870 vs 0.807. Hypotheses, ablation, explanations and limitations: **[docs/m4_results.md](docs/m4_results.md)**.

## M5 — Deployment

ONNX export (`src/part_b/export_onnx.py`), FastAPI service (`api/main.py`: `/health`, `/model-info`, `/predict`, `/explain`),
`api/Dockerfile`, load test (`api/load_test.py`) and latency tables: **[docs/m5_deployment.md](docs/m5_deployment.md)**.

```bash
uv run uvicorn api.main:app --port 8000                 # needs models/ (see docs/m5_deployment.md)
docker build -f api/Dockerfile -t p2-api . && docker run --rm -p 8000:8000 p2-api
```

## M6 — Pitch

5-minute pitch with one key result: [docs/m6_pitch.md](docs/m6_pitch.md), deck `report/pitch/m6_pitch.pptx`.

## Tests

```bash
uv run pytest
```