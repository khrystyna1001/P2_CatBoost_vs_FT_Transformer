# Contribution statement

Team: Khrystyna Koshova (236523), single-member team.

| Area | Who | Notes |
|---|---|---|
| Proposal (M1), research question, hypotheses | Khrystyna Koshova | `report/m1_proposal.tex` |
| Data loading, split protocol (seed 2026, 60/20/20) | Khrystyna Koshova | `data/` |
| Part A pipelines, tuning, SHAP | Khrystyna Koshova | `src/part_a/` |
| Part B models (MLP, FT-Transformer), training loop, attention / embedding analysis | Khrystyna Koshova | `src/part_b/` |
| Joint analysis, statistics, figures | Khrystyna Koshova | `src/eval/` |
| Deployment (ONNX, FastAPI, Docker, load test) | Khrystyna Koshova | `api/`, `src/part_b/export_onnx.py` |
| Report and pitch | Khrystyna Koshova | `report/` |

## Code reused from papers

* `rtdl_revisiting_models` (Gorishniy et al., 2021; MIT licence) is used **only** as a reference
  implementation to check our own FT-Transformer (`src/part_b/check_rtdl.py`). The model that is
  trained, analysed and served is our own implementation in `src/part_b/models.py`.
* CatBoost (Prokhorenkova et al., 2018; Apache 2.0) is used as a library.
