# M4 — Full results: CatBoost vs. FT-Transformer

Test split, first opened in M4. Primary metric ROC-AUC, secondary average precision (AP). Part A: 3 seeds
(logistic regression is deterministic); Part B: 5 seeds. Mean ± sd over seeds unless marked. Part A details are in
[part_a_results.md](part_a_results.md); deviations from the brief in [deviations.md](deviations.md). Reproduce:

```bash
uv run python src/part_a/tune.py          # ~20 min on 2 CPU cores
uv run python src/part_b/tune.py          # ~2 h on 2 CPU cores
uv run python src/part_a/analysis.py      # Part A tables and figures
uv run python src/eval/joint_analysis.py  # joint comparison, bootstrap, explanations, ablation
uv run python src/eval/cost.py            # training time, latency, size
```

## Protocol

* **Equal tuning budget**: 12 Optuna TPE trials (seed 2026) for every model family, objective validation ROC-AUC, same
  training and validation split; Part A: 4 pipelines, Part B: MLP and FT-Transformer. Wall-clock tuning time and the
  chosen parameters are in `results/best_params/`, every trial in `results/optuna/`. Hardware: 2 vCPU, no GPU.
* **FT-Transformer search space**: d ∈ {64, 96}, 1–2 blocks, attention / FFN / residual dropout, learning rate, weight decay.
  Capped for CPU, so smaller than the M3 baseline (d = 192, 3 blocks). Tuning trials are limited to 12 epochs, final
  runs to 25 (patience 4), early stopping on validation ROC-AUC.
* **Statistics**: paired bootstrap over the test rows (2 000 resamples, seed 2026) of
  *mean over CatBoost seeds − mean over FT-Transformer seeds*. Single-model scores are averaged (not predictions), so the family with more seeds gets no ensemble
  advantage. Region of practical equivalence fixed before the test scores were opened: ±0.005 ROC-AUC, ±0.01 AP (`configs/m4_joint.yaml`).

## 1. Test performance (ROC-AUC)

| Data set | One-hot + LogReg | Ordinal + HistGB | TE + LightGBM | CatBoost | MLP + emb. | FT-Transformer |
|---|---|---|---|---|---|---|
| adult | 0.9085 | 0.9289 ± 0.0008 | 0.9291 ± 0.0003 | **0.9306 ± 0.0002** | 0.9106 ± 0.0012 | 0.9182 ± 0.0007 |
| bank-marketing | 0.9102 | 0.9333 ± 0.0006 | 0.9345 ± 0.0027 | **0.9392 ± 0.0004** | 0.9335 ± 0.0012 | 0.9376 ± 0.0008 |
| Amazon | 0.8322 | 0.8069 ± 0.0015 | 0.8341 ± 0.0053 | **0.8702 ± 0.0030** | 0.8181 ± 0.0054 | 0.8074 ± 0.0083 |

AP (CatBoost / FT-Transformer / MLP): adult 0.830 / 0.796 / 0.766; bank-marketing 0.643 / 0.635 / 0.619;
Amazon 0.987 / 0.981 / 0.982. All models: `results/m4_test_scores.csv`. Figure: `report/figures/m4_test_scores.pdf`.

**CatBoost vs FT-Transformer** (`results/m4_bootstrap.csv`; difference CatBoost − FT-T, 95 % CI):

| Data set | ΔROC-AUC | Verdict (ROPE ±0.005) | ΔAP | Verdict (ROPE ±0.01) |
|---|---|---|---|---|
| adult | +0.0124 [0.0104, 0.0144] | CatBoost better | +0.034 [0.029, 0.040] | CatBoost better |
| bank-marketing | +0.0016 [−0.0007, 0.0037] | practically equivalent | +0.008 [−0.007, 0.021] | undecided |
| Amazon | **+0.0629** [0.0470, 0.0784] | CatBoost better | +0.006 [0.004, 0.009] | practically equivalent |

Seed variation: every one of the 15 CatBoost × FT-Transformer seed pairs has the same sign as the bootstrap on adult and
Amazon (min–max ΔAUC +0.011 to +0.013 and +0.049 to +0.076). On bank-marketing the pairs range from +0.000 to +0.003,
so the two are indistinguishable. The Amazon AP difference is small only because AP is saturated (94 % positive rows).

## 2. Seen vs. unseen categories

Amazon `RESOURCE` (7 518 levels). Test rows by training frequency of their level: seen (≥ 10 rows) 3 058, rare (1–9) 2 409,
unseen 1 087. ROC-AUC, mean over seeds (`results/m4_seen_unseen.csv`, figure/table in the report):

| | all | seen | rare | **unseen** | stress: all unseen | drop seen → unseen |
|---|---|---|---|---|---|---|
| One-hot + LogReg | 0.832 | 0.849 | 0.830 | 0.796 | 0.814 | −0.054 |
| TE + LightGBM | 0.834 | 0.848 | 0.829 | 0.805 | 0.813 | −0.043 |
| CatBoost | **0.870** | **0.883** | **0.867** | **0.825** | **0.835** | −0.058 |
| MLP + emb. | 0.818 | 0.831 | 0.815 | 0.797 | 0.808 | −0.034 |
| FT-Transformer | 0.807 | 0.821 | 0.808 | 0.780 | 0.798 | −0.042 |

The *stress test* replaces `RESOURCE` by a never-seen level in **every** test row. On adult and bank-marketing, which have
no natural unseen levels, it costs ≤ 0.002 AUC for CatBoost and the FT-Transformer and 0.005 for the MLP.

**Ablation: no rare-category bucket** (3 seeds; levels with < 10 rows get their own embedding instead of index 0).
No effect on adult (0.9181 vs 0.9182) or bank-marketing (0.9379 vs 0.9376). On Amazon the bucket gives 0.8074 ± 0.0083 vs
0.8025 ± 0.0142 overall and 0.798 vs 0.784 on rare/unseen rows, a small advantage that is within the seed noise
(`results/m4_ablations.csv`).

## 3. Explanations

CatBoost SHAP (`get_feature_importance(type="ShapValues")`, mean |SHAP|), LightGBM + target encoding (TreeSHAP) and the
FT-Transformer's [CLS]-to-feature attention averaged over heads, layers and test rows (`results/m4_explanations.csv`):

| Data set | Kendall τ CatBoost SHAP vs attention | p | Attention stability across 5 seeds (τ) | Kendall τ CatBoost vs LightGBM SHAP |
|---|---|---|---|---|
| adult (14 features) | 0.36 | 0.080 | 0.13 | 0.58 |
| bank-marketing (16) | 0.45 | 0.015 | 0.41 | 0.90 |
| Amazon (9) | 0.61 | 0.025 | 0.71 | 0.56 |

Rank of the highest-cardinality feature (1 = most important), `results/m4_importance_rank_high_card_feature.csv`:
Amazon `RESOURCE` (7 518 levels): CatBoost 1, LightGBM + TE 3, FT-T attention 3 of 9. Adult `native-country` (42 levels):
14 / 14 / 12 of 14. Figure: `report/figures/m4_importance.pdf`.

**Why attention is not an explanation in the SHAP sense.** SHAP attributes the *output* to the features and its values
add up to the prediction; the [CLS] attention only says how much the [CLS] token reads each feature token in each layer
and head, before further non-linear layers and value projections transform it. High attention does not mean large
influence on the output and it carries no sign. Empirically the attention profile is nearly flat (most features 4–10 % of
the total, `report/figures/m4_importance.pdf`) and unstable: its rank agreement between seeds is only τ = 0.13 on adult.
τ with SHAP is therefore bounded by the noise of the attention ranking itself.

**Embeddings** (`report/figures/m4_embeddings_*.pdf`): the first principal component of the 305 learned `RESOURCE`
embeddings has a Spearman correlation of **0.81 with the level's training target mean**: the FT-Transformer learns
a "target-like" direction without seeing any target statistic. Index 0 (unseen / rare) lies near the centre of the cloud.

## 4. Cost (1 CPU thread, `results/cost.csv`)

| Data set | Model | Train (s) | Size / parameters | Latency batch 1, p50 / p95 (ms) | Latency batch 1000, p50 (ms) |
|---|---|---|---|---|---|
| adult | CatBoost | 63 | 2 919 trees, 3.2 MB | 0.32 / 0.52 | 5.2 |
| adult | MLP + emb. | 4 | 149 k, 0.6 MB | 3.3 / 4.6 | 7.5 |
| adult | FT-Transformer | 115 | 162 k, 0.7 MB | 3.8 / 4.9 | 153 |
| bank-marketing | CatBoost | 34 | 2 076 trees, 1.8 MB | 0.28 / 0.33 | 4.2 |
| bank-marketing | FT-Transformer | 147 | 157 k, 0.6 MB | 4.0 / 4.3 | 120 |
| Amazon | CatBoost | 15 | 290 trees, 11.8 MB | 0.35 / 0.44 | 3.8 |
| Amazon | FT-Transformer | 33 | 272 k, 1.1 MB | 3.9 / 5.6 | 46 |

Neural latencies include the Python pre-processing (quantiles, vocabularies), the CatBoost ones the pandas call.
ONNX Runtime timings are added in M5.

## Hypotheses of the M1 proposal

* **H1 — partly confirmed.** CatBoost is clearly better on the purely high-cardinality Amazon data (+0.063 AUC). The
  second half (FT-Transformer matches or beats CatBoost on mixed data) holds on bank-marketing (equivalent) but **not on
  adult**, where CatBoost is better by 0.012.
* **H2 — not confirmed.** CatBoost does *not* degrade less. It loses the most from seen to unseen (−0.058 vs −0.042 for the
  FT-Transformer) and most in the all-unseen stress test (−0.036 vs −0.009), because it relies most on `RESOURCE`. In
  absolute terms it stays best on unseen rows (0.825 vs 0.780). The reserved index 0 makes the FT-Transformer robust but not
  better, and the ablation cannot tell whether the bucket helps.
* **H3 — not confirmed as stated.** Agreement is highest on the data set where *all* features are high-cardinality
  (τ = 0.61 on Amazon) and lowest on adult (τ = 0.36), the opposite of the hypothesis. Given the instability of attention
  across seeds, these values should not be over-interpreted. (The proposal's Spearman / Jaccard measures were replaced by
  Kendall τ and top-10 overlap, as the brief prescribes.)

## Threats to validity

* **12 Optuna trials** (brief default: 60). The FT-Transformer is hurt most: its tuned model on Amazon (validation 0.823)
  is *below* the M3 baseline with d = 192, 3 blocks (0.829), because the CPU-sized search space and 12-epoch cap exclude the
  baseline. The CatBoost – FT-T gap on Amazon (+0.063) is therefore likely an overestimate of what a fully tuned FT-Transformer would show.
* Seeds only vary initialisation and batch order; the split is fixed (one split per data set, brief rule 2).
* Three data sets: conclusions about "categorical data" are limited to these.
* bank-marketing keeps the `duration` feature (leakage checklist L2); absolute AUCs there are optimistic for both families.
* SHAP and attention are computed on one trained model per family (seed 0) for the importance rankings; attention stability across seeds is reported above.
