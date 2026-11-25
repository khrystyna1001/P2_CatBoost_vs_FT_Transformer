# Part A — Applied ML: results

All numbers are on the locked **test** split (60/20/20, seed 2026), unless marked as validation.
Primary metric ROC-AUC, secondary average precision (AP). Reproduce with:

```bash
uv run python src/part_a/encoding_comparison.py   # M2 baseline (validation only)
uv run python src/part_a/tune.py                  # Optuna + final fits + SHAP (~20 min, 2 CPU cores)
uv run python src/part_a/analysis.py              # tables in results/, figures in report/figures/
```

## Protocol

* Four pipelines (`src/part_a/pipelines.py`); every encoder is fitted on training rows only:
  one-hot + logistic regression, ordinal + HistGradientBoosting, cross-fitted `TargetEncoder` (5 folds) +
  LightGBM, CatBoost with native categorical features (ordered target statistics).
* Tuning: Optuna TPE (seed 2026), **12 trials per pipeline and data set**, objective validation ROC-AUC.
  CatBoost early-stops on the validation split (≤ 3 000 trees). Search spaces in `src/part_a/tune.py`,
  best parameters in `results/best_params/`, every trial in `results/optuna/`.
  *Deviation:* the brief's default is 60 trials and the proposal said 30. 12 was chosen so that Part B
  (FT-Transformer on a 2-core CPU) can get the same budget; the equal-budget rule takes priority.
  Rerun with more by changing `n_trials` in `configs/m4_part_a_tuning.yaml`.
* Final models: best configuration refitted with 3 seeds (logistic regression is deterministic: 1 run).

## 1. Encoding comparison (test ROC-AUC, mean ± sd over seeds)

| Data set | One-hot + LogReg | Ordinal + HistGB | TE + LightGBM | CatBoost |
|---|---|---|---|---|
| adult | 0.9085 | 0.9289 ± 0.0008 | 0.9291 ± 0.0003 | **0.9306 ± 0.0002** |
| bank-marketing | 0.9102 | 0.9333 ± 0.0006 | 0.9345 ± 0.0027 | **0.9392 ± 0.0004** |
| Amazon_employee_access | 0.8322 | 0.8069 ± 0.0015 | 0.8341 ± 0.0053 | **0.8702 ± 0.0030** |

AP: `results/part_a_test_scores.csv`. Figure: `report/figures/part_a_test_scores.pdf`.

Paired bootstrap on the test rows (2 000 resamples, seed-averaged predictions; `results/part_a_bootstrap.csv`):
CatBoost beats cross-fitted target encoding + LightGBM by **+0.001** AUC on adult (95 % CI [0.0003, 0.0020]),
**+0.003** on bank-marketing ([0.001, 0.006]) and **+0.035** on Amazon ([0.023, 0.049]).
On the two mixed data sets the gap is statistically clear but practically tiny (below the 0.005 ROPE).
On the purely high-cardinality Amazon data it is large. Ordinal encoding is the worst choice there (0.807):
arbitrary integer codes of 7 518 resources give the trees nothing to split on, while one-hot + a linear model
(0.832) is a strong baseline for ID-like features.

## 2. Leakage demonstration (naive target encoding, no cross-fitting)

`naive_te_lgbm` replaces every category by its target mean over **all** training rows, including the row itself,
and reuses the tuned `te_lgbm` hyper-parameters so the encoder is the only difference.

| Data set | Pipeline | Train AUC | Val AUC | Test AUC | Train − test |
|---|---|---|---|---|---|
| adult | cross-fitted TE | 0.943 | 0.927 | 0.929 | 0.014 |
| adult | naive TE | 0.948 | 0.928 | 0.929 | 0.019 |
| bank-marketing | cross-fitted TE | 0.985 | 0.934 | 0.935 | 0.051 |
| bank-marketing | naive TE | 0.998 | 0.939 | 0.938 | 0.060 |
| Amazon | cross-fitted TE | 0.963 | 0.848 | 0.834 | 0.128 |
| Amazon | **naive TE** | **0.998** | **0.803** | **0.793** | **0.205** |

**Explanation.** For a level seen in only a few training rows, the naive encoding is (almost) that row's own label.
LightGBM learns "encoded value ≈ label", which is true on the training rows and false on any new row, where the
encoding is an honest average of *other* rows. So the training score is inflated (Amazon: 0.998) and the model
generalises worse (test 0.793 vs 0.834). The validation and test scores of the naive pipeline agree with each other
(0.803 vs 0.793): both are clean held-out splits, so **the leak shows as a train → held-out gap, not as a
validation → test gap**. It would only show up as validation > test if the encoder had also been fitted on the
validation rows, or if validation were used for model selection after seeing the leak. On adult and bank-marketing
(at most 42 levels, each with hundreds of rows) a level's mean barely depends on any single row, so the naive
encoder does not hurt and is even slightly better on bank. **Leakage through target statistics scales with
cardinality.** None of the final models uses the naive encoder.

## 3. Rare and unseen categories

Test rows are labelled by the training frequency of their level of the highest-cardinality feature
(`src/eval/slices.py`): *seen* ≥ 10 training rows, *rare* 1–9, *unseen* 0. Only Amazon has rare/unseen levels:
`RESOURCE` is unseen in 1 087 test rows (16.6 %) and rare in 2 409. Adult (`native-country`) and bank-marketing
(`V2`, job) have none, so for every data set we also run a **stress test** that replaces the feature by a level
that never occurs in training for *all* test rows.

| Amazon, `RESOURCE` | n | One-hot + LogReg | Ordinal + HistGB | TE + LightGBM | CatBoost |
|---|---|---|---|---|---|
| seen | 3 058 | 0.849 | 0.836 | 0.852 | **0.887** |
| rare | 2 409 | 0.830 | 0.812 | 0.834 | **0.871** |
| unseen | 1 087 | 0.796 | 0.789 | 0.813 | **0.827** |
| stress test (all unseen) | 6 554 | 0.814 | 0.786 | 0.818 | **0.838** |

(ROC-AUC of the seed-averaged prediction; `results/part_a_seen_unseen.csv`.)
Every model degrades from seen → rare → unseen. CatBoost stays best in every slice, but it loses the most when the
feature is removed (−0.036 in the stress test, vs −0.018 for one-hot + LogReg): it relies most on `RESOURCE`, and an
unseen level falls back to the prior. On adult and bank the stress test costs at most 0.003 AUC: the
highest-cardinality feature there carries little signal.

## 4. Explainability (SHAP)

Global: mean |SHAP| on the test split, CatBoost (`get_feature_importance(type="ShapValues")`) vs LightGBM with target
encoding (TreeSHAP, `pred_contrib`). Figure `report/figures/part_a_shap_top10.pdf`, table
`results/part_a_shap_agreement.csv`.

| Data set | Kendall τ (all features) | Top-10 overlap | Top-5 overlap | Same top-1 |
|---|---|---|---|---|
| adult (14 features) | 0.58 (p = 0.003) | 8/10 | 3/5 | no (relationship vs age) |
| bank-marketing (16) | 0.90 (p < 0.001) | 10/10 | 4/5 | yes (V12 = call duration) |
| Amazon (9) | 0.56 (p = 0.045) | 9/9 | 4/5 | no (RESOURCE vs MGR_ID) |

* The two boosters agree on *which* features matter but disagree on the order. On adult the disagreement is
  between `relationship` and `marital-status`, two strongly correlated features: SHAP splits credit between
  correlated features differently depending on which one the trees happened to split on. This is a limit of
  SHAP, not a disagreement about the data.
* On bank-marketing both models rank V12 (`duration`) first by a wide margin, the feature flagged as leaky in
  `docs/leakage_checklist.md` (L2).
* On Amazon CatBoost puts most credit on `RESOURCE`, while LightGBM + TE spreads it over `MGR_ID` and
  `ROLE_FAMILY_DESC`. With cross-fitted encodings a single resource ID is a noisy feature, so the trees use
  the coarser IDs instead.

Local: `report/figures/part_a_local_shap_amazon.pdf` explains two Amazon test rows with CatBoost. For a row with a
*seen* resource, `RESOURCE` is the largest positive contribution (+0.37 log-odds). For a row with an *unseen*
resource, `RESOURCE` gets −0.30. The model learned nothing about this resource; the negative value only says that
the prior used for unseen levels is lower than the average encoded value. **SHAP explains the model, not the
world**: an attribution to an unseen category describes CatBoost's fallback rule, not evidence about that
resource.

**Limits.** SHAP values are computed on the encoded/internal representation (target statistics), so they explain
"how much the encoded value moved the prediction". They assume feature independence when marginalising, which
correlated census features violate. Global rankings come from one seed. TreeSHAP is exact for the trees, but
not causal.
