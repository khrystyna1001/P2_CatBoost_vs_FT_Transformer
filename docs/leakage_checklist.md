# Leakage checklist (Kapoor & Narayanan, 2023)

Taxonomy from Kapoor, S. & Narayanan, A. (2023), *Leakage and the reproducibility crisis in
machine-learning-based science*, Patterns 4(9). Eight types, L1.1–L3.3. Numbers come from
`src/eval/leakage_audit.py` (`results/leakage_audit.csv`).

| Type | Question | Status | Evidence |
|---|---|---|---|
| **L1.1** No train–test split | Is there a held-out test set never used for fitting or model selection? | ✅ | One stratified 60/20/20 split per data set, seed 2026, indices in `data/splits/*.npz`. The audit asserts that train, validation and test are disjoint. Test scores are first computed in M4 (`src/part_a/tune.py`, `src/part_b/tune.py`); M2 and M3 report validation only. |
| **L1.2** Pre-processing on train + test | Are scalers, encoders and vocabularies fitted on training rows only? | ✅ | Part A: every encoder and scaler sits inside a scikit-learn `Pipeline` (or inside CatBoost), so it is fitted by `fit(X_train)`. `TargetEncoder` is cross-fitted (5 folds) within the training rows. Part B: `TabularPreprocessor.fit` sees only training rows: quantiles, vocabularies and the rare-category threshold are computed on them (`tests/test_preprocessing.py`). |
| **L1.3** Feature selection on train + test | Is any feature selection done on held-out data? | ✅ | No feature selection. All features from OpenML are used. |
| **L1.4** Duplicates across splits | Do identical rows occur in train and test? | ⚠️ minor | `adult`: 14 validation and 14 test rows have exact feature duplicates in the training split (0.14 %). These are genuine distinct people with the same census record, not copies, so they stay in. `bank-marketing` and `Amazon_employee_access`: 0. |
| **L2** Illegitimate features | Is any feature unavailable at prediction time or a proxy of the label? | ⚠️ documented | `bank-marketing` V12 (`duration` of the last call) is known only after the call, when the outcome is already known. UCI notes it should be discarded for a realistic model. We **keep it** for comparability with the OpenML benchmark and the published results: both model families see it, so the CatBoost vs FT-Transformer comparison is unaffected, but absolute bank-marketing AUCs are optimistic. Discussed under threats to validity in the report. |
| **L3.1** Temporal leakage | Is a model trained on the future and tested on the past? | ✅ n/a | No time ordering is used or claimed. `bank-marketing` is ordered by date in the original data, but the task is defined on i.i.d. stratified splits as in OpenML. |
| **L3.2** Non-independence between train and test | Do train and test share entities (patients, users…)? | ⚠️ by design | `Amazon_employee_access` rows share resources, managers and roles across splits. That is the point of the task (target statistics / embeddings of high-cardinality IDs). 1 087 test rows (16.6 %) have a `RESOURCE` never seen in training and 2 409 a rare one (< 10 training rows); we report these slices separately (M4). |
| **L3.3** Sampling bias in the test distribution | Does the test set differ in distribution from deployment? | ✅ | Stratified split: positive rate is identical in train, validation and test to three decimals (adult 0.239, bank 0.117, amazon 0.942). |

## Pitfalls specific to this project

* **Target statistics outside cross-fitting** (brief, Part A step 2). The leak is demonstrated on purpose by
  `naive_te_lgbm`: on Amazon it reaches train ROC-AUC 1.000 but validation 0.798, against 0.831 for the
  cross-fitted `te_lgbm` (`results/m2_part_a_baseline.csv`). The naive encoder is **never** used by a final
  model, and `tests/test_leakage.py` checks that it memorises labels of singleton levels.
* **Category vocabularies built on validation/test rows.** Not possible by construction: the vocabulary is
  built in `TabularPreprocessor.fit(X_train)`; unseen and rare (< 10 training rows) levels map to index 0.
* **Early stopping on the test set.** CatBoost and the neural networks early-stop on the validation split.
  The test split is read only for final evaluation.
* **Tuning budget.** Every model family gets the same number of Optuna trials on the same validation split
  (configs `m4_part_a_tuning.yaml`, `m4_part_b_tuning.yaml`).
