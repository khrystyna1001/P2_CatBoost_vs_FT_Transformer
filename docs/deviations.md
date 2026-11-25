# Deviations from the brief and from the M1 proposal

The joint-project brief (version 15 Oct 2026) is the reference. Where this repository differs from it, or from
`report/m1_proposal.tex`, the difference and its reason are listed here and repeated in the report's threats to validity.

## Brief → this repository

| # | Brief | This repository | Why |
|---|---|---|---|
| 1 | Optuna budget: default **60 trials** per model family | **12 trials** per model family, identical for Part A (4 pipelines) and Part B (MLP, FT-Transformer), same validation split | All experiments run on a 2-core CPU (see Hardware). One FT-Transformer epoch is 20–40 s, so 60 trials × 3 data sets was not feasible. The rule that matters for fairness (equal budget for every family) is kept. Change `n_trials` in `configs/m4_*_tuning.yaml` to rerun with 60. |
| 2 | FT-Transformer: `nn.TransformerEncoder` | `nn.TransformerEncoder` (pre-norm, 8 heads) whose layers are given the paper's **ReGLU** feed-forward (`linear1`: d → 2·d_ffn, ReGLU, `linear2`: d_ffn → d). The first layer keeps its pre-attention LayerNorm (rtdl skips it) | A first version with a plain ReLU feed-forward scored 0.907 vs rtdl's 0.914 validation AUC on adult after 6 epochs (`results/m3_rtdl_check_relu_variant.csv`, FAIL). With ReGLU the 6-epoch gap fell to 0.003 (`results/m3_rtdl_check_reglu_6epochs.csv`), but every run was still improving, so the final check compares both models trained to convergence (`results/m3_rtdl_check.csv`): ours 0.9122 ± 0.0006 vs rtdl 0.9145 ± 0.0003 validation AUC on adult. The −0.0023 gap is marginally outside the tolerance set in advance, so the check is **not passed**; see `docs/part_b_baseline.md`. |
| 3 | FT-Transformer size: 3 layers, d = 192 "as a starting point" | Baseline (M3) at 3 layers, d = 192. Tuned model (M4) searches d ∈ {64, 96}, 1–2 layers | CPU cost. |
| 4 | Deep models: ≥ 5 seeds per **final** configuration | 5 seeds for every final configuration (M4). The M3 baseline FT-Transformer uses 3 seeds (the MLP 5) | M3 is a baseline; the rule applies to final models. |
| 5 | Part B ablations are not prescribed for P2 | One extra ablation of the FT-Transformer (M4): no rare-category bucket. The mean-pooling ablation (stretch goal) was not run (CPU budget) | Test H2 and the stretch goal. |
| 6 | Statistical comparison for one data set: corrected resampled t-test or 5×2cv; ≥ 5 data sets: Friedman + Nemenyi | P2 prescribes a paired bootstrap (2 000 resamples) per data set plus seed variation; with 3 data sets Friedman has almost no power, so it is not used | Follows the P2 "Joint analysis" section. The region of practical equivalence is fixed in `configs/m4_joint.yaml` before test scores are seen. |
| 7 | `results/`: raw CSV of every run (model, seed, fold, metric) | One row per (data set, model, seed); there is no fold column because there is a single train/validation/test split per data set | n ≥ 10 000 → one split (brief, rule 2). |

## M1 proposal → this repository

The M1 proposal was written before the evaluation protocol was fixed. The code follows the brief. Where the two differ:

| Item | M1 proposal | Implemented |
|---|---|---|
| Split | 70/15/15, seed 42 | **60/20/20, seed 2026** (brief, rules 1–2; `data/splits/`) |
| Tuning budget | 30 trials | 12 trials (see #1) |
| Explanation agreement | Spearman ρ and top-3 Jaccard, SHAP vs KernelSHAP/GradientSHAP for the FT-Transformer | **Kendall τ** between CatBoost SHAP importance and [CLS]-to-feature attention importance, as the P2 brief says; top-10 overlap as a secondary number. The FT-Transformer is explained by attention only, not by KernelSHAP |
| Unseen-category metric | ΔAUC on rows with an unseen level + 5 % random masking on adult / bank | Seen / rare / unseen slices of the highest-cardinality feature, plus an "all test rows unseen" stress test on all three data sets (adult and bank have no natural unseen levels) |
| FT-Transformer search space | n_blocks 1–4, d_block up to 256 | n_blocks 1–2, d ≤ 96 in the tuned model (see #3) |
| Hypotheses H1–H3 | as proposed | unchanged; each is judged in the report as confirmed / partly confirmed / not confirmed |

## Hardware

Every experiment ran on one cloud container: 2 vCPU cores, 7 GB RAM, no GPU, Linux, Python 3.12, PyTorch 2.15 (CPU), `torch.set_num_threads(2)`.
Wall-clock tuning and training times are in `results/best_params/*.json` and `results/m4_*_runs_*.csv`.
