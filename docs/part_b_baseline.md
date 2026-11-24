# Part B baseline (M3): MLP with embeddings and FT-Transformer

First deep models on the **same splits** as Part A (`data/splits/`, seed 2026, 60/20/20). Only **validation** scores
appear here; the test split stays locked until M4. Reproduce with:

```bash
uv run python src/part_b/baseline.py     # configs/m3_part_b_baseline.yaml -> results/m3_part_b_baseline.csv, m3_training_curves.csv
uv run python src/part_b/check_rtdl.py --ours-from-baseline   # configs/m3_rtdl_check.yaml -> results/m3_rtdl_check.csv
uv run python src/eval/plot_m3.py        # report/figures/m3_training_curves.*, m3_seed_variance.*
uv run pytest                            # preprocessing / model tests
```

## Setup

* **Pre-processing** (`src/part_b/preprocessing.py`, fitted on training rows only): numerical features are mapped to a
  standard normal through their training quantiles; each categorical feature gets a vocabulary built from the training rows.
  Index 0 is shared by **unseen and rare categories** (fewer than 10 training rows). Tested in `tests/test_preprocessing.py`.
* **MLP with embeddings**: one `nn.Embedding` per categorical feature, dimension `min(50, ceil(levels / 2))`,
  concatenated with the numericals; 3 hidden layers of 256, dropout 0.1.
* **FT-Transformer** (`src/part_b/models.py`): feature tokenizer (one learned vector per numerical feature scaled by its
  value, one embedding table per categorical feature), prepended [CLS] token, `nn.TransformerEncoder`
  (3 layers, 8 heads, d = 192, pre-norm, ReGLU feed-forward), prediction from the final [CLS] vector.
* **Training** (`src/part_b/train.py`): AdamW, batch 512, binary cross-entropy, early stopping on validation ROC-AUC
  (patience 5 for both models, at most 30 epochs), best-epoch weights restored. No tuning yet: these are the brief's
  starting configurations. Seeds: MLP 5, FT-Transformer 3.

## Validation results (mean ± sd over seeds)

| Data set | Model | ROC-AUC | AP | Best epoch | Train time (s) | Parameters |
|---|---|---|---|---|---|---|
| adult | MLP + embeddings | 0.9049 ± 0.0002 | 0.761 ± 0.003 | 6.6 | 5 | 149 k |
| adult | FT-Transformer | **0.9122 ± 0.0006** | 0.788 ± 0.002 | 9.0 | 397 | 916 k |
| bank-marketing | MLP + embeddings | 0.9313 ± 0.0017 | 0.629 ± 0.009 | 9.6 | 6 | 142 k |
| bank-marketing | FT-Transformer | **0.9376 ± 0.0010** | 0.640 ± 0.006 | 8.7 | 389 | 907 k |
| Amazon | MLP + embeddings | 0.8281 ± 0.0090 | 0.983 ± 0.001 | 5.0 | 5 | 341 k |
| Amazon | FT-Transformer | 0.8285 ± 0.0024 | 0.983 ± 0.000 | 9.7 | 172 | 1.29 M |

For reference, the Part A baselines on the same validation split (M2, default hyper-parameters):
CatBoost 0.929 / 0.940 / 0.877 and TE + LightGBM 0.926 / 0.931 / 0.831 on adult / bank / Amazon.

Figures: `report/figures/m3_training_curves.pdf` (train and validation log-loss, validation ROC-AUC per epoch, every seed)
and `report/figures/m3_seed_variance.pdf` (one dot per seed).

## Observations

* **FT-Transformer > MLP on the two mixed data sets** (+0.007 on adult, +0.006 on bank-marketing validation AUC), far
  outside seed noise (sd ≤ 0.002). On **Amazon the two are tied** (0.8285 vs 0.8281), but the MLP's seed sd is nearly four
  times larger (0.009 vs 0.002): attention-based pooling of the nine ID embeddings is more stable.
* **Both networks are still behind the untuned CatBoost**: by 0.017 / 0.002 on adult / bank (FT-Transformer) and by
  0.05 on Amazon. The gap is largest on the purely high-cardinality data, in line with hypothesis H1.
* **Amazon overfits within a few epochs**: training loss keeps falling while validation loss rises from epoch ~4, and
  early stopping selects epochs 5–10. The tuning step in M4 therefore has to search regularisation (dropout, weight decay)
  more than capacity.
* **Cost**: the FT-Transformer takes 70× longer than the MLP on adult (397 s vs 5 s on 2 CPU cores) and has 4–6× more parameters.

## Implementation check against `rtdl_revisiting_models` (brief: "within one seed's noise")

Same architecture, optimiser, splits, seeds and early stopping, adult, validation ROC-AUC, 3 seeds each:

| | ours | rtdl |
|---|---|---|
| ROC-AUC | 0.9122 ± 0.0006 | 0.9145 ± 0.0003 |

The difference of **−0.0023 is slightly outside the tolerance I set in advance** (max of 2 × pooled seed sd and 0.002),
so **the check is not passed**: our model is about 0.002 AUC weaker than the reference, with the same parameter count
(916 k vs 916 k). It is smaller than the FT-Transformer vs MLP gap (0.007) and than the gap to CatBoost, so it does not change
any conclusion, but it is reported as a limitation. A first version with a plain ReLU feed-forward was 0.007 below rtdl and was
replaced by ReGLU (`docs/deviations.md`, item 2). Remaining differences, not yet isolated: our first layer keeps its
pre-attention LayerNorm (rtdl skips it), the attention-weight initialisation differs (PyTorch `nn.MultiheadAttention` vs
rtdl's), and weight decay is applied to all parameters.
