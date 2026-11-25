| dataset                | model                       |   n_seeds |   val_auc_mean | test_auc        | test_ap         |
|:-----------------------|:----------------------------|----------:|---------------:|:----------------|:----------------|
| adult                  | CatBoost                    |         3 |         0.9289 | 0.9306 ± 0.0002 | 0.8300 ± 0.0002 |
| adult                  | FT-T (no rare bucket)       |         3 |         0.9125 | 0.9181 ± 0.0008 | 0.7943 ± 0.0046 |
| adult                  | FT-Transformer              |         5 |         0.9135 | 0.9182 ± 0.0007 | 0.7960 ± 0.0023 |
| adult                  | MLP + embeddings            |         5 |         0.9051 | 0.9106 ± 0.0012 | 0.7660 ± 0.0036 |
| adult                  | Naive TE + LightGBM (leaky) |         3 |         0.9275 | 0.9294 ± 0.0002 | 0.8275 ± 0.0005 |
| adult                  | One-hot + LogReg            |         1 |         0.9055 | 0.9085          | 0.7679          |
| adult                  | Ordinal + HistGB            |         3 |         0.9252 | 0.9289 ± 0.0008 | 0.8273 ± 0.0021 |
| adult                  | TargetEnc + LightGBM        |         3 |         0.9269 | 0.9291 ± 0.0003 | 0.8268 ± 0.0002 |
| bank-marketing         | CatBoost                    |         3 |         0.9397 | 0.9392 ± 0.0004 | 0.6428 ± 0.0014 |
| bank-marketing         | FT-T (no rare bucket)       |         3 |         0.9397 | 0.9379 ± 0.0010 | 0.6382 ± 0.0060 |
| bank-marketing         | FT-Transformer              |         5 |         0.9396 | 0.9376 ± 0.0008 | 0.6349 ± 0.0063 |
| bank-marketing         | MLP + embeddings            |         5 |         0.9329 | 0.9335 ± 0.0012 | 0.6187 ± 0.0065 |
| bank-marketing         | Naive TE + LightGBM (leaky) |         3 |         0.9385 | 0.9378 ± 0.0004 | 0.6269 ± 0.0030 |
| bank-marketing         | One-hot + LogReg            |         1 |         0.9098 | 0.9102          | 0.5591          |
| bank-marketing         | Ordinal + HistGB            |         3 |         0.9350 | 0.9333 ± 0.0006 | 0.6095 ± 0.0032 |
| bank-marketing         | TargetEnc + LightGBM        |         3 |         0.9340 | 0.9345 ± 0.0027 | 0.6189 ± 0.0085 |
| amazon_employee_access | CatBoost                    |         3 |         0.8772 | 0.8702 ± 0.0030 | 0.9872 ± 0.0003 |
| amazon_employee_access | FT-T (no rare bucket)       |         3 |         0.8117 | 0.8025 ± 0.0142 | 0.9810 ± 0.0022 |
| amazon_employee_access | FT-Transformer              |         5 |         0.8227 | 0.8074 ± 0.0083 | 0.9809 ± 0.0016 |
| amazon_employee_access | MLP + embeddings            |         5 |         0.8276 | 0.8181 ± 0.0054 | 0.9824 ± 0.0008 |
| amazon_employee_access | Naive TE + LightGBM (leaky) |         3 |         0.8034 | 0.7928 ± 0.0010 | 0.9784 ± 0.0005 |
| amazon_employee_access | One-hot + LogReg            |         1 |         0.8499 | 0.8322          | 0.9834          |
| amazon_employee_access | Ordinal + HistGB            |         3 |         0.8090 | 0.8069 ± 0.0015 | 0.9811 ± 0.0003 |
| amazon_employee_access | TargetEnc + LightGBM        |         3 |         0.8477 | 0.8341 ± 0.0053 | 0.9843 ± 0.0010 |