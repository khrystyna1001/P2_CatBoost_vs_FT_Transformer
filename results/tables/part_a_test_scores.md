| dataset                | model       |   n_seeds |   val_auc |   test_auc |   test_auc_sd |   test_ap |   test_ap_sd |
|:-----------------------|:------------|----------:|----------:|-----------:|--------------:|----------:|-------------:|
| adult                  | onehot_lr   |         1 |    0.9055 |     0.9085 |      nan      |    0.7679 |     nan      |
| adult                  | ordinal_hgb |         3 |    0.9252 |     0.9289 |        0.0008 |    0.8273 |       0.0021 |
| adult                  | te_lgbm     |         3 |    0.9269 |     0.9291 |        0.0003 |    0.8268 |       0.0002 |
| adult                  | catboost    |         3 |    0.9289 |     0.9306 |        0.0002 |    0.8300 |       0.0002 |
| bank-marketing         | onehot_lr   |         1 |    0.9098 |     0.9102 |      nan      |    0.5591 |     nan      |
| bank-marketing         | ordinal_hgb |         3 |    0.9350 |     0.9333 |        0.0006 |    0.6095 |       0.0032 |
| bank-marketing         | te_lgbm     |         3 |    0.9340 |     0.9345 |        0.0027 |    0.6189 |       0.0085 |
| bank-marketing         | catboost    |         3 |    0.9397 |     0.9392 |        0.0004 |    0.6428 |       0.0014 |
| amazon_employee_access | onehot_lr   |         1 |    0.8499 |     0.8322 |      nan      |    0.9834 |     nan      |
| amazon_employee_access | ordinal_hgb |         3 |    0.8090 |     0.8069 |        0.0015 |    0.9811 |       0.0003 |
| amazon_employee_access | te_lgbm     |         3 |    0.8477 |     0.8341 |        0.0053 |    0.9843 |       0.0010 |
| amazon_employee_access | catboost    |         3 |    0.8772 |     0.8702 |        0.0030 |    0.9872 |       0.0003 |