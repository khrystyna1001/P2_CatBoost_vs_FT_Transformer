| dataset                | model         |   train_auc |   val_auc |   test_auc |   train_minus_test |   val_minus_test |
|:-----------------------|:--------------|------------:|----------:|-----------:|-------------------:|-----------------:|
| adult                  | te_lgbm       |      0.9429 |    0.9269 |     0.9291 |             0.0138 |          -0.0022 |
| adult                  | naive_te_lgbm |      0.9482 |    0.9275 |     0.9294 |             0.0188 |          -0.0019 |
| bank-marketing         | te_lgbm       |      0.9852 |    0.9340 |     0.9345 |             0.0507 |          -0.0005 |
| bank-marketing         | naive_te_lgbm |      0.9980 |    0.9385 |     0.9378 |             0.0602 |           0.0007 |
| amazon_employee_access | te_lgbm       |      0.9625 |    0.8477 |     0.8341 |             0.1283 |           0.0135 |
| amazon_employee_access | naive_te_lgbm |      0.9980 |    0.8034 |     0.7928 |             0.2051 |           0.0105 |