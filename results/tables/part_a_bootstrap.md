| dataset                | comparison             |   diff_auc |   ci_low |   ci_high |   p_catboost_better |
|:-----------------------|:-----------------------|-----------:|---------:|----------:|--------------------:|
| adult                  | catboost - te_lgbm     |     0.0011 |   0.0003 |    0.0020 |              0.9980 |
| adult                  | catboost - onehot_lr   |     0.0224 |   0.0196 |    0.0254 |              1.0000 |
| adult                  | catboost - ordinal_hgb |     0.0011 |  -0.0001 |    0.0022 |              0.9560 |
| bank-marketing         | catboost - te_lgbm     |     0.0034 |   0.0011 |    0.0057 |              0.9950 |
| bank-marketing         | catboost - onehot_lr   |     0.0294 |   0.0238 |    0.0353 |              1.0000 |
| bank-marketing         | catboost - ordinal_hgb |     0.0051 |   0.0024 |    0.0078 |              1.0000 |
| amazon_employee_access | catboost - te_lgbm     |     0.0352 |   0.0225 |    0.0486 |              1.0000 |
| amazon_employee_access | catboost - onehot_lr   |     0.0421 |   0.0280 |    0.0561 |              1.0000 |
| amazon_employee_access | catboost - ordinal_hgb |     0.0575 |   0.0399 |    0.0760 |              1.0000 |