| dataset                | variant                       |   n_seeds |   test_auc_mean |   test_auc_sd |   auc_rare_or_unseen |   auc_stress_all_unseen |
|:-----------------------|:------------------------------|----------:|----------------:|--------------:|---------------------:|------------------------:|
| adult                  | ft_transformer                |         5 |          0.9182 |        0.0007 |             nan      |                  0.9179 |
| adult                  | ft_transformer_no_rare_bucket |         3 |          0.9181 |        0.0008 |             nan      |                  0.9181 |
| bank-marketing         | ft_transformer                |         5 |          0.9376 |        0.0008 |             nan      |                  0.9375 |
| bank-marketing         | ft_transformer_no_rare_bucket |         3 |          0.9379 |        0.0010 |             nan      |                  0.9375 |
| amazon_employee_access | ft_transformer                |         5 |          0.8074 |        0.0083 |               0.7980 |                  0.7984 |
| amazon_employee_access | ft_transformer_no_rare_bucket |         3 |          0.8025 |        0.0142 |               0.7843 |                  0.7972 |