| dataset                | feature        | model          |    all |   n_seen |   auc_seen |   n_rare |   auc_rare |   n_unseen |   auc_unseen |   auc_stress_all_unseen |   drop_stress |
|:-----------------------|:---------------|:---------------|-------:|---------:|-----------:|---------:|-----------:|-----------:|-------------:|------------------------:|--------------:|
| adult                  | native-country | catboost       | 0.9306 |     9767 |     0.9306 |        2 |   nan      |          0 |     nan      |                  0.9288 |        0.0018 |
| adult                  | native-country | te_lgbm        | 0.9291 |     9767 |     0.9292 |        2 |   nan      |          0 |     nan      |                  0.9289 |        0.0002 |
| adult                  | native-country | onehot_lr      | 0.9085 |     9767 |     0.9085 |        2 |   nan      |          0 |     nan      |                  0.9087 |       -0.0003 |
| adult                  | native-country | mlp            | 0.9106 |     9767 |     0.9106 |        2 |   nan      |          0 |     nan      |                  0.9058 |        0.0048 |
| adult                  | native-country | ft_transformer | 0.9182 |     9767 |     0.9183 |        2 |   nan      |          0 |     nan      |                  0.9179 |        0.0003 |
| bank-marketing         | V2             | catboost       | 0.9392 |     9043 |     0.9392 |        0 |   nan      |          0 |     nan      |                  0.9376 |        0.0016 |
| bank-marketing         | V2             | te_lgbm        | 0.9345 |     9043 |     0.9345 |        0 |   nan      |          0 |     nan      |                  0.9349 |       -0.0004 |
| bank-marketing         | V2             | onehot_lr      | 0.9102 |     9043 |     0.9102 |        0 |   nan      |          0 |     nan      |                  0.9093 |        0.0009 |
| bank-marketing         | V2             | mlp            | 0.9335 |     9043 |     0.9335 |        0 |   nan      |          0 |     nan      |                  0.9282 |        0.0052 |
| bank-marketing         | V2             | ft_transformer | 0.9376 |     9043 |     0.9376 |        0 |   nan      |          0 |     nan      |                  0.9375 |        0.0001 |
| amazon_employee_access | RESOURCE       | catboost       | 0.8702 |     3058 |     0.8828 |     2409 |     0.8667 |       1087 |       0.8250 |                  0.8347 |        0.0355 |
| amazon_employee_access | RESOURCE       | te_lgbm        | 0.8341 |     3058 |     0.8479 |     2409 |     0.8294 |       1087 |       0.8049 |                  0.8127 |        0.0214 |
| amazon_employee_access | RESOURCE       | onehot_lr      | 0.8322 |     3058 |     0.8490 |     2409 |     0.8295 |       1087 |       0.7955 |                  0.8144 |        0.0178 |
| amazon_employee_access | RESOURCE       | mlp            | 0.8181 |     3058 |     0.8313 |     2409 |     0.8151 |       1087 |       0.7970 |                  0.8079 |        0.0102 |
| amazon_employee_access | RESOURCE       | ft_transformer | 0.8074 |     3058 |     0.8213 |     2409 |     0.8083 |       1087 |       0.7797 |                  0.7984 |        0.0090 |