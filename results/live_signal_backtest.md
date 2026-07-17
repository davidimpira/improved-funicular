# Walk-forward backtest of the live reversal signal

Out-of-sample: 2020-01-01 to 2026-07-17 (57,340 bars), each year scored by a model trained only on prior years.

Target: confirmed reversal within the next 3 bars. Base rate: **10.98%**.

- AUC: **0.694**
- Brier score: 0.0941 (always-predict-base-rate: 0.0977)

## Calibration (out-of-sample deciles)

|   decile |   mean_pred |   actual_rate |    n |
|---------:|------------:|--------------:|-----:|
|        0 |      0.0221 |        0.0249 | 5734 |
|        1 |      0.0331 |        0.0413 | 5734 |
|        2 |      0.0415 |        0.0593 | 5734 |
|        3 |      0.0501 |        0.0774 | 5734 |
|        4 |      0.0593 |        0.0814 | 5734 |
|        5 |      0.0696 |        0.0959 | 5734 |
|        6 |      0.0829 |        0.1256 | 5734 |
|        7 |      0.101  |        0.1472 | 5734 |
|        8 |      0.1284 |        0.1856 | 5734 |
|        9 |      0.2218 |        0.2592 | 5734 |

## Alert-threshold trade-off (pick your knob here)

|   lift_threshold |   prob_threshold |   alerts_per_week |   precision |   realized_lift |   recall |
|-----------------:|-----------------:|------------------:|------------:|----------------:|---------:|
|              1.5 |           0.1647 |             12.62 |      0.2823 |            2.57 |   0.1932 |
|              2   |           0.2196 |              5.38 |      0.3457 |            3.15 |   0.1009 |
|              2.5 |           0.2745 |              2.75 |      0.3817 |            3.48 |   0.0569 |
|              3   |           0.3294 |              1.51 |      0.4047 |            3.69 |   0.033  |
|              4   |           0.4391 |              0.6  |      0.4078 |            3.71 |   0.0133 |

`lift_threshold` = alert when predicted probability >= that multiple of the base rate. `precision` = fraction of alerts followed by a confirmed reversal within 3 bars.

## Per-year stability

|   year |    auc |   base_rate |
|-------:|-------:|------------:|
|   2020 | 0.7289 |      0.1013 |
|   2021 | 0.6494 |      0.1189 |
|   2022 | 0.7029 |      0.0967 |
|   2023 | 0.6967 |      0.0946 |
|   2024 | 0.708  |      0.1208 |
|   2025 | 0.6862 |      0.1208 |
|   2026 | 0.6779 |      0.12   |
