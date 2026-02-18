# Configuration Parameters Reference

This project uses a two-file JSON config system. Each config set lives in `config/{project_name}/`.

---

## data_config.json

### Root Level

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `ticker` | string | Stock ticker to download | `"SPY"` |
| `start_date` | string | Start date (YYYY-MM-DD) | `"2006-01-01"` |
| `end_date` | string | End date or `"auto"` (resolves to yesterday) | `"auto"` |

### `data.features`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bollinger_period` | int | 20 | Bollinger Band rolling window |
| `bollinger_std` | int | 2 | Bollinger Band standard deviation multiplier |
| `ema_short` | int | 8 | Short EMA period |
| `ema_medium` | int | 21 | Medium EMA period |
| `sma_short` | int | 50 | Short SMA period |
| `sma_long` | int | 200 | Long SMA period (also used for Market_Trend) |
| `adx_period` | int | 14 | ADX indicator period |
| `rsi_period` | int | 14 | RSI indicator period |

### `data.normalization`

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `normalization` | string | `"rolling"`, `"standard"` | Normalization strategy |
| `normalization_window` | int | — | Rolling window size (trading days, ~3 months = 63) |

**Rolling (default)**: Backward-looking rolling Z-score on full timeline. No lookahead.  
**Standard**: Fit `StandardScaler` on train, apply to val/test.  
`Market_Trend` (binary) is always excluded from normalization.

### `target`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `horizon_days` | int | 3 | Forward return prediction horizon (trading days) |

### `split`

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `test_years` | int | — | Number of years to reserve for test set |
| `validation_type` | string | `"expanding"`, `"sliding"` | Cross-validation strategy |
| `n_folds` | int | — | Number of validation folds |
| `sliding_window_years` | int | — | Train window size for sliding window CV |

**Expanding window**: Starts with 2 years of training data; each fold adds more data.  
**Sliding window**: Fixed-size training window slides forward by one fold period.

### `output`

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_dir` | string | Base directory for data files |
| `results_dir` | string | Base directory for results/plots |
| `models_dir` | string | Base directory for saved models |

---

## model_config.json

### Root Level

| Parameter | Type | Description |
|-----------|------|-------------|
| `project_name` | string | Must match the config folder name |
| `eval_metric` | string | Metric used for HPT and model selection: `"f1"`, `"accuracy"`, `"roc_auc"` |

### `models`

Each model has `enabled` (bool) and `hyperparameters` (dict of lists for grid search).

#### `logistic_regression`

| Parameter | Description |
|-----------|-------------|
| `C` | Regularization strength (list → grid search) |
| `max_iter` | Max solver iterations |

#### `random_forest`

| Parameter | Description |
|-----------|-------------|
| `n_estimators` | Number of trees |
| `max_depth` | Max tree depth (`null` = unlimited) |
| `min_samples_split` | Min samples to split a node |

#### `xgboost`

| Parameter | Description |
|-----------|-------------|
| `n_estimators` | Number of boosting rounds |
| `max_depth` | Max tree depth |
| `learning_rate` | Step size shrinkage |

### `learning_curves`

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled` | bool | Whether to compute learning curves |
| `train_sizes` | list[float] | Fractions of training data to evaluate (0.0–1.0) |

### `portfolio`

| Parameter | Type | Description |
|-----------|------|-------------|
| `initial_balance` | float | Starting cash ($) |
| `max_shares` | int | Maximum shares per trade |
| `confidence_threshold` | float | Min probability for Strategy 2 to buy |
| `stop_loss_pct` | float | Exit if loss exceeds this % from entry |
| `take_profit_pct` | float | Exit if gain exceeds this % from entry |

#### Variable Shares (Strategy 3)

| Probability Range | Shares |
|-------------------|--------|
| [0.875, 1.0] | 100% of max_shares |
| [0.75, 0.875) | 75% of max_shares |
| [0.625, 0.75) | 50% of max_shares |
| [0.5, 0.625) | 25% of max_shares |
| < 0.5 | No trade |

---

## Config Sets

| Set | Description | Use Case |
|-----|-------------|----------|
| `dry_run` | 3-year window, 3 folds, small HPT | Quick testing |
| `default_run` | 2006–present, 5 folds, full HPT | Production training |
