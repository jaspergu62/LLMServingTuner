# Predictor Accuracy Evaluation

This directory contains scripts for evaluating the prediction accuracy of XGBoost and Vidur execution time predictors.

## Usage

### Basic Evaluation

Compare both predictors:
```bash
python evaluate_predictors.py \
    --test-data /path/to/test_data.csv \
    --xgb-model /path/to/xgb_model.ubj \
    --config /path/to/config.toml \
    --output-dir ./results
```

Evaluate only Vidur:
```bash
python evaluate_predictors.py \
    --test-data /path/to/test_data.csv \
    --predictor vidur \
    --config /path/to/config.toml
```

Evaluate only XGBoost:
```bash
python evaluate_predictors.py \
    --test-data /path/to/test_data.csv \
    --predictor xgboost \
    --xgb-model /path/to/xgb_model.ubj
```

### Test Data Format

The test data CSV should contain the following columns:

| Column | Description |
|--------|-------------|
| `batch_stage` | "prefill" or "decode" |
| `batch_size` | Number of requests in batch |
| `input_length` | Input token count (prefill tokens) |
| `output_length` | Output token count (decode tokens) |
| `model_execute_time` | Ground truth latency in milliseconds |
| `max_seq_len` | Maximum sequence length (optional) |

Alternative column names are also supported:
- `num_prefill_tokens` → `input_length`
- `num_decode_tokens` → `output_length`
- `latency_ms` or `latency` → `model_execute_time`

### Output

The script generates:

1. **accuracy_metrics.json** - Detailed metrics in JSON format
2. **accuracy_metrics.csv** - Metrics summary as CSV
3. **detailed_predictions.csv** - Per-sample predictions and errors
4. **{predictor}_scatter.png** - Scatter plots (prediction vs ground truth)
5. **comparison_error_hist.png** - Error distribution comparison

### Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error (ms) |
| MAPE | Mean Absolute Percentage Error (%) |
| RMSE | Root Mean Squared Error (ms) |
| R² | Coefficient of determination |
| Median Error | Median absolute error |
| P90 Error | 90th percentile error |
| P99 Error | 99th percentile error |

## Example Output

```
================================================================================
PREDICTOR ACCURACY EVALUATION REPORT
================================================================================

XGBoost - ALL
----------------------------------------
  Samples:      1000
  MAE:          2.3456 ms
  MAPE:         5.67%
  RMSE:         3.4567 ms
  R²:           0.9234
  Median Error: 1.2345 ms
  P90 Error:    5.6789 ms
  P99 Error:    12.3456 ms

Vidur - ALL
----------------------------------------
  Samples:      1000
  MAE:          3.1234 ms
  MAPE:         7.89%
  RMSE:         4.5678 ms
  R²:           0.8901
  ...
```

## Creating Test Data

You can create test data from vLLM profiling:

```python
import pandas as pd

# Load profiling data
df = pd.read_csv("vllm_profile.csv")

# Select relevant columns and rename
test_data = df[[
    'batch_stage',
    'batch_size',
    'num_prefill_tokens',
    'num_decode_tokens',
    'model_execute_time'
]].copy()

test_data.to_csv("test_data.csv", index=False)
```

Or from Vidur simulation output:
```python
# Use the request_metrics.csv from vidur simulation
df = pd.read_csv("simulator_output/.../request_metrics.csv")
# Map fields appropriately
```
