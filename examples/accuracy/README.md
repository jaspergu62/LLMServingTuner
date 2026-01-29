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

#### vLLM Profile CSV Format (Input)

The vLLM profile CSV has **two columns with tuple values**:

| Column | Description |
|--------|-------------|
| `"('batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len', 'model_execute_time')"` | Batch-level info as tuple |
| `"('input_length', 'need_blocks', 'output_length')"` | Per-request info as tuple of tuples |

Example row values:
- Column 1: `('BatchStage.PREFILL', 4, 128, 2048, 512, 15.234)`
- Column 2: `((512, 32, 0), (512, 32, 0), (512, 32, 0), (512, 32, 0))`

#### Processed Test Data Format (Output)

After processing with `generate_test_data.py`, the output CSV has these columns:

| Column | Description |
|--------|-------------|
| `batch_stage` | "prefill" or "decode" |
| `batch_size` | Number of requests in batch |
| `total_need_blocks` | Total KV cache blocks needed |
| `total_prefill_token` | Total prefill tokens in batch |
| `max_seq_len` | Maximum sequence length |
| `model_execute_time` | Ground truth latency in milliseconds |
| `input_length` | Aggregated input tokens from all requests |
| `output_length` | Aggregated output tokens from all requests |

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

### From vLLM Profiling

Enable profiling in vLLM simulation to collect execution times:

```python
from llmservingtuner.inference.simulate_vllm import SimulateVllm

# Initialize with profiling enabled
SimulateVllm.init(profile_flag=True)
# Profile data will be saved to /tmp/profile/<timestamp>.csv
```

Then convert to test data format:

```bash
python generate_test_data.py \
    --profile-csv /tmp/profile/20240101-1200.csv \
    --output test_data.csv
```

### Generate Synthetic Data

For quick testing without real profiling data:

```bash
python generate_test_data.py --synthetic -n 1000 -o test_data.csv
```
