# Benchmark Result Comparison Tool

Compare performance metrics between multiple vLLM benchmarks and Vidur simulations.

## Features

- **Multi-Result Support**: Compare 2+ benchmark results on the same chart
- **Metrics Comparison**: TTFT, TPOT (Mean, P90, P99), Throughput (TPS)
- **Progress Visualization**: Request completion timeline with multiple lines
- **Export**: CSV summary and PNG plots

## Usage

### Compare Multiple Results (Recommended)

```bash
# Compare two results
python compare_benchmarks.py \
    --result vllm:/path/to/vllm_result.json:vLLM-Real \
    --result vidur:/path/to/vidur_output/:Vidur-Sim \
    --output-dir ./comparison_output

# Compare multiple configurations
python compare_benchmarks.py \
    --result vllm:./results/qps1.json:vLLM-QPS1 \
    --result vllm:./results/qps2.json:vLLM-QPS2 \
    --result vidur:./vidur_out/qps1/:Vidur-QPS1 \
    --result vidur:./vidur_out/qps2/:Vidur-QPS2 \
    --output-dir ./comparison_output

# Compare different QPS settings
python compare_benchmarks.py \
    -r vllm:./bench_qps0.5.json:QPS-0.5 \
    -r vllm:./bench_qps1.0.json:QPS-1.0 \
    -r vllm:./bench_qps2.0.json:QPS-2.0 \
    -o ./qps_comparison
```

### Legacy Mode (Two Results Only)

```bash
python compare_benchmarks.py \
    --vllm-result /path/to/vllm.json \
    --vidur-result /path/to/vidur_output/
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--result`, `-r` | Result spec: `type:path:name` (type=vllm\|vidur, name optional). Can be repeated. |
| `--vllm-result`, `-v` | [Legacy] Path to vLLM benchmark JSON file |
| `--vllm-csv` | [Legacy] Path to per-request CSV file |
| `--vidur-result`, `-d` | [Legacy] Path to Vidur `simulator_output/<timestamp>/` directory |
| `--output-dir`, `-o` | Output directory (default: `./comparison_output`) |
| `--no-plot` | Skip generating plots |
| `--no-align` | Do not align start times in progress plot |

## Result Specification Format

```
type:path:name

- type:  'vllm' or 'vidur'
- path:  Path to result file (vllm: JSON or CSV) or directory (vidur)
- name:  Display name for legends (optional)
```

Examples:
- `vllm:./result.json:My-vLLM` (JSON file)
- `vllm:./result_requests.csv:My-vLLM` (CSV file)
- `vidur:./simulator_output/2024-01-01/:Vidur-Sim`
- `vllm:./result.json` (name auto-generated as VLLM-1)

## Output

```
comparison_output/
├── comparison_summary.csv    # Metrics comparison table
├── request_progress.png      # Request completion timeline (multi-line)
└── metrics_comparison.png    # Bar charts for TTFT/TPOT/Throughput
```

## Example Output

### Console Output
```
================================================================================
BENCHMARK COMPARISON
================================================================================
Metric                                         vLLM-QPS1       vLLM-QPS2     Vidur-QPS1
---------------------------------------------------------------------------------
TTFT Mean (ms)                                    123.45          156.78         125.30
TTFT P90 (ms)                                     200.00          250.00         198.50
TPOT Mean (ms)                                     15.50           18.20          15.80
Throughput (tokens/s)                             500.00          480.00         485.00
================================================================================
```

### Progress Plot
The progress plot shows multiple lines with different colors and styles:
- Each benchmark gets a unique color from a 10-color palette
- Line styles cycle through: solid, dashed, dash-dot, dotted

## Expected Input Formats

### vLLM Benchmark Result (JSON)

```json
{
  "mean_ttft_ms": 123.45,
  "p90_ttft_ms": 200.0,
  "p99_ttft_ms": 300.0,
  "mean_tpot_ms": 15.5,
  "p90_tpot_ms": 20.0,
  "p99_tpot_ms": 25.0,
  "output_throughput": 500.0,
  "request_throughput": 10.0,
  "completed": 1000,
  "total_output": 50000,
  "submit_times": [...],
  "finish_times": [...]
}
```

Or with per-request CSV file `*_requests.csv`:
```csv
request_id,prompt_len,output_len,ttft,tpot,latency,submit_time,finish_time
0,100,50,0.123,0.015,0.9,0.0,0.9
...
```

### Vidur Simulation Output

```
simulator_output/<timestamp>/
├── config.json
└── plots/
    ├── prefill_e2e_time.csv
    ├── decode_time_execution_plus_preemption_normalized.csv
    ├── request_arrived_at.csv
    ├── request_completed_at.csv
    ├── request_num_prefill_tokens.csv
    └── request_num_decode_tokens.csv
```

## Requirements

```
numpy
pandas
matplotlib
```
