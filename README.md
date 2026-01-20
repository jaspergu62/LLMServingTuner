# LLMServingTuner

An LLM serving auto‑tuning framework.

## Platforms
- Ascend NPU (MindIE, vLLM-Ascend)
- NVIDIA GPU (vLLM)

## Example Workflow

### 0) Environment
Reference workflow of tuning vllm configuration on H20:
- msserviceprofiler == 1.2.0
- vllm == 0.10.1
- torch == 2.7.1
- nvidia-cuda-runtime-cu12 == 12.6.77

### 1) Apply patch (one‑time)
Apply the patch to your serving framework so it can collect features / enable simulation:
```bash
python -c "from llmservingtuner.patch import enable_patch; enable_patch('LLMSERVINGTUNER_SIMULATE')"
```

### 2) Profile (single CSV)
Set profile mode and run your normal workload. A single CSV will be written to `/tmp/profile/`.
```bash
export LLMSERVINGTUNER_PROFILE=true
# start your service and run workload, e.g. /tmp/profile/20250101-1200.csv
```

For profiling with more information, can check `msserviceprofiler.ms_service_profiler` and use `llmserving/train/pretrain.py` instead.

### 3) Train (from the single profile CSV)
```bash
python examples/train_from_vllm_profile.py \
  --profile-csv /tmp/profile/20250101-1200.csv \
  --output-dir /path/to/model_output
```

### 4) Simulate benchmark
```bash
export LLMSERVINGTUNER_PROFILE=false
export LLMSERVINGTUNER_SIMULATE=true
# run your usual benchmark/workload (simulation model will be used)
```

### 5) Configuration tunning (parameter optimization)

```bash
python examples/run_tune.py \
    --engine vllm \
    --benchmark vllm_benchmark \
    --config ./config.toml
```

Alternatively, use python code with plugins: 

```python
from argparse import Namespace
from llmservingtuner.optimizer.optimizer import plugin_main

args = Namespace(
    engine="vllm",            # or "mindie"
    benchmark_policy="vllm_benchmark",  # or "ais_bench"
    load_breakpoint=False,
    backup=False,
    pd="competition",
)
plugin_main(args)
```

Config example (`config.toml`):

```toml
n_particles = 10
iters = 5

[vllm.command]
host = "127.0.0.1"
port = "8000"
model = "/path/to/model"
served_model_name = "my_model"

[[vllm.target_field]]
name = "MAX_NUM_BATCHED_TOKENS"
config_position = "env"
min = 8192
max = 65536
dtype = "int"
value = 8192
```

## Using Vidur Predictor

LLMServingTuner supports using [Vidur](https://github.com/microsoft/vidur) as an alternative execution time predictor. Vidur provides fine-grained, component-level latency prediction based on profiling data.

### Prerequisites

1. Install Vidur:
```bash
cd /path/to/vidur
pip install -e .
```

2. Prepare profiling data for your model and device (see Vidur documentation for profiling instructions).

### Configuration

To use Vidur predictor instead of the default XGBoost model, modify your `config.toml`:

```toml
[latency_model]
# Switch to Vidur predictor
predictor_type = "vidur"

# Model configuration (must match your profiling data)
vidur_model_name = "meta-llama/Llama-2-7b-hf"
vidur_device = "a100"
vidur_network_device = "a100_pairwise_nvlink"

# Parallelism configuration
vidur_tensor_parallel_size = 1
vidur_num_pipeline_stages = 1

# Scheduler configuration
vidur_block_size = 16

# Predictor type: "random_forest" or "linear_regression"
vidur_predictor_type = "random_forest"

# Cache directory for trained models
vidur_cache_dir = "cache"

# Prediction limits
vidur_prediction_max_batch_size = 128
vidur_prediction_max_tokens_per_request = 4096

# Optional: Override default profiling data paths
# vidur_compute_input_file = "./data/profiling/compute/a100/llama-2-7b/mlp.csv"
# vidur_attention_input_file = "./data/profiling/compute/a100/llama-2-7b/attention.csv"
# vidur_all_reduce_input_file = "./data/profiling/network/a100_pairwise_nvlink/all_reduce.csv"
# vidur_send_recv_input_file = "./data/profiling/network/a100_pairwise_nvlink/send_recv.csv"
```

### Supported Models

Vidur supports various model configurations. Common model names include:
- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-2-70b-hf`
- `meta-llama/Meta-Llama-3-8B`
- `meta-llama/Meta-Llama-3-70B`
- `Qwen/Qwen-72B`
- `microsoft/phi-2`

### Supported Devices

Common device configurations:
- `a100` - NVIDIA A100 GPU
- `h100` - NVIDIA H100 GPU

Network devices (for tensor parallelism):
- `a100_pairwise_nvlink`
- `h100_pairwise_nvlink`

### Switching Between Predictors

You can easily switch between XGBoost and Vidur predictors by changing `predictor_type`:

```toml
[latency_model]
# Use XGBoost (default)
predictor_type = "xgboost"
model_path = "/path/to/xgb_model.ubj"

# Or use Vidur
# predictor_type = "vidur"
# vidur_model_name = "meta-llama/Llama-2-7b-hf"
# ...
```
