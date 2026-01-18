# LLMServingTuner

An LLM serving auto‑tuning framework.

## Platforms
- Ascend NPU (MindIE, vLLM-Ascend)
- NVIDIA GPU (vLLM)

## Workflow & Examples

### 0) Apply patch (one‑time)
Apply the patch to your serving framework so it can collect features / enable simulation:
```bash
python -c "from llmservingtuner.patch import enable_patch; enable_patch('LLMSERVINGTUNER_SIMULATE')"
```

### 1) Profile (single CSV)
Set profile mode and run your normal workload. A single CSV will be written to `/tmp/profile/`.
```bash
export LLMSERVINGTUNER_PROFILE=true
# start your service and run workload, e.g. /tmp/profile/20250101-1200.csv
```

For profiling with more information, can check `msserviceprofiler.ms_service_profiler` and use `llmserving/train/pretrain.py` instead.

### 2) Train (from the single profile CSV)
```bash
python examples/train_from_vllm_profile.py \
  --profile-csv /tmp/profile/20250101-1200.csv \
  --output-dir /path/to/model_output
```

### 3) Simulate benchmark
```bash
export LLMSERVINGTUNER_SIMULATE=true
# run your usual benchmark/workload (simulation model will be used)
```

### 4) Configuration tunning (parameter optimization)

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
