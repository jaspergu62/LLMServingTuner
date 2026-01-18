# LLMServingTuner

An LLM serving auto‑tuning framework that works on both Ascend NPU and NVIDIA GPU.

## Platforms
- Ascend NPU (MindIE)
- NVIDIA GPU (vLLM)

## Workflow & Examples

### 1) Profile (collect runtime data)
Generate a profile output directory that contains at least `profiler.db` and `request.csv` (vLLM also needs `kvcache.csv`).

Example (GPU/vLLM, enable profiling):
```bash
export LLMSERVINGTUNER_PROFILE=true
# start your service and run workload to produce profiler.db/request.csv/kvcache.csv
```

### 2) Train (train the simulation model)
```python
from llmservingtuner.train.source_to_train import source_to_model, req_decodetimes
from llmservingtuner.train.pretrain import pretrain

profile_dir = "/path/to/profile_output"  # contains profiler.db/request.csv (+ kvcache.csv for vLLM)
source_to_model(profile_dir, model_type="vllm")  # or "mindie"
pretrain(f"{profile_dir}/output_csv", "/path/to/model_output")
req_decodetimes(profile_dir, "/path/to/model_output")
```

### 3) Simulate benchmark
```bash
export LLMSERVINGTUNER_SIMULATE=true
# run your usual benchmark/workload (simulation model will be used)
```

### 4) Config tune (parameter optimization)
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

## Note
There is no full documentation yet. The steps above are enough to run profile → train → simulate benchmark → config tune.
