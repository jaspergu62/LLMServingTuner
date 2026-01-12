# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project Rules
- **SAFETY CRITICAL**: NEVER use the `rm` command.
- If you need to delete a file, use the `safe_rm` command instead.
- `safe_rm` is available in the path and moves files to `~/.trash`.

## Project Overview

**msserviceprofiler.modelevalstate** is an LLM inference service optimization framework that uses Particle Swarm Optimization (PSO) to tune service parameters (batch sizes, queue delays, TP/DP configurations) for optimal performance. It supports multiple inference backends (Mindie, VLLM) and benchmarking tools.

## Commands

Run the optimizer with specific engine and benchmark:
```bash
msserviceprofiler optimizer -e <engine> -b <benchmark>
# e.g., msserviceprofiler optimizer -e vllm -b vllm_benchmark
```

List available engines and benchmarks:
```bash
msserviceprofiler optimizer -h
```

Continue from last optimization checkpoint:
```bash
msserviceprofiler optimizer -lb --backup
```

Install a plugin for development:
```bash
pip install -e .  # From plugin directory containing pyproject.toml
```

## Architecture

### Core Flow
```
Config (TOML/Pydantic) → PSOOptimizer → SimulatorInterface → Service Update
                                      ↓
                              BenchmarkInterface → PerformanceTuner → Fitness Score
```

### Key Modules

**config/** - Pydantic-based configuration
- `config.py` - Main `Settings` class, `OptimizerConfigField` for tuning parameters, `PerformanceIndex` for metrics
- `base_config.py` - Constants, enums (`EnginePolicy`, `BenchMarkPolicy`, `ServiceType`)
- `custom_command.py` - Service launch command builders (Mindie, VLLM, AisBench)

**optimizer/** - PSO optimization engine
- `optimizer.py` - `PSOOptimizer` class (inherits `PerformanceTuner`)
- `performance_tuner.py` - Fitness calculation with weighted metrics (generate_speed: 0.4, TTFT: 0.2, TPOT: 0.3, success_rate: 0.1)
- `scheduler.py` - Task scheduling via file-based IPC
- `communication.py` - File-based inter-process communication with `CustomCommand` pattern

**optimizer/interfaces/** - Abstract base classes
- `simulator.py` - `SimulatorInterface` for service control (start/stop/update)
- `benchmark.py` - `BenchmarkInterface` for performance testing

**optimizer/plugins/** - Built-in implementations
- `simulate.py` - `Simulator` (Mindie), `VllmSimulator`
- `benchmark.py` - `VllmBenchMark`, `AisBench`

**train/** - XGBoost model training for latency prediction
- `pretrain.py` - `PretrainModel` for training state prediction models

**inference/** - Latency simulation
- `simulate.py` - `ServiceField` for runtime inference state management

**optimizer/parallel/** - Multi-node parallel PSO evaluation
- `config.py` - `NodeConfig`, `ServiceGroupConfig`, `ParallelConfig` data classes
- `service_group.py` - `ServiceGroup` class for managing multi-node service instances
- `remote.py` - `RemoteExecutor` for SSH-based remote command execution
- `pool.py` - `ServiceGroupPool` for managing multiple service groups
- `dispatcher.py` - `ParticleDispatcher` for coordinating parallel particle evaluation
- `monitoring.py` - `HealthMonitor`, `CircuitBreaker` for fault tolerance

**exceptions.py** - Custom exception hierarchy
- `ModelEvalStateError` (base), `ConfigurationError`, `OptimizationError`, `SimulatorError`, `BenchmarkError`, `CommunicationError`

### Plugin System

Register custom implementations via entry points:
```toml
# pyproject.toml
[project.entry-points.'msserviceprofiler.modelevalstate.plugins']
my_plugin = "my_package:register"
```

Registration pattern:
```python
from msserviceprofiler.modelevalstate.config.config import register_settings
from msserviceprofiler.modelevalstate.optimizer.register import register_simulator, register_benchmarks

def register():
    register_settings(lambda: MyCustomSettings())
    register_simulator("my_engine", MySimulator)
    register_benchmarks("my_benchmark", MyBenchmark)
```

### OptimizerConfigField dtypes

| dtype | Description |
|-------|-------------|
| `int`, `float` | Numeric with min/max bounds |
| `bool` | Boolean (0/1) |
| `enum` | Discrete choices from `dtype_param` list |
| `ratio` | Fraction relative to another parameter (`dtype_param` = target field name) |
| `range` | Generate enum from 0 to max with step `dtype_param` |
| `factories` | Derived from another parameter (e.g., dp = 16/tp) |
| `env` | Set as environment variable |

### Configuration

- Main config: `config.toml` (TOML format, loaded via Pydantic settings)
- Environment variables:
  - `MODEL_EVAL_STATE_CONFIG_PATH` - Custom config path
  - `MODEL_EVAL_STATE_SIMULATE` - Enable simulation mode
  - `MODELEVALSTATE_LEVEL` - Log level (ERROR, INFO, DEBUG)

### Parallel PSO Evaluation

Parallel PSO enables evaluating multiple particles simultaneously across distributed service groups.

**Architecture:**
```
PSOOptimizer → ParticleDispatcher → ServiceGroupPool → ServiceGroup(s) → RemoteExecutor(s)
```

**CLI Options:**
```bash
# Using hostfile
msserviceprofiler optimizer --parallel --hostfile /path/to/hosts.txt --nodes-per-group 2

# Using node string
msserviceprofiler optimizer --node-string "node0,node1:node2,node3"

# Options:
#   --parallel          Enable parallel evaluation
#   --hostfile PATH     Hostfile path (one host per line)
#   --nodes-per-group N Nodes per service group (default: 2)
#   --npus-per-node N   NPUs per node (default: 8)
#   --node-string STR   Node spec: "node0,node1:node2,node3" (groups separated by :)
```

**Config File (`config.toml`):**
```toml
[parallel]
enabled = true
evaluation_timeout = 600
retry_count = 2
retry_delay = 10

[[parallel.service_groups]]
group_id = 0
start_script = "./start_service.sh"
health_check_url = "http://{host}:8000/health"
[[parallel.service_groups.nodes]]
host = "node0.cluster"
npu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
[[parallel.service_groups.nodes]]
host = "node1.cluster"
npu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
```

**Programmatic Usage:**
```python
from msserviceprofiler.modelevalstate.optimizer.parallel import (
    ParallelConfig, ParticleDispatcher, HealthMonitor
)

config = ParallelConfig.from_hostfile("hosts.txt", nodes_per_group=2)
dispatcher = ParticleDispatcher(config, target_field, fitness_func)

with dispatcher:
    fitness = dispatcher.evaluate_particles(particles)
```

**Fault Tolerance:**
- `CircuitBreaker`: Prevents cascading failures (opens after 5 consecutive failures)
- `HealthMonitor`: Background health checking with automatic recovery
- `RetryPolicy`: Configurable retry with exponential backoff

### IPC Commands (optimizer/communication.py)

Commands for scheduler coordination: `start`, `check_success`, `process_poll`, `stop`, `backup`, `init`, `eof`

## Dependencies

Core: pydantic, pydantic-settings, loguru, numpy, pandas, xgboost, scikit-learn, matplotlib, psutil, filelock

Security: Uses `msserviceprofiler.msguard` for secure file operations (`open_s`, `mkdir_s`, `walk_s`)
