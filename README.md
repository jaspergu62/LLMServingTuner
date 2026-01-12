# LLMServingTuner

**ModelEvalState** - LLM Inference Service Optimization Framework for Ascend NPU

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Ascend%20NPU-orange.svg)]()

## 📋 Overview

**ModelEvalState** is a Python framework developed for optimizing Large Language Model (LLM) inference performance on Ascend NPU hardware. It uses Particle Swarm Optimization (PSO) to automatically tune service parameters (batch sizes, queue delays, TP/DP configurations) for optimal performance.

### Key Features

🎯 **State Performance Modeling**
- Uses XGBoost to predict model inference latency based on batch configuration
- Enables fast simulation-based optimization without real benchmark overhead

⚡ **Parameter Optimization**
- Particle Swarm Optimization (PSO) algorithm finds optimal service configuration
- Supports multiple inference backends (Mindie, VLLM)
- Configurable optimization objectives with weighted metrics

🔄 **Dual-Mode Evaluation**
- **Real Benchmark**: Execute actual benchmarks (VLLM, AisBench) for precise measurements
- **Simulation**: Use pre-trained XGBoost models for fast parameter exploration

🔌 **Plugin System**
- Easy integration of custom inference engines and benchmarks
- Register via Python entry points in `pyproject.toml`

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PSOOptimizer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Particle     │  │ Fitness      │  │ Performance  │      │
│  │ Evolution    │──│ Calculation  │──│ Tuner        │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────────┐  ┌─────────────┐
│  Simulator  │  │  Benchmark  │
│  Interface  │  │  Interface  │
├─────────────┤  ├─────────────┤
│ • Mindie    │  │ • VLLM      │
│ • VLLM      │  │ • AisBench  │
└─────────────┘  └─────────────┘
```

### Core Components

**Configuration System** (`config/`)
- Pydantic-based settings management
- `OptimizerConfigField`: Parameter search space definition
- `PerformanceIndex`: Metrics tracking (TTFT, TPOT, throughput, success rate)

**Optimization Engine** (`optimizer/`)
- `PSOOptimizer`: Core PSO implementation
- `PerformanceTuner`: Fitness calculation with weighted metrics
- `Scheduler`: Task coordination via file-based IPC

**Simulator Interface** (`optimizer/interfaces/simulator.py`)
- Abstract base for service lifecycle management
- Implementations: `Simulator` (Mindie), `VllmSimulator`

**Benchmark Interface** (`optimizer/interfaces/benchmark.py`)
- Abstract base for performance testing
- Implementations: `VllmBenchMark`, `AisBench`

**State Prediction** (`inference/`)
- `ServiceField`: Runtime inference state simulation
- XGBoost-based latency prediction

**Model Training** (`train/`)
- `PretrainModel`: Train XGBoost models for latency prediction

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/jaspergu62/LLMServingTuner.git
cd LLMServingTuner

# Install (development mode)
pip install -e .
```

### Basic Usage

```bash
# Run optimizer with specific engine and benchmark
msserviceprofiler optimizer -e vllm -b vllm_benchmark

# List available engines and benchmarks
msserviceprofiler optimizer -h

# Continue from last checkpoint
msserviceprofiler optimizer -lb --backup
```

### Configuration

Edit `config.toml` to customize optimization parameters:

```toml
[data_storage]
base_path = "./msserviceprofiler_data"

[optimizer_engine]
engine = "mindie"  # or "vllm"
benchmark = "aisbench"  # or "vllm_benchmark"

[[optimizer_config_field]]
name = "max_batch_size"
dtype = "int"
min = 16
max = 512
config_position = "BackendConfig.ScheduleConfig.maxBatchSize"

[[optimizer_config_field]]
name = "max_queue_delay_microseconds"
dtype = "int"
min = 100
max = 10000
config_position = "BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds"
```

### OptimizerConfigField Types

| dtype | Description | Example |
|-------|-------------|---------|
| `int`, `float` | Numeric within min/max | batch_size: 16-512 |
| `bool` | Boolean (0/1) | enable_feature: 0/1 |
| `enum` | Discrete choices | dtype_param: [32, 64, 128] |
| `ratio` | Fraction of another param | dp = total_npus / tp |
| `range` | Generate enum from 0 to max | step = dtype_param |
| `factories` | Derived from other params | Computed values |
| `env` | Environment variable | Set as ENV var |

## 📊 Performance Metrics

Default fitness weights:
- **Generate Speed**: 0.4 (maximize throughput)
- **TTFT** (Time to First Token): 0.2
- **TPOT** (Time per Output Token): 0.3
- **Success Rate**: 0.1

Customize in `PerformanceTuner` class:

```python
tuner = PerformanceTuner(
    ttft_penalty=3.0,
    tpot_penalty=3.0,
    success_rate_penalty=5.0,
    ttft_slo=0.5,
    tpot_slo=0.05,
    generate_speed_target=5300
)
```

## 🔌 Plugin Development

### Register Custom Simulator

```python
# my_plugin/simulator.py
from msserviceprofiler.modelevalstate.optimizer.interfaces import SimulatorInterface

class MySimulator(SimulatorInterface):
    def start(self):
        # Start service
        pass

    def stop(self):
        # Stop service
        pass

    def update_config(self, params):
        # Update configuration
        pass
```

```toml
# pyproject.toml
[project.entry-points.'msserviceprofiler.modelevalstate.plugins']
my_plugin = "my_package:register"
```

```python
# my_package/__init__.py
from msserviceprofiler.modelevalstate.optimizer.register import register_simulator

def register():
    register_simulator("my_engine", MySimulator)
```

## 📁 Project Structure

```
.
├── config/              # Configuration management
│   ├── config.py        # Pydantic settings
│   ├── base_config.py   # Constants and enums
│   └── custom_command.py # Service launch commands
├── optimizer/           # PSO optimization engine
│   ├── optimizer.py     # Core PSO implementation
│   ├── performance_tuner.py # Fitness calculation
│   ├── scheduler.py     # Task scheduling
│   ├── interfaces/      # Abstract interfaces
│   │   ├── simulator.py
│   │   └── benchmark.py
│   └── plugins/         # Built-in implementations
│       ├── simulate.py  # Mindie/VLLM simulators
│       └── benchmark.py # VLLM/AisBench benchmarks
├── inference/           # State prediction
│   └── simulate.py      # ServiceField for simulation
├── train/               # Model training
│   └── pretrain.py      # XGBoost training
├── model/               # Saved models
├── data_feature/        # Feature engineering
└── config.toml          # Main configuration file
```

## 🛠️ Development

### Run Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black . --line-length 120

# Sort imports
isort . --profile black

# Lint
flake8 . --max-line-length 120
```

## 📖 Documentation

- **Architecture Report**: See `docs/architecture-report.html`
- **Plugin Guide**: See `optimizer/plugins/plugin.md`
- **API Documentation**: (Coming soon)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Developed by Huawei Technologies Co., Ltd.

For questions or support, please open an issue on GitHub.

---

**Note**: For parallel PSO evaluation with multi-node support, see the [`parallel_pso`](https://github.com/jaspergu62/LLMServingTuner/tree/parallel_pso) branch.
