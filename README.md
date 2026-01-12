# LLMServingTuner (Parallel PSO)

**ModelEvalState with Parallel PSO** - High-Performance LLM Inference Optimization Framework

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Ascend%20NPU-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-155%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-77%25-yellow.svg)]()

## 🚀 What's New in Parallel PSO

This branch adds **parallel particle evaluation** to the PSO optimizer, enabling simultaneous optimization across **distributed multi-node service groups**. This significantly reduces optimization time for large-scale LLM deployments.

### Key Improvements

⚡ **~4x Faster Optimization**
- Evaluate 4+ particles simultaneously across distributed service groups
- Reduce optimization time from 30 minutes to ~9 minutes (10 particles, 4 groups)

🌐 **Multi-Node Support**
- Each service group spans multiple nodes (e.g., 2 nodes × 8 NPUs)
- SSH-based remote execution and coordination
- Support for heterogeneous cluster configurations

🛡️ **Fault Tolerance**
- Circuit breaker pattern prevents cascading failures
- Automatic retry with exponential backoff
- Health monitoring with automatic recovery

📊 **Enhanced Observability**
- Real-time progress tracking
- Detailed dispatcher statistics
- Service group health monitoring

## 📋 Overview

**ModelEvalState** is a Python framework for optimizing Large Language Model (LLM) inference performance on Ascend NPU hardware. It uses Particle Swarm Optimization (PSO) with **parallel evaluation** to automatically tune service parameters for optimal performance.

### Use Case Example: Qwen3-235B

```
Optimization Scenario:
- Model: Qwen3-235B (requires 16 NPUs per service instance)
- Available: 8 servers × 8 NPUs = 64 NPUs total
- Parallel Capacity: 4 service groups

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Service Group 0 │  │ Service Group 1 │  │ Service Group 2 │  │ Service Group 3 │
│ Node 0 + Node 1 │  │ Node 2 + Node 3 │  │ Node 4 + Node 5 │  │ Node 6 + Node 7 │
│ Particle 0      │  │ Particle 1      │  │ Particle 2      │  │ Particle 3      │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘

Result: Evaluate 4 particles simultaneously → ~4x speedup
```

## 🏗️ Parallel Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Master Node                               │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │
│  │ PSOOptimizer│──│ParticleDispatcher│──│   HealthMonitor   │   │
│  └─────────────┘  └─────────────────┘  └─────────────────────┘   │
└────────────────────────┬─────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┬────────────────┐
        │                │                │                │
        ▼                ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ServiceGroup 0│  │ServiceGroup 1│  │ServiceGroup 2│  │ServiceGroup 3│
│  (Node 0+1)  │  │  (Node 2+3)  │  │  (Node 4+5)  │  │  (Node 6+7)  │
│              │  │              │  │              │  │              │
│ RemoteExec   │  │ RemoteExec   │  │ RemoteExec   │  │ RemoteExec   │
│ Service      │  │ Service      │  │ Service      │  │ Service      │
│ Benchmark    │  │ Benchmark    │  │ Benchmark    │  │ Benchmark    │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository (parallel_pso branch)
git clone -b parallel_pso https://github.com/jaspergu62/LLMServingTuner.git
cd LLMServingTuner

# Install dependencies
pip install -e .
pip install paramiko  # For SSH remote execution
```

### Parallel PSO Usage

#### Option 1: Using Hostfile (Recommended)

```bash
# Create hostfile (one host per line)
cat > hosts.txt << EOF
node0.cluster
node1.cluster
node2.cluster
node3.cluster
node4.cluster
node5.cluster
node6.cluster
node7.cluster
EOF

# Run parallel optimization
msserviceprofiler optimizer \
  --parallel \
  --hostfile hosts.txt \
  --nodes-per-group 2 \
  --npus-per-node 8
```

#### Option 2: Using Node String

```bash
# Specify node groups separated by colon
msserviceprofiler optimizer \
  --parallel \
  --node-string "node0,node1:node2,node3:node4,node5:node6,node7"
```

#### Option 3: Using Config File

```toml
# config.toml
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

# Add more service groups...
```

```bash
msserviceprofiler optimizer -e vllm -b vllm_benchmark
```

### Programmatic Usage

```python
from msserviceprofiler.modelevalstate.optimizer.parallel import (
    ParallelConfig,
    ParticleDispatcher,
    HealthMonitor
)

# Create configuration from hostfile
config = ParallelConfig.from_hostfile(
    "hosts.txt",
    nodes_per_group=2,
    npus_per_node=8
)

# Create dispatcher
dispatcher = ParticleDispatcher(
    config=config,
    target_field=target_field,
    fitness_func=fitness_func
)

# Evaluate particles in parallel
with dispatcher:
    fitness_values = dispatcher.evaluate_particles(particles)

# Check statistics
stats = dispatcher.get_stats()
print(f"Particles evaluated: {stats.particles_evaluated}")
print(f"Success rate: {stats.success_rate}")
```

## 📦 Parallel PSO Components

### Core Modules (`optimizer/parallel/`)

**config.py** (~338 lines)
- `NodeConfig`: Single node configuration
- `ServiceGroupConfig`: Multi-node service group config
- `ParallelConfig`: Overall parallel configuration with factory methods

**remote.py** (~564 lines)
- `RemoteExecutor`: SSH-based command execution
- File transfer (upload/download)
- Service lifecycle management
- Connection pooling

**service_group.py** (~589 lines)
- `ServiceGroup`: Manages multi-node service instance
- Service start/stop coordination
- Config distribution
- Benchmark execution

**pool.py** (~371 lines)
- `ServiceGroupPool`: Manages multiple service groups
- Batch particle evaluation
- Load balancing

**dispatcher.py** (~335 lines)
- `ParticleDispatcher`: Coordinates parallel particle evaluation
- Thread pool management
- Result collection
- Statistics tracking

**monitoring.py** (~551 lines)
- `HealthMonitor`: Background health checking
- `CircuitBreaker`: Prevents cascading failures
- `RetryPolicy`: Configurable retry with exponential backoff

### Test Coverage

**Test Suite** (~2,397 lines, 155 tests, 77% coverage)

| Test File | Tests | Focus |
|-----------|-------|-------|
| `test_parallel_config.py` | 27 | Configuration validation |
| `test_parallel_dispatcher.py` | 15 | Particle dispatch logic |
| `test_parallel_monitoring.py` | 38 | Fault tolerance mechanisms |
| `test_parallel_remote.py` | 32 | Remote execution via SSH |
| `test_parallel_service_group.py` | 43 | Service lifecycle management |

Run tests:
```bash
pytest tests/test_parallel_*.py -v
```

## 🔧 Configuration Options

### Parallel Configuration

```toml
[parallel]
enabled = true                    # Enable parallel evaluation
evaluation_timeout = 600          # Max seconds per particle (default: 600)
retry_count = 2                   # Number of retries on failure (default: 2)
retry_delay = 10                  # Delay between retries in seconds (default: 10)

[[parallel.service_groups]]
group_id = 0                      # Unique group identifier
start_script = "./start.sh"       # Service startup script
health_check_url = "http://{host}:8000/health"  # Health check endpoint

[[parallel.service_groups.nodes]]
host = "node0.cluster"            # Node hostname
ssh_port = 22                     # SSH port (default: 22)
npu_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # NPU IDs on this node
work_dir = "/tmp/modelevalstate"  # Working directory on node
```

### Circuit Breaker Configuration

```python
from msserviceprofiler.modelevalstate.optimizer.parallel.monitoring import (
    CircuitBreakerConfig,
    CircuitBreaker
)

breaker_config = CircuitBreakerConfig(
    failure_threshold=5,          # Open after 5 consecutive failures
    success_threshold=2,          # Close after 2 consecutive successes
    timeout=60,                   # Half-open after 60 seconds
    half_open_max_calls=3         # Max calls in half-open state
)

breaker = CircuitBreaker(config=breaker_config)
```

## 📊 Performance Comparison

### Sequential vs Parallel (10 particles, 4 service groups)

| Mode | Time | Speedup |
|------|------|---------|
| Sequential | ~30 min (3 min/particle) | 1x |
| Parallel (4 groups) | ~9 min (3 batches) | ~3.3x |
| Theoretical Max | ~7.5 min | 4x |

### Optimization Timeline

```
Sequential: P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9
            │    │    │    │    │    │    │    │    │    │
            3'   3'   3'   3'   3'   3'   3'   3'   3'   3'  = 30 minutes

Parallel:   [P0, P1, P2, P3] → [P4, P5, P6, P7] → [P8, P9]
            │                │                  │
            3 min            3 min              3 min         = ~9 minutes
```

## 🛡️ Fault Tolerance

### Circuit Breaker States

```
CLOSED ──[5 failures]──> OPEN ──[60s timeout]──> HALF_OPEN
  ↑                                                    │
  └──────────[2 successes]───────────────────────────┘
```

### Automatic Recovery

- **Health Monitor**: Background thread checks service health every 30s
- **Circuit Breaker**: Prevents sending requests to unhealthy services
- **Retry Policy**: Exponential backoff (initial: 1s, max: 60s)

### Error Handling

```python
from msserviceprofiler.modelevalstate.optimizer.parallel.monitoring import with_retry

@with_retry(max_attempts=3, initial_delay=1.0, max_delay=10.0)
def unstable_operation():
    # Your code here
    pass
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: SSH connection fails
```bash
# Solution: Verify SSH key is configured
ssh-copy-id user@node0.cluster

# Test connection
ssh node0.cluster "echo OK"
```

**Issue**: Service fails to start on remote node
```bash
# Solution: Check start script permissions
chmod +x start_service.sh

# Test script locally
./start_service.sh --config /path/to/config.json
```

**Issue**: Health check fails
```bash
# Solution: Verify health check URL
curl http://node0.cluster:8000/health

# Check firewall
sudo firewall-cmd --list-ports
```

### Debug Mode

```bash
# Enable debug logging
export MODELEVALSTATE_LEVEL=DEBUG

# Run optimizer
msserviceprofiler optimizer --parallel --hostfile hosts.txt
```

## 📖 Documentation

- **Architecture Report**: `docs/architecture-report.html`
- **Parallel PSO Guide**: `docs/parallel-pso-guide.html`
- **Code Review**: `docs/parallel-pso-code-review.html`
- **Test Report**: `docs/parallel-pso-test-report.html`
- **Improvements Changelog**: `docs/improvements-2-4-7-changelog.html`
- **Design Document**: `docs/plans/2026-01-07-parallel-pso-design.md`

## 🎯 API Reference

### ParticleDispatcher

```python
from msserviceprofiler.modelevalstate.optimizer.parallel import ParticleDispatcher

dispatcher = ParticleDispatcher(
    config: ParallelConfig,
    target_field: Tuple[OptimizerConfigField],
    fitness_func: Callable[[PerformanceIndex], float]
)

# Context manager usage (recommended)
with dispatcher:
    fitness = dispatcher.evaluate_particles(particles: np.ndarray)

# Manual lifecycle
dispatcher.start()
try:
    fitness = dispatcher.evaluate_particles(particles)
finally:
    dispatcher.stop()

# Get statistics
stats = dispatcher.get_stats()
print(f"Success rate: {stats.success_rate}")
```

### ParallelConfig Factory Methods

```python
from msserviceprofiler.modelevalstate.optimizer.parallel import ParallelConfig

# From hostfile
config = ParallelConfig.from_hostfile(
    hostfile_path="hosts.txt",
    nodes_per_group=2,
    npus_per_node=8
)

# From node string
config = ParallelConfig.from_node_string(
    node_string="node0,node1:node2,node3",
    npus_per_node=8
)

# From TOML config
config = ParallelConfig.from_toml("config.toml")
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new features
4. Ensure tests pass: `pytest tests/test_parallel_*.py`
5. Run code quality checks: `black . && isort . && flake8 .`
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run all tests with coverage
pytest tests/test_parallel_*.py --cov=optimizer/parallel --cov-report=html
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

**Developed by**: Huawei Technologies Co., Ltd.

**Contributors**:
- Core PSO framework: Huawei ModelEvalState Team
- Parallel PSO feature: Co-developed with Claude Sonnet 4.5

## 📞 Support

For questions or support:
- **Issues**: https://github.com/jaspergu62/LLMServingTuner/issues
- **Pull Requests**: https://github.com/jaspergu62/LLMServingTuner/pulls
- **Documentation**: See `docs/` directory

---

**For the baseline version without parallel PSO**, see the [`main`](https://github.com/jaspergu62/LLMServingTuner/tree/main) branch.
