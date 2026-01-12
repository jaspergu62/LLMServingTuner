# Parallel PSO Evaluation Design (Multi-Node)

## Overview

Design for parallel particle evaluation in PSO optimizer across multiple service groups, where each service group spans 1-2 nodes.

## Scenario

```
Example: Qwen3-235B optimization
- Each service needs: 2 nodes × 8 NPUs = 16 NPUs
- Available: 8 servers (8 × 8 = 64 NPUs total)
- Parallel capacity: 4 service groups

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Service Group 0 │  │  Service Group 1 │  │  Service Group 2 │  │  Service Group 3 │
│  Node 0 + Node 1 │  │  Node 2 + Node 3 │  │  Node 4 + Node 5 │  │  Node 6 + Node 7 │
│  Particle 0      │  │  Particle 1      │  │  Particle 2      │  │  Particle 3      │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Architecture

### Master-Worker Model

```
┌──────────────────────────────────────────────────────────────────┐
│                         Master Node                               │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │
│  │ PSOOptimizer │──│ ParticleDispatcher│──│ ResultCollector   │   │
│  └─────────────┘  └─────────────────┘  └─────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
         │                    │                      ▲
         │    dispatch        │                      │ results
         ▼                    ▼                      │
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Worker 0   │  │  Worker 1   │  │  Worker 2   │  │  Worker 3   │
│  (Group 0)  │  │  (Group 1)  │  │  (Group 2)  │  │  (Group 3)  │
│ Node 0 + 1  │  │ Node 2 + 3  │  │ Node 4 + 5  │  │ Node 6 + 7  │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

## Phased Implementation

### Phase 1: Service Group Abstraction (Foundation)

**Goal**: Abstract the concept of "service group" that spans multiple nodes.

```python
# optimizer/service_group.py

@dataclass
class NodeConfig:
    """Configuration for a single node in the service group."""
    host: str
    ssh_port: int = 22
    npu_ids: List[int] = field(default_factory=lambda: list(range(8)))
    work_dir: str = "/tmp/modelevalstate"


@dataclass
class ServiceGroupConfig:
    """Configuration for a service group spanning 1+ nodes."""
    group_id: int
    nodes: List[NodeConfig]
    master_node_index: int = 0  # Which node runs the main service process


class ServiceGroup:
    """
    Manages a service instance that spans multiple nodes.
    Handles config distribution, service start/stop across nodes.
    """

    def __init__(self, config: ServiceGroupConfig,
                 simulator_cls: Type[SimulatorInterface],
                 benchmark_cls: Type[BenchmarkInterface]):
        self.config = config
        self.simulator_cls = simulator_cls
        self.benchmark_cls = benchmark_cls
        self._remote_executors: Dict[str, RemoteExecutor] = {}

    def setup(self):
        """Establish connections to all nodes."""
        for node in self.config.nodes:
            self._remote_executors[node.host] = RemoteExecutor(node)

    def update_config(self, params: Tuple[OptimizerConfigField]) -> bool:
        """Distribute config to all nodes."""
        master = self.config.nodes[self.config.master_node_index]
        # Update config on master node, it will sync to workers
        return self._remote_executors[master.host].update_config(params)

    def start(self) -> bool:
        """Start service across all nodes."""
        # Start workers first, then master
        ...

    def run_benchmark(self) -> PerformanceIndex:
        """Run benchmark from master node."""
        ...

    def stop(self):
        """Stop service on all nodes."""
        ...

    def cleanup(self):
        """Release all resources."""
        for executor in self._remote_executors.values():
            executor.close()
```

**Deliverables**:
- `ServiceGroup` class
- `NodeConfig` / `ServiceGroupConfig` dataclasses
- Unit tests for local node execution

---

### Phase 2: Remote Execution Layer

**Goal**: Enable command execution on remote nodes via SSH.

```python
# optimizer/remote.py

class RemoteExecutor:
    """Execute commands on a remote node via SSH."""

    def __init__(self, node: NodeConfig):
        self.node = node
        self._ssh_client = None

    def connect(self):
        """Establish SSH connection."""
        import paramiko
        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh_client.connect(
            self.node.host,
            port=self.node.ssh_port,
            # Use SSH key authentication (no password)
        )

    def execute(self, command: str, timeout: int = 300) -> Tuple[int, str, str]:
        """Execute command and return (exit_code, stdout, stderr)."""
        ...

    def upload_file(self, local_path: str, remote_path: str):
        """Upload file to remote node."""
        ...

    def download_file(self, remote_path: str, local_path: str):
        """Download file from remote node."""
        ...

    def update_config(self, params: Tuple[OptimizerConfigField]) -> bool:
        """Update service config on remote node."""
        # 1. Serialize params to JSON
        # 2. Upload to remote work_dir
        # 3. Execute config update command
        ...

    def start_service(self, command: str, env: Dict[str, str]) -> int:
        """Start service and return PID."""
        ...

    def stop_service(self, pid: int):
        """Stop service by PID."""
        ...

    def close(self):
        """Close SSH connection."""
        if self._ssh_client:
            self._ssh_client.close()
```

**Deliverables**:
- `RemoteExecutor` class with SSH support
- Config/file transfer utilities
- Integration tests with 2-node setup

---

### Phase 3: Parallel Dispatcher

**Goal**: Dispatch particles to service groups and collect results.

```python
# optimizer/parallel.py

class ParticleDispatcher:
    """
    Dispatches particles to available service groups for evaluation.
    Manages the work queue and result collection.
    """

    def __init__(self,
                 service_groups: List[ServiceGroup],
                 fitness_func: Callable):
        self.service_groups = service_groups
        self.fitness_func = fitness_func
        self.n_workers = len(service_groups)

    def evaluate_batch(self,
                       particles: np.ndarray,
                       target_field: Tuple) -> np.ndarray:
        """
        Evaluate all particles across service groups.
        Uses thread pool for I/O-bound coordination.
        """
        n_particles = particles.shape[0]
        results = [float('inf')] * n_particles

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Process in batches matching worker count
            for batch_start in range(0, n_particles, self.n_workers):
                batch_end = min(batch_start + self.n_workers, n_particles)
                batch = particles[batch_start:batch_end]

                futures = {}
                for i, particle in enumerate(batch):
                    group_idx = i % len(self.service_groups)
                    future = executor.submit(
                        self._evaluate_on_group,
                        self.service_groups[group_idx],
                        particle,
                        target_field
                    )
                    futures[future] = batch_start + i

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result(timeout=600)
                    except Exception as e:
                        logger.error(f"Particle {idx} failed: {e}")
                        results[idx] = float('inf')

        return np.array(results)

    def _evaluate_on_group(self,
                           group: ServiceGroup,
                           particle: np.ndarray,
                           target_field: Tuple) -> float:
        """Evaluate single particle on a service group."""
        try:
            # 1. Update config
            params = self._particle_to_params(particle, target_field)
            group.update_config(params)

            # 2. Restart service
            group.stop()
            group.start()

            # 3. Run benchmark
            perf = group.run_benchmark()

            # 4. Calculate fitness
            return self.fitness_func(perf)

        except Exception as e:
            logger.error(f"Evaluation failed on group {group.config.group_id}: {e}")
            return float('inf')
```

**Deliverables**:
- `ParticleDispatcher` class
- Batch processing logic
- Error handling and retry logic

---

### Phase 4: Configuration & CLI Integration

**Goal**: User-friendly configuration for parallel execution.

```toml
# config.toml additions

[parallel]
enabled = true
mode = "multi_node"  # "single_node" | "multi_node"

[[parallel.service_groups]]
group_id = 0
[[parallel.service_groups.nodes]]
host = "node0.cluster"
npu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
[[parallel.service_groups.nodes]]
host = "node1.cluster"
npu_ids = [0, 1, 2, 3, 4, 5, 6, 7]

[[parallel.service_groups]]
group_id = 1
[[parallel.service_groups.nodes]]
host = "node2.cluster"
npu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
[[parallel.service_groups.nodes]]
host = "node3.cluster"
npu_ids = [0, 1, 2, 3, 4, 5, 6, 7]

# ... more groups
```

**CLI**:
```bash
# Auto-detect from hostfile (like MPI)
msserviceprofiler optimizer --parallel --hostfile hosts.txt

# Explicit node specification
msserviceprofiler optimizer --parallel \
  --nodes "node0,node1:node2,node3:node4,node5:node6,node7" \
  --npus-per-node 8
```

**Deliverables**:
- Config schema for parallel mode
- CLI arguments parsing
- Hostfile parsing utility

---

### Phase 5: Fault Tolerance & Monitoring

**Goal**: Handle failures gracefully, provide visibility.

```python
class ServiceGroupHealthMonitor:
    """Monitor health of service groups."""

    def __init__(self, groups: List[ServiceGroup]):
        self.groups = groups
        self._health_status: Dict[int, bool] = {}

    async def check_all(self) -> Dict[int, bool]:
        """Check health of all groups."""
        ...

    def mark_unhealthy(self, group_id: int):
        """Mark a group as unhealthy, exclude from dispatch."""
        ...

    def get_healthy_groups(self) -> List[ServiceGroup]:
        """Return only healthy groups."""
        ...


class ParallelOptimizationMonitor:
    """Real-time monitoring of parallel optimization."""

    def __init__(self, dispatcher: ParticleDispatcher):
        self.dispatcher = dispatcher
        self.metrics = {
            "particles_evaluated": 0,
            "particles_failed": 0,
            "avg_eval_time": 0.0,
            "groups_healthy": 0,
        }

    def update(self, event: str, data: dict):
        """Update metrics based on events."""
        ...

    def report(self) -> str:
        """Generate status report."""
        ...
```

**Deliverables**:
- Health monitoring
- Automatic retry on failure
- Progress reporting

---

## Implementation Timeline

| Phase | Description | Dependencies |
|-------|-------------|--------------|
| 1 | Service Group Abstraction | None |
| 2 | Remote Execution Layer | Phase 1 |
| 3 | Parallel Dispatcher | Phase 1, 2 |
| 4 | Configuration & CLI | Phase 3 |
| 5 | Fault Tolerance | Phase 3, 4 |

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SSH Authentication | Key-based, pre-configured | Simplifies implementation, standard practice |
| Service Start | Master starts via script | Script handles multi-node coordination |
| Config Storage | Master node only | Reduces sync complexity |
| Benchmark Location | Any node with service access | Flexible deployment |
| Result Storage | Local + centralized on PSO master | Local for debugging, master for aggregation |

## Detailed Design

### Service Start Flow

```
ServiceGroup.start():
    1. SSH to master node of service group
    2. Execute start script: `./start_service.sh --config /path/to/config.json`
    3. Script internally handles multi-node startup
    4. Wait for health check to pass
    5. Return success/failure
```

### Config Update Flow

```
ServiceGroup.update_config(params):
    1. Generate config JSON from params
    2. Upload config to master node: /work_dir/config.json
    3. Service start script reads from this path
```

### Result Collection Flow

```
ParticleDispatcher.evaluate_batch():
    1. Dispatch particles to service groups (parallel)
    2. Each group:
       - Updates config on its master node
       - Runs start script
       - Runs benchmark
       - Stores local result in /work_dir/results/
       - Returns fitness to dispatcher
    3. Dispatcher collects all fitness values
    4. Results aggregated on PSO master node
```
