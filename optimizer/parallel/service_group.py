# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ServiceGroup implementation for parallel PSO evaluation.

A ServiceGroup manages a service instance that may span multiple nodes,
handling configuration distribution, service lifecycle, and benchmarking.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import requests
from loguru import logger

from msserviceprofiler.modelevalstate.config.config import (
    OptimizerConfigField,
    PerformanceIndex,
)
from msserviceprofiler.modelevalstate.exceptions import (
    SimulatorStartError,
    SimulatorStopError,
    SimulatorHealthCheckError,
    BenchmarkExecutionError,
)
from msserviceprofiler.modelevalstate.optimizer.parallel.config import (
    ServiceGroupConfig,
    NodeConfig,
)

# Import RemoteExecutor - may not be available if paramiko is not installed
try:
    from msserviceprofiler.modelevalstate.optimizer.parallel.remote import (
        RemoteExecutor,
        PARAMIKO_AVAILABLE,
    )
except ImportError:
    RemoteExecutor = None
    PARAMIKO_AVAILABLE = False


def default_executor_factory(node: NodeConfig) -> "RemoteExecutor":
    """
    Default factory function to create RemoteExecutor instances.

    Args:
        node: Node configuration

    Returns:
        RemoteExecutor instance

    Raises:
        ImportError: If paramiko is not installed
    """
    if not PARAMIKO_AVAILABLE:
        raise ImportError(
            "paramiko is required for remote execution. "
            "Install with: pip install paramiko"
        )
    return RemoteExecutor(node)


class ServiceGroupStatus(Enum):
    """Status of a service group."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EvaluationResult:
    """Result of a particle evaluation."""
    fitness: float
    performance_index: Optional[PerformanceIndex] = None
    params: Optional[Tuple[OptimizerConfigField, ...]] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    group_id: int = 0


class ServiceGroup:
    """
    Manages a service instance spanning multiple nodes.

    This class handles:
    - Configuration distribution to the master node
    - Service start/stop via scripts
    - Health monitoring
    - Benchmark execution
    - Result collection

    The actual remote execution is delegated to a RemoteExecutor
    (to be implemented in Phase 2). For now, this class supports
    local execution for testing.

    Example:
        config = ServiceGroupConfig(
            group_id=0,
            nodes=[NodeConfig(host="node0"), NodeConfig(host="node1")]
        )
        group = ServiceGroup(config)
        group.setup()

        # Evaluate a particle
        result = group.evaluate(params, target_field, fitness_func)

        group.cleanup()
    """

    def __init__(self,
                 config: ServiceGroupConfig,
                 executor_factory: Optional[Callable[[NodeConfig], Any]] = None,
                 use_remote: bool = True):
        """
        Initialize a ServiceGroup.

        Args:
            config: Service group configuration
            executor_factory: Factory function to create remote executors.
                             If None and use_remote=True, uses default_executor_factory.
                             If None and use_remote=False, uses local mode (for testing).
            use_remote: If True, use remote execution via SSH.
                       If False, use local mode (for testing/development).
        """
        self.config = config
        self.use_remote = use_remote

        # Set up executor factory
        if executor_factory:
            self.executor_factory = executor_factory
        elif use_remote:
            self.executor_factory = default_executor_factory
        else:
            self.executor_factory = None

        self._status = ServiceGroupStatus.IDLE
        self._executors: Dict[str, Any] = {}
        self._current_params: Optional[Tuple[OptimizerConfigField, ...]] = None
        self._last_error: Optional[str] = None

    @property
    def group_id(self) -> int:
        """Get the service group ID."""
        return self.config.group_id

    @property
    def status(self) -> ServiceGroupStatus:
        """Get current status."""
        return self._status

    @property
    def is_healthy(self) -> bool:
        """Check if the service group is in a healthy state."""
        return self._status in (ServiceGroupStatus.IDLE, ServiceGroupStatus.RUNNING)

    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error

    def setup(self) -> bool:
        """
        Initialize connections to all nodes.

        For remote mode, creates RemoteExecutor instances and establishes
        SSH connections to all nodes in the service group.

        Returns:
            True if setup successful, False otherwise.
        """
        logger.info(f"Setting up ServiceGroup {self.group_id}")
        try:
            for node in self.config.nodes:
                if self.executor_factory:
                    executor = self.executor_factory(node)
                    self._executors[node.host] = executor
                    logger.debug(f"Created executor for {node.host}")

                    # Connect to the node
                    if hasattr(executor, 'connect'):
                        if not executor.connect():
                            raise ConnectionError(
                                f"Failed to connect to {node.host}: "
                                f"{getattr(executor, 'last_error', 'unknown error')}"
                            )
                        logger.info(f"Connected to {node.host}")
                else:
                    # Local mode - no remote executors needed
                    logger.debug(f"Local mode: skipping executor for {node.host}")

            logger.info(f"ServiceGroup {self.group_id} setup complete with {len(self._executors)} executors")
            return True
        except Exception as e:
            self._status = ServiceGroupStatus.ERROR
            self._last_error = str(e)
            logger.error(f"Failed to setup ServiceGroup {self.group_id}: {e}")
            # Clean up any already created executors
            self.cleanup()
            return False

    def cleanup(self):
        """Release all resources and connections."""
        logger.info(f"Cleaning up ServiceGroup {self.group_id}")
        for host, executor in self._executors.items():
            try:
                if hasattr(executor, 'close'):
                    executor.close()
                logger.debug(f"Closed executor for {host}")
            except Exception as e:
                logger.warning(f"Error closing executor for {host}: {e}")
        self._executors.clear()
        self._status = ServiceGroupStatus.IDLE

    def update_config(self,
                      params: Tuple[OptimizerConfigField, ...]) -> bool:
        """
        Update service configuration on the master node.

        Args:
            params: Tuple of optimizer config fields with values

        Returns:
            True if update successful, False otherwise.
        """
        logger.debug(f"Updating config for ServiceGroup {self.group_id}")
        try:
            # Convert params to config dict
            config_dict = self._params_to_config_dict(params)

            if self.executor_factory and self.config.master_node.host in self._executors:
                # Remote mode - upload config to master node
                executor = self._executors[self.config.master_node.host]
                config_json = json.dumps(config_dict, indent=2)

                if hasattr(executor, 'write_file'):
                    executor.write_file(self.config.config_path, config_json)
                else:
                    logger.warning("Executor does not support write_file, skipping upload")
            else:
                # Local mode - just log the config
                logger.debug(f"Config update (local mode): {config_dict}")

            self._current_params = params
            return True

        except Exception as e:
            self._last_error = f"Config update failed: {e}"
            logger.error(self._last_error)
            return False

    def start(self) -> bool:
        """
        Start the service on all nodes.

        The start script on the master node is responsible for
        coordinating multi-node startup.

        Returns:
            True if start successful, False otherwise.
        """
        logger.info(f"Starting service for ServiceGroup {self.group_id}")
        self._status = ServiceGroupStatus.STARTING

        try:
            master = self.config.master_node
            start_cmd = f"{self.config.start_script} --config {self.config.config_path}"

            if self.executor_factory and master.host in self._executors:
                # Remote mode - execute start script on master
                executor = self._executors[master.host]
                if hasattr(executor, 'execute'):
                    result = executor.execute(start_cmd)
                    # Handle both CommandResult objects and tuples
                    if hasattr(result, 'exit_code'):
                        exit_code = result.exit_code
                        stderr = result.stderr
                    else:
                        exit_code, _, stderr = result

                    if exit_code != 0:
                        raise SimulatorStartError(
                            simulator_type="service_group",
                            reason=f"Start script failed with exit code {exit_code}: {stderr}",
                            command=start_cmd
                        )
                    logger.debug(f"Start script executed successfully on {master.host}")
                else:
                    logger.warning("Executor does not support execute")
            else:
                # Local mode - just log
                logger.debug(f"Start command (local mode): {start_cmd}")

            # Wait for health check
            if not self._wait_for_healthy():
                raise SimulatorStartError(
                    simulator_type="service_group",
                    reason="Health check timeout",
                    command=start_cmd
                )

            self._status = ServiceGroupStatus.RUNNING
            logger.info(f"ServiceGroup {self.group_id} started successfully")
            return True

        except SimulatorStartError:
            self._status = ServiceGroupStatus.ERROR
            raise
        except Exception as e:
            self._status = ServiceGroupStatus.ERROR
            self._last_error = str(e)
            raise SimulatorStartError(
                simulator_type="service_group",
                reason=str(e)
            )

    def stop(self) -> bool:
        """
        Stop the service on all nodes.

        Returns:
            True if stop successful, False otherwise.
        """
        logger.info(f"Stopping service for ServiceGroup {self.group_id}")
        self._status = ServiceGroupStatus.STOPPING

        try:
            master = self.config.master_node

            if self.config.stop_script:
                stop_cmd = self.config.stop_script
            else:
                # Default: kill processes
                stop_cmd = "pkill -f 'start_service'"

            if self.executor_factory and master.host in self._executors:
                executor = self._executors[master.host]
                if hasattr(executor, 'execute'):
                    executor.execute(stop_cmd)
            else:
                logger.debug(f"Stop command (local mode): {stop_cmd}")

            self._status = ServiceGroupStatus.IDLE
            logger.info(f"ServiceGroup {self.group_id} stopped")
            return True

        except Exception as e:
            self._status = ServiceGroupStatus.ERROR
            self._last_error = str(e)
            logger.error(f"Failed to stop ServiceGroup {self.group_id}: {e}")
            return False

    def run_benchmark(self,
                      benchmark_func: Optional[Callable[[], PerformanceIndex]] = None
                      ) -> PerformanceIndex:
        """
        Run benchmark and collect performance metrics.

        Args:
            benchmark_func: Optional benchmark function. If None, uses
                           default benchmark execution.

        Returns:
            PerformanceIndex with collected metrics.

        Raises:
            BenchmarkExecutionError: If benchmark fails.
        """
        logger.debug(f"Running benchmark for ServiceGroup {self.group_id}")

        try:
            if benchmark_func:
                return benchmark_func()
            else:
                # Default: return empty metrics (to be implemented with actual benchmark)
                logger.warning("No benchmark function provided, returning empty metrics")
                return PerformanceIndex()

        except Exception as e:
            raise BenchmarkExecutionError(
                benchmark_type="service_group",
                reason=str(e)
            )

    def evaluate(self,
                 params: Tuple[OptimizerConfigField, ...],
                 fitness_func: Callable[[PerformanceIndex], float],
                 benchmark_func: Optional[Callable[[], PerformanceIndex]] = None
                 ) -> EvaluationResult:
        """
        Evaluate a single particle (parameter set).

        This is the main entry point for particle evaluation:
        1. Update config with params
        2. Restart service
        3. Run benchmark
        4. Calculate fitness

        Args:
            params: Parameter values to evaluate
            fitness_func: Function to calculate fitness from performance metrics
            benchmark_func: Optional custom benchmark function

        Returns:
            EvaluationResult with fitness and metrics.
        """
        start_time = time.time()
        logger.info(f"Evaluating particle on ServiceGroup {self.group_id}")

        try:
            # 1. Update configuration
            if not self.update_config(params):
                return EvaluationResult(
                    fitness=float('inf'),
                    error="Config update failed",
                    group_id=self.group_id,
                    duration_seconds=time.time() - start_time
                )

            # 2. Stop existing service (if running)
            if self._status == ServiceGroupStatus.RUNNING:
                self.stop()

            # 3. Start service with new config
            self.start()

            # 4. Run benchmark
            perf_index = self.run_benchmark(benchmark_func)

            # 5. Calculate fitness
            fitness = fitness_func(perf_index)

            duration = time.time() - start_time
            logger.info(
                f"ServiceGroup {self.group_id} evaluation complete: "
                f"fitness={fitness:.4f}, duration={duration:.1f}s"
            )

            return EvaluationResult(
                fitness=fitness,
                performance_index=perf_index,
                params=params,
                duration_seconds=duration,
                group_id=self.group_id
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Evaluation failed on ServiceGroup {self.group_id}: {error_msg}")

            return EvaluationResult(
                fitness=float('inf'),
                error=error_msg,
                params=params,
                duration_seconds=duration,
                group_id=self.group_id
            )

    def health_check(self) -> bool:
        """
        Check if the service is healthy.

        Returns:
            True if service is responding, False otherwise.
        """
        url = self.config.get_health_check_url()
        try:
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _wait_for_healthy(self, timeout: Optional[int] = None) -> bool:
        """
        Wait for service to become healthy.

        Args:
            timeout: Timeout in seconds. Uses config default if None.

        Returns:
            True if service became healthy, False on timeout.
        """
        timeout = timeout or self.config.health_check_timeout
        start_time = time.time()
        check_interval = 5  # seconds

        logger.debug(f"Waiting for ServiceGroup {self.group_id} to become healthy")

        while time.time() - start_time < timeout:
            if self.health_check():
                logger.debug(f"ServiceGroup {self.group_id} is healthy")
                return True
            time.sleep(check_interval)

        logger.warning(f"Health check timeout for ServiceGroup {self.group_id}")
        return False

    def _params_to_config_dict(self,
                               params: Tuple[OptimizerConfigField, ...]
                               ) -> Dict[str, Any]:
        """
        Convert optimizer config fields to a config dictionary.

        Args:
            params: Tuple of optimizer config fields

        Returns:
            Dictionary suitable for JSON serialization.
        """
        config = {}
        for param in params:
            # Use config_position as the key path
            if param.config_position == "env":
                # Environment variables handled separately
                if "env" not in config:
                    config["env"] = {}
                config["env"][param.name] = param.value
            else:
                # Nested config path
                self._set_nested_value(config, param.config_position, param.value)
        return config

    def _set_nested_value(self,
                          config: Dict[str, Any],
                          path: str,
                          value: Any):
        """
        Set a value in a nested dictionary using a dotted path.

        Args:
            config: The config dictionary to modify
            path: Dotted path like "BackendConfig.ScheduleConfig.maxBatchSize"
            value: Value to set
        """
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key.isdigit():
                # Handle array index - ensure parent is a list
                idx = int(key)
                if not isinstance(current, list):
                    continue
                while len(current) <= idx:
                    current.append({})
                current = current[idx]
            else:
                if key not in current:
                    # Check if next key is numeric (array)
                    next_idx = keys.index(key) + 1
                    if next_idx < len(keys) and keys[next_idx].isdigit():
                        current[key] = []
                    else:
                        current[key] = {}
                current = current[key]

        # Set the final value
        final_key = keys[-1]
        if final_key.isdigit() and isinstance(current, list):
            idx = int(final_key)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        else:
            current[final_key] = value

    def __repr__(self) -> str:
        return (
            f"ServiceGroup(id={self.group_id}, "
            f"status={self._status.value}, "
            f"nodes={self.config.node_count})"
        )
