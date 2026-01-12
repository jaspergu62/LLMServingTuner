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
Unit tests for ServiceGroup and ServiceGroupPool classes.
"""

from unittest.mock import Mock, MagicMock, patch
import pytest
import numpy as np

from msserviceprofiler.modelevalstate.optimizer.parallel.config import (
    NodeConfig,
    ServiceGroupConfig,
    ParallelConfig,
)
from msserviceprofiler.modelevalstate.optimizer.parallel.service_group import (
    ServiceGroup,
    ServiceGroupStatus,
    EvaluationResult,
)
from msserviceprofiler.modelevalstate.optimizer.parallel.pool import (
    ServiceGroupPool,
    PoolStats,
)


class MockExecutor:
    """Mock executor for testing without SSH."""

    def __init__(self, node: NodeConfig):
        self.node = node
        self.host = node.host
        self.connected = False
        self.last_error = None
        self.commands = []
        self.files = {}

    def connect(self) -> bool:
        self.connected = True
        return True

    def close(self):
        self.connected = False

    def execute(self, command: str, **kwargs):
        self.commands.append(command)
        # Return mock CommandResult
        result = Mock()
        result.exit_code = 0
        result.stdout = "OK"
        result.stderr = ""
        return result

    def write_file(self, path: str, content: str):
        self.files[path] = content
        return True


class TestServiceGroupStatus:
    """Tests for ServiceGroupStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert ServiceGroupStatus.IDLE.value == "idle"
        assert ServiceGroupStatus.STARTING.value == "starting"
        assert ServiceGroupStatus.RUNNING.value == "running"
        assert ServiceGroupStatus.STOPPING.value == "stopping"
        assert ServiceGroupStatus.ERROR.value == "error"


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_successful_result(self):
        """Test successful evaluation result."""
        result = EvaluationResult(
            fitness=0.5,
            duration_seconds=10.5,
            group_id=0
        )
        assert result.fitness == 0.5
        assert result.error is None
        assert result.duration_seconds == 10.5

    def test_failed_result(self):
        """Test failed evaluation result."""
        result = EvaluationResult(
            fitness=float('inf'),
            error="Connection failed",
            group_id=1
        )
        assert result.fitness == float('inf')
        assert result.error == "Connection failed"


class TestServiceGroup:
    """Tests for ServiceGroup class."""

    @pytest.fixture
    def simple_config(self):
        """Create a simple service group config for testing."""
        return ServiceGroupConfig(
            group_id=0,
            nodes=[NodeConfig(host="localhost")],
            start_script="./start.sh",
            config_path="/tmp/config.json"
        )

    @pytest.fixture
    def multi_node_config(self):
        """Create a multi-node service group config."""
        return ServiceGroupConfig(
            group_id=1,
            nodes=[
                NodeConfig(host="master"),
                NodeConfig(host="worker1"),
                NodeConfig(host="worker2")
            ],
            master_node_index=0
        )

    def test_init_local_mode(self, simple_config):
        """Test ServiceGroup initialization in local mode."""
        group = ServiceGroup(simple_config, use_remote=False)
        assert group.group_id == 0
        assert group.status == ServiceGroupStatus.IDLE
        assert group.is_healthy is True
        assert group.executor_factory is None

    def test_init_with_custom_executor(self, simple_config):
        """Test ServiceGroup with custom executor factory."""
        group = ServiceGroup(
            simple_config,
            executor_factory=MockExecutor,
            use_remote=True
        )
        assert group.executor_factory == MockExecutor

    def test_setup_local_mode(self, simple_config):
        """Test setup in local mode (no remote executors)."""
        group = ServiceGroup(simple_config, use_remote=False)
        result = group.setup()
        assert result is True
        assert group.status == ServiceGroupStatus.IDLE

    def test_setup_with_mock_executor(self, simple_config):
        """Test setup with mock executor."""
        group = ServiceGroup(
            simple_config,
            executor_factory=MockExecutor,
            use_remote=True
        )
        result = group.setup()
        assert result is True
        assert "localhost" in group._executors

    def test_cleanup(self, simple_config):
        """Test cleanup releases resources."""
        group = ServiceGroup(
            simple_config,
            executor_factory=MockExecutor,
            use_remote=True
        )
        group.setup()
        assert len(group._executors) == 1

        group.cleanup()
        assert len(group._executors) == 0
        assert group.status == ServiceGroupStatus.IDLE

    def test_update_config_local_mode(self, simple_config):
        """Test config update in local mode."""
        from msserviceprofiler.modelevalstate.config.config import OptimizerConfigField

        group = ServiceGroup(simple_config, use_remote=False)
        group.setup()

        params = (
            OptimizerConfigField(
                name="max_batch_size",
                config_position="BackendConfig.max_batch_size",
                min=1,
                max=256,
                value=64,
                dtype="int"
            ),
        )
        result = group.update_config(params)
        assert result is True
        assert group._current_params == params

    def test_update_config_with_executor(self, simple_config):
        """Test config update writes to remote file."""
        from msserviceprofiler.modelevalstate.config.config import OptimizerConfigField

        group = ServiceGroup(
            simple_config,
            executor_factory=MockExecutor,
            use_remote=True
        )
        group.setup()

        params = (
            OptimizerConfigField(
                name="max_batch_size",
                config_position="BackendConfig.ScheduleConfig.maxBatchSize",
                min=1,
                max=256,
                value=128,
                dtype="int"
            ),
        )
        result = group.update_config(params)
        assert result is True

        # Verify file was written to executor
        executor = group._executors["localhost"]
        assert simple_config.config_path in executor.files

    def test_params_to_config_dict(self, simple_config):
        """Test conversion of params to config dict."""
        from msserviceprofiler.modelevalstate.config.config import OptimizerConfigField

        group = ServiceGroup(simple_config, use_remote=False)

        params = (
            OptimizerConfigField(
                name="max_batch_size",
                config_position="BackendConfig.maxBatchSize",
                min=1,
                max=256,
                value=64,
                dtype="int"
            ),
            OptimizerConfigField(
                name="prefix_cache",
                config_position="BackendConfig.prefixCache",
                min=0,
                max=1,
                value=True,
                dtype="bool"
            ),
        )

        config_dict = group._params_to_config_dict(params)
        assert config_dict["BackendConfig"]["maxBatchSize"] == 64
        assert config_dict["BackendConfig"]["prefixCache"] is True

    def test_params_to_config_dict_env_vars(self, simple_config):
        """Test env variable params in config dict."""
        from msserviceprofiler.modelevalstate.config.config import OptimizerConfigField

        group = ServiceGroup(simple_config, use_remote=False)

        params = (
            OptimizerConfigField(
                name="SOME_ENV_VAR",
                config_position="env",
                min=0,
                max=100,
                value=42,
                dtype="int"
            ),
        )

        config_dict = group._params_to_config_dict(params)
        assert config_dict["env"]["SOME_ENV_VAR"] == 42

    def test_repr(self, simple_config):
        """Test string representation."""
        group = ServiceGroup(simple_config, use_remote=False)
        repr_str = repr(group)
        assert "id=0" in repr_str
        assert "idle" in repr_str
        assert "nodes=1" in repr_str


class TestServiceGroupPool:
    """Tests for ServiceGroupPool class."""

    @pytest.fixture
    def pool_config(self):
        """Create a ParallelConfig for pool testing."""
        return ParallelConfig(
            enabled=True,
            service_groups=[
                ServiceGroupConfig(
                    group_id=0,
                    nodes=[NodeConfig(host="node0")]
                ),
                ServiceGroupConfig(
                    group_id=1,
                    nodes=[NodeConfig(host="node1")]
                )
            ],
            evaluation_timeout=60,
            retry_count=1
        )

    def test_init(self, pool_config):
        """Test pool initialization."""
        pool = ServiceGroupPool(pool_config, use_remote=False)
        assert pool.worker_count == 0  # Not setup yet
        assert pool._is_setup is False

    def test_setup_local_mode(self, pool_config):
        """Test pool setup in local mode."""
        pool = ServiceGroupPool(pool_config, use_remote=False)
        result = pool.setup()
        assert result is True
        assert pool._is_setup is True
        assert pool.worker_count == 2
        assert len(pool.groups) == 2
        assert len(pool.healthy_groups) == 2

    def test_cleanup(self, pool_config):
        """Test pool cleanup."""
        pool = ServiceGroupPool(pool_config, use_remote=False)
        pool.setup()
        assert pool.worker_count == 2

        pool.cleanup()
        assert pool.worker_count == 0
        assert pool._is_setup is False

    def test_context_manager(self, pool_config):
        """Test pool as context manager."""
        with ServiceGroupPool(pool_config, use_remote=False) as pool:
            assert pool.worker_count == 2
            assert pool._is_setup is True
        assert pool._is_setup is False

    def test_get_group(self, pool_config):
        """Test getting group by ID."""
        pool = ServiceGroupPool(pool_config, use_remote=False)
        pool.setup()

        group0 = pool.get_group(0)
        assert group0 is not None
        assert group0.group_id == 0

        group1 = pool.get_group(1)
        assert group1 is not None
        assert group1.group_id == 1

        assert pool.get_group(99) is None

    def test_stats(self, pool_config):
        """Test pool statistics."""
        pool = ServiceGroupPool(pool_config, use_remote=False)
        pool.setup()

        stats = pool.stats
        assert stats.total_groups == 2
        assert stats.healthy_groups == 2
        assert stats.evaluations_completed == 0
        assert stats.evaluations_failed == 0

    def test_evaluate_batch_not_setup(self, pool_config):
        """Test evaluate_batch raises error when not setup."""
        from msserviceprofiler.modelevalstate.config.config import OptimizerConfigField

        pool = ServiceGroupPool(pool_config, use_remote=False)
        particles = np.array([[1.0, 2.0], [3.0, 4.0]])
        target_field = (
            OptimizerConfigField(
                name="param1",
                config_position="config.param1",
                min=0,
                max=10,
                dtype="float"
            ),
        )

        with pytest.raises(RuntimeError, match="Pool not setup"):
            pool.evaluate_batch(particles, target_field, lambda x: 0.0)

    def test_repr(self, pool_config):
        """Test string representation."""
        pool = ServiceGroupPool(pool_config, use_remote=False)
        pool.setup()

        repr_str = repr(pool)
        assert "groups=2" in repr_str
        assert "healthy=2" in repr_str


class TestPoolStats:
    """Tests for PoolStats dataclass."""

    def test_default_values(self):
        """Test default stat values."""
        stats = PoolStats()
        assert stats.total_groups == 0
        assert stats.healthy_groups == 0
        assert stats.evaluations_completed == 0
        assert stats.evaluations_failed == 0
        assert stats.total_evaluation_time == 0.0

    def test_success_rate_no_evaluations(self):
        """Test success rate with no evaluations."""
        stats = PoolStats()
        assert stats.success_rate == 0.0

    def test_success_rate_all_success(self):
        """Test success rate with all successful evaluations."""
        stats = PoolStats(evaluations_completed=10, evaluations_failed=0)
        assert stats.success_rate == 1.0

    def test_success_rate_mixed(self):
        """Test success rate with mixed results."""
        stats = PoolStats(evaluations_completed=7, evaluations_failed=3)
        assert stats.success_rate == 0.7

    def test_avg_evaluation_time_no_completions(self):
        """Test avg time with no completions."""
        stats = PoolStats(total_evaluation_time=100.0, evaluations_completed=0)
        assert stats.avg_evaluation_time == 0.0

    def test_avg_evaluation_time(self):
        """Test avg time calculation."""
        stats = PoolStats(total_evaluation_time=50.0, evaluations_completed=5)
        assert stats.avg_evaluation_time == 10.0
