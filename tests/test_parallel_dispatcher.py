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
Unit tests for ParticleDispatcher class.
"""

from unittest.mock import Mock, MagicMock, patch
import pytest
import numpy as np

from msserviceprofiler.modelevalstate.config.config import (
    OptimizerConfigField,
    PerformanceIndex,
)
from msserviceprofiler.modelevalstate.optimizer.parallel.config import (
    NodeConfig,
    ServiceGroupConfig,
    ParallelConfig,
)
from msserviceprofiler.modelevalstate.optimizer.parallel.dispatcher import (
    ParticleDispatcher,
    DispatcherStats,
    create_parallel_op_func,
)


class TestDispatcherStats:
    """Tests for DispatcherStats dataclass."""

    def test_default_values(self):
        """Test default stat values."""
        stats = DispatcherStats()
        assert stats.total_dispatches == 0
        assert stats.successful_evaluations == 0
        assert stats.failed_evaluations == 0
        assert stats.total_particles == 0

    def test_success_rate_no_evaluations(self):
        """Test success rate with no evaluations."""
        stats = DispatcherStats()
        assert stats.success_rate == 0.0

    def test_success_rate(self):
        """Test success rate calculation."""
        stats = DispatcherStats(
            successful_evaluations=8,
            failed_evaluations=2
        )
        assert stats.success_rate == 0.8


class TestParticleDispatcher:
    """Tests for ParticleDispatcher class."""

    @pytest.fixture
    def parallel_config(self):
        """Create a ParallelConfig for testing."""
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
            evaluation_timeout=60
        )

    @pytest.fixture
    def disabled_config(self):
        """Create a disabled ParallelConfig."""
        return ParallelConfig(enabled=False)

    @pytest.fixture
    def target_field(self):
        """Create target field for testing."""
        return (
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
                value=1,
                dtype="bool"
            ),
        )

    @pytest.fixture
    def fitness_func(self):
        """Create a simple fitness function."""
        def func(perf_index: PerformanceIndex) -> float:
            if perf_index.generate_speed:
                return -perf_index.generate_speed
            return float('inf')
        return func

    def test_init(self, parallel_config, target_field, fitness_func):
        """Test dispatcher initialization."""
        dispatcher = ParticleDispatcher(
            parallel_config=parallel_config,
            target_field=target_field,
            fitness_func=fitness_func,
            use_remote=False
        )
        assert dispatcher.parallel_config == parallel_config
        assert dispatcher.target_field == target_field
        assert dispatcher.fitness_func == fitness_func
        assert dispatcher._pool is None
        assert dispatcher._is_setup is False

    def test_is_ready_not_setup(self, parallel_config, target_field, fitness_func):
        """Test is_ready returns False when not setup."""
        dispatcher = ParticleDispatcher(
            parallel_config=parallel_config,
            target_field=target_field,
            fitness_func=fitness_func,
            use_remote=False
        )
        assert dispatcher.is_ready is False
        assert dispatcher.worker_count == 0

    def test_setup_disabled_config(self, disabled_config, target_field, fitness_func):
        """Test setup with disabled config returns False."""
        dispatcher = ParticleDispatcher(
            parallel_config=disabled_config,
            target_field=target_field,
            fitness_func=fitness_func
        )
        result = dispatcher.setup()
        assert result is False
        assert dispatcher.is_ready is False

    def test_setup_success(self, parallel_config, target_field, fitness_func):
        """Test successful setup."""
        dispatcher = ParticleDispatcher(
            parallel_config=parallel_config,
            target_field=target_field,
            fitness_func=fitness_func,
            use_remote=False
        )
        result = dispatcher.setup()
        assert result is True
        assert dispatcher.is_ready is True
        assert dispatcher.worker_count == 2

    def test_cleanup(self, parallel_config, target_field, fitness_func):
        """Test cleanup releases resources."""
        dispatcher = ParticleDispatcher(
            parallel_config=parallel_config,
            target_field=target_field,
            fitness_func=fitness_func,
            use_remote=False
        )
        dispatcher.setup()
        assert dispatcher.is_ready is True

        dispatcher.cleanup()
        assert dispatcher.is_ready is False
        assert dispatcher._pool is None

    def test_context_manager(self, parallel_config, target_field, fitness_func):
        """Test dispatcher as context manager."""
        dispatcher = ParticleDispatcher(
            parallel_config=parallel_config,
            target_field=target_field,
            fitness_func=fitness_func,
            use_remote=False
        )

        with dispatcher:
            assert dispatcher.is_ready is True
            assert dispatcher.worker_count == 2

        assert dispatcher.is_ready is False

    def test_evaluate_particles_fallback(self, parallel_config, target_field, fitness_func):
        """Test evaluate_particles uses fallback when not ready."""
        dispatcher = ParticleDispatcher(
            parallel_config=parallel_config,
            target_field=target_field,
            fitness_func=fitness_func,
            use_remote=False
        )
        # Don't setup - should use fallback

        particles = np.array([[64.0, 1.0], [128.0, 0.0]])
        results = dispatcher.evaluate_particles(particles)

        # Fallback returns inf for all
        assert len(results) == 2
        assert all(np.isinf(results))
        assert dispatcher.stats.failed_evaluations == 2

    def test_result_callback(self, parallel_config, target_field, fitness_func):
        """Test result callback is called."""
        callback_results = []

        def callback(fitness):
            callback_results.append(fitness)

        dispatcher = ParticleDispatcher(
            parallel_config=parallel_config,
            target_field=target_field,
            fitness_func=fitness_func,
            result_callback=callback,
            use_remote=False
        )

        particles = np.array([[64.0, 1.0], [128.0, 0.0]])
        dispatcher.evaluate_particles(particles)

        # Callback should be called for each particle
        assert len(callback_results) == 2

    def test_stats_tracking(self, parallel_config, target_field, fitness_func):
        """Test statistics are tracked correctly."""
        dispatcher = ParticleDispatcher(
            parallel_config=parallel_config,
            target_field=target_field,
            fitness_func=fitness_func,
            use_remote=False
        )

        particles = np.array([[64.0, 1.0], [128.0, 0.0], [32.0, 1.0]])
        dispatcher.evaluate_particles(particles)

        stats = dispatcher.stats
        assert stats.total_dispatches == 1
        assert stats.total_particles == 3

    def test_get_summary(self, parallel_config, target_field, fitness_func):
        """Test get_summary returns correct structure."""
        dispatcher = ParticleDispatcher(
            parallel_config=parallel_config,
            target_field=target_field,
            fitness_func=fitness_func,
            use_remote=False
        )
        dispatcher.setup()

        summary = dispatcher.get_summary()
        assert "is_ready" in summary
        assert "worker_count" in summary
        assert "stats" in summary
        assert "pool_stats" in summary

        assert summary["is_ready"] is True
        assert summary["worker_count"] == 2

    def test_repr(self, parallel_config, target_field, fitness_func):
        """Test string representation."""
        dispatcher = ParticleDispatcher(
            parallel_config=parallel_config,
            target_field=target_field,
            fitness_func=fitness_func,
            use_remote=False
        )

        repr_str = repr(dispatcher)
        assert "ParticleDispatcher" in repr_str
        assert "ready=False" in repr_str

        dispatcher.setup()
        repr_str = repr(dispatcher)
        assert "ready=True" in repr_str
        assert "workers=2" in repr_str

    def test_pool_stats(self, parallel_config, target_field, fitness_func):
        """Test pool_stats property."""
        dispatcher = ParticleDispatcher(
            parallel_config=parallel_config,
            target_field=target_field,
            fitness_func=fitness_func,
            use_remote=False
        )

        # No pool before setup
        assert dispatcher.pool_stats is None

        dispatcher.setup()
        pool_stats = dispatcher.pool_stats
        assert pool_stats is not None
        assert pool_stats.total_groups == 2


class TestCreateParallelOpFunc:
    """Tests for create_parallel_op_func factory function."""

    @pytest.fixture
    def mock_dispatcher(self):
        """Create a mock dispatcher."""
        dispatcher = Mock()
        dispatcher.is_ready = True
        dispatcher.evaluate_particles = Mock(
            return_value=np.array([0.1, 0.2, 0.3])
        )
        return dispatcher

    def test_uses_dispatcher_when_ready(self, mock_dispatcher):
        """Test uses dispatcher when ready."""
        op_func = create_parallel_op_func(mock_dispatcher)

        particles = np.array([[1.0], [2.0], [3.0]])
        result = op_func(particles)

        mock_dispatcher.evaluate_particles.assert_called_once()
        np.testing.assert_array_equal(result, [0.1, 0.2, 0.3])

    def test_uses_fallback_when_not_ready(self, mock_dispatcher):
        """Test uses fallback when dispatcher not ready."""
        mock_dispatcher.is_ready = False
        fallback = Mock(return_value=np.array([1.0, 2.0, 3.0]))

        op_func = create_parallel_op_func(mock_dispatcher, fallback)

        particles = np.array([[1.0], [2.0], [3.0]])
        result = op_func(particles)

        mock_dispatcher.evaluate_particles.assert_not_called()
        fallback.assert_called_once()
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_returns_inf_when_no_fallback(self, mock_dispatcher):
        """Test returns inf when not ready and no fallback."""
        mock_dispatcher.is_ready = False

        op_func = create_parallel_op_func(mock_dispatcher)

        particles = np.array([[1.0], [2.0], [3.0]])
        result = op_func(particles)

        assert len(result) == 3
        assert all(np.isinf(result))
