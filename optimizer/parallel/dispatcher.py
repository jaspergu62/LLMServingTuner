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
Particle dispatcher for parallel PSO evaluation.

This module provides the bridge between PSOOptimizer and ServiceGroupPool,
enabling parallel particle evaluation across multiple service groups.
"""

from dataclasses import dataclass, field
from math import inf
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from msserviceprofiler.modelevalstate.config.config import (
    OptimizerConfigField,
    PerformanceIndex,
)
from msserviceprofiler.modelevalstate.optimizer.parallel.config import ParallelConfig
from msserviceprofiler.modelevalstate.optimizer.parallel.pool import (
    ServiceGroupPool,
    PoolStats,
)
from msserviceprofiler.modelevalstate.optimizer.parallel.service_group import (
    EvaluationResult,
)


@dataclass
class DispatcherStats:
    """Statistics for the particle dispatcher."""
    total_dispatches: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    total_particles: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        total = self.successful_evaluations + self.failed_evaluations
        if total == 0:
            return 0.0
        return self.successful_evaluations / total


class ParticleDispatcher:
    """
    Dispatches particles to service groups for parallel evaluation.

    This class coordinates between the PSOOptimizer's particle swarm and
    the ServiceGroupPool's distributed evaluation infrastructure.

    Features:
    - Automatic batch sizing based on available workers
    - Result collection and fitness calculation
    - Fallback to sequential evaluation if pool unavailable
    - Statistics tracking

    Example:
        dispatcher = ParticleDispatcher(
            parallel_config=config,
            target_field=target_field,
            fitness_func=minimum_algorithm,
            result_callback=scheduler.save_result
        )

        with dispatcher:
            fitness_values = dispatcher.evaluate_particles(particles)
    """

    def __init__(self,
                 parallel_config: ParallelConfig,
                 target_field: Tuple[OptimizerConfigField, ...],
                 fitness_func: Callable[[PerformanceIndex], float],
                 result_callback: Optional[Callable[[float], None]] = None,
                 benchmark_func: Optional[Callable[[], PerformanceIndex]] = None,
                 use_remote: bool = True):
        """
        Initialize the particle dispatcher.

        Args:
            parallel_config: Configuration for parallel evaluation
            target_field: Target optimization fields
            fitness_func: Function to calculate fitness from performance metrics
            result_callback: Optional callback to save results (e.g., scheduler.save_result)
            benchmark_func: Optional custom benchmark function
            use_remote: Whether to use remote execution
        """
        self.parallel_config = parallel_config
        self.target_field = target_field
        self.fitness_func = fitness_func
        self.result_callback = result_callback
        self.benchmark_func = benchmark_func
        self.use_remote = use_remote

        self._pool: Optional[ServiceGroupPool] = None
        self._stats = DispatcherStats()
        self._is_setup = False

    @property
    def stats(self) -> DispatcherStats:
        """Get dispatcher statistics."""
        return self._stats

    @property
    def pool_stats(self) -> Optional[PoolStats]:
        """Get underlying pool statistics."""
        if self._pool:
            return self._pool.stats
        return None

    @property
    def worker_count(self) -> int:
        """Number of available parallel workers."""
        if self._pool:
            return self._pool.worker_count
        return 0

    @property
    def is_ready(self) -> bool:
        """Check if dispatcher is ready for parallel evaluation."""
        return self._is_setup and self._pool is not None and self.worker_count > 0

    def setup(self) -> bool:
        """
        Initialize the service group pool.

        Returns:
            True if setup successful and pool is ready for evaluation.
        """
        if not self.parallel_config.enabled:
            logger.info("Parallel evaluation disabled in config")
            return False

        if not self.parallel_config.service_groups:
            logger.warning("No service groups configured")
            return False

        logger.info("Setting up ParticleDispatcher")

        self._pool = ServiceGroupPool(
            config=self.parallel_config,
            use_remote=self.use_remote
        )

        if not self._pool.setup():
            logger.error("Failed to setup ServiceGroupPool")
            self._pool = None
            return False

        self._is_setup = True
        logger.info(
            f"ParticleDispatcher ready with {self.worker_count} workers"
        )
        return True

    def cleanup(self):
        """Release all resources."""
        logger.info("Cleaning up ParticleDispatcher")
        if self._pool:
            self._pool.cleanup()
            self._pool = None
        self._is_setup = False

    def evaluate_particles(self, particles: np.ndarray) -> np.ndarray:
        """
        Evaluate all particles in parallel.

        Args:
            particles: Array of particle positions (n_particles x dimensions)

        Returns:
            Array of fitness values for each particle.
        """
        n_particles = particles.shape[0]
        self._stats.total_dispatches += 1
        self._stats.total_particles += n_particles

        logger.info(f"Dispatching {n_particles} particles for evaluation")

        if not self.is_ready:
            logger.warning(
                "Parallel evaluation not available, using sequential fallback"
            )
            return self._evaluate_sequential(particles)

        # Use pool for parallel evaluation
        fitness_values = self._pool.evaluate_batch(
            particles=particles,
            target_field=self.target_field,
            fitness_func=self.fitness_func,
            benchmark_func=self.benchmark_func
        )

        # Track statistics and call result callback
        for i, fitness in enumerate(fitness_values):
            if np.isinf(fitness):
                self._stats.failed_evaluations += 1
            else:
                self._stats.successful_evaluations += 1

            # Call result callback for each particle
            if self.result_callback:
                self.result_callback(fitness=fitness)

        return fitness_values

    def _evaluate_sequential(self, particles: np.ndarray) -> np.ndarray:
        """
        Fallback sequential evaluation.

        This is used when parallel evaluation is not available.
        Note: This requires a scheduler to be set up separately.

        Args:
            particles: Array of particle positions

        Returns:
            Array of fitness values (all inf in fallback mode).
        """
        n_particles = particles.shape[0]
        logger.warning(
            f"Sequential fallback: {n_particles} particles will return inf. "
            "Configure parallel evaluation or use PSOOptimizer's native op_func."
        )

        # Return inf for all particles - actual sequential evaluation
        # should be handled by the original PSOOptimizer.op_func
        results = [inf] * n_particles

        for fitness in results:
            self._stats.failed_evaluations += 1
            if self.result_callback:
                self.result_callback(fitness=fitness)

        return np.array(results)

    def refresh_pool(self):
        """Refresh health status of all service groups."""
        if self._pool:
            self._pool.refresh_health()
            logger.info(f"Pool refreshed: {self.worker_count} healthy workers")

    def get_summary(self) -> Dict:
        """Get a summary of dispatcher state and statistics."""
        summary = {
            "is_ready": self.is_ready,
            "worker_count": self.worker_count,
            "stats": {
                "total_dispatches": self._stats.total_dispatches,
                "total_particles": self._stats.total_particles,
                "successful_evaluations": self._stats.successful_evaluations,
                "failed_evaluations": self._stats.failed_evaluations,
                "success_rate": self._stats.success_rate,
            }
        }

        if self._pool:
            pool_stats = self._pool.stats
            summary["pool_stats"] = {
                "total_groups": pool_stats.total_groups,
                "healthy_groups": pool_stats.healthy_groups,
                "avg_evaluation_time": pool_stats.avg_evaluation_time,
            }

        return summary

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False

    def __repr__(self) -> str:
        return (
            f"ParticleDispatcher(ready={self.is_ready}, "
            f"workers={self.worker_count})"
        )


def create_parallel_op_func(
    dispatcher: ParticleDispatcher,
    fallback_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a parallel-aware objective function for PSO.

    This factory function creates an op_func that can be used directly
    with CustomGlobalBestPSO, automatically handling parallel evaluation
    when available and falling back to sequential evaluation otherwise.

    Args:
        dispatcher: The particle dispatcher for parallel evaluation
        fallback_func: Optional fallback function for sequential evaluation

    Returns:
        A function compatible with PSO's op_func signature.

    Example:
        dispatcher = ParticleDispatcher(config, target_field, fitness_func)
        op_func = create_parallel_op_func(dispatcher, original_op_func)

        # Use in PSO
        optimizer = CustomGlobalBestPSO(...)
        cost, pos = optimizer.optimize(op_func, iters=100)
    """
    def parallel_op_func(x: np.ndarray) -> np.ndarray:
        if dispatcher.is_ready:
            return dispatcher.evaluate_particles(x)
        elif fallback_func:
            logger.info("Using fallback sequential evaluation")
            return fallback_func(x)
        else:
            logger.error("No evaluation method available")
            return np.full(x.shape[0], inf)

    return parallel_op_func
