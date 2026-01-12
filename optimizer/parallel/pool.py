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
Connection pool and service group management for parallel PSO.

This module provides:
- ServiceGroupPool: Manages multiple ServiceGroup instances
- Parallel evaluation coordination
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from msserviceprofiler.modelevalstate.config.config import (
    OptimizerConfigField,
    PerformanceIndex,
)
from msserviceprofiler.modelevalstate.optimizer.parallel.config import (
    ParallelConfig,
    ServiceGroupConfig,
)
from msserviceprofiler.modelevalstate.optimizer.parallel.service_group import (
    ServiceGroup,
    ServiceGroupStatus,
    EvaluationResult,
)


@dataclass
class PoolStats:
    """Statistics for the service group pool."""
    total_groups: int = 0
    healthy_groups: int = 0
    evaluations_completed: int = 0
    evaluations_failed: int = 0
    total_evaluation_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.evaluations_completed + self.evaluations_failed
        if total == 0:
            return 0.0
        return self.evaluations_completed / total

    @property
    def avg_evaluation_time(self) -> float:
        """Calculate average evaluation time."""
        if self.evaluations_completed == 0:
            return 0.0
        return self.total_evaluation_time / self.evaluations_completed


class ServiceGroupPool:
    """
    Manages a pool of ServiceGroup instances for parallel evaluation.

    This class handles:
    - Lifecycle management of multiple service groups
    - Parallel particle evaluation across groups
    - Health monitoring and failure handling
    - Statistics collection

    Example:
        config = ParallelConfig(enabled=True, service_groups=[...])
        pool = ServiceGroupPool(config)
        pool.setup()

        results = pool.evaluate_batch(particles, target_field, fitness_func)

        pool.cleanup()
    """

    def __init__(self,
                 config: ParallelConfig,
                 use_remote: bool = True):
        """
        Initialize the service group pool.

        Args:
            config: Parallel configuration with service groups
            use_remote: Whether to use remote execution (SSH)
        """
        self.config = config
        self.use_remote = use_remote
        self._groups: Dict[int, ServiceGroup] = {}
        self._stats = PoolStats()
        self._is_setup = False

    @property
    def worker_count(self) -> int:
        """Number of available workers (healthy service groups)."""
        return len(self.healthy_groups)

    @property
    def groups(self) -> List[ServiceGroup]:
        """Get all service groups."""
        return list(self._groups.values())

    @property
    def healthy_groups(self) -> List[ServiceGroup]:
        """Get all healthy service groups."""
        return [g for g in self._groups.values() if g.is_healthy]

    @property
    def stats(self) -> PoolStats:
        """Get pool statistics."""
        return self._stats

    def setup(self) -> bool:
        """
        Initialize all service groups in the pool.

        Returns:
            True if at least one group is healthy, False otherwise.
        """
        logger.info(f"Setting up ServiceGroupPool with {len(self.config.service_groups)} groups")

        for sg_config in self.config.service_groups:
            group = ServiceGroup(
                config=sg_config,
                use_remote=self.use_remote
            )

            if group.setup():
                self._groups[sg_config.group_id] = group
                logger.info(f"ServiceGroup {sg_config.group_id} initialized")
            else:
                logger.error(f"Failed to initialize ServiceGroup {sg_config.group_id}")

        self._stats.total_groups = len(self.config.service_groups)
        self._stats.healthy_groups = len(self.healthy_groups)
        self._is_setup = True

        if self._stats.healthy_groups == 0:
            logger.error("No healthy service groups available")
            return False

        logger.info(
            f"ServiceGroupPool ready: {self._stats.healthy_groups}/{self._stats.total_groups} "
            f"groups healthy"
        )
        return True

    def cleanup(self):
        """Clean up all service groups."""
        logger.info("Cleaning up ServiceGroupPool")
        for group in self._groups.values():
            try:
                group.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up group {group.group_id}: {e}")
        self._groups.clear()
        self._is_setup = False

    def evaluate_batch(self,
                       particles: np.ndarray,
                       target_field: Tuple[OptimizerConfigField, ...],
                       fitness_func: Callable[[PerformanceIndex], float],
                       benchmark_func: Optional[Callable[[], PerformanceIndex]] = None
                       ) -> np.ndarray:
        """
        Evaluate all particles in parallel across service groups.

        Args:
            particles: Array of particle positions (n_particles x dimensions)
            target_field: Tuple of optimizer config fields
            fitness_func: Function to calculate fitness from performance
            benchmark_func: Optional custom benchmark function

        Returns:
            Array of fitness values for each particle
        """
        if not self._is_setup:
            raise RuntimeError("Pool not setup. Call setup() first.")

        n_particles = particles.shape[0]
        results = [float('inf')] * n_particles
        healthy = self.healthy_groups

        if not healthy:
            logger.error("No healthy service groups available for evaluation")
            return np.array(results)

        logger.info(
            f"Evaluating {n_particles} particles across {len(healthy)} service groups"
        )
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=len(healthy)) as executor:
            # Process particles in batches matching worker count
            for batch_start in range(0, n_particles, len(healthy)):
                batch_end = min(batch_start + len(healthy), n_particles)
                batch_size = batch_end - batch_start

                # Submit evaluation tasks
                futures: Dict[Future, int] = {}
                for i in range(batch_size):
                    particle_idx = batch_start + i
                    group = healthy[i % len(healthy)]

                    # Convert particle array to params
                    params = self._particle_to_params(
                        particles[particle_idx],
                        target_field
                    )

                    future = executor.submit(
                        self._evaluate_with_retry,
                        group,
                        params,
                        fitness_func,
                        benchmark_func
                    )
                    futures[future] = particle_idx

                # Collect results
                for future in as_completed(futures, timeout=self.config.evaluation_timeout):
                    particle_idx = futures[future]
                    try:
                        eval_result = future.result()
                        results[particle_idx] = eval_result.fitness

                        if eval_result.error:
                            self._stats.evaluations_failed += 1
                            logger.warning(
                                f"Particle {particle_idx} evaluation failed: {eval_result.error}"
                            )
                        else:
                            self._stats.evaluations_completed += 1
                            self._stats.total_evaluation_time += eval_result.duration_seconds

                    except Exception as e:
                        logger.error(f"Particle {particle_idx} evaluation exception: {e}")
                        results[particle_idx] = float('inf')
                        self._stats.evaluations_failed += 1

        total_time = time.time() - start_time
        logger.info(
            f"Batch evaluation complete: {n_particles} particles in {total_time:.1f}s"
        )

        return np.array(results)

    def _evaluate_with_retry(self,
                             group: ServiceGroup,
                             params: Tuple[OptimizerConfigField, ...],
                             fitness_func: Callable[[PerformanceIndex], float],
                             benchmark_func: Optional[Callable[[], PerformanceIndex]] = None
                             ) -> EvaluationResult:
        """
        Evaluate a particle with retry logic.

        Args:
            group: Service group to use
            params: Parameter values
            fitness_func: Fitness function
            benchmark_func: Optional benchmark function

        Returns:
            EvaluationResult
        """
        last_error = None

        for attempt in range(self.config.retry_count + 1):
            try:
                result = group.evaluate(params, fitness_func, benchmark_func)
                if result.error is None:
                    return result
                last_error = result.error

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Evaluation attempt {attempt + 1}/{self.config.retry_count + 1} "
                    f"failed on group {group.group_id}: {e}"
                )

            if attempt < self.config.retry_count:
                time.sleep(self.config.retry_delay)

        # All retries failed
        return EvaluationResult(
            fitness=float('inf'),
            error=f"All {self.config.retry_count + 1} attempts failed: {last_error}",
            group_id=group.group_id
        )

    def _particle_to_params(self,
                            particle: np.ndarray,
                            target_field: Tuple[OptimizerConfigField, ...]
                            ) -> Tuple[OptimizerConfigField, ...]:
        """
        Convert particle array to parameter tuple.

        Args:
            particle: Particle position array
            target_field: Target field definitions

        Returns:
            Tuple of OptimizerConfigField with values set
        """
        from copy import deepcopy

        params = []
        particle_idx = 0

        for field in target_field:
            field_copy = deepcopy(field)

            # Skip constant fields
            if field.constant is not None:
                field_copy.value = field.constant
            else:
                # Get value from particle
                if particle_idx < len(particle):
                    raw_value = particle[particle_idx]
                    field_copy.value = field_copy.find_available_value(raw_value)
                    particle_idx += 1

            params.append(field_copy)

        return tuple(params)

    def get_group(self, group_id: int) -> Optional[ServiceGroup]:
        """Get a service group by ID."""
        return self._groups.get(group_id)

    def refresh_health(self):
        """Refresh health status of all groups."""
        for group in self._groups.values():
            if group.status == ServiceGroupStatus.ERROR:
                # Try to recover
                logger.info(f"Attempting to recover group {group.group_id}")
                group.cleanup()
                group.setup()

        self._stats.healthy_groups = len(self.healthy_groups)

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
            f"ServiceGroupPool(groups={len(self._groups)}, "
            f"healthy={len(self.healthy_groups)})"
        )
