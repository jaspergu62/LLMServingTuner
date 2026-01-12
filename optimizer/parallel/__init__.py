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
Parallel PSO evaluation module.

This module provides support for parallel particle evaluation across
multiple service groups, where each service group can span multiple nodes.

Example:
    8 servers with 8 NPUs each, running Qwen3-235B (requires 16 NPUs per service):
    - 4 service groups (2 nodes each)
    - 4 particles evaluated in parallel per iteration
"""

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
from msserviceprofiler.modelevalstate.optimizer.parallel.remote import (
    RemoteExecutor,
    CommandResult,
    ConnectionStatus,
    PARAMIKO_AVAILABLE,
)
from msserviceprofiler.modelevalstate.optimizer.parallel.pool import (
    ServiceGroupPool,
    PoolStats,
)
from msserviceprofiler.modelevalstate.optimizer.parallel.dispatcher import (
    ParticleDispatcher,
    DispatcherStats,
    create_parallel_op_func,
)
from msserviceprofiler.modelevalstate.optimizer.parallel.monitoring import (
    HealthMonitor,
    HealthStatus,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryPolicy,
    with_retry,
)

__all__ = [
    # Config
    "NodeConfig",
    "ServiceGroupConfig",
    "ParallelConfig",
    # Service Group
    "ServiceGroup",
    "ServiceGroupStatus",
    "EvaluationResult",
    # Remote Execution
    "RemoteExecutor",
    "CommandResult",
    "ConnectionStatus",
    "PARAMIKO_AVAILABLE",
    # Pool
    "ServiceGroupPool",
    "PoolStats",
    # Dispatcher
    "ParticleDispatcher",
    "DispatcherStats",
    "create_parallel_op_func",
    # Monitoring & Fault Tolerance
    "HealthMonitor",
    "HealthStatus",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "RetryPolicy",
    "with_retry",
]
