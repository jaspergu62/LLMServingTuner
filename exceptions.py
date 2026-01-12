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
Custom exception hierarchy for ModelEvalState.

Exception Hierarchy:
    ModelEvalStateError (base)
    ├── ConfigurationError
    │   ├── ConfigPathError
    │   └── ConfigValidationError
    ├── OptimizationError
    │   ├── PSOConvergenceError
    │   └── FitnessCalculationError
    ├── SimulatorError
    │   ├── SimulatorStartError
    │   ├── SimulatorStopError
    │   └── SimulatorHealthCheckError
    ├── BenchmarkError
    │   ├── BenchmarkExecutionError
    │   └── BenchmarkParseError
    └── CommunicationError
        ├── IPCTimeoutError
        └── IPCCommandError
"""
from typing import Any, Optional


class ModelEvalStateError(Exception):
    """Base exception for all ModelEvalState errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# Configuration Errors
class ConfigurationError(ModelEvalStateError):
    """Base exception for configuration-related errors."""

    pass


class ConfigPathError(ConfigurationError):
    """Error when config_position path is invalid or cannot be resolved."""

    def __init__(self, path: str, reason: str, config_file: Optional[str] = None):
        details = {"path": path, "reason": reason}
        if config_file:
            details["config_file"] = config_file
        message = f"Invalid configuration path '{path}': {reason}"
        super().__init__(message, details)


class ConfigValidationError(ConfigurationError):
    """Error when configuration validation fails."""

    def __init__(self, field: str, value: Any, constraint: str):
        details = {"field": field, "value": value, "constraint": constraint}
        message = f"Configuration validation failed for '{field}': {constraint}"
        super().__init__(message, details)


# Optimization Errors
class OptimizationError(ModelEvalStateError):
    """Base exception for optimization-related errors."""

    pass


class PSOConvergenceError(OptimizationError):
    """Error when PSO fails to converge within specified iterations."""

    def __init__(self, iterations: int, best_fitness: float, target_fitness: Optional[float] = None):
        details = {"iterations": iterations, "best_fitness": best_fitness}
        if target_fitness:
            details["target_fitness"] = target_fitness
        message = f"PSO failed to converge after {iterations} iterations (best fitness: {best_fitness})"
        super().__init__(message, details)


class FitnessCalculationError(OptimizationError):
    """Error when fitness calculation fails."""

    def __init__(self, reason: str, performance_index: Optional[dict] = None):
        details = {"reason": reason}
        if performance_index:
            details["performance_index"] = performance_index
        message = f"Fitness calculation failed: {reason}"
        super().__init__(message, details)


# Simulator Errors
class SimulatorError(ModelEvalStateError):
    """Base exception for simulator-related errors."""

    pass


class SimulatorStartError(SimulatorError):
    """Error when simulator fails to start."""

    def __init__(self, simulator_type: str, reason: str, command: Optional[str] = None):
        details = {"simulator_type": simulator_type, "reason": reason}
        if command:
            details["command"] = command
        message = f"Failed to start {simulator_type} simulator: {reason}"
        super().__init__(message, details)


class SimulatorStopError(SimulatorError):
    """Error when simulator fails to stop gracefully."""

    def __init__(self, simulator_type: str, pid: Optional[int] = None, reason: Optional[str] = None):
        details = {"simulator_type": simulator_type}
        if pid:
            details["pid"] = pid
        if reason:
            details["reason"] = reason
        message = f"Failed to stop {simulator_type} simulator"
        if reason:
            message += f": {reason}"
        super().__init__(message, details)


class SimulatorHealthCheckError(SimulatorError):
    """Error when simulator health check fails."""

    def __init__(self, simulator_type: str, url: str, status_code: Optional[int] = None):
        details = {"simulator_type": simulator_type, "url": url}
        if status_code:
            details["status_code"] = status_code
        message = f"Health check failed for {simulator_type} simulator at {url}"
        super().__init__(message, details)


# Benchmark Errors
class BenchmarkError(ModelEvalStateError):
    """Base exception for benchmark-related errors."""

    pass


class BenchmarkExecutionError(BenchmarkError):
    """Error when benchmark execution fails."""

    def __init__(self, benchmark_type: str, reason: str, exit_code: Optional[int] = None):
        details = {"benchmark_type": benchmark_type, "reason": reason}
        if exit_code is not None:
            details["exit_code"] = exit_code
        message = f"Benchmark execution failed for {benchmark_type}: {reason}"
        super().__init__(message, details)


class BenchmarkParseError(BenchmarkError):
    """Error when benchmark output parsing fails."""

    def __init__(self, benchmark_type: str, output: str, reason: str):
        details = {"benchmark_type": benchmark_type, "reason": reason, "output_preview": output[:200]}
        message = f"Failed to parse {benchmark_type} benchmark output: {reason}"
        super().__init__(message, details)


# Communication Errors
class CommunicationError(ModelEvalStateError):
    """Base exception for IPC-related errors."""

    pass


class IPCTimeoutError(CommunicationError):
    """Error when IPC operation times out."""

    def __init__(self, operation: str, timeout_seconds: float):
        details = {"operation": operation, "timeout_seconds": timeout_seconds}
        message = f"IPC operation '{operation}' timed out after {timeout_seconds}s"
        super().__init__(message, details)


class IPCCommandError(CommunicationError):
    """Error when IPC command fails."""

    def __init__(self, command: str, reason: str, response: Optional[str] = None):
        details = {"command": command, "reason": reason}
        if response:
            details["response"] = response
        message = f"IPC command '{command}' failed: {reason}"
        super().__init__(message, details)
