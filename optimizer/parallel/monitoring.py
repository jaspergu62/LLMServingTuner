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
Health monitoring and fault tolerance for parallel PSO evaluation.

This module provides:
- HealthMonitor: Periodic health checking and recovery
- CircuitBreaker: Prevents cascading failures
- RetryPolicy: Configurable retry strategies
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Any

from loguru import logger


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests allowed
    OPEN = "open"          # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class HealthStatus:
    """Health status of a service group."""
    group_id: int
    is_healthy: bool
    last_check: datetime
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_error: Optional[str] = None
    response_time_ms: float = 0.0

    @property
    def status_summary(self) -> str:
        """Get a summary of the health status."""
        status = "healthy" if self.is_healthy else "unhealthy"
        return (
            f"Group {self.group_id}: {status} "
            f"(failures={self.consecutive_failures}, "
            f"response_time={self.response_time_ms:.1f}ms)"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5       # Failures before opening circuit
    success_threshold: int = 3       # Successes in half-open before closing
    timeout_seconds: int = 60        # Time before half-open
    half_open_max_calls: int = 1     # Max calls in half-open state


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by stopping requests to failing services
    and allowing gradual recovery testing.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is failing, requests are blocked
    - HALF_OPEN: Testing if service has recovered

    Example:
        breaker = CircuitBreaker(group_id=0)

        if breaker.allow_request():
            try:
                result = call_service()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
    """

    def __init__(self,
                 group_id: int,
                 config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            group_id: ID of the service group
            config: Circuit breaker configuration
        """
        self.group_id = group_id
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.state == CircuitState.OPEN

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed.

        Returns:
            True if request can proceed, False if blocked.
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                return False
            else:  # HALF_OPEN
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

    def record_success(self):
        """Record a successful request."""
        with self._lock:
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(
                        f"Circuit breaker for group {self.group_id} closed "
                        f"(service recovered)"
                    )

    def record_failure(self, error: Optional[str] = None):
        """Record a failed request."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    f"Circuit breaker for group {self.group_id} reopened "
                    f"(recovery test failed)"
                )
            elif (self._state == CircuitState.CLOSED and
                  self._failure_count >= self.config.failure_threshold):
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    f"Circuit breaker for group {self.group_id} opened "
                    f"(failure threshold exceeded: {error})"
                )

    def reset(self):
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None

    def _check_state_transition(self):
        """Check if state should transition based on timeout."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = (datetime.now() - self._last_failure_time).total_seconds()
            if elapsed >= self.config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
                logger.info(
                    f"Circuit breaker for group {self.group_id} half-open "
                    f"(testing recovery)"
                )

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._success_count = 0
        self._half_open_calls = 0
        logger.debug(
            f"Circuit breaker group {self.group_id}: "
            f"{old_state.value} -> {new_state.value}"
        )


class HealthMonitor:
    """
    Monitors health of service groups and manages recovery.

    Features:
    - Periodic health checks in background thread
    - Circuit breaker integration
    - Automatic recovery attempts
    - Health status reporting

    Example:
        monitor = HealthMonitor(pool, check_interval=30)
        monitor.start()

        # Check status
        for status in monitor.get_all_status():
            print(status.status_summary)

        monitor.stop()
    """

    def __init__(self,
                 pool,  # ServiceGroupPool
                 check_interval: int = 30,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize health monitor.

        Args:
            pool: ServiceGroupPool to monitor
            check_interval: Seconds between health checks
            circuit_breaker_config: Configuration for circuit breakers
        """
        self.pool = pool
        self.check_interval = check_interval
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()

        self._health_status: Dict[int, HealthStatus] = {}
        self._circuit_breakers: Dict[int, CircuitBreaker] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Initialize circuit breakers for each group
        for group in pool.groups:
            self._circuit_breakers[group.group_id] = CircuitBreaker(
                group.group_id,
                self.circuit_breaker_config
            )
            self._health_status[group.group_id] = HealthStatus(
                group_id=group.group_id,
                is_healthy=True,
                last_check=datetime.now()
            )

    def start(self):
        """Start background health monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="health-monitor"
        )
        self._thread.start()
        logger.info(
            f"Health monitor started (interval={self.check_interval}s, "
            f"groups={len(self._health_status)})"
        )

    def stop(self):
        """Stop background health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Health monitor stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self.check_all()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            # Sleep in small increments to allow quick shutdown
            for _ in range(self.check_interval):
                if not self._running:
                    break
                time.sleep(1)

    def check_all(self):
        """Perform health check on all groups."""
        for group in self.pool.groups:
            self.check_group(group.group_id)

    def check_group(self, group_id: int) -> HealthStatus:
        """
        Check health of a specific group.

        Args:
            group_id: ID of the service group

        Returns:
            HealthStatus for the group
        """
        group = self.pool.get_group(group_id)
        if not group:
            return HealthStatus(
                group_id=group_id,
                is_healthy=False,
                last_check=datetime.now(),
                last_error="Group not found"
            )

        start_time = time.time()
        is_healthy = False
        error_msg = None

        try:
            is_healthy = group.health_check()
        except Exception as e:
            error_msg = str(e)
            is_healthy = False

        response_time = (time.time() - start_time) * 1000

        with self._lock:
            status = self._health_status.get(group_id)
            if status:
                if is_healthy:
                    status.consecutive_failures = 0
                    status.consecutive_successes += 1
                    self._circuit_breakers[group_id].record_success()
                else:
                    status.consecutive_failures += 1
                    status.consecutive_successes = 0
                    self._circuit_breakers[group_id].record_failure(error_msg)

                status.is_healthy = is_healthy
                status.last_check = datetime.now()
                status.last_error = error_msg
                status.response_time_ms = response_time
            else:
                status = HealthStatus(
                    group_id=group_id,
                    is_healthy=is_healthy,
                    last_check=datetime.now(),
                    last_error=error_msg,
                    response_time_ms=response_time
                )
                self._health_status[group_id] = status

        if not is_healthy:
            logger.warning(
                f"Health check failed for group {group_id}: {error_msg}"
            )

        return status

    def get_status(self, group_id: int) -> Optional[HealthStatus]:
        """Get health status for a group."""
        with self._lock:
            return self._health_status.get(group_id)

    def get_all_status(self) -> List[HealthStatus]:
        """Get health status for all groups."""
        with self._lock:
            return list(self._health_status.values())

    def get_healthy_groups(self) -> List[int]:
        """Get list of healthy group IDs."""
        with self._lock:
            return [
                gid for gid, status in self._health_status.items()
                if status.is_healthy
            ]

    def is_group_available(self, group_id: int) -> bool:
        """
        Check if a group is available for requests.

        Considers both health status and circuit breaker state.
        """
        with self._lock:
            status = self._health_status.get(group_id)
            breaker = self._circuit_breakers.get(group_id)

            if not status or not breaker:
                return False

            return status.is_healthy and breaker.allow_request()

    def attempt_recovery(self, group_id: int) -> bool:
        """
        Attempt to recover a failed group.

        Args:
            group_id: ID of the group to recover

        Returns:
            True if recovery successful
        """
        logger.info(f"Attempting recovery for group {group_id}")

        group = self.pool.get_group(group_id)
        if not group:
            return False

        try:
            # Try to cleanup and re-setup
            group.cleanup()
            if group.setup():
                # Check if it's actually healthy
                if group.health_check():
                    with self._lock:
                        self._health_status[group_id].is_healthy = True
                        self._health_status[group_id].consecutive_failures = 0
                        self._circuit_breakers[group_id].reset()
                    logger.info(f"Recovery successful for group {group_id}")
                    return True
        except Exception as e:
            logger.error(f"Recovery failed for group {group_id}: {e}")

        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of health status."""
        with self._lock:
            healthy = sum(1 for s in self._health_status.values() if s.is_healthy)
            total = len(self._health_status)
            avg_response = (
                sum(s.response_time_ms for s in self._health_status.values()) / total
                if total > 0 else 0
            )

            return {
                "healthy_groups": healthy,
                "total_groups": total,
                "health_percentage": healthy / total * 100 if total > 0 else 0,
                "avg_response_time_ms": avg_response,
                "groups": [s.status_summary for s in self._health_status.values()]
            }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


@dataclass
class RetryPolicy:
    """
    Configurable retry policy for operations.

    Supports:
    - Fixed delay retries
    - Exponential backoff
    - Jitter for thundering herd prevention
    """
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1  # Random factor (0-1)

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds before next retry
        """
        import random

        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def should_retry(self, attempt: int) -> bool:
        """Check if another retry should be attempted."""
        return attempt < self.max_retries


def with_retry(policy: RetryPolicy,
               on_retry: Optional[Callable[[int, Exception], None]] = None):
    """
    Decorator for adding retry logic to functions.

    Args:
        policy: RetryPolicy to use
        on_retry: Optional callback called on each retry

    Example:
        @with_retry(RetryPolicy(max_retries=3))
        def call_service():
            return requests.get(url)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(policy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(attempt):
                        break

                    if on_retry:
                        on_retry(attempt, e)

                    delay = policy.get_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{policy.max_retries} "
                        f"in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator
