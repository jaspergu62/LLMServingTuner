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
Unit tests for monitoring module (CircuitBreaker, HealthMonitor, RetryPolicy).
"""

from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import pytest
import time
import threading

from msserviceprofiler.modelevalstate.optimizer.parallel.monitoring import (
    CircuitState,
    HealthStatus,
    CircuitBreakerConfig,
    CircuitBreaker,
    HealthMonitor,
    RetryPolicy,
    with_retry,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_state_values(self):
        """Test state enum values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_all_states_exist(self):
        """Test all expected states exist."""
        states = list(CircuitState)
        assert len(states) == 3


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_healthy_status(self):
        """Test healthy status."""
        status = HealthStatus(
            group_id=0,
            is_healthy=True,
            last_check=datetime.now()
        )
        assert status.group_id == 0
        assert status.is_healthy is True
        assert status.consecutive_failures == 0
        assert status.consecutive_successes == 0
        assert status.last_error is None
        assert status.response_time_ms == 0.0

    def test_unhealthy_status(self):
        """Test unhealthy status with error."""
        status = HealthStatus(
            group_id=1,
            is_healthy=False,
            last_check=datetime.now(),
            consecutive_failures=3,
            last_error="Connection refused",
            response_time_ms=150.5
        )
        assert status.is_healthy is False
        assert status.consecutive_failures == 3
        assert status.last_error == "Connection refused"
        assert status.response_time_ms == 150.5

    def test_status_summary_healthy(self):
        """Test status summary for healthy group."""
        status = HealthStatus(
            group_id=0,
            is_healthy=True,
            last_check=datetime.now(),
            response_time_ms=25.3
        )
        summary = status.status_summary
        assert "Group 0" in summary
        assert "healthy" in summary
        assert "25.3ms" in summary

    def test_status_summary_unhealthy(self):
        """Test status summary for unhealthy group."""
        status = HealthStatus(
            group_id=2,
            is_healthy=False,
            last_check=datetime.now(),
            consecutive_failures=5,
            response_time_ms=0.0
        )
        summary = status.status_summary
        assert "Group 2" in summary
        assert "unhealthy" in summary
        assert "failures=5" in summary


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_seconds == 60
        assert config.half_open_max_calls == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            timeout_seconds=120,
            half_open_max_calls=3
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 5
        assert config.timeout_seconds == 120
        assert config.half_open_max_calls == 3


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        breaker = CircuitBreaker(group_id=0)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_open is False

    def test_allow_request_closed(self):
        """Test requests allowed in closed state."""
        breaker = CircuitBreaker(group_id=0)
        assert breaker.allow_request() is True

    def test_transition_to_open_on_failures(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(group_id=0, config=config)

        # Record failures
        for i in range(3):
            breaker.record_failure(f"Error {i}")

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open is True

    def test_block_requests_when_open(self):
        """Test requests blocked in open state."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(group_id=0, config=config)

        breaker.record_failure("Error 1")
        breaker.record_failure("Error 2")

        assert breaker.allow_request() is False

    def test_success_resets_failure_count(self):
        """Test success resets failure count."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker(group_id=0, config=config)

        # Record some failures (not enough to open)
        breaker.record_failure("Error 1")
        breaker.record_failure("Error 2")
        assert breaker._failure_count == 2

        # Record success - should reset
        breaker.record_success()
        assert breaker._failure_count == 0
        assert breaker.state == CircuitState.CLOSED

    def test_transition_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=1  # 1 second timeout for testing
        )
        breaker = CircuitBreaker(group_id=0, config=config)

        # Open the circuit
        breaker.record_failure("Error 1")
        breaker.record_failure("Error 2")
        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(1.1)

        # Should transition to half-open
        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_allows_limited_requests(self):
        """Test half-open state allows limited requests."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0,  # Immediate transition for testing
            half_open_max_calls=2
        )
        breaker = CircuitBreaker(group_id=0, config=config)

        # Open the circuit
        breaker.record_failure("Error 1")
        breaker.record_failure("Error 2")

        # Manually set to half-open for testing
        breaker._state = CircuitState.HALF_OPEN
        breaker._half_open_calls = 0

        # Should allow limited requests
        assert breaker.allow_request() is True  # 1st call
        assert breaker.allow_request() is True  # 2nd call
        assert breaker.allow_request() is False  # Exceeded limit

    def test_half_open_success_closes_circuit(self):
        """Test successful requests in half-open close the circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2
        )
        breaker = CircuitBreaker(group_id=0, config=config)

        # Set to half-open
        breaker._state = CircuitState.HALF_OPEN
        breaker._success_count = 0

        # Record successes
        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN  # Not enough yet

        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED  # Recovered

    def test_half_open_failure_reopens_circuit(self):
        """Test failure in half-open reopens the circuit."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(group_id=0, config=config)

        # Set to half-open
        breaker._state = CircuitState.HALF_OPEN

        # Record failure
        breaker.record_failure("Error during recovery")

        assert breaker.state == CircuitState.OPEN

    def test_reset(self):
        """Test reset returns circuit to closed state."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(group_id=0, config=config)

        # Open the circuit
        breaker.record_failure("Error 1")
        breaker.record_failure("Error 2")
        assert breaker.state == CircuitState.OPEN

        # Reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
        assert breaker._success_count == 0

    def test_thread_safety(self):
        """Test circuit breaker is thread-safe."""
        config = CircuitBreakerConfig(failure_threshold=100)
        breaker = CircuitBreaker(group_id=0, config=config)

        errors = []

        def record_failures():
            try:
                for _ in range(50):
                    breaker.record_failure("Error")
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [threading.Thread(target=record_failures) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have opened after 100 failures
        assert breaker.state == CircuitState.OPEN


class TestHealthMonitor:
    """Tests for HealthMonitor class."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock ServiceGroupPool."""
        pool = Mock()
        group1 = Mock()
        group1.group_id = 0
        group1.health_check.return_value = True

        group2 = Mock()
        group2.group_id = 1
        group2.health_check.return_value = True

        pool.groups = [group1, group2]
        pool.get_group.side_effect = lambda gid: group1 if gid == 0 else (group2 if gid == 1 else None)

        return pool

    def test_init(self, mock_pool):
        """Test health monitor initialization."""
        monitor = HealthMonitor(mock_pool, check_interval=30)

        assert monitor.pool == mock_pool
        assert monitor.check_interval == 30
        assert len(monitor._health_status) == 2
        assert len(monitor._circuit_breakers) == 2
        assert monitor._running is False

    def test_start_stop(self, mock_pool):
        """Test starting and stopping health monitor."""
        monitor = HealthMonitor(mock_pool, check_interval=1)

        monitor.start()
        assert monitor._running is True
        assert monitor._thread is not None
        assert monitor._thread.is_alive()

        monitor.stop()
        assert monitor._running is False

    def test_context_manager(self, mock_pool):
        """Test health monitor as context manager."""
        with HealthMonitor(mock_pool, check_interval=60) as monitor:
            assert monitor._running is True

        assert monitor._running is False

    def test_check_group_healthy(self, mock_pool):
        """Test checking a healthy group."""
        monitor = HealthMonitor(mock_pool)

        status = monitor.check_group(0)

        assert status.is_healthy is True
        assert status.group_id == 0
        assert status.last_error is None

    def test_check_group_unhealthy(self, mock_pool):
        """Test checking an unhealthy group."""
        mock_pool.groups[0].health_check.return_value = False

        monitor = HealthMonitor(mock_pool)
        status = monitor.check_group(0)

        assert status.is_healthy is False

    def test_check_group_exception(self, mock_pool):
        """Test checking group that throws exception."""
        mock_pool.groups[0].health_check.side_effect = Exception("Connection failed")

        monitor = HealthMonitor(mock_pool)
        status = monitor.check_group(0)

        assert status.is_healthy is False
        assert "Connection failed" in status.last_error

    def test_check_group_not_found(self, mock_pool):
        """Test checking non-existent group."""
        monitor = HealthMonitor(mock_pool)
        status = monitor.check_group(99)

        assert status.is_healthy is False
        assert "not found" in status.last_error

    def test_check_all(self, mock_pool):
        """Test checking all groups."""
        monitor = HealthMonitor(mock_pool)
        monitor.check_all()

        # Both groups should have been checked
        mock_pool.groups[0].health_check.assert_called()
        mock_pool.groups[1].health_check.assert_called()

    def test_get_status(self, mock_pool):
        """Test getting status for a group."""
        monitor = HealthMonitor(mock_pool)
        monitor.check_group(0)

        status = monitor.get_status(0)
        assert status is not None
        assert status.group_id == 0

    def test_get_all_status(self, mock_pool):
        """Test getting status for all groups."""
        monitor = HealthMonitor(mock_pool)

        statuses = monitor.get_all_status()
        assert len(statuses) == 2

    def test_get_healthy_groups(self, mock_pool):
        """Test getting list of healthy groups."""
        # Make group 1 unhealthy
        mock_pool.groups[1].health_check.return_value = False

        monitor = HealthMonitor(mock_pool)
        monitor.check_all()

        healthy = monitor.get_healthy_groups()
        assert 0 in healthy
        assert 1 not in healthy

    def test_is_group_available(self, mock_pool):
        """Test checking if group is available."""
        monitor = HealthMonitor(mock_pool)
        monitor.check_all()

        assert monitor.is_group_available(0) is True

    def test_is_group_available_circuit_open(self, mock_pool):
        """Test group unavailable when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=2)
        monitor = HealthMonitor(mock_pool, circuit_breaker_config=config)

        # Record failures to open circuit
        monitor._circuit_breakers[0].record_failure("Error 1")
        monitor._circuit_breakers[0].record_failure("Error 2")

        assert monitor.is_group_available(0) is False

    def test_attempt_recovery_success(self, mock_pool):
        """Test successful recovery attempt."""
        mock_pool.groups[0].cleanup.return_value = None
        mock_pool.groups[0].setup.return_value = True
        mock_pool.groups[0].health_check.return_value = True

        monitor = HealthMonitor(mock_pool)

        # Set group as unhealthy
        monitor._health_status[0].is_healthy = False
        monitor._health_status[0].consecutive_failures = 3

        result = monitor.attempt_recovery(0)

        assert result is True
        assert monitor._health_status[0].is_healthy is True
        assert monitor._health_status[0].consecutive_failures == 0

    def test_attempt_recovery_failure(self, mock_pool):
        """Test failed recovery attempt."""
        mock_pool.groups[0].cleanup.return_value = None
        mock_pool.groups[0].setup.return_value = False

        monitor = HealthMonitor(mock_pool)
        monitor._health_status[0].is_healthy = False

        result = monitor.attempt_recovery(0)

        assert result is False

    def test_get_summary(self, mock_pool):
        """Test getting health summary."""
        monitor = HealthMonitor(mock_pool)
        monitor.check_all()

        summary = monitor.get_summary()

        assert "healthy_groups" in summary
        assert "total_groups" in summary
        assert "health_percentage" in summary
        assert "avg_response_time_ms" in summary
        assert "groups" in summary

        assert summary["total_groups"] == 2
        assert summary["healthy_groups"] == 2
        assert summary["health_percentage"] == 100.0

    def test_consecutive_failures_tracking(self, mock_pool):
        """Test consecutive failures are tracked."""
        mock_pool.groups[0].health_check.return_value = False

        monitor = HealthMonitor(mock_pool)

        # Check multiple times
        monitor.check_group(0)
        assert monitor._health_status[0].consecutive_failures == 1

        monitor.check_group(0)
        assert monitor._health_status[0].consecutive_failures == 2

        # Success should reset
        mock_pool.groups[0].health_check.return_value = True
        monitor.check_group(0)
        assert monitor._health_status[0].consecutive_failures == 0
        assert monitor._health_status[0].consecutive_successes == 1


class TestRetryPolicy:
    """Tests for RetryPolicy dataclass."""

    def test_default_values(self):
        """Test default retry policy values."""
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.initial_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.exponential_base == 2.0
        assert policy.jitter == 0.1

    def test_custom_values(self):
        """Test custom retry policy values."""
        policy = RetryPolicy(
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=0.2
        )
        assert policy.max_retries == 5
        assert policy.initial_delay == 2.0
        assert policy.max_delay == 120.0

    def test_get_delay_exponential(self):
        """Test exponential delay calculation."""
        policy = RetryPolicy(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=0.0  # No jitter for predictable testing
        )

        # Delay should double each attempt
        assert policy.get_delay(0) == 1.0   # 1.0 * 2^0 = 1.0
        assert policy.get_delay(1) == 2.0   # 1.0 * 2^1 = 2.0
        assert policy.get_delay(2) == 4.0   # 1.0 * 2^2 = 4.0
        assert policy.get_delay(3) == 8.0   # 1.0 * 2^3 = 8.0

    def test_get_delay_max_cap(self):
        """Test delay is capped at max_delay."""
        policy = RetryPolicy(
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=0.0
        )

        # After several attempts, should hit cap
        assert policy.get_delay(10) == 10.0  # Would be 1024, but capped

    def test_get_delay_with_jitter(self):
        """Test delay includes jitter."""
        policy = RetryPolicy(
            initial_delay=10.0,
            jitter=0.2,  # 20% jitter
            exponential_base=1.0  # No exponential growth
        )

        delays = [policy.get_delay(0) for _ in range(100)]

        # With 20% jitter on 10.0, delays should be between 8.0 and 12.0
        assert min(delays) >= 8.0
        assert max(delays) <= 12.0
        # Should have some variation
        assert min(delays) != max(delays)

    def test_should_retry(self):
        """Test should_retry logic."""
        policy = RetryPolicy(max_retries=3)

        assert policy.should_retry(0) is True
        assert policy.should_retry(1) is True
        assert policy.should_retry(2) is True
        assert policy.should_retry(3) is False
        assert policy.should_retry(4) is False


class TestWithRetry:
    """Tests for with_retry decorator."""

    def test_success_no_retry(self):
        """Test successful function doesn't retry."""
        call_count = [0]

        @with_retry(RetryPolicy(max_retries=3))
        def successful_func():
            call_count[0] += 1
            return "success"

        result = successful_func()

        assert result == "success"
        assert call_count[0] == 1

    def test_retry_then_success(self):
        """Test function retries and eventually succeeds."""
        call_count = [0]

        @with_retry(RetryPolicy(max_retries=3, initial_delay=0.01))
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception(f"Attempt {call_count[0]} failed")
            return "success"

        result = flaky_func()

        assert result == "success"
        assert call_count[0] == 3

    def test_exhausted_retries(self):
        """Test exception raised after exhausting retries."""
        call_count = [0]

        @with_retry(RetryPolicy(max_retries=2, initial_delay=0.01))
        def always_fails():
            call_count[0] += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError) as exc_info:
            always_fails()

        assert "Always fails" in str(exc_info.value)
        assert call_count[0] == 3  # Initial + 2 retries

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        retry_attempts = []

        def on_retry(attempt, exception):
            retry_attempts.append((attempt, str(exception)))

        call_count = [0]

        @with_retry(
            RetryPolicy(max_retries=2, initial_delay=0.01),
            on_retry=on_retry
        )
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception(f"Error {call_count[0]}")
            return "success"

        flaky_func()

        assert len(retry_attempts) == 2
        assert retry_attempts[0][0] == 0
        assert "Error 1" in retry_attempts[0][1]
        assert retry_attempts[1][0] == 1
        assert "Error 2" in retry_attempts[1][1]

    def test_with_args_and_kwargs(self):
        """Test decorator works with function arguments."""
        @with_retry(RetryPolicy(max_retries=1))
        def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = func_with_args(1, 2, c=3)
        assert result == "1-2-3"

    def test_preserves_exception_type(self):
        """Test original exception type is preserved."""
        class CustomError(Exception):
            pass

        @with_retry(RetryPolicy(max_retries=1, initial_delay=0.01))
        def raises_custom():
            raise CustomError("Custom error")

        with pytest.raises(CustomError):
            raises_custom()
