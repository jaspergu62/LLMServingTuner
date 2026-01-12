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
Unit tests for RemoteExecutor class and related types.
"""

from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pytest
import socket

from msserviceprofiler.modelevalstate.optimizer.parallel.config import NodeConfig
from msserviceprofiler.modelevalstate.optimizer.parallel.remote import (
    ConnectionStatus,
    CommandResult,
    RemoteExecutor,
    PARAMIKO_AVAILABLE,
)


class TestConnectionStatus:
    """Tests for ConnectionStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert ConnectionStatus.DISCONNECTED.value == "disconnected"
        assert ConnectionStatus.CONNECTING.value == "connecting"
        assert ConnectionStatus.CONNECTED.value == "connected"
        assert ConnectionStatus.ERROR.value == "error"

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        statuses = list(ConnectionStatus)
        assert len(statuses) == 4


class TestCommandResult:
    """Tests for CommandResult dataclass."""

    def test_successful_result(self):
        """Test successful command result."""
        result = CommandResult(
            exit_code=0,
            stdout="output",
            stderr="",
            duration_seconds=1.5,
            command="ls -la"
        )
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "output"
        assert result.stderr == ""
        assert result.duration_seconds == 1.5
        assert result.command == "ls -la"

    def test_failed_result(self):
        """Test failed command result."""
        result = CommandResult(
            exit_code=1,
            stdout="",
            stderr="error message",
            duration_seconds=0.5,
            command="invalid_cmd"
        )
        assert result.success is False
        assert result.exit_code == 1
        assert result.stderr == "error message"

    def test_default_values(self):
        """Test default values."""
        result = CommandResult(exit_code=0, stdout="", stderr="")
        assert result.duration_seconds == 0.0
        assert result.command == ""

    def test_repr_success(self):
        """Test string representation for success."""
        result = CommandResult(exit_code=0, stdout="", stderr="", duration_seconds=2.5)
        repr_str = repr(result)
        assert "OK" in repr_str
        assert "2.50s" in repr_str

    def test_repr_failure(self):
        """Test string representation for failure."""
        result = CommandResult(exit_code=127, stdout="", stderr="", duration_seconds=0.1)
        repr_str = repr(result)
        assert "FAIL(127)" in repr_str


@pytest.mark.skipif(not PARAMIKO_AVAILABLE, reason="paramiko not installed")
class TestRemoteExecutor:
    """Tests for RemoteExecutor class."""

    @pytest.fixture
    def node_config(self):
        """Create a NodeConfig for testing."""
        return NodeConfig(
            host="test-node.cluster",
            ssh_port=22,
            username="testuser",
            npu_ids=[0, 1, 2, 3]
        )

    @pytest.fixture
    def mock_ssh_client(self):
        """Create a mock SSHClient."""
        with patch('msserviceprofiler.modelevalstate.optimizer.parallel.remote.paramiko') as mock_paramiko:
            mock_client = MagicMock()
            mock_paramiko.SSHClient.return_value = mock_client
            mock_paramiko.AutoAddPolicy.return_value = MagicMock()

            # Create proper exception classes
            class MockAuthException(Exception):
                pass

            class MockSSHException(Exception):
                pass

            mock_paramiko.AuthenticationException = MockAuthException
            mock_paramiko.SSHException = MockSSHException

            # Setup transport mock
            mock_transport = MagicMock()
            mock_transport.is_active.return_value = True
            mock_client.get_transport.return_value = mock_transport

            yield mock_client, mock_paramiko

    def test_init(self, node_config):
        """Test executor initialization."""
        with patch('msserviceprofiler.modelevalstate.optimizer.parallel.remote.PARAMIKO_AVAILABLE', True):
            with patch('msserviceprofiler.modelevalstate.optimizer.parallel.remote.paramiko'):
                executor = RemoteExecutor(node_config)
                assert executor.host == "test-node.cluster"
                assert executor.connect_timeout == 30
                assert executor.command_timeout == 300
                assert executor.retry_count == 3
                assert executor.status == ConnectionStatus.DISCONNECTED

    def test_init_with_custom_timeouts(self, node_config):
        """Test initialization with custom timeouts."""
        with patch('msserviceprofiler.modelevalstate.optimizer.parallel.remote.PARAMIKO_AVAILABLE', True):
            with patch('msserviceprofiler.modelevalstate.optimizer.parallel.remote.paramiko'):
                executor = RemoteExecutor(
                    node_config,
                    connect_timeout=60,
                    command_timeout=600,
                    retry_count=5,
                    retry_delay=10
                )
                assert executor.connect_timeout == 60
                assert executor.command_timeout == 600
                assert executor.retry_count == 5
                assert executor.retry_delay == 10

    def test_init_without_paramiko(self, node_config):
        """Test initialization raises error without paramiko."""
        with patch('msserviceprofiler.modelevalstate.optimizer.parallel.remote.PARAMIKO_AVAILABLE', False):
            with pytest.raises(ImportError) as exc_info:
                RemoteExecutor(node_config)
            assert "paramiko is required" in str(exc_info.value)

    def test_connect_success(self, node_config, mock_ssh_client):
        """Test successful connection."""
        mock_client, mock_paramiko = mock_ssh_client

        executor = RemoteExecutor(node_config)
        result = executor.connect()

        assert result is True
        assert executor.status == ConnectionStatus.CONNECTED
        mock_client.connect.assert_called_once()

    def test_connect_already_connected(self, node_config, mock_ssh_client):
        """Test connect when already connected."""
        mock_client, mock_paramiko = mock_ssh_client

        executor = RemoteExecutor(node_config)
        executor.connect()

        # Reset mock
        mock_client.connect.reset_mock()

        # Try to connect again
        result = executor.connect()
        assert result is True
        mock_client.connect.assert_not_called()

    def test_connect_auth_failure(self, node_config, mock_ssh_client):
        """Test connection with authentication failure."""
        mock_client, mock_paramiko = mock_ssh_client
        # Use the exception class from fixture
        AuthException = mock_paramiko.AuthenticationException
        mock_client.connect.side_effect = AuthException("Auth failed")
        mock_client.get_transport.return_value = None

        executor = RemoteExecutor(node_config)
        result = executor.connect()

        assert result is False
        assert executor.status == ConnectionStatus.ERROR
        assert "Authentication failed" in executor.last_error

    def test_connect_retry_on_network_error(self, node_config, mock_ssh_client):
        """Test connection retries on network error."""
        mock_client, mock_paramiko = mock_ssh_client
        mock_client.get_transport.return_value = None

        # Use the exception class from fixture
        SSHException = mock_paramiko.SSHException

        def setup_transport_after_connect():
            mock_transport = MagicMock()
            mock_transport.is_active.return_value = True
            mock_client.get_transport.return_value = mock_transport

        # Success on third try sets up transport
        call_count = [0]
        def connect_with_setup(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise SSHException("Network error")
            setup_transport_after_connect()

        mock_client.connect.side_effect = connect_with_setup

        with patch('time.sleep'):  # Skip actual sleep
            executor = RemoteExecutor(node_config, retry_delay=0)
            result = executor.connect()

        assert result is True
        assert mock_client.connect.call_count == 3

    def test_close(self, node_config, mock_ssh_client):
        """Test closing connections."""
        mock_client, mock_paramiko = mock_ssh_client
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp

        executor = RemoteExecutor(node_config)
        executor.connect()
        executor._sftp_client = mock_sftp

        executor.close()

        assert executor.status == ConnectionStatus.DISCONNECTED
        mock_client.close.assert_called_once()
        mock_sftp.close.assert_called_once()

    def test_execute_success(self, node_config, mock_ssh_client):
        """Test successful command execution."""
        mock_client, mock_paramiko = mock_ssh_client

        # Setup exec_command mock
        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        mock_stdout.read.return_value = b"command output\n"
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        executor = RemoteExecutor(node_config)
        executor.connect()
        result = executor.execute("ls -la")

        assert result.success is True
        assert result.exit_code == 0
        assert "command output" in result.stdout
        assert result.command == "ls -la"

    def test_execute_failure(self, node_config, mock_ssh_client):
        """Test failed command execution."""
        mock_client, mock_paramiko = mock_ssh_client

        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stderr.read.return_value = b"command not found\n"
        mock_stdout.channel.recv_exit_status.return_value = 127
        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        executor = RemoteExecutor(node_config)
        executor.connect()
        result = executor.execute("invalid_command")

        assert result.success is False
        assert result.exit_code == 127
        assert "command not found" in result.stderr

    def test_execute_with_env_and_cwd(self, node_config, mock_ssh_client):
        """Test command execution with environment and working directory."""
        mock_client, mock_paramiko = mock_ssh_client

        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        mock_stdout.read.return_value = b"ok"
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        executor = RemoteExecutor(node_config)
        executor.connect()
        executor.execute(
            "echo $MY_VAR",
            env={"MY_VAR": "test_value"},
            cwd="/tmp"
        )

        # Verify command includes env and cwd
        call_args = mock_client.exec_command.call_args
        full_command = call_args[0][0]
        assert "export MY_VAR=test_value" in full_command
        assert "cd /tmp" in full_command

    def test_execute_timeout(self, node_config, mock_ssh_client):
        """Test command timeout."""
        mock_client, mock_paramiko = mock_ssh_client
        mock_client.exec_command.side_effect = socket.timeout("Command timed out")

        from msserviceprofiler.modelevalstate.exceptions import IPCTimeoutError

        executor = RemoteExecutor(node_config)
        executor.connect()

        with pytest.raises(IPCTimeoutError):
            executor.execute("sleep 1000", timeout=1)

    def test_execute_auto_reconnect(self, node_config, mock_ssh_client):
        """Test execute auto-reconnects if disconnected."""
        mock_client, mock_paramiko = mock_ssh_client

        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        mock_stdout.read.return_value = b"ok"
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        executor = RemoteExecutor(node_config)
        # Don't call connect() first

        result = executor.execute("echo ok")

        # Should have connected automatically
        mock_client.connect.assert_called()
        assert result.success is True

    def test_execute_background(self, node_config, mock_ssh_client):
        """Test background command execution."""
        mock_client, mock_paramiko = mock_ssh_client

        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        executor = RemoteExecutor(node_config)
        executor.connect()
        result = executor.execute_background("my_service", log_file="/var/log/service.log")

        call_args = mock_client.exec_command.call_args
        full_command = call_args[0][0]
        assert "nohup" in full_command
        assert "/var/log/service.log" in full_command
        assert "&" in full_command

    def test_write_file(self, node_config, mock_ssh_client):
        """Test writing file to remote."""
        mock_client, mock_paramiko = mock_ssh_client
        mock_sftp = MagicMock()
        mock_file = MagicMock()
        mock_sftp.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_sftp.open.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.open_sftp.return_value = mock_sftp

        # Mock execute for mkdir
        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        executor = RemoteExecutor(node_config)
        executor.connect()
        result = executor.write_file("/tmp/test.txt", "file content")

        assert result is True
        mock_sftp.open.assert_called_with("/tmp/test.txt", 'w')

    def test_read_file(self, node_config, mock_ssh_client):
        """Test reading file from remote."""
        mock_client, mock_paramiko = mock_ssh_client
        mock_sftp = MagicMock()
        mock_file = MagicMock()
        mock_file.read.return_value = b"file content"
        mock_sftp.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_sftp.open.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.open_sftp.return_value = mock_sftp

        executor = RemoteExecutor(node_config)
        executor.connect()
        content = executor.read_file("/tmp/test.txt")

        assert content == "file content"

    def test_read_file_not_found(self, node_config, mock_ssh_client):
        """Test reading non-existent file."""
        mock_client, mock_paramiko = mock_ssh_client
        mock_sftp = MagicMock()
        mock_sftp.open.side_effect = FileNotFoundError("No such file")
        mock_client.open_sftp.return_value = mock_sftp

        executor = RemoteExecutor(node_config)
        executor.connect()
        content = executor.read_file("/tmp/nonexistent.txt")

        assert content is None

    def test_file_exists(self, node_config, mock_ssh_client):
        """Test checking if file exists."""
        mock_client, mock_paramiko = mock_ssh_client
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp

        executor = RemoteExecutor(node_config)
        executor.connect()

        # File exists
        mock_sftp.stat.return_value = MagicMock()
        assert executor.file_exists("/tmp/exists.txt") is True

        # File doesn't exist
        mock_sftp.stat.side_effect = FileNotFoundError()
        assert executor.file_exists("/tmp/notexists.txt") is False

    def test_upload_file(self, node_config, mock_ssh_client, tmp_path):
        """Test uploading file to remote."""
        mock_client, mock_paramiko = mock_ssh_client
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp

        # Create temp local file
        local_file = tmp_path / "test.txt"
        local_file.write_text("test content")

        # Mock execute for mkdir
        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        executor = RemoteExecutor(node_config)
        executor.connect()
        result = executor.upload_file(str(local_file), "/remote/test.txt")

        assert result is True
        mock_sftp.put.assert_called_once()

    def test_upload_file_not_found(self, node_config, mock_ssh_client):
        """Test uploading non-existent file."""
        mock_client, mock_paramiko = mock_ssh_client

        executor = RemoteExecutor(node_config)
        executor.connect()
        result = executor.upload_file("/nonexistent/file.txt", "/remote/file.txt")

        assert result is False

    def test_download_file(self, node_config, mock_ssh_client, tmp_path):
        """Test downloading file from remote."""
        mock_client, mock_paramiko = mock_ssh_client
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp

        local_file = tmp_path / "downloaded.txt"

        executor = RemoteExecutor(node_config)
        executor.connect()
        result = executor.download_file("/remote/file.txt", str(local_file))

        assert result is True
        mock_sftp.get.assert_called_once()

    def test_mkdir(self, node_config, mock_ssh_client):
        """Test creating remote directory."""
        mock_client, mock_paramiko = mock_ssh_client

        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        executor = RemoteExecutor(node_config)
        executor.connect()
        result = executor.mkdir("/tmp/newdir", parents=True)

        assert result is True
        call_args = mock_client.exec_command.call_args
        assert "mkdir -p" in call_args[0][0]

    def test_check_connectivity(self, node_config, mock_ssh_client):
        """Test connectivity check."""
        mock_client, mock_paramiko = mock_ssh_client

        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        mock_stdout.read.return_value = b"ok\n"
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        executor = RemoteExecutor(node_config)
        result = executor.check_connectivity()

        assert result is True

    def test_check_connectivity_failure(self, node_config, mock_ssh_client):
        """Test connectivity check failure."""
        mock_client, mock_paramiko = mock_ssh_client
        mock_client.connect.side_effect = Exception("Connection refused")
        mock_client.get_transport.return_value = None

        executor = RemoteExecutor(node_config)
        result = executor.check_connectivity()

        assert result is False

    def test_context_manager(self, node_config, mock_ssh_client):
        """Test context manager usage."""
        mock_client, mock_paramiko = mock_ssh_client

        with RemoteExecutor(node_config) as executor:
            assert executor.status == ConnectionStatus.CONNECTED

        assert executor.status == ConnectionStatus.DISCONNECTED
        mock_client.close.assert_called()

    def test_repr(self, node_config, mock_ssh_client):
        """Test string representation."""
        mock_client, mock_paramiko = mock_ssh_client

        executor = RemoteExecutor(node_config)
        repr_str = repr(executor)

        assert "RemoteExecutor" in repr_str
        assert "test-node.cluster" in repr_str
        assert "disconnected" in repr_str

    def test_build_command_with_npu_devices(self, node_config, mock_ssh_client):
        """Test command building includes NPU devices."""
        mock_client, mock_paramiko = mock_ssh_client

        executor = RemoteExecutor(node_config)
        command = executor._build_command("python train.py", cwd="/app")

        assert "ASCEND_VISIBLE_DEVICES" in command
        assert "cd /app" in command


class TestRemoteExecutorWithoutParamiko:
    """Tests for RemoteExecutor when paramiko is not available."""

    def test_init_raises_import_error(self):
        """Test that initialization raises ImportError without paramiko."""
        with patch('msserviceprofiler.modelevalstate.optimizer.parallel.remote.PARAMIKO_AVAILABLE', False):
            node = NodeConfig(host="test.cluster")
            with pytest.raises(ImportError) as exc_info:
                RemoteExecutor(node)
            assert "paramiko is required" in str(exc_info.value)
