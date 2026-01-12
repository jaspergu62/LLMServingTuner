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
Remote execution layer for parallel PSO evaluation.

This module provides SSH-based remote command execution and file transfer
capabilities for managing services across multiple nodes.
"""

import os
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

from msserviceprofiler.modelevalstate.exceptions import CommunicationError, IPCTimeoutError
from msserviceprofiler.modelevalstate.optimizer.parallel.config import NodeConfig

# Optional paramiko import - will be checked at runtime
try:
    import paramiko
    from paramiko import SSHClient, SFTPClient, AutoAddPolicy
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    paramiko = None
    SSHClient = None
    SFTPClient = None
    AutoAddPolicy = None


class ConnectionStatus(Enum):
    """Status of SSH connection."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class CommandResult:
    """Result of a remote command execution."""
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float = 0.0
    command: str = ""

    @property
    def success(self) -> bool:
        """Check if command succeeded (exit code 0)."""
        return self.exit_code == 0

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAIL({self.exit_code})"
        return f"CommandResult({status}, duration={self.duration_seconds:.2f}s)"


class RemoteExecutor:
    """
    Execute commands on a remote node via SSH.

    This class provides:
    - SSH connection management with auto-reconnect
    - Remote command execution with timeout
    - File upload/download via SFTP
    - Environment variable handling

    Assumes SSH key-based authentication is pre-configured.

    Example:
        node = NodeConfig(host="node0.cluster")
        executor = RemoteExecutor(node)
        executor.connect()

        result = executor.execute("ls -la")
        print(result.stdout)

        executor.upload_file("local.txt", "/remote/path/file.txt")
        executor.close()
    """

    def __init__(self,
                 node: NodeConfig,
                 connect_timeout: int = 30,
                 command_timeout: int = 300,
                 retry_count: int = 3,
                 retry_delay: int = 5):
        """
        Initialize RemoteExecutor.

        Args:
            node: Node configuration with host, port, etc.
            connect_timeout: SSH connection timeout in seconds
            command_timeout: Default command execution timeout in seconds
            retry_count: Number of connection retry attempts
            retry_delay: Delay between retries in seconds
        """
        if not PARAMIKO_AVAILABLE:
            raise ImportError(
                "paramiko is required for remote execution. "
                "Install with: pip install paramiko"
            )

        self.node = node
        self.connect_timeout = connect_timeout
        self.command_timeout = command_timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        self._ssh_client: Optional[SSHClient] = None
        self._sftp_client: Optional[SFTPClient] = None
        self._status = ConnectionStatus.DISCONNECTED
        self._last_error: Optional[str] = None

    @property
    def host(self) -> str:
        """Get the host address."""
        return self.node.host

    @property
    def status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if connected to remote host."""
        if self._status != ConnectionStatus.CONNECTED:
            return False
        # Verify connection is still alive
        try:
            if self._ssh_client and self._ssh_client.get_transport():
                return self._ssh_client.get_transport().is_active()
        except Exception:
            pass
        return False

    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error

    def connect(self) -> bool:
        """
        Establish SSH connection to the remote node.

        Uses SSH key-based authentication. Retries on failure.

        Returns:
            True if connection successful, False otherwise.
        """
        if self.is_connected:
            logger.debug(f"Already connected to {self.host}")
            return True

        self._status = ConnectionStatus.CONNECTING
        logger.info(f"Connecting to {self.host}:{self.node.ssh_port}")

        for attempt in range(self.retry_count):
            try:
                self._ssh_client = paramiko.SSHClient()
                self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                # Build connection kwargs
                connect_kwargs = {
                    "hostname": self.node.host,
                    "port": self.node.ssh_port,
                    "timeout": self.connect_timeout,
                    "allow_agent": True,
                    "look_for_keys": True,
                }

                # Add username if specified
                if self.node.username:
                    connect_kwargs["username"] = self.node.username

                self._ssh_client.connect(**connect_kwargs)
                self._status = ConnectionStatus.CONNECTED
                logger.info(f"Connected to {self.host}")
                return True

            except paramiko.AuthenticationException as e:
                self._last_error = f"Authentication failed: {e}"
                logger.error(f"Auth failed for {self.host}: {e}")
                self._status = ConnectionStatus.ERROR
                return False  # Don't retry auth failures

            except (paramiko.SSHException, socket.error, socket.timeout) as e:
                self._last_error = f"Connection failed: {e}"
                logger.warning(
                    f"Connection attempt {attempt + 1}/{self.retry_count} "
                    f"to {self.host} failed: {e}"
                )
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)

            except Exception as e:
                self._last_error = f"Unexpected error: {e}"
                logger.error(f"Unexpected error connecting to {self.host}: {e}")
                self._status = ConnectionStatus.ERROR
                return False

        self._status = ConnectionStatus.ERROR
        logger.error(f"Failed to connect to {self.host} after {self.retry_count} attempts")
        return False

    def close(self):
        """Close SSH and SFTP connections."""
        logger.debug(f"Closing connection to {self.host}")

        if self._sftp_client:
            try:
                self._sftp_client.close()
            except Exception as e:
                logger.warning(f"Error closing SFTP to {self.host}: {e}")
            self._sftp_client = None

        if self._ssh_client:
            try:
                self._ssh_client.close()
            except Exception as e:
                logger.warning(f"Error closing SSH to {self.host}: {e}")
            self._ssh_client = None

        self._status = ConnectionStatus.DISCONNECTED
        logger.info(f"Disconnected from {self.host}")

    def execute(self,
                command: str,
                timeout: Optional[int] = None,
                env: Optional[Dict[str, str]] = None,
                cwd: Optional[str] = None) -> CommandResult:
        """
        Execute a command on the remote node.

        Args:
            command: Command to execute
            timeout: Command timeout in seconds (uses default if None)
            env: Environment variables to set
            cwd: Working directory for command

        Returns:
            CommandResult with exit code, stdout, stderr

        Raises:
            CommunicationError: If not connected or execution fails
            IPCTimeoutError: If command times out
        """
        if not self.is_connected:
            if not self.connect():
                raise CommunicationError(
                    f"Not connected to {self.host}",
                    details={"host": self.host, "error": self._last_error}
                )

        timeout = timeout or self.command_timeout
        start_time = time.time()

        # Build full command with environment and cwd
        full_command = self._build_command(command, env, cwd)
        logger.debug(f"Executing on {self.host}: {command[:100]}...")

        try:
            stdin, stdout, stderr = self._ssh_client.exec_command(
                full_command,
                timeout=timeout
            )

            # Read output
            stdout_str = stdout.read().decode('utf-8', errors='replace')
            stderr_str = stderr.read().decode('utf-8', errors='replace')
            exit_code = stdout.channel.recv_exit_status()

            duration = time.time() - start_time

            result = CommandResult(
                exit_code=exit_code,
                stdout=stdout_str,
                stderr=stderr_str,
                duration_seconds=duration,
                command=command
            )

            if exit_code != 0:
                logger.warning(
                    f"Command on {self.host} exited with code {exit_code}: "
                    f"{stderr_str[:200]}"
                )
            else:
                logger.debug(f"Command on {self.host} completed in {duration:.2f}s")

            return result

        except socket.timeout:
            duration = time.time() - start_time
            raise IPCTimeoutError(
                operation=f"execute({command[:50]}...)",
                timeout_seconds=timeout
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Command execution failed on {self.host}: {e}")
            raise CommunicationError(
                f"Command execution failed: {e}",
                details={"host": self.host, "command": command[:100]}
            )

    def execute_background(self,
                           command: str,
                           log_file: Optional[str] = None,
                           env: Optional[Dict[str, str]] = None,
                           cwd: Optional[str] = None) -> CommandResult:
        """
        Execute a command in the background (non-blocking).

        Args:
            command: Command to execute
            log_file: File to redirect output (default: /dev/null)
            env: Environment variables to set
            cwd: Working directory for command

        Returns:
            CommandResult (exit_code will be 0 if backgrounding succeeded)
        """
        log_file = log_file or "/dev/null"
        bg_command = f"nohup {command} > {log_file} 2>&1 &"
        return self.execute(bg_command, env=env, cwd=cwd, timeout=30)

    def _build_command(self,
                       command: str,
                       env: Optional[Dict[str, str]] = None,
                       cwd: Optional[str] = None) -> str:
        """Build full command with environment and working directory."""
        parts = []

        # Add environment variables
        if env:
            env_exports = " ".join(f"{k}={v}" for k, v in env.items())
            parts.append(f"export {env_exports};")

        # Add NPU visible devices from node config
        if self.node.npu_ids:
            npu_devices = self.node.get_npu_visible_devices()
            parts.append(f"export ASCEND_VISIBLE_DEVICES={npu_devices};")

        # Add working directory
        if cwd:
            parts.append(f"cd {cwd} &&")

        parts.append(command)
        return " ".join(parts)

    def _get_sftp(self) -> SFTPClient:
        """Get or create SFTP client."""
        if not self.is_connected:
            if not self.connect():
                raise CommunicationError(
                    f"Not connected to {self.host}",
                    details={"host": self.host}
                )

        if self._sftp_client is None:
            self._sftp_client = self._ssh_client.open_sftp()

        return self._sftp_client

    def upload_file(self,
                    local_path: Union[str, Path],
                    remote_path: str,
                    create_dirs: bool = True) -> bool:
        """
        Upload a file to the remote node.

        Args:
            local_path: Local file path
            remote_path: Remote destination path
            create_dirs: Create parent directories if needed

        Returns:
            True if upload successful, False otherwise.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            logger.error(f"Local file not found: {local_path}")
            return False

        logger.debug(f"Uploading {local_path} to {self.host}:{remote_path}")

        try:
            # Create parent directories if needed
            if create_dirs:
                remote_dir = str(Path(remote_path).parent)
                self.execute(f"mkdir -p {remote_dir}", timeout=30)

            sftp = self._get_sftp()
            sftp.put(str(local_path), remote_path)
            logger.info(f"Uploaded {local_path.name} to {self.host}:{remote_path}")
            return True

        except Exception as e:
            logger.error(f"Upload failed to {self.host}: {e}")
            return False

    def download_file(self,
                      remote_path: str,
                      local_path: Union[str, Path],
                      create_dirs: bool = True) -> bool:
        """
        Download a file from the remote node.

        Args:
            remote_path: Remote file path
            local_path: Local destination path
            create_dirs: Create parent directories if needed

        Returns:
            True if download successful, False otherwise.
        """
        local_path = Path(local_path)
        logger.debug(f"Downloading {self.host}:{remote_path} to {local_path}")

        try:
            # Create local parent directories if needed
            if create_dirs:
                local_path.parent.mkdir(parents=True, exist_ok=True)

            sftp = self._get_sftp()
            sftp.get(remote_path, str(local_path))
            logger.info(f"Downloaded {remote_path} from {self.host}")
            return True

        except Exception as e:
            logger.error(f"Download failed from {self.host}: {e}")
            return False

    def write_file(self,
                   remote_path: str,
                   content: str,
                   create_dirs: bool = True) -> bool:
        """
        Write content directly to a remote file.

        Args:
            remote_path: Remote file path
            content: Content to write
            create_dirs: Create parent directories if needed

        Returns:
            True if write successful, False otherwise.
        """
        logger.debug(f"Writing to {self.host}:{remote_path}")

        try:
            # Create parent directories if needed
            if create_dirs:
                remote_dir = str(Path(remote_path).parent)
                self.execute(f"mkdir -p {remote_dir}", timeout=30)

            sftp = self._get_sftp()
            with sftp.open(remote_path, 'w') as f:
                f.write(content)

            logger.info(f"Wrote {len(content)} bytes to {self.host}:{remote_path}")
            return True

        except Exception as e:
            logger.error(f"Write failed to {self.host}: {e}")
            return False

    def read_file(self, remote_path: str) -> Optional[str]:
        """
        Read content from a remote file.

        Args:
            remote_path: Remote file path

        Returns:
            File content as string, or None if failed.
        """
        logger.debug(f"Reading from {self.host}:{remote_path}")

        try:
            sftp = self._get_sftp()
            with sftp.open(remote_path, 'r') as f:
                content = f.read().decode('utf-8', errors='replace')

            logger.debug(f"Read {len(content)} bytes from {self.host}:{remote_path}")
            return content

        except Exception as e:
            logger.error(f"Read failed from {self.host}: {e}")
            return None

    def file_exists(self, remote_path: str) -> bool:
        """Check if a remote file exists."""
        try:
            sftp = self._get_sftp()
            sftp.stat(remote_path)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def mkdir(self, remote_path: str, parents: bool = True) -> bool:
        """
        Create directory on remote node.

        Args:
            remote_path: Directory path to create
            parents: Create parent directories if True

        Returns:
            True if successful.
        """
        flag = "-p" if parents else ""
        result = self.execute(f"mkdir {flag} {remote_path}", timeout=30)
        return result.success

    def check_connectivity(self) -> bool:
        """
        Quick connectivity check.

        Returns:
            True if can connect and execute simple command.
        """
        try:
            if not self.connect():
                return False
            result = self.execute("echo ok", timeout=10)
            return result.success and "ok" in result.stdout
        except Exception:
            return False

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __repr__(self) -> str:
        return f"RemoteExecutor({self.host}, status={self._status.value})"
