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
Configuration classes for parallel PSO evaluation.

This module defines the data structures for configuring multi-node
parallel particle evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class NodeConfig:
    """
    Configuration for a single node in a service group.

    Attributes:
        host: Hostname or IP address of the node
        ssh_port: SSH port for remote connection (default: 22)
        npu_ids: List of NPU device IDs available on this node
        work_dir: Working directory for service files on this node
        username: SSH username (default: current user)
    """
    host: str
    ssh_port: int = 22
    npu_ids: List[int] = field(default_factory=lambda: list(range(8)))
    work_dir: str = "/tmp/modelevalstate"
    username: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.host:
            raise ValueError("host cannot be empty")
        if self.ssh_port <= 0 or self.ssh_port > 65535:
            raise ValueError(f"Invalid SSH port: {self.ssh_port}")
        if not self.npu_ids:
            raise ValueError("npu_ids cannot be empty")

    @property
    def npu_count(self) -> int:
        """Number of NPUs on this node."""
        return len(self.npu_ids)

    def get_npu_visible_devices(self) -> str:
        """Get ASCEND_VISIBLE_DEVICES environment variable value."""
        return ",".join(str(i) for i in self.npu_ids)

    def __repr__(self) -> str:
        return f"NodeConfig(host={self.host}, npus={self.npu_count})"


@dataclass
class ServiceGroupConfig:
    """
    Configuration for a service group spanning one or more nodes.

    A service group represents a single service instance that may span
    multiple nodes (e.g., a large model requiring 16 NPUs across 2 nodes).

    Attributes:
        group_id: Unique identifier for this service group
        nodes: List of nodes in this service group
        master_node_index: Index of the master node (runs main service process)
        start_script: Path to the service start script on master node
        stop_script: Path to the service stop script (optional)
        config_path: Path where config file should be uploaded
        health_check_url: URL template for health check (use {host} placeholder)
        health_check_timeout: Timeout for health check in seconds
    """
    group_id: int
    nodes: List[NodeConfig]
    master_node_index: int = 0
    start_script: str = "./start_service.sh"
    stop_script: Optional[str] = None
    config_path: str = "/tmp/modelevalstate/config.json"
    health_check_url: str = "http://{host}:8000/health"
    health_check_timeout: int = 300

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.nodes:
            raise ValueError("nodes cannot be empty")
        if self.master_node_index < 0 or self.master_node_index >= len(self.nodes):
            raise ValueError(
                f"master_node_index ({self.master_node_index}) out of range "
                f"for {len(self.nodes)} nodes"
            )
        if self.health_check_timeout <= 0:
            raise ValueError("health_check_timeout must be positive")

    @property
    def master_node(self) -> NodeConfig:
        """Get the master node configuration."""
        return self.nodes[self.master_node_index]

    @property
    def worker_nodes(self) -> List[NodeConfig]:
        """Get all worker (non-master) nodes."""
        return [n for i, n in enumerate(self.nodes) if i != self.master_node_index]

    @property
    def total_npus(self) -> int:
        """Total number of NPUs across all nodes."""
        return sum(node.npu_count for node in self.nodes)

    @property
    def node_count(self) -> int:
        """Number of nodes in this service group."""
        return len(self.nodes)

    def get_health_check_url(self) -> str:
        """Get the health check URL with master host substituted."""
        return self.health_check_url.format(host=self.master_node.host)

    def __repr__(self) -> str:
        return (
            f"ServiceGroupConfig(id={self.group_id}, "
            f"nodes={self.node_count}, npus={self.total_npus})"
        )


@dataclass
class ParallelConfig:
    """
    Top-level configuration for parallel PSO evaluation.

    Attributes:
        enabled: Whether parallel evaluation is enabled
        service_groups: List of service group configurations
        evaluation_timeout: Timeout for single particle evaluation in seconds
        retry_count: Number of retries on evaluation failure
        retry_delay: Delay between retries in seconds
    """
    enabled: bool = False
    service_groups: List[ServiceGroupConfig] = field(default_factory=list)
    evaluation_timeout: int = 600
    retry_count: int = 2
    retry_delay: int = 10

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.enabled and not self.service_groups:
            raise ValueError("service_groups cannot be empty when parallel is enabled")
        if self.evaluation_timeout <= 0:
            raise ValueError("evaluation_timeout must be positive")
        if self.retry_count < 0:
            raise ValueError("retry_count cannot be negative")

    @property
    def worker_count(self) -> int:
        """Number of parallel workers (service groups)."""
        return len(self.service_groups)

    @property
    def total_npus(self) -> int:
        """Total NPUs across all service groups."""
        return sum(sg.total_npus for sg in self.service_groups)

    def get_group_by_id(self, group_id: int) -> Optional[ServiceGroupConfig]:
        """Get service group by ID."""
        for sg in self.service_groups:
            if sg.group_id == group_id:
                return sg
        return None

    @classmethod
    def from_hostfile(cls,
                      hostfile_path: str,
                      nodes_per_group: int = 2,
                      npus_per_node: int = 8,
                      **kwargs) -> "ParallelConfig":
        """
        Create ParallelConfig from a hostfile.

        Hostfile format (one host per line):
            node0.cluster
            node1.cluster
            node2.cluster
            ...

        Args:
            hostfile_path: Path to hostfile
            nodes_per_group: Number of nodes per service group
            npus_per_node: Number of NPUs per node
            **kwargs: Additional ParallelConfig arguments

        Returns:
            ParallelConfig instance
        """
        hosts = []
        with open(hostfile_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    hosts.append(line)

        if len(hosts) < nodes_per_group:
            raise ValueError(
                f"Not enough hosts ({len(hosts)}) for {nodes_per_group} nodes per group"
            )

        service_groups = []
        group_id = 0

        for i in range(0, len(hosts), nodes_per_group):
            group_hosts = hosts[i:i + nodes_per_group]
            if len(group_hosts) < nodes_per_group:
                break  # Not enough hosts for a complete group

            nodes = [
                NodeConfig(
                    host=host,
                    npu_ids=list(range(npus_per_node))
                )
                for host in group_hosts
            ]

            service_groups.append(
                ServiceGroupConfig(
                    group_id=group_id,
                    nodes=nodes
                )
            )
            group_id += 1

        return cls(enabled=True, service_groups=service_groups, **kwargs)

    @classmethod
    def from_node_string(cls,
                         node_string: str,
                         npus_per_node: int = 8,
                         **kwargs) -> "ParallelConfig":
        """
        Create ParallelConfig from a node specification string.

        Format: "node0,node1:node2,node3:node4,node5"
        - Groups separated by ":"
        - Nodes within group separated by ","

        Args:
            node_string: Node specification string
            npus_per_node: Number of NPUs per node
            **kwargs: Additional ParallelConfig arguments

        Returns:
            ParallelConfig instance
        """
        service_groups = []
        groups = node_string.split(":")

        for group_id, group_str in enumerate(groups):
            hosts = [h.strip() for h in group_str.split(",") if h.strip()]
            if not hosts:
                continue

            nodes = [
                NodeConfig(
                    host=host,
                    npu_ids=list(range(npus_per_node))
                )
                for host in hosts
            ]

            service_groups.append(
                ServiceGroupConfig(
                    group_id=group_id,
                    nodes=nodes
                )
            )

        return cls(enabled=True, service_groups=service_groups, **kwargs)

    @classmethod
    def from_settings(cls, settings) -> "ParallelConfig":
        """
        Create ParallelConfig from ParallelSettings (pydantic model).

        Args:
            settings: ParallelSettings instance from config

        Returns:
            ParallelConfig instance
        """
        service_groups = []

        for sg_settings in settings.service_groups:
            nodes = [
                NodeConfig(
                    host=node.host,
                    ssh_port=node.ssh_port,
                    npu_ids=node.npu_ids,
                    work_dir=node.work_dir,
                    username=node.username
                )
                for node in sg_settings.nodes
            ]

            service_groups.append(
                ServiceGroupConfig(
                    group_id=sg_settings.group_id,
                    nodes=nodes,
                    master_node_index=sg_settings.master_node_index,
                    start_script=sg_settings.start_script,
                    stop_script=sg_settings.stop_script,
                    config_path=sg_settings.config_path,
                    health_check_url=sg_settings.health_check_url,
                    health_check_timeout=sg_settings.health_check_timeout
                )
            )

        return cls(
            enabled=settings.enabled,
            service_groups=service_groups,
            evaluation_timeout=settings.evaluation_timeout,
            retry_count=settings.retry_count,
            retry_delay=settings.retry_delay
        )

    def __repr__(self) -> str:
        return (
            f"ParallelConfig(enabled={self.enabled}, "
            f"groups={self.worker_count}, total_npus={self.total_npus})"
        )
