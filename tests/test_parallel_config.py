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
Unit tests for parallel configuration classes.
"""

import tempfile
import pytest

from msserviceprofiler.modelevalstate.optimizer.parallel.config import (
    NodeConfig,
    ServiceGroupConfig,
    ParallelConfig,
)


class TestNodeConfig:
    """Tests for NodeConfig dataclass."""

    def test_default_values(self):
        """Test NodeConfig with default values."""
        node = NodeConfig(host="node0.cluster")
        assert node.host == "node0.cluster"
        assert node.ssh_port == 22
        assert node.npu_ids == [0, 1, 2, 3, 4, 5, 6, 7]
        assert node.work_dir == "/tmp/modelevalstate"
        assert node.username is None

    def test_custom_values(self):
        """Test NodeConfig with custom values."""
        node = NodeConfig(
            host="192.168.1.100",
            ssh_port=2222,
            npu_ids=[0, 1, 2, 3],
            work_dir="/opt/service",
            username="admin"
        )
        assert node.host == "192.168.1.100"
        assert node.ssh_port == 2222
        assert node.npu_ids == [0, 1, 2, 3]
        assert node.npu_count == 4
        assert node.username == "admin"

    def test_npu_visible_devices(self):
        """Test get_npu_visible_devices method."""
        node = NodeConfig(host="node0", npu_ids=[0, 2, 4, 6])
        assert node.get_npu_visible_devices() == "0,2,4,6"

    def test_empty_host_raises_error(self):
        """Test that empty host raises ValueError."""
        with pytest.raises(ValueError, match="host cannot be empty"):
            NodeConfig(host="")

    def test_invalid_ssh_port_raises_error(self):
        """Test that invalid SSH port raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SSH port"):
            NodeConfig(host="node0", ssh_port=0)
        with pytest.raises(ValueError, match="Invalid SSH port"):
            NodeConfig(host="node0", ssh_port=70000)

    def test_empty_npu_ids_raises_error(self):
        """Test that empty npu_ids raises ValueError."""
        with pytest.raises(ValueError, match="npu_ids cannot be empty"):
            NodeConfig(host="node0", npu_ids=[])

    def test_repr(self):
        """Test NodeConfig string representation."""
        node = NodeConfig(host="node0", npu_ids=[0, 1, 2, 3])
        assert "node0" in repr(node)
        assert "4" in repr(node)  # npu_count


class TestServiceGroupConfig:
    """Tests for ServiceGroupConfig dataclass."""

    def test_single_node_group(self):
        """Test ServiceGroupConfig with single node."""
        nodes = [NodeConfig(host="node0")]
        group = ServiceGroupConfig(group_id=0, nodes=nodes)
        assert group.group_id == 0
        assert group.node_count == 1
        assert group.master_node.host == "node0"
        assert group.worker_nodes == []

    def test_multi_node_group(self):
        """Test ServiceGroupConfig with multiple nodes."""
        nodes = [
            NodeConfig(host="node0"),
            NodeConfig(host="node1"),
            NodeConfig(host="node2")
        ]
        group = ServiceGroupConfig(group_id=1, nodes=nodes, master_node_index=0)
        assert group.node_count == 3
        assert group.master_node.host == "node0"
        assert len(group.worker_nodes) == 2
        assert group.worker_nodes[0].host == "node1"
        assert group.worker_nodes[1].host == "node2"

    def test_total_npus(self):
        """Test total_npus calculation."""
        nodes = [
            NodeConfig(host="node0", npu_ids=[0, 1, 2, 3]),
            NodeConfig(host="node1", npu_ids=[0, 1, 2, 3, 4, 5, 6, 7])
        ]
        group = ServiceGroupConfig(group_id=0, nodes=nodes)
        assert group.total_npus == 12

    def test_health_check_url(self):
        """Test get_health_check_url method."""
        nodes = [NodeConfig(host="192.168.1.100")]
        group = ServiceGroupConfig(
            group_id=0,
            nodes=nodes,
            health_check_url="http://{host}:8000/health"
        )
        assert group.get_health_check_url() == "http://192.168.1.100:8000/health"

    def test_empty_nodes_raises_error(self):
        """Test that empty nodes raises ValueError."""
        with pytest.raises(ValueError, match="nodes cannot be empty"):
            ServiceGroupConfig(group_id=0, nodes=[])

    def test_invalid_master_index_raises_error(self):
        """Test that invalid master_node_index raises ValueError."""
        nodes = [NodeConfig(host="node0")]
        with pytest.raises(ValueError, match="master_node_index.*out of range"):
            ServiceGroupConfig(group_id=0, nodes=nodes, master_node_index=5)

    def test_negative_master_index_raises_error(self):
        """Test that negative master_node_index raises ValueError."""
        nodes = [NodeConfig(host="node0")]
        with pytest.raises(ValueError, match="master_node_index.*out of range"):
            ServiceGroupConfig(group_id=0, nodes=nodes, master_node_index=-1)

    def test_custom_scripts(self):
        """Test custom start/stop scripts."""
        nodes = [NodeConfig(host="node0")]
        group = ServiceGroupConfig(
            group_id=0,
            nodes=nodes,
            start_script="/opt/bin/start.sh",
            stop_script="/opt/bin/stop.sh"
        )
        assert group.start_script == "/opt/bin/start.sh"
        assert group.stop_script == "/opt/bin/stop.sh"


class TestParallelConfig:
    """Tests for ParallelConfig dataclass."""

    def test_disabled_config(self):
        """Test disabled ParallelConfig."""
        config = ParallelConfig(enabled=False)
        assert config.enabled is False
        assert config.worker_count == 0
        assert config.total_npus == 0

    def test_enabled_config(self):
        """Test enabled ParallelConfig with service groups."""
        groups = [
            ServiceGroupConfig(
                group_id=0,
                nodes=[NodeConfig(host="node0"), NodeConfig(host="node1")]
            ),
            ServiceGroupConfig(
                group_id=1,
                nodes=[NodeConfig(host="node2"), NodeConfig(host="node3")]
            )
        ]
        config = ParallelConfig(enabled=True, service_groups=groups)
        assert config.enabled is True
        assert config.worker_count == 2
        assert config.total_npus == 32  # 4 nodes * 8 NPUs

    def test_get_group_by_id(self):
        """Test get_group_by_id method."""
        groups = [
            ServiceGroupConfig(group_id=0, nodes=[NodeConfig(host="node0")]),
            ServiceGroupConfig(group_id=5, nodes=[NodeConfig(host="node1")])
        ]
        config = ParallelConfig(enabled=True, service_groups=groups)

        assert config.get_group_by_id(0).nodes[0].host == "node0"
        assert config.get_group_by_id(5).nodes[0].host == "node1"
        assert config.get_group_by_id(99) is None

    def test_enabled_without_groups_raises_error(self):
        """Test that enabled config without groups raises ValueError."""
        with pytest.raises(ValueError, match="service_groups cannot be empty"):
            ParallelConfig(enabled=True, service_groups=[])

    def test_invalid_timeout_raises_error(self):
        """Test that non-positive timeout raises ValueError."""
        with pytest.raises(ValueError, match="evaluation_timeout must be positive"):
            ParallelConfig(enabled=False, evaluation_timeout=0)

    def test_negative_retry_raises_error(self):
        """Test that negative retry_count raises ValueError."""
        with pytest.raises(ValueError, match="retry_count cannot be negative"):
            ParallelConfig(enabled=False, retry_count=-1)

    def test_from_hostfile(self):
        """Test creating ParallelConfig from hostfile."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("node0.cluster\n")
            f.write("node1.cluster\n")
            f.write("node2.cluster\n")
            f.write("node3.cluster\n")
            f.write("# comment line\n")
            f.write("node4.cluster\n")
            f.write("node5.cluster\n")
            f.flush()

            config = ParallelConfig.from_hostfile(
                f.name,
                nodes_per_group=2,
                npus_per_node=16
            )

        assert config.enabled is True
        assert config.worker_count == 3  # 6 nodes / 2 per group
        assert config.service_groups[0].nodes[0].host == "node0.cluster"
        assert config.service_groups[0].nodes[1].host == "node1.cluster"
        assert config.service_groups[0].nodes[0].npu_count == 16

    def test_from_hostfile_not_enough_hosts(self):
        """Test from_hostfile with insufficient hosts."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("node0.cluster\n")
            f.flush()

            with pytest.raises(ValueError, match="Not enough hosts"):
                ParallelConfig.from_hostfile(f.name, nodes_per_group=2)

    def test_from_node_string(self):
        """Test creating ParallelConfig from node string."""
        config = ParallelConfig.from_node_string(
            "node0,node1:node2,node3:node4,node5",
            npus_per_node=8
        )

        assert config.enabled is True
        assert config.worker_count == 3
        assert config.service_groups[0].group_id == 0
        assert config.service_groups[0].nodes[0].host == "node0"
        assert config.service_groups[0].nodes[1].host == "node1"
        assert config.service_groups[2].nodes[0].host == "node4"

    def test_from_node_string_single_node_groups(self):
        """Test from_node_string with single-node groups."""
        config = ParallelConfig.from_node_string("node0:node1:node2")

        assert config.worker_count == 3
        assert config.service_groups[0].node_count == 1
        assert config.service_groups[1].node_count == 1

    def test_from_node_string_with_spaces(self):
        """Test from_node_string handles whitespace."""
        config = ParallelConfig.from_node_string(" node0 , node1 : node2 , node3 ")

        assert config.worker_count == 2
        assert config.service_groups[0].nodes[0].host == "node0"


class TestParallelConfigIntegration:
    """Integration tests for ParallelConfig."""

    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        # Create nodes
        nodes_group0 = [
            NodeConfig(host="master0", npu_ids=list(range(16))),
            NodeConfig(host="worker0", npu_ids=list(range(16)))
        ]
        nodes_group1 = [
            NodeConfig(host="master1", npu_ids=list(range(16))),
            NodeConfig(host="worker1", npu_ids=list(range(16)))
        ]

        # Create service groups
        groups = [
            ServiceGroupConfig(
                group_id=0,
                nodes=nodes_group0,
                start_script="/opt/llm/start.sh",
                health_check_url="http://{host}:8000/v1/health",
                health_check_timeout=600
            ),
            ServiceGroupConfig(
                group_id=1,
                nodes=nodes_group1,
                start_script="/opt/llm/start.sh",
                health_check_url="http://{host}:8000/v1/health",
                health_check_timeout=600
            )
        ]

        # Create parallel config
        config = ParallelConfig(
            enabled=True,
            service_groups=groups,
            evaluation_timeout=900,
            retry_count=3,
            retry_delay=15
        )

        # Verify configuration
        assert config.worker_count == 2
        assert config.total_npus == 64
        assert config.evaluation_timeout == 900

        # Verify group access
        group0 = config.get_group_by_id(0)
        assert group0.master_node.host == "master0"
        assert group0.total_npus == 32
        assert group0.get_health_check_url() == "http://master0:8000/v1/health"
