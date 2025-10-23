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

import json
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

from msserviceprofiler.modelevalstate.config.config import VllmBenchmarkConfig, get_settings, PerformanceIndex, \
    OptimizerConfigField
from msserviceprofiler.modelevalstate.config.custom_command import VllmBenchmarkCommand
from msserviceprofiler.modelevalstate.optimizer.interfaces.benchmark import BenchmarkInterface
from msserviceprofiler.modelevalstate.optimizer.utils import remove_file
from msserviceprofiler.msguard.security import open_s, walk_s


MS_TO_S = 10 ** 3


class VllmBenchMark(BenchmarkInterface):
    def __init__(self, config: Optional[VllmBenchmarkConfig] = None, *args, **kwargs):

        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.vllm_benchmark
        super().__init__(*args, **kwargs)
        self.command = VllmBenchmarkCommand(self.config.command).command

    @property
    def data_field(self) -> Optional[Tuple[OptimizerConfigField, ...]]:
        """
        获取data field 属性
        Returns:  Optional[Tuple[OptimizerConfigField]]

        """
        return self.config.target_field

    @data_field.setter
    def data_field(self, value: Optional[Tuple[OptimizerConfigField]] = None) -> None:
        """
        提供新的数据，更新替换data field属性。
        Args:
            value:

        Returns:

        """
        self.config.target_field = value

    @property
    def num_prompts(self) -> int:
        """
        获取服务的进程名属性
        Returns:""

        """
        return self.config.command.num_prompts

    @num_prompts.setter
    def num_prompts(self, value):
        """
        获取服务的进程名属性
        Returns:""

        """
        self.config.command.num_prompts = value

    @property
    def model_name(self) -> str:
        """
        获取当前运行运行模型的名字
        Returns:

        """
        return ""

    @property
    def dataset_path(self) -> str:
        """
        获取当前使用的数据集
        Returns:

        """
        return ""

    @property
    def max_output_len(self) -> 0:
        """
        获取当前设置的最大输出长度。
        Returns:

        """
        return 0

    def update_command(self):
        self.command = VllmBenchmarkCommand(self.config.command).command

    def stop(self, del_log: bool = True):
        # 删除输出的文件
        output_path = Path(self.config.command.result_dir)
        remove_file(output_path)
        super().stop(del_log)

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField, ...]] = None):
        # 启动前清理输出目录 因为get_performance_index是从里面获取其中一条数据，防止获取到错误数据
        output_path = Path(self.config.command.result_dir)
        remove_file(output_path)
        super().before_run(run_params)

    def get_performance_index(self):
        output_path = Path(self.config.command.result_dir)
        performance_index = PerformanceIndex()
        for file in walk_s(output_path):
            file = Path(file)
            if not file.name.endswith(".json"):
                continue
            with open_s(file, mode='r', encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Failed in parse vllm benchmark result. file: {file}")
                    continue

            performance_index.generate_speed = data.get("output_throughput", 0)
            performance_index.time_to_first_token = data.get("mean_ttft_ms", 0) / MS_TO_S
            performance_index.time_per_output_token = data.get("mean_tpot_ms", 0) / MS_TO_S
            num_prompts = data.get("num_prompts", 1)
            completed = data.get("completed", 0)
            performance_index.success_rate = 0
            if num_prompts > 0:
                performance_index.success_rate = completed / num_prompts
            performance_index.throughput = float(data.get("request_throughput", 3.0))
        return performance_index
