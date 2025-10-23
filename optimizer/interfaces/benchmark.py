# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

from abc import ABC, abstractmethod
from typing import Tuple, Optional

from msserviceprofiler.modelevalstate.config.base_config import VLLM_CUSTOM_OUTPUT, MINDIE_BENCHMARK_PERF_COLUMNS, \
    AnalyzeTool
from msserviceprofiler.modelevalstate.optimizer.interfaces.custom_process import CustomProcess
from msserviceprofiler.modelevalstate.config.config import PerformanceIndex, OptimizerConfigField


MS_TO_S = 10 ** 3
US_TO_S = 10 ** 6


class BenchmarkInterface(CustomProcess, ABC):
    """
    操作benchmark程序，测试性能。
    """

    @property
    @abstractmethod
    def data_field(self) -> Optional[Tuple[OptimizerConfigField]]:
        """
        获取data field 属性
        Returns:  Optional[Tuple[OptimizerConfigField]]

        """
        pass

    @data_field.setter
    @abstractmethod
    def data_field(self, value: Optional[Tuple[OptimizerConfigField]] = None) -> None:
        """
        提供新的数据，更新替换data field属性。
        Args:
            value:

        Returns:

        """
        pass

    @property
    def num_prompts(self) -> int:
        """
        获取服务的进程名属性
        Returns:""

        """
        return 0

    @num_prompts.setter
    def num_prompts(self, value):
        """
        获取服务的进程名属性
        Returns:""

        """
        pass

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

    @abstractmethod
    def update_command(self) -> None:
        """
        更新服务启动命令。更新self.command属性。
        Returns: None

        """
        pass

    @abstractmethod
    def get_performance_index(self) -> PerformanceIndex:
        """
        获取性能指标
        Returns: 指标数据类

        """
        pass