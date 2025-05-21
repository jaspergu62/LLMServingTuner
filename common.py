# !/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict

_PREFILL = "prefill"
_DECODE = "decode"


class StateType(Enum):
    DEFAULT = 0
    LINE = 1


@dataclass
class State:
    prefill: int = 0
    decode: int = 0
    batch_prefill: int = 0
    batch_decode: int = 0

    def __repr__(self):
        return f"TT_{self.prefill}_{self.decode}_{self.batch_prefill}_{self.batch_decode}"

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __ne__(self, other):
        return self.__repr__() != other.__repr__()

    def sum(self):
        return self.prefill + self.decode + self.batch_prefill + self.batch_decode


def computer_speed(line_node, field):
    return 1 / (getattr(line_node, field) * 10 ** -3)


def computer_speed_with_second(line_node, field):
    return 1 / (getattr(line_node, field) * 10 ** -6)


def my_std(nums):
    n = len(nums)
    avg = sum(nums) / n
    return (sum(map(lambda e: (e - avg) * (e - avg), nums)) / n) ** 0.5


def get_train_sub_path(base_path: Path = Path(r"D:\PyProject\state_eval\tmp\pd_content\train")):
    # 给训练输出目录生成新的目录
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)
    _sub_len = len([0 for _ in base_path.iterdir()])
    _sub_dir = base_path.joinpath(f"{_sub_len + 1}")
    _sub_dir.mkdir(parents=True, exist_ok=True)
    return _sub_dir


def update_global_coefficient(global_coefficient: Dict, key: State, value: float) -> None:
    if key not in global_coefficient:
        global_coefficient[key] = [value]
    else:
        global_coefficient[key].append(value)


def get_module_version(module_name):
    try:
        # 方法1：直接导入模块
        # fix
        import importlib
        module = importlib.import_module(module_name)
        if hasattr(module, "__version__"):
            return module.__version__
        elif hasattr(module, "version"):
            return module.version
    except ImportError:
        pass

    try:
        # 方法2：使用 pkg_resources
        import pkg_resources
        return pkg_resources.get_distribution(module_name).version
    except (ImportError, pkg_resources.DistributionNotFound):
        pass

    try:
        # 方法3：使用 importlib.metadata（Python 3.8+）
        import importlib.metadata
        return importlib.metadata.version(module_name)
    except (ImportError):
        pass

    # 方法4：最后尝试 pip show
    try:
        import subprocess
        output = subprocess.check_output(
            ["/usr/bin/pip", "show", module_name],
            universal_newlines=True
        )
        for line in output.splitlines():
            if line.startswith("Version:"):
                return line.split(":")[1].strip()
    except subprocess.CalledProcessError:
        pass

    raise ValueError("模块未安装或无法获取版本")
