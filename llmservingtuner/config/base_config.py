# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import time
from enum import Enum
from pathlib import Path
from typing import Optional
import llmservingtuner

RUN_TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())
INSTALL_PATH = Path(llmservingtuner.__path__[0])
RUN_PATH = Path(os.getcwd())
LLMSERVINGTUNER_CONFIG_PATH = "LLMSERVINGTUNER_CONFIG_PATH"


def _get_env_value(key: str) -> Optional[str]:
    return os.getenv(key) or os.getenv(key.lower())


_config_path = _get_env_value(LLMSERVINGTUNER_CONFIG_PATH)
if not _config_path:
    _config_path = RUN_PATH.joinpath("config.toml")
llmservingtuner_config_path = Path(_config_path).absolute().resolve()

LLMSERVINGTUNER_OUTPUT = "LLMSERVINGTUNER_OUTPUT"
CUSTOM_OUTPUT = LLMSERVINGTUNER_OUTPUT
custom_output = _get_env_value(LLMSERVINGTUNER_OUTPUT)
if custom_output:
    custom_output = Path(custom_output).resolve()
else:
    custom_output = RUN_PATH
LLMSERVINGTUNER_VLLM_CUSTOM_OUTPUT = "LLMSERVINGTUNER_VLLM_CUSTOM_OUTPUT"
VLLM_CUSTOM_OUTPUT = LLMSERVINGTUNER_VLLM_CUSTOM_OUTPUT
LLMSERVINGTUNER_SIMULATE = "LLMSERVINGTUNER_SIMULATE"
LLMSERVINGTUNER_ALL = "LLMSERVINGTUNER_ALL"
SIMULATE = "simulate"
REAL_EVALUATION = "real_evaluation"
REQUESTRATES = ("REQUESTRATE",)
CONCURRENCYS = ("CONCURRENCY", "MAXCONCURRENCY")
simulate_env = _get_env_value(LLMSERVINGTUNER_SIMULATE)
simulate_flag = simulate_env and (simulate_env.lower() == "true" or simulate_env.lower() != "false")
optimizer_env = _get_env_value(LLMSERVINGTUNER_ALL)
optimizer_flag = optimizer_env and (optimizer_env.lower() == "true" or optimizer_env.lower() != "false")


MINDIE_BENCHMARK_PERF_COLUMNS = ["average", "max", "min", "p75", "p90", "slo_p90", "p99", "n"]
FOLDER_LIMIT_SIZE = 1024 * 1024 * 1024  # 1GB


class EnginePolicy(Enum):
    mindie = "mindie"
    vllm = "vllm"


class AnalyzeTool(Enum):
    default = "default"
    profiler = "profiler"
    vllm_benchmark = "vllm"


class BenchMarkPolicy(Enum):
    benchmark = "benchmark"
    profiler_benchmark = "profiler_benchmark"
    vllm_benchmark = "vllm_benchmark"
    ais_bench = "ais_bench"


class DeployPolicy(Enum):
    single = "single"
    multiple = "multiple"


class PDPolicy(Enum):
    competition = "competition"
    disaggregation = "disaggregation"

    
class ServiceType(Enum):
    master = "master"
    slave = "slave"
