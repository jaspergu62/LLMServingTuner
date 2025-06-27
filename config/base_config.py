# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import time
from enum import Enum
from pathlib import Path

import msserviceprofiler.modelevalstate

RUN_TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())
INSTALL_PATH = Path(msserviceprofiler.modelevalstate.__path__[0])
RUN_PATH = Path(os.getcwd())
MODEL_EVAL_STATE_CONFIG_PATH = "MODEL_EVAL_STATE_CONFIG_PATH"
modelevalstate_config_path = os.getenv(MODEL_EVAL_STATE_CONFIG_PATH) or os.getenv(MODEL_EVAL_STATE_CONFIG_PATH.lower())
if not modelevalstate_config_path:
    modelevalstate_config_path = RUN_PATH.joinpath("config.json")
modelevalstate_config_path = Path(modelevalstate_config_path).absolute().resolve()

CUSTOM_OUTPUT = "MODEL_EVAL_STATE_CUSTOM_OUTPUT"
custom_output = os.getenv(CUSTOM_OUTPUT) or os.getenv(CUSTOM_OUTPUT.lower())
if custom_output:
    custom_output = Path(custom_output).resolve()
else:
    custom_output = RUN_PATH


class DeployPolicy(Enum):
    single = "single"
    multiple = "multiple"


class BenchMarkPolicy(Enum):
    benchmark = "benchmark"
    profiler_benchmark = "profiler_benchmark"
    vllm_benchmark = "vllm_benchmark"


class AnalyzeTool(Enum):
    default = "default"
    profiler = "profiler"
    vllm_benchmark = "vllm"


class ServiceType(Enum):
    master = "master"
    slave = "slave"