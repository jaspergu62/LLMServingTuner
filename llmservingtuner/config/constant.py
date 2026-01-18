# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
from typing import Optional

LLMSERVINGTUNER_SIMULATE = "LLMSERVINGTUNER_SIMULATE"
LLMSERVINGTUNER_ALL = "LLMSERVINGTUNER_ALL"
SIMULATE = "simulate"

def _get_env_value(key: str) -> Optional[str]:
    return os.getenv(key) or os.getenv(key.lower())


simulate_env = _get_env_value(LLMSERVINGTUNER_SIMULATE)
simulate_flag = simulate_env and (simulate_env.lower() == "true" or simulate_env.lower() != "false")

REAL_EVALUATION = "real_evaluation"

REQUESTRATES = ("REQUESTRATE",)
CONCURRENCYS = ("CONCURRENCY", "MAXCONCURRENCY")
METRIC_TTFT = 'ttft'
METRIC_TPOT = 'tpot'
