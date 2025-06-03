# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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


from warnings import warn

from loguru import logger

from msserviceprofiler.modelevalstate.common import get_module_version

MINDIE_LLM = "mindie_llm"
VLLM_Ascend = "vllm_ascend"

simulate_patch = []
optimize_patch = []
collection_patch = []
simulate_patch_elegant = []
optimize_patch_elegant = []
collection_patch_elegant = []
vllm_simulate_patch = []
vllm_optimize_patch = []

env_patch = {
    "MODEL_EVAL_STATE_COLLECT": collection_patch,
    "MODEL_EVAL_STATE_SIMULATE": simulate_patch,
    "MODEL_EVAL_STATE_ALL": optimize_patch,
    "MODEL_EVAL_STATE_COLLECT_ELEGANT": collection_patch_elegant,
    "MODEL_EVAL_STATE_SIMULATE_ELEGANT": simulate_patch_elegant,
    "MODEL_EVAL_STATE_ALL_ELEGANT": optimize_patch_elegant
}

try:
    from modelevalstate.patch.patch_manager import Patch2rc1

    simulate_patch.append(Patch2rc1)
    optimize_patch.append(Patch2rc1)
    collection_patch.append(Patch2rc1)
except ImportError as e:
    warn(f"Failed from .patch_manager import Patch2rc1. error: {e}")

try:
    from modelevalstate.patch.plugin_simulate_patch import Patch2rc1

    simulate_patch_elegant.append(Patch2rc1)
    optimize_patch_elegant.append(Patch2rc1)
    collection_patch_elegant.append(Patch2rc1)
except ImportError as e:
    warn(f"Failed from .patch_manager import Patch2rc1. error: {e}")


try:
    from modelevalstate.patch.patch_vllm import PatchVllm

    vllm_simulate_patch.append(PatchVllm)
    vllm_optimize_patch.append(PatchVllm)
except ImportError as e:
    warn(f"Failed from .patch_manager import PatchVllm. error: {e}")


def enable_patch(targer_env):
    # mindie_llm_version = get_module_version(MINDIE_LLM)
    # flag = []
    # for _p in env_patch.get(targer_env):
    #     if _p.check_version(mindie_llm_version):
    #         _p.patch()
    #         flag.append(_p)
    flag = []
    try:
        mindie_llm_version = get_module_version(MINDIE_LLM)
        for _p in env_patch.get(targer_env):
             if _p.check_version(mindie_llm_version):
                _p.patch()
                flag.append(_p)
    except (ModuleNotFoundError, ValueError): 
        pass

    try:
        vllm_ascend_version = get_module_version(VLLM_Ascend)
        for _p in env_patch.get(targer_env):
             if _p.check_version(vllm_ascend_version):
                _p.patch()
                flag.append(_p)
    except (ModuleNotFoundError, ValueError):
        pass

    if flag:
        logger.info(f"Installed patch list {flag}.")
    else:
        logger.error(
            f"No match patch version is found. current version: {mindie_llm_version}, "
            f"support mindie_llm version {[getattr(_p, MINDIE_LLM) for _p in env_patch.get(targer_env)]}")
        raise ValueError(
            f"No match patch version is found. current version: {mindie_llm_version}, "
            f"support mindie_llm version {[getattr(_p, MINDIE_LLM) for _p in env_patch.get(targer_env)]}")
