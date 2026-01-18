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

"""
Patch module for LLM serving frameworks.

This module provides patches for various LLM serving frameworks:
- MindIE LLM (Huawei)
- vLLM Ascend (Huawei NPU)
- vLLM GPU (NVIDIA GPU)
"""

from loguru import logger

from llmservingtuner.common import get_module_version

# Module names
MINDIE_LLM = "mindie_llm"
VLLM_ASCEND = "vllm_ascend"
VLLM_GPU = "vllm"

# Patch registries for different environments
mindie_simulate_patch = []
mindie_optimize_patch = []
vllm_ascend_simulate_patch = []
vllm_ascend_optimize_patch = []
vllm_gpu_simulate_patch = []
vllm_gpu_optimize_patch = []

# Environment to patch mapping
LLMSERVINGTUNER_SIMULATE = "LLMSERVINGTUNER_SIMULATE"
LLMSERVINGTUNER_ALL = "LLMSERVINGTUNER_ALL"

env_patch = {
    LLMSERVINGTUNER_SIMULATE: mindie_simulate_patch,
    LLMSERVINGTUNER_ALL: mindie_optimize_patch,
}

vllm_ascend_env_patch = {
    LLMSERVINGTUNER_SIMULATE: vllm_ascend_simulate_patch,
    LLMSERVINGTUNER_ALL: vllm_ascend_optimize_patch,
}

vllm_gpu_env_patch = {
    LLMSERVINGTUNER_SIMULATE: vllm_gpu_simulate_patch,
    LLMSERVINGTUNER_ALL: vllm_gpu_optimize_patch,
}

# Register MindIE patches
try:
    from llmservingtuner.patch.patch_manager import Patch2rc1  # mindie 2.1rc1-2.2
    mindie_simulate_patch.append(Patch2rc1)
    mindie_optimize_patch.append(Patch2rc1)
except ImportError as e:
    logger.warning(f"Failed to import Patch2rc1: {e}")

try:
    from llmservingtuner.patch.patch_mindie import PatchMindie2rc1  # mindie 2.0a9-2.0
    mindie_simulate_patch.append(PatchMindie2rc1)
    mindie_optimize_patch.append(PatchMindie2rc1)
except ImportError as e:
    logger.warning(f"Failed to import PatchMindie2rc1: {e}")

# Register vLLM Ascend patches
try:
    from llmservingtuner.patch.patch_vllm import PatchVllm
    vllm_ascend_optimize_patch.append(PatchVllm)
    vllm_ascend_simulate_patch.append(PatchVllm)
except ImportError as e:
    logger.warning(f"Failed to import PatchVllm: {e}")

try:
    from llmservingtuner.patch.patch_vllm_ascend import PatchVllmAscend
    vllm_ascend_optimize_patch.append(PatchVllmAscend)
    vllm_ascend_simulate_patch.append(PatchVllmAscend)
except ImportError as e:
    logger.warning(f"Failed to import PatchVllmAscend: {e}")

# Register vLLM GPU patches
try:
    from llmservingtuner.patch.patch_vllm_gpu import PatchVllmGPU, PatchVllmGPUV1
    vllm_gpu_optimize_patch.append(PatchVllmGPU)
    vllm_gpu_simulate_patch.append(PatchVllmGPU)
    vllm_gpu_optimize_patch.append(PatchVllmGPUV1)
    vllm_gpu_simulate_patch.append(PatchVllmGPUV1)
except ImportError as e:
    logger.warning(f"Failed to import PatchVllmGPU/PatchVllmGPUV1: {e}")


def enable_patch(target_env):
    """
    Enable patches for the specified environment.

    Args:
        target_env: Environment variable name (LLMSERVINGTUNER_SIMULATE or LLMSERVINGTUNER_ALL)

    Returns:
        List of successfully applied patch classes
    """
    applied_patches = []

    # Apply MindIE patches
    try:
        mindie_llm_version = get_module_version(MINDIE_LLM)
        for patch_cls in env_patch.get(target_env, []):
            if patch_cls.check_version(mindie_llm_version):
                patch_cls.patch()
                applied_patches.append(patch_cls)
    except (ModuleNotFoundError, ValueError):
        pass

    # Apply vLLM Ascend patches
    try:
        vllm_ascend_version = get_module_version(VLLM_ASCEND)
        for patch_cls in vllm_ascend_env_patch.get(target_env, []):
            if patch_cls.check_version(vllm_ascend_version):
                patch_cls.patch()
                applied_patches.append(patch_cls)
    except (ModuleNotFoundError, ValueError):
        pass

    # Apply vLLM GPU patches
    try:
        vllm_gpu_version = get_module_version(VLLM_GPU)
        for patch_cls in vllm_gpu_env_patch.get(target_env, []):
            if patch_cls.check_version(vllm_gpu_version):
                patch_cls.patch()
                applied_patches.append(patch_cls)
    except (ModuleNotFoundError, ValueError):
        pass

    if applied_patches:
        logger.info(f"Applied patches: {[p.__name__ for p in applied_patches]}")

    return applied_patches
