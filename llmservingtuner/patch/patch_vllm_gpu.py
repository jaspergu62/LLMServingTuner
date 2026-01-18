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

from pathlib import Path

from loguru import logger
from packaging import version
from llmservingtuner.patch.patch_manager import check_append_patch, add_append_patch, add_diff_patch, add_replace_patch


_patch_dir = Path(__file__).absolute().expanduser().parent.resolve()


class PatchVllmGPU:
    version_patch = {
        "0.10.0": "patches/vllm_gpu.0.10.0.model_runner.patch",
        "0.10.1": "patches/vllm_gpu.0.10.1.model_runner.patch",
    }
    patch_file = None

    @staticmethod
    def check_version(target_version):
        _t_v = version.parse(target_version)
        _c_v_list = [version.parse(v) for v in PatchVllmGPU.version_patch.keys()]
        if _t_v in _c_v_list:
            PatchVllmGPU.patch_file = PatchVllmGPU.version_patch[target_version]
            return True
        else:
            logger.warning(f"The version {target_version} is not supported.")
            return False
    
    @staticmethod
    def patch():
        import vllm
        file_path = vllm.__path__[0]
        file = Path(file_path).joinpath("worker/model_runner.py").resolve()
        patch = _patch_dir.joinpath(PatchVllmGPU.patch_file)
        if not check_append_patch(file, patch):
            logger.info("The patch already exists.")
            return
        add_append_patch(file, patch)



class PatchVllmGPUV1:
    version_patch = {
        "0.10.0": "patches/vllm_gpu.0.10.0.v1.gpu_model_runner.patch",
        "0.10.1": "patches/vllm_gpu.0.10.1.v1.gpu_model_runner.patch",
    }
    patch_file = None
    
    @staticmethod
    def check_version(target_version):
        _t_v = version.parse(target_version)
        _c_v_list = [version.parse(v) for v in PatchVllmGPUV1.version_patch.keys()]
        if _t_v in _c_v_list:
            PatchVllmGPUV1.patch_file = PatchVllmGPUV1.version_patch[target_version]
            return True
        else:
            logger.warning(f"The version {target_version} is not supported.")
            return False

    # Diff patch
    @staticmethod
    def patch():
        import vllm
        file_path = vllm.__path__[0]
        file = Path(file_path).joinpath("v1/worker/gpu_model_runner.py").resolve()
        patch = _patch_dir.joinpath(PatchVllmGPUV1.patch_file)
        
        # first apply / restore and then apply
        # add_diff_patch(file, patch)
        add_replace_patch(file, patch)
