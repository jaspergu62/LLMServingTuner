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


from pathlib import Path

from loguru import logger
from packaging import version
from msserviceprofiler.msguard import Rule
from llmservingtuner.patch.patch_manager import check_append_patch, add_append_patch

_patch_dir = Path(__file__).absolute().expanduser().parent.resolve()



class Patch2rc1:
    mindie_llm = "2.0"
    mindie_llm_low = "2.0a9"

    @staticmethod
    def check_version(target_version):
        _t_v = version.parse(target_version)
        _c_v_up = version.parse(Patch2rc1.mindie_llm)
        _c_v_low = version.parse(Patch2rc1.mindie_llm_low)
        if _c_v_low < _t_v <= _c_v_up:
            pass
        else:
            logger.warning("The version may not match.")
        return True

    @staticmethod
    def patch():
        import mindie_llm
        file_path = mindie_llm.__path__[0]
        # 检查文件是否存在
        file = Path(file_path).joinpath("text_generator/plugins/plugin_manager.py").resolve()
        patch = _patch_dir.joinpath("patches/mindie.plugin_manager.patch")
        if not check_append_patch(file, patch):
            # 已经打过补丁，不需要打了
            logger.info("The patch aleady exists.")
            return
        add_append_patch(file, patch)

