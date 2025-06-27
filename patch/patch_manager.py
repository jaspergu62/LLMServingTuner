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


import os
import stat
from pathlib import Path

from loguru import logger
from packaging import version
from msguard.constraints.rule import validate_params, Rule

_patch_dir = Path(__file__).absolute().expanduser().parent.resolve()


def check_flag(target_file, patch_file):
    with open(target_file, "r", encoding="utf-8") as f:
        data = f.readlines()
    with open(patch_file, "r", encoding="utf-8") as f:
        patch_data = f.readlines()
    i = 0
    diff_flag = True
    for _o_row in data:
        # 原来的代码
        if i == 0 and _o_row != patch_data[0]:
            continue
        if i >= len(patch_data):
            break
        # 发现有补丁代码
        if _o_row == patch_data[i]:
            i += 1
            diff_flag = False
        else:
            diff_flag = True
    return diff_flag
 
 
@validate_params({'patch_file': Rule.input_file_read})
@validate_params({'target_file': Rule.output_path_write})
def add_patch(target_file, patch_file):
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with open(patch_file, "r", encoding="utf-8") as f:
        patch_data = f.readlines()
    with os.fdopen(os.open(target_file, flags, modes), "a") as f:
        for _row in patch_data:
            f.write(_row)
    # 没有打补丁的，添加补丁文件内容
    logger.info("The patch is installed successfully.")


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
        plugin_manager_file = Path(file_path).joinpath("text_generator/plugins/plugin_manager.py").resolve()
        if not Rule.input_file_read.is_satisfied_by(plugin_manager_file):
            logger.error("not found patch file for mindie")
            return
        plugin_manager_patch = _patch_dir.joinpath("plugin_manager_patch.patch")
        diff_flag = check_flag(plugin_manager_file, plugin_manager_patch)
        if not diff_flag:
            # 已经打过补丁，不需要打了
            logger.info("The patch aleady exists.")
            return
        add_patch(plugin_manager_file, plugin_manager_patch)

