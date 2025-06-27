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
import os
import shutil
from pathlib import Path

import psutil
from loguru import logger

from msserviceprofiler.msguard import validate_params, Rule


def remove_file(output_path: Path):
    if not output_path:
        return
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    if not output_path.exists():
        return
    if output_path.is_file():
        output_path.unlink()
        return
    for file in output_path.iterdir():
        if file.is_file():
            file.unlink()
        else:
            try:
                shutil.rmtree(file)
            except OSError:
                remove_file(file)


def kill_children(children):
    for child in children:
        if not child.is_running():
            continue
        try:
            child.send_signal(9)
            child.wait(10)
        except Exception as e:
            logger.error(f"Failed in kill the {child.pid} process. detail: {e}")
            continue
        if child.is_running():
            logger.error(f"Failed to kill the {child.pid} process.")


def kill_process(process_name):
    for proc in psutil.process_iter(["pid", "name"]):
        if not hasattr(proc, "info"):
            continue
        if process_name not in proc.info["name"]:
            continue
        children = psutil.Process(proc.pid).children(recursive=True)
        kill_children([proc])
        kill_children(children)


def backup(target, bak, class_name="", max_depth=10, current_depth=0):
    if not target or not bak:
        return
    if not isinstance(target, Path):
        target = Path(target)
    if not isinstance(bak, Path):
        bak = Path(bak)
    if not target.exists() or not bak.exists():
        return
    if current_depth >= max_depth:
        logger.warning(f"Reached maximum backup depth {max_depth} for {target}")
        return

    new_file = bak.joinpath(class_name).joinpath(target.name)
    if target.is_file():
        if not Rule.input_file_read.is_satisfied_by(target):
            return
        new_file.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        if not new_file.exists():
            shutil.copy(target, new_file)
    else:
        if not Rule.input_dir_traverse.is_satisfied_by(target):
            return
        if new_file.exists():
            for child in new_file.iterdir():
                backup(child, new_file, class_name, max_depth, current_depth + 1)
        else:
            shutil.copytree(target, new_file)


def close_file_fp(file_fp):
    if not file_fp:
        return
    try:
        # 检查file_fp是否是一个文件对象
        if hasattr(file_fp, 'close'):
            file_fp.close()
        else:
            # 如果file_fp是一个文件描述符，调用os.close()
            os.close(file_fp)
    except (AttributeError, OSError):
        return