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
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Tuple, Optional, List

import psutil
from loguru import logger

from modelevalstate.config.config import CUSTOM_OUTPUT, settings, MODEL_EVAL_STATE_CONFIG_PATH, \
    modelevalstate_config_path
from modelevalstate.config.config import OptimizerConfigField
from modelevalstate.optimizer.utils import close_file_fp, remove_file, kill_children, backup, kill_process
from msserviceprofiler.msguard.security import open_s


class CustomProcess:
    def __init__(self, bak_path: Optional[Path] = None, command: Optional[List[str]] = None,
                 work_path: Optional[Path] = None, print_log: bool = False,
                 process_name: str = ""):
        self.command = command
        self.bak_path = bak_path
        self.work_path = work_path if work_path else os.getcwd()
        self.run_log = None
        self.run_log_offset = None
        self.run_log_fp = None
        self.process = None
        self.print_log = print_log
        self.process_name = process_name

    @staticmethod
    def check_env(process_name):

        logger.info("check env")
        _residual_process = []
        _all_process_name = process_name.split(",")
        for proc in psutil.process_iter(["pid", "name"]):
            if not hasattr(proc, "info"):
                continue
            _proc_flag = []
            for p in _all_process_name:
                if p not in proc.info["name"]:
                    _proc_flag.append(True)
                else:
                    _proc_flag.append(False)
            if all(_proc_flag):
                continue
            _residual_process.append(proc)
        if _residual_process:
            logger.info("kill residual_process")
            for _p_name in _all_process_name:
                try:
                    kill_process(_p_name)
                except Exception as e:
                    logger.error(f"Failed to kill process. {e}")
        time.sleep(1)


    def backup(self):
        backup(self.run_log, self.bak_path, self.__class__.__name__)

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField, ...]] = None):
        self.run_log_fp, self.run_log = tempfile.mkstemp(prefix="modelevalstate_")
        self.run_log_offset = 0
        if run_params:
            for k in run_params:
                if k.config_position == "env":
                    # env 类型的数据，设置环境变量和更新命令中包含的变量,设置时全部为大写
                    os.environ[k.name.upper().strip()] = str(k.value)
                    _var_name = f"${k.name.upper().strip()}"
                    if _var_name not in self.command:
                        continue
                    _i = self.command.index(_var_name)
                    if k.value:
                        self.command[_i] = str(k.value)
                    else:
                        # 这个元素没设置的话，从命令中删除该命令预填值
                        self.command.pop(_i)
                        self.command.pop(_i - 1)
        if CUSTOM_OUTPUT not in os.environ:
            # 设置输出目录
            os.environ[CUSTOM_OUTPUT] = str(settings.output)
        # 设置读取的json文件
        if MODEL_EVAL_STATE_CONFIG_PATH not in os.environ:
            os.environ[MODEL_EVAL_STATE_CONFIG_PATH] = str(modelevalstate_config_path)

    def run(self, run_params: Optional[Tuple[OptimizerConfigField, ...]] = None):
        # 启动测试
        if self.process_name:
            try:
                self.check_env(self.process_name)
            except Exception as e:
                logger.error(f"Failed to check env. {e}")
        self.before_run(run_params)
        try:
            self.process = subprocess.Popen(self.command, env=os.environ, stdout=self.run_log_fp,
                                            stderr=subprocess.STDOUT, text=True, cwd=self.work_path)
        except OSError as e:
            logger.error(f"Failed to run {self.command}. error {e}")
            raise e
        logger.info(f"command: {' '.join(self.command)}, log file: {self.run_log}")

    def get_log(self):
        output = None
        if not self.run_log:
            return output
        run_log_path = Path(self.run_log)
        if run_log_path.exists():
            try:
                with open_s(run_log_path, "r", encoding="utf-8") as f:
                    f.seek(self.run_log_offset)
                    output = f.read()
                    self.run_log_offset = f.tell()
            except (UnicodeError, OSError) as e:
                logger.error(f"Failed read {self.command} log. error {e}")
        return output

    def check_success(self):
        output = self.get_log()
        if self.print_log:
            logger.info(output)
        if self.process.poll() is None:
            return False
        elif self.process.poll() == 0:
            return True
        else:
            raise subprocess.SubprocessError(
                f"Failed in run {self.command}. return code: {self.process.returncode}. log: {self.run_log}")

    def stop(self, del_log: bool = True):
        self.run_log_offset = 0
        close_file_fp(self.run_log_fp)
        if del_log and self.run_log:
            remove_file(Path(self.run_log))
        if not self.process:
            return
        _process_state = self.process.poll()
        if _process_state is not None:
            logger.info(f"The mindie program has exited. exit_code: {_process_state}")
            return
        try:
            children = psutil.Process(self.process.pid).children(recursive=True)
            self.process.kill()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.send_signal(9)
            if self.process.poll() is not None:
                logger.debug(f"The {self.process.pid} process has been shut down.")
            else:
                logger.error(f"The {self.process.pid} process shutdown failed.")
            kill_children(children)
        except Exception as e:
            logger.error(f"Failed to stop simulator process. {e}")

    def get_last_log(self, number: int = 5):
        output = None
        if not self.run_log:
            return output
        run_log_path = Path(self.run_log)
        if run_log_path.exists():
            try:
                with open_s(run_log_path, "r", encoding="utf-8") as f:
                    file_lines = f.readlines()
            except (UnicodeError, OSError) as e:
                logger.error(f"Failed read {self.command} log. error {e}")
            output = ''.join(file_lines[-number:])
        return output
