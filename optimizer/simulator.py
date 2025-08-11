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
import json
import os
import shlex
import stat
import subprocess
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Tuple, Optional
import shutil
import psutil
from loguru import logger

from msserviceprofiler.modelevalstate.config.config import (MindieConfig, MODEL_EVAL_STATE_CONFIG_PATH,
                                                            modelevalstate_config_path, CUSTOM_OUTPUT, custom_output)
from msserviceprofiler.modelevalstate.config.config import OptimizerConfigField
from msserviceprofiler.modelevalstate.optimizer.utils import (backup, kill_process, kill_children,
                                                              remove_file, close_file_fp)
from msserviceprofiler.msguard.security import open_s


class Simulator:
    from msserviceprofiler.modelevalstate.config.custom_command import MindieCommand

    def __init__(self, mindie_config: MindieConfig, bak_path: Optional[Path] = None):
        self.mindie_config = mindie_config
        logger.info(f"config path {self.mindie_config.config_path}", )
        if not self.mindie_config.config_path.exists():
            raise FileNotFoundError(self.mindie_config.config_path)
        with open_s(self.mindie_config.config_path, "r") as f:
            data = json.load(f)
        self.default_config = data
        logger.info(f"config bak path {self.mindie_config.config_bak_path}", )
        if self.mindie_config.config_bak_path.exists():
            self.mindie_config.config_bak_path.unlink()
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(self.mindie_config.config_bak_path, flags, modes), 'w') as fout:
            json.dump(self.default_config, fout, indent=4)
        self.mindie_log = None
        self.mindie_log_offset = 0
        self.bak_path = bak_path
        self.mindie_log_fp = None
        self.process = None
        self.command = self.MindieCommand(self.mindie_config.command).command

    @staticmethod
    def is_int(x):
        try:
            int(x)
            return True
        except ValueError:
            return False
        
    @staticmethod
    def prepare_before_start_server():
        pkill_path = shutil.which("pkill")
        if pkill_path is not None:
            subprocess.run([pkill_path, "-9", "mindie"])
        else:
            logger.error("pkill not found in path")

    @staticmethod
    def set_config_for_dict(origin_config, cur_key, next_key, next_level, value):
        if cur_key in origin_config:
            Simulator.set_config(origin_config[cur_key], next_level, value)
        elif Simulator.is_int(cur_key):
            raise KeyError(f"data: {origin_config}, key: {cur_key}")
        elif Simulator.is_int(next_key):
            origin_config[cur_key] = []
            Simulator.set_config(origin_config[cur_key], next_level, value)
        else:
            origin_config[cur_key] = {}
            Simulator.set_config(origin_config[cur_key], next_level, value)

    @staticmethod
    def set_config_for_list(origin_config, cur_key, next_key, next_level, value):
        if len(origin_config) > int(cur_key):
            Simulator.set_config(origin_config[int(cur_key)], next_level, value)
        elif len(origin_config) == int(cur_key) and Simulator.is_int(next_key):
            origin_config.append([])
            Simulator.set_config(origin_config[int(cur_key)], next_level, value)
        elif len(origin_config) == int(cur_key) and not Simulator.is_int(next_key):
            origin_config.append({})
            Simulator.set_config(origin_config[int(cur_key)], next_level, value)
        else:
            raise IndexError(f"data: {origin_config}, index: {cur_key}")

    @staticmethod
    def set_config(origin_config, key: str, value: Any):
        next_level = None
        try:
            if "." in key:
                _f_index = key.index(".")
                _cur_key, next_level = key[:_f_index], key[_f_index + 1:]
            else:
                _cur_key = key
            if next_level is None:
                if isinstance(origin_config, dict):
                    origin_config[_cur_key] = value
                elif isinstance(origin_config, list):
                    if len(origin_config) > int(_cur_key):
                        origin_config[int(_cur_key)] = value
                    else:
                        origin_config.append(value)
                return
            if "." in next_level:
                _next_index = next_level.index(".")
                _next_key = next_level[:_next_index]
            elif next_level:
                _next_key = next_level
            else:
                _next_key = None
        except Exception as e:
                logger.error(f"Unexpected error occurred at {key}")
                raise e
        if isinstance(origin_config, dict):
            Simulator.set_config_for_dict(origin_config, _cur_key, _next_key, next_level, value)
        elif isinstance(origin_config, list):
            Simulator.set_config_for_list(origin_config, _cur_key, _next_key, next_level, value)
        else:
            raise ValueError(f"Not Support type {type(origin_config)}")

    def backup(self, del_log=True):
        backup(self.mindie_config.config_path, self.bak_path, self.__class__.__name__)
        if not del_log and self.mindie_log:
            backup(self.mindie_log, self.bak_path, self.__class__.__name__)

    def update_config(self, params: Tuple[OptimizerConfigField]):
        # 将params值更新到新的config中
        new_config = deepcopy(self.default_config)
        for p in params:
            if not p.config_position.startswith("BackendConfig"):
                continue
            Simulator.set_config(new_config, p.config_position, p.value)

        # 将新的config写入到config文件中
        logger.debug(f"new config {new_config}")
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        if self.mindie_config.config_path.exists():
            self.mindie_config.config_path.unlink()
        with os.fdopen(os.open(self.mindie_config.config_path, flags, modes), "w") as fout:
            json.dump(new_config, fout, indent=4)

    def check_env(self):
        logger.info("check env")
        _residual_process = []
        _all_process_name = self.mindie_config.process_name.split(",")
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

    def check_success(self, print_log=False):
        with open_s(self.mindie_log, "r") as f:
            try:
                f.seek(self.mindie_log_offset)
                output = f.read()
                self.mindie_log_offset = f.tell()
            except Exception as e:
                logger.info(f"Failed in read mindie log. error: {e}")
        if output:
            if print_log:
                logger.info(f"simulate out: \n{output}")
            if "Daemon start success!" in output:
                return True
        if self.process.poll() is not None:
            raise subprocess.SubprocessError(
                f"Failed in run mindie. return code: {self.process.returncode}. "
                f"Please check the service log or console output.")
        return False

    def start_server(self, run_params: Tuple[OptimizerConfigField]):
        self.prepare_before_start_server()
        self.mindie_log_fp, self.mindie_log = tempfile.mkstemp(prefix="modelevalstate_mindie")
        self.mindie_log_offset = 0
        if self.mindie_config.work_path:
            cwd = self.mindie_config.work_path
        else:
            cwd = os.getcwd()
        for k in run_params:
            if k.config_position == "env":
                os.environ[k.name] = str(k.value)
                _var_name = f"${k.name}"
                if _var_name in self.command:
                    _i = self.command.index(_var_name)
                    self.command[_i] = str(k.value)
        if MODEL_EVAL_STATE_CONFIG_PATH not in os.environ:
            os.environ[MODEL_EVAL_STATE_CONFIG_PATH] = str(modelevalstate_config_path)
        if CUSTOM_OUTPUT not in os.environ:
            os.environ[CUSTOM_OUTPUT] = str(custom_output)
        logger.debug(f"env {os.environ}")
        logger.info(f"run cmd: {self.command}, log path: {self.mindie_log}")
        self.process = subprocess.Popen(self.command, stdout=self.mindie_log_fp, stderr=subprocess.STDOUT, 
                                        env=os.environ, text=True, cwd=cwd)

    def run(self, run_params: Tuple[OptimizerConfigField]):
        logger.info(f'start run in simulator. run params: {run_params}')
        # 根据params 修改配置文件
        self.update_config(run_params)
        # 启动mindie仿真
        try:
            self.check_env()
        except Exception as e:
            logger.error(f"Failed to check env. {e}")
        self.start_server(run_params)

    def stop(self, del_log=True):
        logger.info("Stop simulator process")
        if self.bak_path:
            self.backup()
        close_file_fp(self.mindie_log_fp)
        if del_log:
            remove_file(self.mindie_log)
        self.mindie_log_offset = 0
        if not self.process:
            return
        _process_state = self.process.poll()
        if _process_state is not None:
            logger.info(f"mindie already. exit_code: {_process_state}")
            return
        try:
            children = psutil.Process(self.process.pid).children(recursive=True)
            self.process.kill()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.send_signal(9)
            if self.process.poll() is not None:
                logger.info(f"The {self.process.pid} process has been shut down.")
            else:
                logger.error(f"The {self.process.pid} process shutdown failed.")
            kill_children(children)
            kill_process(self.mindie_config.process_name)
            remove_file(self.mindie_config.config_path)
            flags = os.O_WRONLY | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(self.mindie_config.config_path, flags, modes), "w") as fout:
                json.dump(self.default_config, fout, indent=4)
        except Exception as e:
            logger.error(f"Failed to stop simulator process. {e}")


class VllmSimulator(Simulator):
    from msserviceprofiler.modelevalstate.config.custom_command import VllmCommand

    def __init__(self, vllm_config: MindieConfig, bak_path: Optional[Path] = None):
        try:
            super().__init__(vllm_config, bak_path)
        except Exception as e:
            logger.info('VllmSimulator init failed')
        self.vllm_config = vllm_config
        self.mindie_log = None
        self.mindie_log_offset = 0
        self.bak_path = bak_path
        self.mindie_log_fp = None
        self.process = None
        self.command = self.VllmCommand(self.vllm_config.vllm_command).command

    @staticmethod
    def prepare_before_start_server():
        pkill_path = shutil.which("pkill")
        if pkill_path is not None:
            subprocess.run([pkill_path, "-15", "vllm"])
        else:
            logger.error("pkill not found in path")

    def run(self, run_params: Tuple[OptimizerConfigField]):
        logger.info(f'start run in simulator. run params: {run_params}')
        # 启动mindie仿真
        try:
            self.check_env()
        except Exception as e:
            logger.error(f"Failed to check env. {e}")
        self.start_server(run_params)

    def check_success(self, print_log=False):
        with open_s(self.mindie_log, "r") as f:
            try:
                f.seek(self.mindie_log_offset)
                output = f.read()
                self.mindie_log_offset = f.tell()
            except Exception as e:
                logger.info(f"Failed in read vllm log. error: {e}")
        if output:
            if print_log:
                logger.info(f"simulate out: \n{output}")
            if "Application startup complete." in output:
                return True
        if self.process.poll() is not None:
            raise subprocess.SubprocessError(
                f"Failed in run vllm. return code: {self.process.returncode}. "
                f"Please check the service log or console output.")
        return False

    def stop(self, del_log=True):
        logger.info("Stop simulator process")
        if self.bak_path:
            self.backup()
        close_file_fp(self.mindie_log_fp)
        if del_log:
            remove_file(self.mindie_log)
        self.mindie_log_offset = 0
        if not self.process:
            return
        _process_state = self.process.poll()
        if _process_state is not None:
            logger.info(f"vllm already. exit_code: {_process_state}")
            return
        try:
            children = psutil.Process(self.process.pid).children(recursive=True)
            self.process.kill()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.send_signal(9)
            if self.process.poll() is not None:
                logger.info(f"The {self.process.pid} process has been shut down.")
            else:
                logger.error(f"The {self.process.pid} process shutdown failed.")
            kill_children(children)
            kill_process(self.mindie_config.process_name)
        except Exception as e:
            logger.error(f"Failed to stop simulator process. {e}")