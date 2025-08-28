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
import subprocess
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Tuple, Optional, Union
from loguru import logger

from msserviceprofiler.modelevalstate.config.config import MindieConfig, VllmConfig, OptimizerConfigField
from msserviceprofiler.modelevalstate.config.base_config import simulate_flag, SIMULATE
from msserviceprofiler.modelevalstate.config.custom_command import MindieCommand, VllmCommand
from msserviceprofiler.modelevalstate.optimizer.custom_process import CustomProcess
from msserviceprofiler.modelevalstate.optimizer.utils import backup, remove_file
from msserviceprofiler.msguard.security import open_s


class Simulator(CustomProcess):
    def __init__(self, mindie_config: MindieConfig, bak_path: Optional[Path] = None,
                 print_log: bool = False):
        super().__init__(bak_path=bak_path, print_log=print_log, process_name=mindie_config.process_name)
        self.mindie_config = mindie_config
        logger.debug(f"config path {self.mindie_config.config_path!r}", )
        if not self.mindie_config.config_path.exists():
            raise FileNotFoundError(self.mindie_config.config_path)
        with open_s(self.mindie_config.config_path, "r") as f:
            data = json.load(f)
        self.default_config = data
        logger.debug(f"config bak path {self.mindie_config.config_bak_path!r}", )
        if self.mindie_config.config_bak_path.exists():
            self.mindie_config.config_bak_path.unlink()
        with open_s(self.mindie_config.config_bak_path, 'w') as fout:
            json.dump(self.default_config, fout, indent=4)
        self.command = MindieCommand(self.mindie_config.command).command

    @staticmethod
    def is_int(x):
        try:
            int(x)
            return True
        except ValueError:
            return False
        
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

    def update_command(self):
        self.command = MindieCommand(self.mindie_config.command).command
    
    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField]] = None):
        # 根据params 修改配置文件
        # 启动mindie仿真
        self.update_command()
        self.update_config(run_params)
        super().before_run(run_params)
        subprocess.run(["pkill", "-9", "mindie"], env=os.environ, stdout=self.run_log_fp,
                       stderr=subprocess.STDOUT, text=True, cwd=self.work_path)
        subprocess.run(["npu-smi", "info"], env=os.environ, stdout=self.run_log_fp,
                       stderr=subprocess.STDOUT, text=True, cwd=self.work_path)

    def backup(self):
        super().backup()
        backup(self.mindie_config.config_path, self.bak_path, self.__class__.__name__)

    def update_config(self, params: Optional[Tuple[OptimizerConfigField]] = None):
        # 将params值更新到新的config中
        if not params:
            return
        new_config = deepcopy(self.default_config)
        for p in params:
            if not p.config_position.startswith("BackendConfig"):
                continue
            Simulator.set_config(new_config, p.config_position, p.value)

        # 将新的config写入到config文件中
        logger.debug(f"new config {new_config}")
        if self.mindie_config.config_path.exists():
            self.mindie_config.config_path.unlink()
        with open_s(self.mindie_config.config_path, "w") as fout:
            json.dump(new_config, fout, indent=4, ensure_ascii=False)

    def check_success(self, print_log=False):
        output = self.get_log()
        if self.print_log:
            logger.info(output)
        if output and "Daemon start success!" in output:
            return True
        if self.process.poll() is None:
            return False
        elif self.process.poll() == 0:
            return True
        else:
            raise subprocess.SubprocessError(
                f"Failed in run {self.command}. return code: {self.process.returncode}. log: {self.run_log}")

    def stop(self, del_log: bool = True):
        # 恢复默认的mindie 配置
        remove_file(self.mindie_config.config_path)
        with open_s(self.mindie_config.config_path, "w") as fout:
            json.dump(self.default_config, fout, indent=4, ensure_ascii=False)
        super().stop(del_log)


class VllmSimulator(CustomProcess):
    def __init__(self, vllm_config: VllmConfig, bak_path: Optional[Path] = None,
                 print_log: bool = False):
        super().__init__(bak_path=bak_path, print_log=print_log, process_name=vllm_config.process_name)
        self.vllm_config = vllm_config
        self.command = VllmCommand(vllm_config.command).command

    def update_command(self):
        self.command = VllmCommand(self.vllm_config.command).command

    def backup(self):
        super().backup()
        backup(self.vllm_config.output, self.bak_path, self.__class__.__name__)

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField]] = None):
        self.update_command()
        super().before_run(run_params)
        subprocess.run(["pkill", "-15", "vllm"], env=os.environ, stdout=self.run_log_fp,
                       stderr=subprocess.STDOUT, text=True, cwd=self.work_path)
        subprocess.run(["npu-smi", "info"], env=os.environ, stdout=self.run_log_fp,
                       stderr=subprocess.STDOUT, text=True, cwd=self.work_path)

    def check_success(self, print_log=False):
        output = self.get_log()
        if self.print_log:
            logger.info(output)
        if output and "Application startup complete." in output:
            return True
        if self.process.poll() is None:
            return False
        elif self.process.poll() == 0:
            return True
        else:
            raise subprocess.SubprocessError(
                f"Failed in run {self.command}. return code: {self.process.returncode}. log: {self.run_log}")


@contextmanager
def enable_simulate(simulate):
    if simulate_flag and isinstance(simulate, Simulator):
        origin_data = simulate.default_config
        data = deepcopy(origin_data)
        simulate.default_config = data
        model_config = data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]
        if "plugin_params" in model_config:
            _plugin_params = json.loads(model_config["plugin_params"])
            if SIMULATE not in _plugin_params["plugin_type"]:
                _plugin_params["plugin_type"] += "," + SIMULATE
                model_config["plugin_params"] = json.dumps(_plugin_params)
        else:
            model_config["plugin_params"] = json.dumps({"plugin_type": SIMULATE})
        with open_s(simulate.mindie_config.config_path, 'w') as f:
            json.dump(data, f, indent=4)
        yield simulate_flag
        if simulate.mindie_config.config_path.exists():
            simulate.mindie_config.config_path.unlink()
        with open_s(simulate.mindie_config.config_path, 'w') as f:
            json.dump(origin_data, f, indent=4)
    else:
        yield simulate_flag
    return