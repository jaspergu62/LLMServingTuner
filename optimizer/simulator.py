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
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Tuple, Optional, Union
import shutil
import tempfile
import time
from loguru import logger
import yaml
from msserviceprofiler.modelevalstate.config.config import MindieConfig, VllmConfig, OptimizerConfigField, KubectlConfig
from msserviceprofiler.modelevalstate.config.base_config import simulate_flag, SIMULATE
from msserviceprofiler.modelevalstate.config.custom_command import MindieCommand, VllmCommand
from msserviceprofiler.modelevalstate.optimizer.custom_process import CustomProcess
from msserviceprofiler.modelevalstate.optimizer.utils import backup, remove_file, close_file_fp
from msserviceprofiler.msguard.security import open_s
from msserviceprofiler.msguard import Rule


@dataclass
class ConfigContextdict:
    origin_config: dict
    cur_key: str
    next_key: str
    next_level: str
    value: Any
    current_depth: int


@dataclass
class ConfigContextlist:
    origin_config: list
    cur_key: str
    next_key: str
    next_level: str
    value: Any
    current_depth: int


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
    def set_config_for_dict(context: ConfigContextdict):
        if context.cur_key in context.origin_config:
            Simulator.set_config(context.origin_config[context.cur_key], context.next_level, context.value, 
                                 context.current_depth)
        elif Simulator.is_int(context.cur_key):
            raise KeyError(f"data: {context.origin_config}, key: {context.cur_key}")
        elif Simulator.is_int(context.next_key):
            context.origin_config[context.cur_key] = []
            Simulator.set_config(context.origin_config[context.cur_key], context.next_level, context.value, 
                                 context.current_depth)
        else:
            context.origin_config[context.cur_key] = {}
            Simulator.set_config(context.origin_config[context.cur_key], context.next_level, context.value, 
                                 context.current_depth)

    @staticmethod
    def set_config_for_list(context: ConfigContextlist):
        if len(context.origin_config) > int(context.cur_key):
            Simulator.set_config(context.origin_config[int(context.cur_key)], context.next_level, context.value, 
                                 context.current_depth)
        elif len(context.origin_config) == int(context.cur_key) and Simulator.is_int(context.next_key):
            context.origin_config.append([])
            Simulator.set_config(context.origin_config[int(context.cur_key)], context.next_level, context.value, 
                                 context.current_depth)
        elif len(context.origin_config) == int(context.cur_key) and not Simulator.is_int(context.next_key):
            context.origin_config.append({})
            Simulator.set_config(context.origin_config[int(context.cur_key)], context.next_level, context.value, 
                                 context.current_depth)
        else:
            raise IndexError(f"data: {context.origin_config}, index: {context.cur_key}")

    @staticmethod
    def set_config(origin_config, key: str, value: Any, current_depth=0):
        if current_depth > 10:
            raise RecursionError("Exceeded maximum recursion depth")
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
            context = ConfigContextdict(
                origin_config=origin_config,
                cur_key=_cur_key,
                next_key=_next_key,
                next_level=next_level,
                value=value,
                current_depth=current_depth + 1
            )
            Simulator.set_config_for_dict(context)
        elif isinstance(origin_config, list):
            context = ConfigContextlist(
                origin_config=origin_config,
                cur_key=_cur_key,
                next_key=_next_key,
                next_level=next_level,
                value=value,
                current_depth=current_depth + 1
            )
            Simulator.set_config_for_list(context)
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
        subprocess.run(["pkill", "-9", "mindie"], env=self.env, stdout=self.run_log_fp,
                       stderr=subprocess.STDOUT, text=True, cwd=self.work_path)
        subprocess.run(["npu-smi", "info"], env=self.env, stdout=self.run_log_fp,
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
            logger.debug(output)
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


class DisaggregationSimulator(CustomProcess):
    from msserviceprofiler.modelevalstate.config.custom_command import KubectlCommand

    def __init__(self, mindie_config: KubectlConfig, bak_path: Optional[Path] = None):
        super().__init__(bak_path=bak_path)
        self.mindie_config = mindie_config
        logger.debug(f"config path {self.mindie_config.config_single_path!r}", )
        if not self.mindie_config.config_single_path.exists():
            raise FileNotFoundError(self.mindie_config.config_single_path)
        with open_s(self.mindie_config.config_single_path, "r") as f:
            data = json.load(f)
        with open_s(self.mindie_config.config_single_pd_path, "r") as f:
            pd_data = json.load(f)
        self.default_pd_config = pd_data
        self.default_config = data
        logger.debug(f"config bak path {self.mindie_config.config_single_bak_path!r}", )
        if self.mindie_config.config_single_bak_path.exists():
            self.mindie_config.config_single_bak_path.unlink()
        with open_s(self.mindie_config.config_single_bak_path, "w") as fout:
            json.dump(self.default_config, fout, indent=4)
        self.run_log = None
        self.mindie_log_offset = 0
        self.bak_path = bak_path
        self.mindie_log_fp = None
        self.process = None
        self.command = self.KubectlCommand(self.mindie_config.command).command
        self.log_command = self.KubectlCommand(self.mindie_config.command).log_command
        self.monitor_command = self.KubectlCommand(self.mindie_config.command).monitor_command

    @staticmethod
    def is_int(x):
        try:
            int(x)
            return True
        except ValueError:
            return False

    @staticmethod
    def set_config_for_dict(context: ConfigContextdict):
        if context.cur_key in context.origin_config:
            DisaggregationSimulator.set_config(context.origin_config[context.cur_key], context.next_level, 
                                               context.value, context.current_depth)
        elif DisaggregationSimulator.is_int(context.cur_key):
            raise KeyError(f"data: {context.origin_config}, key: {context.cur_key}")
        elif DisaggregationSimulator.is_int(context.next_key):
            context.origin_config[context.cur_key] = []
            DisaggregationSimulator.set_config(context.origin_config[context.cur_key], context.next_level, 
                                               context.value, context.current_depth)
        else:
            context.origin_config[context.cur_key] = {}
            DisaggregationSimulator.set_config(context.origin_config[context.cur_key], context.next_level, 
                                               context.value, context.current_depth)

    @staticmethod
    def set_config_for_list(context: ConfigContextlist):
        if len(context.origin_config) > int(context.cur_key):
            DisaggregationSimulator.set_config(context.origin_config[int(context.cur_key)], context.next_level, 
                                               context.value, context.current_depth)
        elif len(context.origin_config) == int(context.cur_key) and DisaggregationSimulator.is_int(context.next_key):
            context.origin_config.append([])
            DisaggregationSimulator.set_config(context.origin_config[int(context.cur_key)], context.next_level, 
                                               context.value, context.current_depth)
        elif len(context.origin_config) == int(context.cur_key) and not \
            DisaggregationSimulator.is_int(context.next_key):
            context.origin_config.append({})
            DisaggregationSimulator.set_config(context.origin_config[int(context.cur_key)], context.next_level, 
                                               context.value, context.current_depth)
        else:
            raise IndexError(f"data: {context.origin_config}, index: {context.cur_key}")

    @staticmethod
    def set_config(origin_config, key: str, value: Any, current_depth=0):
        if current_depth > 10:
            raise RecursionError("Exceeded maximum recursion depth")
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
            context = ConfigContextdict(
                origin_config=origin_config,
                cur_key=_cur_key,
                next_key=_next_key,
                next_level=next_level,
                value=value,
                current_depth=current_depth + 1
            )
            DisaggregationSimulator.set_config_for_dict(context)
        elif isinstance(origin_config, list):
            context = ConfigContextlist(
                origin_config=origin_config,
                cur_key=_cur_key,
                next_key=_next_key,
                next_level=next_level,
                value=value,
                current_depth=current_depth + 1
            )
            DisaggregationSimulator.set_config_for_list(context)
        else:
            raise ValueError(f"Not Support type {type(origin_config)}")
    
    def prepare_before_start_server(self):
        bash_path = shutil.which("bash")
        if bash_path is not None:
            if not Rule.input_file_read.is_satisfied_by(self.mindie_config.delete_path):
                raise PermissionError("the file of delete_path is not safe, please check")
            subprocess.run([bash_path, self.mindie_config.delete_path, "mindie", "."], 
                           cwd=self.mindie_config.kubectl_default_path)
            while True:
                singal = True
                proc = subprocess.run(self.log_command, stdout=subprocess.PIPE, text=True, 
                                      cwd=self.mindie_config.kubectl_default_path)
                lines = proc.stdout.splitlines()
                for line in lines:
                    if line.startswith('mindie'):
                        singal = False
                if singal is True:
                    break
                time.sleep(1)
        else:
            logger.error("bash not found in path")

    def backup(self, del_log=True):
        backup(self.mindie_config.config_single_path, self.bak_path, self.__class__.__name__)
        if not del_log and self.run_log:
            backup(self.run_log, self.bak_path, self.__class__.__name__)

    def update_config(self, params: Tuple[OptimizerConfigField]):
        # 将params值更新到新的config中
        new_config = deepcopy(self.default_config)
        pd_config = deepcopy(self.default_pd_config)
        for p in params:
            if p.config_position.startswith("default"):
                DisaggregationSimulator.set_config(pd_config, p.config_position, p.value)
            if not p.config_position.startswith("BackendConfig"):
                continue
            DisaggregationSimulator.set_config(new_config, p.config_position, p.value)

        # 将新的config写入到config文件中
        logger.debug(f"new config {new_config}")
        if self.mindie_config.config_single_path.exists():
            self.mindie_config.config_single_path.unlink()
        with open_s(self.mindie_config.config_single_path, "w") as fout:
            json.dump(new_config, fout, indent=4)
        if self.mindie_config.config_single_pd_path.exists():
            self.mindie_config.config_single_pd_path.unlink()
        with open_s(self.mindie_config.config_single_pd_path, "w") as fout:
            json.dump(pd_config, fout, indent=4)

    def test_curl(self):
        import requests
        curl_port = None
        yaml_dir = self.mindie_config.kubectl_single_path.parent
        yaml_path = os.path.join(yaml_dir, "deployment/mindie_service_single_container.yaml")
        with open_s(yaml_path, 'r') as file:
            all_documents = yaml.safe_load_all(file)
            for doc in all_documents:
                # 检查文档中是否存在 spec.ports 部分
                if 'spec' in doc and 'ports' in doc['spec']:
                    ports = doc['spec']['ports']
                    for port in ports:
                        if 'nodePort' in port:
                            curl_port = port['nodePort']
        if curl_port:
            url = f"http://127.0.0.1:{curl_port}"
        else:
            raise ("cannot find port from mindie_service_single_container.yaml, please check")

        # 定义请求体的 JSON 数据
        payload = {
            "inputs": "Please introduce yourself.",
            "parameters": {
                "max_new_tokens": 20,
                "temperature": 0.3,
                "top_p": 0.3,
                "top_k": 5,
                "do_sample": True,
                "repetition_penalty": 1.05,
                "seed": 128
            }
        }

        # 定义请求头（如果需要）
        headers = {
            "Content-Type": "application/json"
        }

        try:
            # 发送 POST 请求
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            # 检查响应状态码
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.exceptions.RequestException as e:
            return False

    def check_success(self, print_log=False):
        with open_s(self.run_log, "r") as f:
            try:
                f.seek(self.mindie_log_offset)
                output = f.read()
                self.mindie_log_offset = f.tell()
            except Exception as e:
                logger.warning(f"Failed in read mindie log. error: {e}")
        if output:
            if print_log:
                logger.debug(f"simulate out: \n{output}")
            if "MindIE-MS coordinator is ready!!!" in output:
                while True:
                    if self.test_curl() is True:
                        return True
                    time.sleep(1)
        return False

    def start_server(self, run_params: Tuple[OptimizerConfigField]):
        self.prepare_before_start_server()
        self.mindie_log_fp, self.run_log = tempfile.mkstemp(prefix="modelevalstate_mindie")
        self.mindie_log_offset = 0
        if self.mindie_config.work_path:
            cwd = self.mindie_config.work_path
        else:
            cwd = os.getcwd()
        logger.info(f"start running the command: {self.command}")
        self.process = subprocess.run(self.command, env=self.env, text=True, 
                                      cwd=self.mindie_config.kubectl_default_path)
        logger.info(f"self.log_command: {self.log_command}")
        while True:
            proc = subprocess.run(self.log_command, stdout=subprocess.PIPE, text=True, 
                                cwd=self.mindie_config.kubectl_default_path)
            
            lines = proc.stdout.splitlines()
            for line in lines:
                if line.startswith('mindie'):
                    mindie_name = line.split()
                    break
            if mindie_name[3] == 'Running':
                break
            time.sleep(1)
        kubectl_monitor_command = self.KubectlCommand(self.mindie_config.command).monitor_command
        kubectl_monitor_command.append(mindie_name[1])
        logger.debug(f"mindie_name: {mindie_name[1]}")
        self.log_process = subprocess.Popen(kubectl_monitor_command, stdout=self.mindie_log_fp, 
                                            stderr=subprocess.STDOUT, env=self.env, text=True, 
                                            cwd=self.mindie_config.kubectl_default_path)
        logger.info(f"Start running the command: {' '.join(kubectl_monitor_command)}, log file: {self.run_log}")

    def run(self, run_params: Tuple[OptimizerConfigField]):
        logger.info(f'Start running in simulator. params are: {run_params}')
        # 根据params 修改配置文件
        self.update_config(run_params)
        # 启动mindie仿真
        self.start_server(run_params)

    def stop(self, del_log=True):
        logger.debug("Stop simulator process")
        if self.bak_path:
            self.backup()
        close_file_fp(self.mindie_log_fp)
        if del_log:
            remove_file(self.run_log)
        self.mindie_log_offset = 0
        try:
            bash_path = shutil.which("bash")
            if not Rule.input_file_read.is_satisfied_by(self.mindie_config.delete_path):
                raise PermissionError("the file of delete_path is not safe, please check")
            if bash_path is not None:
                subprocess.run([bash_path, self.mindie_config.delete_path, "mindie", "./"])
            else:
                logger.error("bash not found in path")
        except Exception as e:
            logger.error(f"Failed to stop simulator process. {e}")


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
        subprocess.run(["pkill", "-15", "vllm"], env=self.env, stdout=self.run_log_fp,
                       stderr=subprocess.STDOUT, text=True, cwd=self.work_path)
        subprocess.run(["npu-smi", "info"], env=self.env, stdout=self.run_log_fp,
                       stderr=subprocess.STDOUT, text=True, cwd=self.work_path)

    def check_success(self, print_log=False):
        output = self.get_log()
        if self.print_log:
            logger.debug(output)
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
def enable_simulate_old(simulate):
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