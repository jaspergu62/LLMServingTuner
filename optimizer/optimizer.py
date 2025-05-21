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
import argparse
import atexit
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
import xmlrpc.client
from copy import deepcopy
from math import exp, inf
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from xmlrpc.client import ServerProxy

import numpy as np
import pandas as pd
import psutil
from loguru import logger

from msserviceprofiler.modelevalstate.common import get_train_sub_path
from msserviceprofiler.modelevalstate.config.config import AnalyzeTool, BenchMarkConfig, MindieConfig, settings, \
    DeployPolicy, map_param_with_value, MODEL_EVAL_STATE_CONFIG_PATH, modelevalstate_config_path, \
    CUSTOM_OUTPUT, custom_output
from msserviceprofiler.modelevalstate.config.config import default_support_field, PsoOptions, \
    PerformanceIndex, OptimizerConfigField
from msserviceprofiler.modelevalstate.inference.constant import IS_SLEEP_FLAG
from msserviceprofiler.modelevalstate.optimizer.analyze_profiler import analyze as analyze_profiler
from msserviceprofiler.modelevalstate.optimizer.global_best_custom import CustomGlobalBestPSO
from msserviceprofiler.modelevalstate.optimizer.store import DataStorage

_analyze_mapping = {
    AnalyzeTool.profiler.value: analyze_profiler
}


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


def backup(target, bak, class_name=""):
    if not target:
        return
    if not bak:
        return
    if not isinstance(target, Path):
        target = Path(target)
    if not isinstance(target, Path):
        bak = Path(bak)
    if not target.exists():
        return
    if not bak.exists():
        return
    new_file = bak.joinpath(class_name).joinpath(target.name)
    if target.is_file():
        new_file.parent.mkdir(parents=True, exist_ok=True)
        if not new_file.exists():
            shutil.copy(target, new_file)
    else:
        if new_file.exists():
            for child in new_file.iterdir():
                backup(child, new_file, class_name)
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


@atexit.register
def clearing_residual_process():
    kill_process(MindieConfig().process_name)


class BenchMark:
    def __init__(self, benchmark_config: BenchMarkConfig, throughput_type: str = "common",
                 bak_path: Optional[Path] = None):
        self.benchmark_config = benchmark_config
        self.throughput_type = throughput_type
        self.bak_path = bak_path
        self.run_log = None
        self.run_log_offset = None
        self.run_log_fp = None
        self.process = None
        self.pattern = re.compile(r"\s*(\d+\.?\d*)\s*\%")

    def backup(self, del_log=True):
        backup(self.benchmark_config.output_path, self.bak_path, self.__class__.__name__)
        if not del_log:
            backup(self.run_log, self.bak_path, self.__class__.__name__)

    def get_performance_index(self):
        output_path = Path(self.benchmark_config.output_path)
        common_generate_speed = None
        first_token_time = None
        perf_generate_token_speed = None
        decode_time = None
        success_rate = None
        for file in output_path.iterdir():
            if "result_common" in file.name:
                try:
                    df = pd.read_csv(file)
                    if "OutputGenerateSpeed" in df.columns:
                        _generate_speed = df["OutputGenerateSpeed"][0]
                    else:
                        _generate_speed = df["GenerateSpeed"][0]
                    if isinstance(_generate_speed, str):
                        common_generate_speed = float(_generate_speed.split()[0])
                    elif isinstance(_generate_speed,
                                    (int, float, np.int64, np.int32, np.float64, np.float32, np.float16)):
                        common_generate_speed = _generate_speed
                    else:
                        raise TypeError(f"GenerateSpeed: {_generate_speed}, type: {type(_generate_speed)}")
                    req_returnd = df["Returned"][0]
                    if not req_returnd:
                        continue
                    _m_res = self.pattern.search(req_returnd)
                    if not _m_res:
                        continue
                    success_rate = float(_m_res.group(1)) / 100
                except (KeyError, AttributeError) as e:
                    logger.error(f"Failed in get GenerateSpeed. error: {e}")
                continue
            if "result_perf" in file.name:
                try:
                    df = pd.read_csv(file)
                    first_token_time = float(df["FirstTokenTime"][0].split()[0])
                    perf_generate_token_speed = float(df["GeneratedTokenSpeed"][0].split()[0])
                    decode_time = float(df["DecodeTime"][0].split()[0])
                except (AttributeError, KeyError) as e:
                    logger.error("Failed in get FirstTokenTime or GeneratedTokenSpeed. error: {}", e)
        if common_generate_speed is None and perf_generate_token_speed is None:
            raise ValueError("Not Found common_generate_speed or perf_generate_token_speed.")
        if first_token_time is None or decode_time is None:
            raise ValueError("Not Found first_token_time.")
        if self.throughput_type == "common":
            generate_speed = common_generate_speed
        else:
            generate_speed = perf_generate_token_speed
        time_to_first_token = first_token_time / 10 ** 3
        time_per_output_token = decode_time / 10 ** 3
        return PerformanceIndex(generate_speed=generate_speed,
                                time_to_first_token=time_to_first_token,
                                time_per_output_token=time_per_output_token,
                                success_rate=success_rate)

    def prepare(self):
        remove_file(Path(self.benchmark_config.output_path))
        remove_file(Path(self.benchmark_config.custom_collect_output_path))

    def check_success(self, print_log=False):
        if self.run_log:
            run_log_path = Path(self.run_log)
            if run_log_path.exists() and print_log:
                try:
                    with open(run_log_path, "r", encoding="utf-8") as f:
                        f.seek(self.run_log_offset)
                        output = f.read()
                        self.run_log_offset = f.tell()
                        logger.info(f"benchmark out: \n{output}")
                except (UnicodeError, OSError) as e:
                    logger.error(f"Failed read benchmark log. error {e}")
        try:
            if self.process.poll() is None:
                return False
            elif self.process.poll() == 0:
                return True
            else:
                raise subprocess.SubprocessError(
                    f"Failed in run benchmark. return code: {self.process.returncode}. ")
        except AttributeError as e:
            logger.error(f"Failed to check process status, error {e}")
            return False

    def run(self, run_params: Tuple[OptimizerConfigField]):
        # 启动测试
        logger.info("Start the benchmark test.")
        self.run_log_fp, self.run_log = tempfile.mkstemp(prefix="modelevalstate_benchmark")
        self.run_log_offset = 0
        if self.benchmark_config.work_path:
            cwd = self.benchmark_config.work_path
        else:
            cwd = os.getcwd()
        for k in run_params:
            if k.config_position == "env":
                try:
                    os.environ[k.name] = str(k.value)
                except KeyError as e:
                    logger.error(f"Failed to set environment variable. error {e}")
        if CUSTOM_OUTPUT not in os.environ:
            os.environ[CUSTOM_OUTPUT] = str(custom_output)
        run_cmd = shlex.split(self.benchmark_config.command)
        try:
            self.process = subprocess.Popen(run_cmd, env=os.environ, stdout=self.run_log_fp, stderr=subprocess.STDOUT,
                                            text=True, cwd=cwd)
        except OSError as e:
            logger.error(f"Failed to run benchmark. error {e}")
            raise e
        logger.info(f"command: {' '.join(run_cmd)}, log file: {self.run_log}")

    def stop(self, del_log=True):
        self.backup(del_log)
        close_file_fp(self.run_log_fp)
        try:
            if self.process and self.process.poll() is None:
                self.process.kill()
        except AttributeError as e:
            logger.error(f"Failed to kill process. error {e}")
        if del_log:
            remove_file(Path(self.run_log))


class ProfilerBenchmark(BenchMark):
    def __init__(self, benchmark_config: BenchMarkConfig, *args, analyze_tool: AnalyzeTool = AnalyzeTool.default,
                 **kwargs):
        super().__init__(benchmark_config, *args, **kwargs)
        self.analyze_tool = analyze_tool
        self.profiler_cmd = ["python", "-m", "ms_service_profiler.parse",
                             f"--input-path={self.benchmark_config.profile_input_path}",
                             f"--output-path={self.benchmark_config.profile_output_path}"]
        self.profiler_log = None
        self.profiler_log_fp = None
        self.profiler_log_offset = 0
        self.profiler_process = None

    def extra_performance_index(self, *args, **kwargs):
        logger.info("extra_performance_index")
        analyze_tool = _analyze_mapping.get(self.analyze_tool)
        if analyze_tool is None:
            raise ValueError(f"Analyze tool not found: {self.analyze_tool}")
        res = analyze_tool(*args, **kwargs)
        time_to_first_token = time_per_output_token = success_rate = None
        if isinstance(res, tuple):
            if len(res) == 1:
                generate_speed = res[0]
            elif len(res) == 2:
                generate_speed, time_to_first_token = res
            elif len(res) == 3:
                generate_speed, time_to_first_token, time_per_output_token = res
            elif len(res) == 4:
                generate_speed, time_to_first_token, time_per_output_token, success_rate = res
            else:
                raise ValueError(f"Not Support. res: {res}")
        else:
            generate_speed = res
        return PerformanceIndex(generate_speed=generate_speed, time_to_first_token=time_to_first_token,
                                time_per_output_token=time_per_output_token, success_rate=success_rate)


    def backup(self, del_log=True):
        super().backup(del_log)
        backup(self.benchmark_config.profile_input_path, self.bak_path, self.__class__.__name__)
        backup(self.benchmark_config.profile_output_path, self.bak_path, self.__class__.__name__)
        if not del_log and self.profiler_log:
            backup(self.profiler_log, self.bak_path, self.__class__.__name__)

    def prepare(self):
        super().prepare()
        remove_file(Path(self.benchmark_config.profile_input_path))
        remove_file(Path(self.benchmark_config.profile_output_path))

    def check_profiler(self, print_log=False):
        if print_log:
            try:
                with open(self.profiler_log, "r") as f:
                    f.seek(self.profiler_log_offset)
                    output = f.read()
                    self.profiler_log_offset = f.tell()
            except (UnicodeError, OSError) as e:
                logger.error(f"Failed read benchmark log. error {e}")
            if output:
                logger.info(f"benchmark out: \n{output}")
        if self.profiler_process.poll() is None:
            return False
        elif self.profiler_process.poll() == 0:
            return True
        else:
            raise subprocess.SubprocessError(
                f"Failed in run benchmark. return code: {self.process.returncode}. ")

    def start_profiler(self):
        self.profiler_log_fp, self.profiler_log = tempfile.mkstemp(prefix="modelevalstate_profiler")
        self.profiler_log_offset = 0
        if not os.path.exists(self.benchmark_config.work_path):
            raise FileNotFoundError(f"Work path not found: {self.benchmark_config.work_path}")
        logger.info(f"command: {' '.join(self.profiler_cmd)}, log file: {self.profiler_log}")
        self.profiler_process = subprocess.Popen(self.profiler_cmd, env=os.environ, stdout=self.profiler_log_fp,
                                                 stderr=subprocess.STDOUT,
                                                 text=True, cwd=self.benchmark_config.work_path)

    def get_performance_index(self):
        logger.info("get_performance_index")
        try:
            self.start_profiler()
            logger.info("wait profiler")
            while True:
                if self.check_profiler(print_log=True):
                    break
                time.sleep(1)
        except Exception as e:
            logger.error(f"Failed in start profiler. relation log: {self.profiler_log}")
            raise e
        collect_path = Path(self.benchmark_config.custom_collect_output_path)
        if self.analyze_tool == AnalyzeTool.profiler.value:
            res = self.extra_performance_index(self.benchmark_config.profile_output_path, collect_path)
            return res
        else:
            return super().get_performance_index()

    def stop(self, del_log=True):
        super().stop(del_log)
        close_file_fp(self.profiler_log_fp)
        if del_log:
            remove_file(Path(self.profiler_log))
        try:
            if self.profiler_process and self.profiler_process.poll() is None:
                self.profiler_process.kill()
        except AttributeError as e:
            logger.error(f"Failed to kill process. error {e}")


class Simulator:
    def __init__(self, mindie_config: MindieConfig, bak_path: Optional[Path] = None):
        self.mindie_config = mindie_config
        logger.info(f"config path {self.mindie_config.config_path}", )
        if not self.mindie_config.config_path.exists():
            raise FileNotFoundError(self.mindie_config.config_path)
        with open(self.mindie_config.config_path, "r") as f:
            data = json.load(f)
        self.default_config = data
        logger.info(f"config bak path {self.mindie_config.config_bak_path}", )
        if not self.mindie_config.config_bak_path.exists():
            flags = os.O_WRONLY | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(self.mindie_config.config_bak_path, flags, modes), 'w') as fout:
                json.dump(self.default_config, fout, indent=4)
        self.mindie_log = None
        self.mindie_log_offset = 0
        self.bak_path = bak_path
        self.mindie_log_fp = None
        self.process = None

    @staticmethod
    def get_new_config(origin_config, params: Tuple[OptimizerConfigField], upper_key: str = "") -> Any:
        if upper_key:
            _keys = [upper_key]
        else:
            _keys = []
        if isinstance(origin_config, dict):
            _dict_config = {}
            for k, v in origin_config.items():
                _root_key = ".".join([*_keys, k])
                new_value = Simulator.get_new_config(v, params, _root_key)
                _dict_config[k] = new_value
            return _dict_config

        elif isinstance(origin_config, list):
            _list_config = []
            for i, v in enumerate(origin_config):
                _root_key = ".".join([*_keys, str(i)])
                new_value = Simulator.get_new_config(v, params, f"{upper_key}.{i}")
                _list_config.append(new_value)
            return _list_config
        else:
            for _p in params:
                if upper_key == _p.config_position:
                    logger.info(f"Update Config key: {upper_key}")
                    return _p.value
            return origin_config

    @staticmethod
    def set_config(origin_config, key: str, value: Any):
        next_level = None
        if "." in key:
            _f_index = key.index(".")
            _cur_key, next_level = key[:_f_index], key[_f_index + 1:]
        else:
            _cur_key = key
        if next_level:
            if isinstance(origin_config, dict):
                Simulator.set_config(origin_config[_cur_key], next_level, value)
            elif isinstance(origin_config, list):
                Simulator.set_config(origin_config[int(_cur_key)], next_level, value)
            else:
                raise ValueError(f"Not Support type {type(origin_config)}")
        else:
            origin_config[_cur_key] = value

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
        with open(self.mindie_log, "r") as f:
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
        self.mindie_log_fp, self.mindie_log = tempfile.mkstemp(prefix="modelevalstate_mindie")
        self.mindie_log_offset = 0
        if self.mindie_config.work_path:
            cwd = self.mindie_config.work_path
        else:
            cwd = os.getcwd()
        for k in run_params:
            if k.config_position == "env":
                os.environ[k.name] = str(k.value)
        if MODEL_EVAL_STATE_CONFIG_PATH not in os.environ:
            os.environ[MODEL_EVAL_STATE_CONFIG_PATH] = str(modelevalstate_config_path)
        if CUSTOM_OUTPUT not in os.environ:
            os.environ[CUSTOM_OUTPUT] = str(custom_output)
        logger.debug(f"env {os.environ}")
        run_cmd = shlex.split(self.mindie_config.command)
        logger.info(f"run cmd: {run_cmd}, log path: {self.mindie_log}")
        self.process = subprocess.Popen(run_cmd, stdout=self.mindie_log_fp, stderr=subprocess.STDOUT, env=os.environ,
                                        text=True, cwd=cwd)

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
        logger.info("Stop mindie simulator process")
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
                json.dump(self.default_config, fout)
        except Exception as e:
            logger.error(f"Failed to stop mindie simulator process. {e}")


class Scheduler:
    def __init__(self, simulator: Simulator, benchmark: BenchMark, data_storage: DataStorage,
                 bak_path: Optional[Path] = None, retry_number: int = 3, wait_start_time=1800):
        self.simulator = simulator
        self.benchmark = benchmark
        self.data_storage = data_storage
        self.bak_path = bak_path
        self.retry_number = retry_number
        self.wait_time = wait_start_time
        self.current_back_path = None
        self.simulate_run_info = None

    def back_up(self):
        if self.bak_path:
            self.current_back_path = get_train_sub_path(self.bak_path)
            self.simulator.bak_path = self.current_back_path
            self.benchmark.bak_path = self.current_back_path

    def wait_simulate(self):
        logger.info("wait run mindie")
        for _ in range(self.wait_time):
            time.sleep(1)
            if self.simulator.check_success():
                logger.info(f"Successfully started the {self.simulator.process.pid} process.")
                return
        raise TimeoutError(self.wait_time)


    def run_simulate(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
        self.benchmark.prepare()
        self.simulator.run(tuple(self.simulate_run_info))
        self.wait_simulate()

    def monitoring_status(self):
        logger.info("monitor status")
        while True:
            if self.simulator.process.poll() is not None:
                self.simulator.stop(del_log=False)
                self.benchmark.stop(del_log=False)
                raise subprocess.SubprocessError(f"Failed in run mindie. "
                                                 f"return code: {self.simulator.process.returncode}.")
            if self.benchmark.check_success():
                return
            time.sleep(1)

    def run_target_server(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
        """
        1. 启动mindie仿真
        2. 启动benchmark 测试
        3. 检查mindie状态，检查benchmark状态
        """
        for _ in range(self.retry_number):
            try:
                self.run_simulate(params, params_field)
            except Exception as e:
                logger.error(f"Failed in Mindie Running. error: {e}， mindie log {self.simulator.mindie_log}")
                logger.exception("What?!")
                self.stop_target_server(del_log=False)
                continue
            time.sleep(1)
            try:
                self.benchmark.run(tuple(self.simulate_run_info))
            except Exception as e:
                logger.error(f"Failed in Benchmark Running. error: {e}, benchmark log {self.benchmark.run_log}")
                logger.exception("What?!")
                self.stop_target_server(del_log=False)
                continue
            time.sleep(1)
            try:
                self.monitoring_status()
            except Exception as e:
                self.stop_target_server(del_log=False)
                logger.error(f"Failed in monitoring status. error: {e}, mindie log {self.simulator.mindie_log}, "
                             f"benchmark log {self.benchmark.run_log}")
                logger.exception("What?!")
                continue
            return
        raise ValueError(
            f"Failed in run_target_server")

    def stop_target_server(self, del_log=True):
        self.simulator.stop(del_log)
        self.benchmark.stop(del_log)

    def run(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]) -> PerformanceIndex:
        """
        1. 启动mindie仿真
        2. 启动benchmark 测试
        3. 获取benchmark测试结果
        4. 关闭mindie仿真
        5. 返回benchmark测试结果
        params: 是一维数组，其值对应mindie 的相关配置。
        """
        logger.info("Start run in scheduler.")
        self.back_up()
        self.simulate_run_info = map_param_with_value(params, params_field)
        error_info = None
        del_log = True
        performance_index = PerformanceIndex()
        try:
            self.run_target_server(params, params_field)
            time.sleep(1)
            performance_index = self.benchmark.get_performance_index()
        except Exception as e:
            logger.error(f"Failed running. bak path: {self.simulator.bak_path}")
            error_info = e
            del_log = False
        self.data_storage.save(performance_index, tuple(self.simulate_run_info), self.benchmark.benchmark_config,
                               error=error_info, bakcup=self.current_back_path)
        self.stop_target_server(del_log)
        if error_info:
            raise error_info
        return performance_index


class ScheduleWithMultiMachine(Scheduler):
    def __init__(self, rpc_clients: List[ServerProxy], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rpc_clients = rpc_clients

    def back_up(self):
        if self.bak_path:
            _cur_bak_path = get_train_sub_path(self.bak_path)
            self.simulator.bak_path = _cur_bak_path
            self.benchmark.bak_path = _cur_bak_path
            for rpc in self.rpc_clients:
                if rpc.simulator:
                    rpc.simulator.bak_path = _cur_bak_path

    def monitoring_status(self):
        logger.info("Start monitoring")
        while True:
            all_poll = [self.simulator.process.poll()]
            for rpc in self.rpc_clients:
                all_poll.append(rpc.process_poll())
            if any([_i is not None for _i in all_poll]):
                self.stop_target_server(del_log=False)
                raise subprocess.SubprocessError(
                    f"Failed in run mindie. all status: {all_poll}, machine info: master, {self.rpc_clients}.")
            if self.benchmark.check_success():
                return
            time.sleep(1)

    def run_simulate(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
        self.benchmark.prepare()
        _simulate_run_info = map_param_with_value(params, params_field)
        [rpc.run_simulator(params.tolist()) for rpc in self.rpc_clients]
        self.simulator.run(tuple(self.simulate_run_info))
        self.wait_simulate()
        [rpc.check_success() for rpc in self.rpc_clients]

    def stop_target_server(self, del_log=True):
        super(ScheduleWithMultiMachine, self).stop_target_server(del_log)
        [rpc.stop_simulator(del_log) for rpc in self.rpc_clients]


class PSOOptimizer:
    def __init__(self, scheduler: Scheduler, n_particles: int = 10, iters=100, pso_options: PsoOptions = None,
                 target_field: Optional[Tuple] = None, prefill_lam: float = 0.5, decode_lam: float = 0.5,
                 success_rate_lam: float = 0.5, prefill_constrain: float = 0.05, decode_constrain: float = 0.05,
                 success_rate_constrain: float = 1, load_history_data: Optional[List] = None,
                 load_breakpoint: bool = False, pso_init_kwargs: Optional[Dict] = None):
        self.scheduler = scheduler
        self.n_particles = n_particles
        self.iters = iters
        self.target_field = target_field if target_field else default_support_field
        if not pso_options:
            self.pso_options = PsoOptions()
        else:
            self.pso_options = pso_options
        self.prefill_lam = prefill_lam  # 优化算法中惩罚系数
        self.decode_lam = decode_lam
        self.success_rate_lam = success_rate_lam
        self.prefill_constrain = prefill_constrain
        self.decode_constrain = decode_constrain
        self.success_rate_constrain = success_rate_constrain
        self.load_history_data = load_history_data
        self.load_breakpoint = load_breakpoint
        self.pso_init_kwargs = pso_init_kwargs
        self.init_pos = None
        self.history_cost, self.history_pos = None, None
        if self.load_history_data and self.load_breakpoint:
            self.history_pos, self.history_cost = self.computer_fitness()

    def computer_fitness(self) -> Tuple:
        all_position = []
        all_cost = []
        for case_data in self.load_history_data:
            _params = {}
            for k in PerformanceIndex.model_fields.keys():
                if k in case_data:
                    _params[k] = case_data[k]
            performance_index = PerformanceIndex(**_params)
            try:
                _fitness = self.minimum_algorithm(performance_index)
                logger.info(f"fitness {_fitness}")
                _pos = [case_data.get(_field.name) for _field in self.target_field]
                if not all(_pos):
                    continue
                all_cost.append(_fitness)
                all_position.append(_pos)
            except KeyError:
                continue
        if len(all_position) != len(all_cost):
            raise ValueError("Failed in computer_fitness.")
        return all_position, all_cost

    def minimum_algorithm(self, performance_index: PerformanceIndex) -> float:
        try:
            fitness = 1 / performance_index.generate_speed
        except OverflowError:
            return inf
        if performance_index.time_to_first_token is not None:
            _var = max(0.0, (
                    performance_index.time_to_first_token - self.prefill_constrain) / self.prefill_constrain)
            try:
                fitness += self.prefill_lam * (exp(_var) - 1)
            except OverflowError:
                return inf
        if performance_index.time_per_output_token is not None:
            _decode_var = max(0.0, (
                    performance_index.time_per_output_token - self.decode_constrain) / self.decode_constrain)
            try:
                fitness += self.decode_lam * (exp(_decode_var) - 1)
            except OverflowError:
                return inf
        if performance_index.success_rate:
            _success_var = max(0.0, (
                    performance_index.success_rate - self.success_rate_constrain) / self.success_rate_constrain)
            try:
                fitness += self.success_rate_lam * (exp(_success_var) - 1)
            except OverflowError:
                return inf
        return fitness

    def op_func(self, x) -> np.ndarray:
        n_particles = x.shape[0]
        logger.info(f"Acquired n_particles: {n_particles}, value: {x}")
        generate_speed = []
        for i in range(n_particles):
            # 调用schedule， 获取采集的数据
            try:
                _res = self.scheduler.run(x[i], self.target_field)
                # 根据采集的数据，计算最优化值
                _fitness = self.minimum_algorithm(_res)
            except Exception as e:
                logger.error(f"Failed. error: {e}, please check.")
                logger.exception("What?!")
                _fitness = inf
            logger.info(f"fitness {_fitness}")
            generate_speed.append(_fitness)
        return np.array(generate_speed)

    def constructing_bounds(self) -> Tuple[Tuple, Tuple]:
        """
        返回示例：((0, 10), (0, 10))
        """
        _min = []
        _max = []
        for _field in self.target_field:
            _min.append(_field.min)
            _max.append(_field.max)
        return (tuple(_min), tuple(_max))

    def run(self):
        optimizer = CustomGlobalBestPSO(n_particles=self.n_particles, dimensions=len(self.target_field),
                                        options=self.pso_options.model_dump(), bounds=self.constructing_bounds(),
                                        init_pos=self.init_pos, breakpoint_pos=self.history_pos,
                                        breakpoint_cost=self.history_cost, **self.pso_init_kwargs)
        cost, joint_vars = optimizer.optimize(self.op_func, iters=self.iters)
        logger.info(
            f"best cost {cost}, best joint_vars: "
            f"{[self.target_field[i].format_func(k) for i, k in enumerate(joint_vars)]}")


def main(args: argparse.Namespace):
    simulator = Simulator(settings.mindie)
    bak_path = None
    if args.backup:
        bak_path = settings.output.joinpath("bak")
        if not bak_path.exists():
            bak_path.mkdir(parents=True)
    rpc_clients = []
    if args.deploy_policy == DeployPolicy.multiple.value:
        for server_address in settings.server:
            rpc = xmlrpc.client.ServerProxy(server_address, allow_none=True)
            logger.info(f"{server_address} support method {rpc.system.listMethods()}")
            rpc_clients.append(rpc)
    # 单机benchmark
    time_sleep = os.getenv(IS_SLEEP_FLAG, "False").lower().strip() == "true"
    if time_sleep:
        benchmark = BenchMark(settings.benchmark, bak_path=bak_path)
    else:
        # 默认 自定义单机
        benchmark = ProfilerBenchmark(settings.benchmark, bak_path=bak_path, analyze_tool=AnalyzeTool.profiler)
    # 存储结果，只在主节点存储结果
    data_storage = DataStorage(settings.data_storage)
    # 初始化调度模块，支持单机和多机。
    if args.deploy_policy == DeployPolicy.multiple.value:
        scheduler = ScheduleWithMultiMachine(rpc_clients, simulator, benchmark, data_storage, bak_path=bak_path)
    else:
        scheduler = Scheduler(simulator, benchmark, data_storage, bak_path=bak_path)
    _load_history_data = None
    _load_history = None
    if args.load_breakpoint:
        _load_history = True
    if _load_history:
        _load_history_data = data_storage.load_history_position(settings.data_storage.store_dir)
    pso = PSOOptimizer(scheduler, n_particles=settings.n_particles, iters=settings.iters,
                       prefill_lam=settings.prefill_lam, target_field=settings.target_field,
                       decode_lam=settings.decode_lam, success_rate_lam=settings.success_rate_lam,
                       decode_constrain=settings.decode_constrain, prefill_constrain=settings.prefill_constrain,
                       success_rate_constrain=settings.success_rate_constrain, load_history_data=_load_history_data,
                       load_breakpoint=args.load_breakpoint,
                       pso_init_kwargs={"ftol": settings.ftol, "ftol_iter": settings.ftol_iter})
    pso.run()


def arg_parse():
    parser = argparse.ArgumentParser(prog='optimizer')
    parser.add_argument("-lb", "--load_breakpoint", default=False, action="store_true",
                        help="Continue from where the last optimization was aborted.")
    parser.add_argument("-d", "--deploy_policy", default=DeployPolicy.single.value,
                        choices=[k.value for k in list(DeployPolicy)],
                        help="Indicates whether the multi-node running policy is used.")
    parser.add_argument("--backup", default=False, action="store_true",
                        help="Whether to back up data.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    _args = arg_parse()
    main(_args)
