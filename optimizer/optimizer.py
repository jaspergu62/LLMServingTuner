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
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from math import exp, inf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
import numpy as np
from loguru import logger

from msserviceprofiler.modelevalstate.config.base_config import AnalyzeTool, BenchMarkPolicy, DeployPolicy, EnginePolicy
from msserviceprofiler.modelevalstate.config.base_config import CUSTOM_OUTPUT, custom_output, FOLDER_LIMIT_SIZE
from msserviceprofiler.modelevalstate.optimizer.utils import backup, remove_file, close_file_fp, get_folder_size
from msserviceprofiler.modelevalstate.optimizer.analyze_profiler import analyze as analyze_profiler
from msserviceprofiler.modelevalstate.common import get_train_sub_path, read_csv_s
from msserviceprofiler.msguard.security import open_s


_analyze_mapping = {AnalyzeTool.profiler.value: analyze_profiler}


def validate_parameters(common_generate_speed, perf_generate_token_speed, first_token_time, decode_time):
    if common_generate_speed is None and perf_generate_token_speed is None:
        raise ValueError("Not Found common_generate_speed or perf_generate_token_speed.")
    if first_token_time is None or decode_time is None:
        raise ValueError("Not Found first_token_time.")
    

def aisbench_validate_parameters(generate_speed, first_token_time, decode_time):
    if generate_speed is None:
        raise ValueError("Not Found generate_speed")
    if first_token_time is None:
        raise ValueError("Not Found first_token_time.")
    if decode_time is None:
        raise ValueError("Not Found decode_time.")


class BenchMark:
    def __init__(self, benchmark_config, throughput_type: str = "common",
                 bak_path: Optional[Path] = None):
        self.benchmark_config = benchmark_config
        self.throughput_type = throughput_type
        self.bak_path = bak_path
        self.run_log = None
        self.run_log_offset = None
        self.run_log_fp = None
        self.process = None
        self.pattern = re.compile(r'\(([^)]+)')
        self.update_command()

    def update_command(self):
        from msserviceprofiler.modelevalstate.config.custom_command import BenchmarkCommand
        self.command = BenchmarkCommand(self.benchmark_config.command).command

    def backup(self, del_log=True):
        backup(self.benchmark_config.output_path, self.bak_path, self.__class__.__name__)
        if not del_log:
            backup(self.run_log, self.bak_path, self.__class__.__name__)

    def get_performance_index(self):
        from msserviceprofiler.modelevalstate.config.config import PerformanceIndex

        output_path = Path(self.benchmark_config.output_path)
        common_generate_speed = None
        first_token_time = None
        perf_generate_token_speed = None
        decode_time = None
        success_rate = None
        for file in output_path.iterdir():
            if "result_common" in file.name:
                try:
                    df = read_csv_s(file)
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
                    cleaned_str = _m_res.group(1).replace(' ', '').rstrip('%')
                    success_rate = float(cleaned_str) / 100
                except (KeyError, AttributeError) as e:
                    logger.error(f"Failed in get GenerateSpeed. error: {e}")
                continue
            if "result_perf" in file.name:
                try:
                    df = read_csv_s(file)
                    first_token_time = float(df["FirstTokenTime"][0].split()[0])
                    perf_generate_token_speed = float(df["GeneratedTokenSpeed"][0].split()[0])
                    decode_time = float(df["DecodeTime"][0].split()[0])
                except (AttributeError, KeyError) as e:
                    logger.error("Failed in get FirstTokenTime or GeneratedTokenSpeed. error: {}", e)
        validate_parameters(common_generate_speed, perf_generate_token_speed, first_token_time, decode_time)
        if self.throughput_type == "common":
            generate_speed = common_generate_speed
        else:
            generate_speed = perf_generate_token_speed
        time_to_first_token = first_token_time / 10 ** 3
        time_per_output_token = decode_time / 10 ** 3
        return PerformanceIndex(generate_speed=generate_speed, time_to_first_token=time_to_first_token,
                                time_per_output_token=time_per_output_token, success_rate=success_rate)

    def prepare(self):
        remove_file(Path(self.benchmark_config.output_path))
        remove_file(Path(self.benchmark_config.custom_collect_output_path))

    def check_success(self, print_log=False):
        if self.run_log:
            run_log_path = Path(self.run_log)
            if run_log_path.exists() and print_log:
                try:
                    with open_s(run_log_path, "r", encoding="utf-8") as f:
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

    def run(self, run_params):
        # 启动测试
        self.update_command()
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
                    _var_name = f"${k.name}"
                    if _var_name in self.command:
                        _i = self.command.index(_var_name)
                        self.command[_i] = str(k.value)
                except KeyError as e:
                    logger.error(f"Failed to set environment variable. error {e}")
        if CUSTOM_OUTPUT not in os.environ:
            os.environ[CUSTOM_OUTPUT] = str(custom_output)
        try:
            self.process = subprocess.Popen(self.command, env=os.environ, stdout=self.run_log_fp, 
                                            stderr=subprocess.STDOUT, text=True, cwd=cwd)
        except OSError as e:
            logger.error(f"Failed to run benchmark. error {e}")
            raise e
        logger.info(f"command: {' '.join(self.command)}, log file: {self.run_log}")

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


class AisBench:
    def __init__(self, benchmark_config, bak_path: Optional[Path] = None):
        self.benchmark_config = benchmark_config
        self.bak_path = bak_path
        self.run_log = None
        self.run_log_offset = None
        self.run_log_fp = None
        self.process = None
        self.update_command()

    def update_command(self):
        from msserviceprofiler.modelevalstate.config.custom_command import AisbenchCommand
        self.command = AisbenchCommand(self.benchmark_config.aisbench_command).command

    def backup(self, del_log=True):
        backup(self.benchmark_config.aisbench_output_path, self.bak_path, self.__class__.__name__)
        if not del_log:
            backup(self.run_log, self.bak_path, self.__class__.__name__)

    def get_performance_index(self):
        from msserviceprofiler.modelevalstate.config.config import PerformanceIndex

        aisbench_output_path = Path(self.benchmark_config.aisbench_output_path)
        first_token_time = None
        decode_time = None
        success_rate = None
        generate_speed = None
        if not aisbench_output_path.exists():
            logger.error("the output of aisbench is not find")
        result_files = glob.glob(f"{aisbench_output_path}/**/*.csv", recursive=True)
        if len(result_files) != 1:
            logger.error("The aisbench result for csv files are not unique; please check")
        else:
            result_file = result_files[0]
            df = read_csv_s(result_file, header=0)
            ttft_average = df[df["Performance Parameters"] == "TTFT"]["Average"].values[0]
            first_token_time = ttft_average.split()[0]
            tpot_average = df[df["Performance Parameters"] == "TPOT"]["Average"].values[0]
            decode_time = tpot_average.split()[0]
            rate_dir = os.path.dirname(result_file)
            rate_files = glob.glob(f"{rate_dir}/*dataset.json", recursive=True)
            if len(rate_files) != 1:
                logger.error("The aisbench result files for json are not unique; please check")
                success_rate = 0
            else:
                json_file = rate_files[0]
                with open_s(json_file, "r") as f:
                    data = json.load(f)
                total_requests = data["Total Requests"]["total"]
                success_req = data["Success Requests"]["total"]
                if total_requests != 0:
                    success_rate = success_req / total_requests
                    output_average = data["Output Token Throughput"]["total"]
                    generate_speed = output_average.split()[0]
                else:
                    logger.error("total_requests can not be 0; please check")
        aisbench_validate_parameters(generate_speed, first_token_time, decode_time)
        time_to_first_token = float(first_token_time) / 10 ** 3
        time_per_output_token = float(decode_time) / 10 ** 3
        return PerformanceIndex(generate_speed=generate_speed, time_to_first_token=time_to_first_token,
                                time_per_output_token=time_per_output_token, success_rate=success_rate)

    def prepare(self):
        remove_file(Path(self.benchmark_config.aisbench_output_path))
        remove_file(Path(self.benchmark_config.custom_collect_output_path))

    def check_success(self, print_log=False):
        if self.run_log:
            run_log_path = Path(self.run_log)
            if run_log_path.exists() and print_log:
                try:
                    with open_s(run_log_path, "r", encoding="utf-8") as f:
                        f.seek(self.run_log_offset)
                        output = f.read()
                        self.run_log_offset = f.tell()
                        logger.info(f"aisbench out: \n{output}")
                except (UnicodeError, OSError) as e:
                    logger.error(f"Failed read aisbench log. error {e}")
        try:
            if self.process.poll() is None:
                return False
            elif self.process.poll() == 0:
                return True
            else:
                raise subprocess.SubprocessError(
                    f"Failed in run aisbench. return code: {self.process.returncode}. ")
        except AttributeError as e:
            logger.error(f"Failed to check process status, error {e}")
            return False

    def run(self, run_params):
        import ast
        import ais_bench
        aisbench_dir = ais_bench.__file__
        ais_dir = Path(aisbench_dir).parent
        api_dir = ais_dir.joinpath("benchmark", "configs", "models")
        # 启动测试
        logger.info("Start the aisbench test.")
        api_name = self.benchmark_config.aisbench_command.models
        for file_path in api_dir.rglob("*.py"):
            if file_path.name == f"{api_name}.py":
                api_path = file_path
        self.run_log_fp, self.run_log = tempfile.mkstemp(prefix="modelevalstate_aisbench")
        self.run_log_offset = 0
        if self.benchmark_config.work_path:
            cwd = self.benchmark_config.work_path
        else:
            cwd = os.getcwd()
        for k in run_params:
            if k.name == "CONCURRENCY":
                concurrency = int(k.value)
            if k.name == "REQUESTRATE":
                rate = int(k.value)
        with open_s(api_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 修改 request_rate 和 batch_size
        for i, line in enumerate(lines):
            if 'request_rate' in line:
                lines[i] = f'        request_rate={rate},\n'
            if 'batch_size' in line:
                lines[i] = f'        batch_size={concurrency},\n'

        # 将修改后的内容写回文件
        with open_s(api_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        if CUSTOM_OUTPUT not in os.environ:
            os.environ[CUSTOM_OUTPUT] = str(custom_output)
        try:
            self.process = subprocess.Popen(self.command, env=os.environ, stdout=self.run_log_fp, 
                                            stderr=subprocess.STDOUT, text=True, cwd=cwd)
        except OSError as e:
            logger.error(f"Failed to run benchmark. error {e}")
            raise e
        logger.info(f"command: {' '.join(self.command)}, log file: {self.run_log}")

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


class ProfilerBenchmark(AisBench):
    def __init__(self, benchmark_config, *args, analyze_tool: AnalyzeTool = AnalyzeTool.default,
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
        from msserviceprofiler.modelevalstate.config.config import PerformanceIndex

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
                with open_s(self.profiler_log, "r") as f:
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


class VllmBenchMark(AisBench):
    def __init__(self, benchmark_config, bak_path: Optional[Path] = None):
        super().__init__(benchmark_config, bak_path)
        self.output_path = benchmark_config.output_path
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True, mode=0o750)
        self.update_command()

    def backup(self, del_log=True):
        backup(self.benchmark_config.output_path, self.bak_path, self.__class__.__name__)
        if not del_log:
            backup(self.run_log, self.bak_path, self.__class__.__name__)
    
    def prepare(self):
        remove_file(Path(self.benchmark_config.output_path))

    def update_command(self):
        from msserviceprofiler.modelevalstate.config.custom_command import VllmBenchmarkCommand
        self.command = VllmBenchmarkCommand(self.benchmark_config.vllm_command).command

    def get_performance_index(self):
        from msserviceprofiler.modelevalstate.config.config import PerformanceIndex

        output_path = Path(self.benchmark_config.output_path)
        generate_speed = None
        time_per_output_token = None
        time_to_first_token = None
        success_rate = None
        for file in output_path.iterdir():
            if not file.name.endswith(".json"):
                continue
            with open_s(file, mode='r', encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as err:
                    logger.warning("Failed to open %r, error: %r" % (file, err))
                    data = {}
            generate_speed = data.get("output_throughput", 0)
            time_to_first_token = data.get("mean_ttft_ms", 0) / 10 ** 3
            time_per_output_token = data.get("mean_tpot_ms", 0) / 10 ** 3
            num_prompts = data.get("num_prompts", 1)
            completed = data.get("completed", 0)
            if num_prompts != 0:
                success_rate = completed / num_prompts
        return PerformanceIndex(generate_speed=generate_speed,
                                time_to_first_token=time_to_first_token,
                                time_per_output_token=time_per_output_token,
                                success_rate=success_rate)

    def run(self, run_params):
        # 启动测试
        self.update_command()
        logger.info("Start the vllm_benchmark test.")
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
                    _var_name = f"${k.name}"
                    if _var_name in self.command:
                        _i = self.command.index(_var_name)
                        self.command[_i] = str(k.value)
                except KeyError as e:
                    logger.error(f"Failed to set environment variable. error {e}")
        if CUSTOM_OUTPUT not in os.environ:
            os.environ[CUSTOM_OUTPUT] = str(custom_output)
        os.environ["MODEL_EVAL_STATE_VLLM_CUSTOM_OUTPUT"] = str(self.output_path)
        _var_name = f"$MODEL_EVAL_STATE_VLLM_CUSTOM_OUTPUT"
        if _var_name in self.command:
            _i = self.command.index(_var_name)
            self.command[_i] = str(self.output_path)
        try:
            self.process = subprocess.Popen(self.command, env=os.environ, stdout=self.run_log_fp, 
                                            stderr=subprocess.STDOUT, text=True, cwd=cwd)
        except OSError as e:
            logger.error(f"Failed to run vllm_benchmark. error {e}")
            raise e
        logger.info(f"command: {' '.join(self.command)}, log file: {self.run_log}")


class Scheduler:
    def __init__(self, simulator, benchmark, data_storage,
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
            if get_folder_size(self.bak_path) > FOLDER_LIMIT_SIZE:
                self.simulator.bak_path = None
                self.benchmark.bak_path = None
            else:
                _cur_bak_path = get_train_sub_path(self.bak_path)
                self.simulator.bak_path = _cur_bak_path
                self.benchmark.bak_path = _cur_bak_path

    def wait_simulate(self):
        logger.info("wait run simulator")
        for _ in range(self.wait_time):
            time.sleep(1)
            if self.simulator.check_success():
                logger.info(f"Successfully started the {self.simulator.process.pid} process.")
                return
        raise TimeoutError(self.wait_time)

    def run_simulate(self, params: np.ndarray, params_field):
        self.benchmark.prepare()
        self.simulator.run(tuple(self.simulate_run_info))
        self.wait_simulate()

    def monitoring_status(self):
        logger.info("monitor status")
        while True:
            if self.simulator.process.poll() is not None:
                self.simulator.stop(del_log=False)
                self.benchmark.stop(del_log=False)
                raise subprocess.SubprocessError(f"Failed in run simulator. "
                                                 f"return code: {self.simulator.process.returncode}.")
            if self.benchmark.check_success():
                return
            time.sleep(1)

    def run_target_server(self, params: np.ndarray, params_field):
        """
        1. 启动mindie仿真
        2. 启动benchmark 测试
        3. 检查mindie状态，检查benchmark状态
        """
        for _ in range(self.retry_number):
            try:
                self.run_simulate(params, params_field)
            except Exception as e:
                logger.error(f"Failed in Simulator Running. error: {e}， simulator log {self.simulator.mindie_log}")
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
                logger.error(f"Failed in monitoring status. error: {e}, simulator log {self.simulator.mindie_log}, "
                             f"benchmark log {self.benchmark.run_log}")
                logger.exception("What?!")
                continue
            return
        raise ValueError(
            f"Failed in run_target_server")

    def stop_target_server(self, del_log=True):
        self.simulator.stop(del_log)
        self.benchmark.stop(del_log)

    def run(self, params: np.ndarray, params_field):
        """
        1. 启动mindie仿真
        2. 启动benchmark 测试
        3. 获取benchmark测试结果
        4. 关闭mindie仿真
        5. 返回benchmark测试结果
        params: 是一维数组，其值对应mindie 的相关配置。
        """
        from msserviceprofiler.modelevalstate.config.config import map_param_with_value, PerformanceIndex

        logger.info("Start run in scheduler.")
        self.back_up()
        self.simulate_run_info = map_param_with_value(params, params_field)
        error_info = None
        del_log = True
        performance_index = PerformanceIndex()
        self.benchmark.update_command()
        try:
            self.run_target_server(params, params_field)
            time.sleep(1)
            performance_index = self.benchmark.get_performance_index()
        except Exception as e:
            logger.error(f"Failed running. bak path: {self.simulator.bak_path}")
            error_info = e
            del_log = False
        self.data_storage.save(performance_index, tuple(self.simulate_run_info), error=error_info, 
                               backup=self.current_back_path)
        self.stop_target_server(del_log)
        if error_info:
            raise error_info
        return performance_index


class ScheduleWithMultiMachine(Scheduler):
    def __init__(self, communication_config, *args, **kwargs):
        from msserviceprofiler.modelevalstate.optimizer.communication import CommunicationForFile, CustomCommand

        super().__init__(*args, **kwargs)
        self.communication_config = communication_config
        self.communication = CommunicationForFile(self.communication_config.cmd_file,
                                                  self.communication_config.res_file)
        self.cmd = CustomCommand()
        _cmd = self.cmd.init
        self.communication.send_command(_cmd)
        self.communication.clear_command(_cmd)

    def back_up(self):
        if self.bak_path:
            if get_folder_size(self.bak_path) > FOLDER_LIMIT_SIZE:
                self.simulator.bak_path = None
                self.benchmark.bak_path = None
            else:
                _cur_bak_path = get_train_sub_path(self.bak_path)
                self.simulator.bak_path = _cur_bak_path
                self.benchmark.bak_path = _cur_bak_path
                _cmd = f"{self.cmd.backup} params:{_cur_bak_path}"
                self.communication.send_command(_cmd)
                self.communication.clear_cmd(_cmd)

    def monitoring_status(self):
        logger.info("Start monitoring")
        while True:
            _cmd = self.cmd.process_poll
            self.communication.send_command(_cmd)
            all_poll = [self.simulator.process.poll(), self.communication.clear_cmd(_cmd)]
            if any([_i is not None for _i in all_poll]):
                self.stop_target_server(del_log=False)
                raise subprocess.SubprocessError(
                    f"Failed in run simulator. all status: {all_poll}")
            if self.benchmark.check_success():
                return
            time.sleep(1)

    def run_simulate(self, params: np.ndarray, params_field):
        _cmd = f"{self.cmd.start} params:{params.tolist()}"
        self.cmd.history = _cmd
        self.communication.send_command(_cmd)
        self.communication.clear_cmd(_cmd)
        self.simulator.run(tuple(self.simulate_run_info))
        self.wait_simulate()
        # wait 其他服务器上服务成功
        _cmd = self.cmd.check_success
        self.cmd.history = _cmd
        self.communication.send_command(_cmd)
        self.communication.clear_cmd(_cmd)

    def stop_target_server(self, del_log=True):
        super(ScheduleWithMultiMachine, self).stop_target_server(del_log)
        # wait 其他服务器上服务成功
        _cmd = f"{self.cmd.stop} params:{del_log}"
        self.communication.send_command(_cmd)
        self.communication.clear_cmd(_cmd)
        self.cmd.history = _cmd


class PSOOptimizer:
    def __init__(self, scheduler: Scheduler, n_particles: int = 10, iters=100, pso_options=None,
                 target_field: Optional[Tuple] = None, prefill_lam: float = 0.5, decode_lam: float = 0.5,
                 success_rate_lam: float = 0.5, prefill_constraint: float = 0.05, decode_constraint: float = 0.05,
                 success_rate_constraint: float = 1, load_history_data: Optional[List] = None,
                 load_breakpoint: bool = False, pso_init_kwargs: Optional[Dict] = None):
        from msserviceprofiler.modelevalstate.config.config import default_support_field, PsoOptions

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
        self.prefill_constraint = prefill_constraint
        self.decode_constraint = decode_constraint
        self.success_rate_constraint = success_rate_constraint
        self.load_history_data = load_history_data
        self.load_breakpoint = load_breakpoint
        self.pso_init_kwargs = pso_init_kwargs
        self.init_pos = None
        self.history_cost, self.history_pos = None, None
        if self.load_history_data and self.load_breakpoint:
            self.history_pos, self.history_cost = self.computer_fitness()

    def computer_fitness(self) -> Tuple:
        from msserviceprofiler.modelevalstate.config.config import PerformanceIndex

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

    def minimum_algorithm(self, performance_index) -> float:
        if not isinstance(performance_index.generate_speed, (int, float)) or performance_index.generate_speed == 0:
            return inf
        try:
            fitness = 1 / performance_index.generate_speed
        except OverflowError:
            return inf
        
        def calculate_metric(value, constraint, lam):
            if constraint == 0:
                return inf
            try:
                _var = max(0.0, (value - constraint) / constraint)
                return lam * (exp(_var) - 1)
            except OverflowError:
                return inf
            
        if performance_index.time_to_first_token is not None:
            fitness += calculate_metric(performance_index.time_to_first_token, 
                                    self.prefill_constraint, 
                                    self.prefill_lam)
        if performance_index.time_per_output_token is not None:
            fitness += calculate_metric(performance_index.time_per_output_token,
                                      self.decode_constraint,
                                      self.decode_lam)
        if performance_index.success_rate:
            fitness += calculate_metric(self.success_rate_constraint - performance_index.success_rate,
                                      self.success_rate_constraint,
                                      self.success_rate_lam)
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
        from msserviceprofiler.modelevalstate.optimizer.global_best_custom import CustomGlobalBestPSO
        from msserviceprofiler.modelevalstate.config.config import map_param_with_value

        optimizer = CustomGlobalBestPSO(n_particles=self.n_particles, dimensions=len(self.target_field),
                                        options=self.pso_options.model_dump(), bounds=self.constructing_bounds(),
                                        init_pos=self.init_pos, breakpoint_pos=self.history_pos,
                                        breakpoint_cost=self.history_cost, **self.pso_init_kwargs)
        cost, joint_vars = optimizer.optimize(self.op_func, iters=self.iters, verbose=False)
        best_position = {_field.name: _field.value for _field in map_param_with_value(joint_vars, self.target_field)}
        logger.info(f"best cost {cost}, best joint_vars: {best_position}")


def arg_parse(subparsers):
    parser = subparsers.add_parser(
        "optimizer", formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="optimize for performance"
    )

    parser.add_argument("-lb", "--load_breakpoint", action="store_true",
                        help="Continue from where the last optimization was aborted.")
    parser.add_argument("-d", "--deploy_policy", default=DeployPolicy.single.value,
                        choices=[k.value for k in list(DeployPolicy)],
                        help="Indicates whether the multi-node running policy is used.")
    parser.add_argument("--backup", default=False, action="store_true",
                        help="Whether to back up data.")
    parser.add_argument("-b", "--benchmark_policy", default=BenchMarkPolicy.benchmark.value,
                        choices=[k.value for k in list(BenchMarkPolicy)],
                        help="Whether to use custom performance indicators.")
    parser.add_argument("-e", "--engine", default=EnginePolicy.mindie.value, 
                        choices=[k.value for k in list(EnginePolicy)],
                        help="Whether to back up data.")
    parser.set_defaults(func=main)


def main(args: argparse.Namespace):
    from msserviceprofiler.modelevalstate.optimizer.server import main as slave_server
    from msserviceprofiler.modelevalstate.optimizer.simulator import Simulator, VllmSimulator
    from msserviceprofiler.modelevalstate.config.config import settings, ServiceType
    from msserviceprofiler.modelevalstate.optimizer.store import DataStorage

    if settings.service == ServiceType.slave.value:
        slave_server()
        return
    if args.engine == EnginePolicy.vllm.value:
        simulator = VllmSimulator(settings.simulator)
    else:
        simulator = Simulator(settings.simulator)
    bak_path = None
    if args.backup:
        bak_path = settings.output.joinpath("bak")
        if not bak_path.exists():
            bak_path.mkdir(parents=True, mode=0o750)
    # 单机benchmark
    if args.benchmark_policy == BenchMarkPolicy.aisbench.value:
        benchmark = AisBench(settings.benchmark, bak_path=bak_path)
    elif args.benchmark_policy == BenchMarkPolicy.vllm_benchmark.value:
        benchmark = VllmBenchMark(settings.benchmark, bak_path=bak_path)
    elif args.benchmark_policy == BenchMarkPolicy.profiler_benchmark:
        benchmark = ProfilerBenchmark(settings.benchmark, bak_path=bak_path, analyze_tool=AnalyzeTool.profiler)
    else:
        benchmark = BenchMark(settings.benchmark, bak_path=bak_path)

    # 存储结果，只在主节点存储结果
    data_storage = DataStorage(settings.data_storage)
    # 初始化调度模块，支持单机和多机。
    if args.deploy_policy == DeployPolicy.multiple.value:
        scheduler = ScheduleWithMultiMachine(settings.communication, simulator, benchmark, data_storage,
                                             bak_path=bak_path)
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
                       decode_constraint=settings.decode_constraint, prefill_constraint=settings.prefill_constraint,
                       success_rate_constraint=settings.success_rate_constraint, load_history_data=_load_history_data,
                       load_breakpoint=args.load_breakpoint,
                       pso_init_kwargs={"ftol": settings.ftol, "ftol_iter": settings.ftol_iter})
    pso.run()
