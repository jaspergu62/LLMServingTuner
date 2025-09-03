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
import glob
import importlib
import json
import os
import re
import time
from math import isnan, isinf
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger

from msserviceprofiler.modelevalstate.config.config import (
    AnalyzeTool, BenchMarkConfig, ProfileConfig, VllmBenchmarkConfig,
    AisBenchConfig, settings, PerformanceIndex, OptimizerConfigField
)
from msserviceprofiler.modelevalstate.config.base_config import VLLM_CUSTOM_OUTPUT, MINDIE_BENCHMARK_PERF_COLUMNS
from msserviceprofiler.modelevalstate.config.custom_command import (
    BenchmarkCommand, VllmBenchmarkCommand, AisBenchCommand
)
from msserviceprofiler.modelevalstate.optimizer.analyze_profiler import analyze as analyze_profiler
from msserviceprofiler.modelevalstate.optimizer.custom_process import CustomProcess
from msserviceprofiler.modelevalstate.optimizer.utils import backup, remove_file
from msserviceprofiler.modelevalstate.common import read_csv_s
from msserviceprofiler.msguard.security import open_s, walk_s


_analyze_mapping = {
    AnalyzeTool.profiler.value: analyze_profiler
}
MS_TO_S = 10 ** 3
US_TO_S = 10 ** 6


def parse_result(res):
    if isinstance(res, str):
        _res = res.strip().split()
        if len(_res) > 1:
            if _res[1].strip().lower() == "ms":
                return float(_res[0]) / 10 ** 3
            elif _res[1].strip().lower() == "us":
                return float(_res[0]) / 10 ** 6
            else:
                return float(_res[0])
        return float(res)
    return res


class AisBench(CustomProcess):
    def __init__(self, benchmark_config: AisBenchConfig, bak_path: Optional[Path] = None, print_log: bool = False):
        super().__init__(bak_path=bak_path, print_log=print_log, process_name=benchmark_config.process_name)
        self.benchmark_config = benchmark_config
        self.work_path = self.benchmark_config.work_path
        self.update_command()
        self.mindie_benchmark_perf_columns = [k.lower().strip() for k in MINDIE_BENCHMARK_PERF_COLUMNS]
 
    def update_command(self):
        self.command = AisBenchCommand(self.benchmark_config.command).command
 
    def backup(self, del_log=True):
        backup(self.benchmark_config.output_path, self.bak_path, self.__class__.__name__)
        if not del_log:
            backup(self.run_log, self.bak_path, self.__class__.__name__)
 
    def get_performance_metric(self, metric_name: str, algorithm: str = "average"):
        output_path = Path(self.benchmark_config.output_path)
        result_files = glob.glob(f"{output_path}/**/*.csv", recursive=True)
        if len(result_files) != 1:
            logger.error("The aisbench result for csv files are not unique; please check")
        metric_name = metric_name.lower().strip()
        algorithm = algorithm.strip().lower()
 
        if algorithm not in self.mindie_benchmark_perf_columns:
            raise ValueError(f"The {algorithm} does not support it; "
                             f"only {self.mindie_benchmark_perf_columns} are supported.")
        algorithm_index = self.mindie_benchmark_perf_columns.index(algorithm)
        for file in result_files:
            df = read_csv_s(file)
            _all_metrics = [k.strip().lower() for k in df["Performance Parameters"].tolist()]
            if metric_name not in _all_metrics:
                continue
            _i = _all_metrics.index(metric_name)
            _columns = [k.lower().strip() for k in df.columns]
            _col_index = _columns.index(algorithm)
            _res = df.iloc[_i, _col_index]
            if not _res:
                continue
            return parse_result(_res)
        raise ValueError("Not Found value.")
 
    def get_performance_index(self):
        output_path = Path(self.benchmark_config.output_path)
        performance_index = PerformanceIndex()
        if not output_path.exists():
            logger.error(f"the output of aisbench is not find: {output_path}")
        result_files = glob.glob(f"{output_path}/**/performances/*/*.csv", recursive=True)
        if len(result_files) < 1:
            raise ValueError("The aisbench result for csv files are not unique; please check")
        for result_file in result_files:
            try:
                df = read_csv_s(result_file, header=0)
            except pd.errors.ParserError as e:
                logger.error(f"{e}, file: {result_file}")
                continue
            ttft_average = df[df["Performance Parameters"] == "TTFT"]["Average"].values[0]
            first_token_time = ttft_average.split()[0]
            tpot_average = df[df["Performance Parameters"] == "TPOT"]["Average"].values[0]
            decode_time = tpot_average.split()[0]
            performance_index.time_to_first_token = float(first_token_time) / MS_TO_S
            performance_index.time_per_output_token = float(decode_time) / MS_TO_S
            performance_index.throughput = float(
                df[df["Performance Parameters"] == "OutputTokenThroughput"]["Average"].values[0].split()[0])
        rate_files = glob.glob(f"{output_path}/**/performances/*/*dataset.json", recursive=True)
        for json_file in rate_files:
            with open_s(json_file, "r") as f:
                try:
                    data = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    logger.error(f"{e}, file: {json_file}")
                    continue
            total_requests = data.get("Total Requests", {}).get("total", 0)
            success_req = data.get("Success Requests", {}).get("total", 0)
            if total_requests != 0:
                performance_index.success_rate = success_req / total_requests
                output_average = data["Output Token Throughput"]["total"]
                performance_index.generate_speed = float(output_average.split()[0])
        return performance_index
 
    def prepare(self):
        remove_file(Path(self.benchmark_config.output_path))
 
    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField]] = None):
        self.update_command()
        super().before_run(run_params)
        module = importlib.import_module("ais_bench")
        aisbench_dir = module.__file__
        ais_dir = Path(aisbench_dir).parent
        api_dir = ais_dir.joinpath("benchmark", "configs", "models")
        # 启动测试
        logger.debug("Start the aisbench test.")
        api_name = self.benchmark_config.command.models
        api_path = None
        for file_path in api_dir.rglob("*.py"):
            if file_path.name == f"{api_name}.py":
                api_path = file_path
        if not api_path:
            raise FileNotFoundError("Not Found {api_name}.py")
        concurrency = rate = None
        for k in run_params:
            try:
                if k.name == "CONCURRENCY" and k.value:
                    concurrency = int(k.value)
                if k.name == "REQUESTRATE" and k.value:
                    rate = k.value
            except ValueError:
                logger.warning(f"the {k.name} is not number; please check: {k.value}")
                concurrency = rate = None
        with open_s(api_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        _request_rate_pattern = re.compile(r"(request_rate=)\d{1,10}(?:\.\d{1,10})?,")
        _batch_size_pattern = re.compile(r"(batch_size=)\d{1,10}(?:\.\d{1,10})?,")
        # 修改 request_rate 和 batch_size
        for i, line in enumerate(lines):
            if 'request_rate=' in line:
                _res = _request_rate_pattern.search(lines[i])
                if _res and rate:
                    lines[i] = lines[i].replace(_res.group(), f"request_rate={rate},")
            if 'batch_size' in line:
                _res = _batch_size_pattern.search(lines[i])
                if _res and concurrency:
                    lines[i] = lines[i].replace(_res.group(), f"batch_size={concurrency},")
 
        # 将修改后的内容写回文件
        with open_s(api_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)


class BenchMark(CustomProcess):
    def __init__(self, benchmark_config: BenchMarkConfig, throughput_type: str = "common",
                 bak_path: Optional[Path] = None, print_log: bool = False):
        super().__init__(bak_path=bak_path, print_log=print_log, process_name=benchmark_config.process_name)
        self.benchmark_config = benchmark_config
        self.throughput_type = throughput_type
        self.process = None
        self.pattern = re.compile(r"^\s*(\d{1,10}(?:\.\d{1,10})?)\s*\%$")
        self.mindie_benchmark_perf_columns = [k.lower().strip() for k in MINDIE_BENCHMARK_PERF_COLUMNS]
        self.command = BenchmarkCommand(self.benchmark_config.command).command

    def update_command(self):
        self.command = BenchmarkCommand(self.benchmark_config.command).command

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField, ...]] = None):
        self.update_command()
        super().before_run(run_params)

    def backup(self):
        backup(self.benchmark_config.output_path, self.bak_path, self.__class__.__name__)
        super().backup()

    def get_req_token_info(self, results_per_request_path: Optional[Path] = None):
        if not results_per_request_path:
            output_path = Path(self.benchmark_config.command.save_path)
            for file_path in walk_s(output_path):
                _file = Path(file_path)
                if _file.name.startswith("results_per_request"):
                    results_per_request_path = _file
                    break
        if results_per_request_path is None:
            raise FileNotFoundError(f"Not Found results_per_request_path: {results_per_request_path!r}")
        with open_s(results_per_request_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON file {results_per_request_path!r}: {e}")
                return None
        _http_rid = []
        _recv_token_size = []
        _reply_token_size = []
        for k, v in data.items():
            if "input_len" not in v.keys():
                continue
            if "output_len" not in v.keys():
                continue
            _http_rid.append(k)
            _recv_token_size.append(v["input_len"])
            _reply_token_size.append(v["output_len"])
        return {"http_rid": _http_rid, "recv_token_size": _recv_token_size,
                "reply_token_size": _reply_token_size}

    def get_ttft_tpot(self, file):
        first_token_time = decode_time = None
        df = read_csv_s(file)
        ttft_metric = self.benchmark_config.performance_config.time_to_first_token.metric
        if ttft_metric in df.columns:
            ttft_algorithm = self.benchmark_config.performance_config.time_to_first_token.algorithm.lower()
            if ttft_algorithm not in self.mindie_benchmark_perf_columns:
                raise ValueError(f"Only one of these is supported {self.mindie_benchmark_perf_columns}. "
                                 f"but {ttft_algorithm} was obtained."
                                 f"please check the configuration file and modify \
                                  benchmark.performance_config.time_to_first_token.algorithm.")
            _index = self.mindie_benchmark_perf_columns.index(ttft_algorithm)
            try:
                first_token_time = float(df[ttft_metric][_index].split()[0])
            except ValueError as e:
                raise ValueError(f"Failed to parse time_to_first_token value: {e}") from e

        tpot_metric = self.benchmark_config.performance_config.time_per_output_token.metric
        if tpot_metric in df.columns:
            tpot_algorithm = self.benchmark_config.performance_config.time_per_output_token.algorithm.lower()
            if tpot_algorithm not in self.mindie_benchmark_perf_columns:
                raise ValueError(f"Only one of these is supported {self.mindie_benchmark_perf_columns}. "
                                 f"but {tpot_algorithm} was obtained."
                                 f"please check the configuration file and modify \
                                  benchmark.performance_config.time_per_output_token.algorithm.")
            _index = self.mindie_benchmark_perf_columns.index(tpot_algorithm)
            try:
                decode_time = float(df[tpot_metric][_index].split()[0])
            except ValueError as e:
                raise ValueError(f"Failed to parse time_per_output_token value: {e}") from e
        try:
            perf_generate_token_speed = float(df["GeneratedTokenSpeed"][0].split()[0])
        except ValueError as e:
            raise ValueError(f"Failed to parse GeneratedTokenSpeed value: {e}") from e
        return first_token_time, decode_time, perf_generate_token_speed

    def get_performance_metric(self, metric_name: str, algorithm: str = "average"):
        output_path = Path(self.benchmark_config.command.save_path)
        metric_name = metric_name.lower().strip()
        algorithm = algorithm.strip().lower()
        if algorithm not in self.mindie_benchmark_perf_columns:
            raise ValueError(f"The {algorithm} does not support it; "
                             f"only {self.mindie_benchmark_perf_columns} are supported.")
        algorithm_index = self.mindie_benchmark_perf_columns.index(algorithm)

        for file_path in walk_s(output_path):
            file = Path(file_path)
            if "result_perf" not in file.name:
                continue
            df = read_csv_s(file)
            _columns = [k.lower().strip() for k in df.columns]
            if metric_name not in _columns:
                continue
            _i = _columns.index(metric_name)
            _res = df.iloc[:, _i][algorithm_index]
            if isinstance(_res, str):
                parts = _res.split()
                if len(parts) > 1:  # 确保分割后有足够的元素
                    if parts[1].strip() == "ms":
                        return float(parts[0]) / MS_TO_S
                    elif parts[1].strip() == "us":
                        return float(parts[0]) / US_TO_S
                # 如果没有单位或者单位不是ms/us，直接转换
                return float(parts[0])
            return _res

    def update_result_common(self, file, performance_index):
        df = read_csv_s(file)
        _generate_speed = common_generate_speed = None
        try:
            if "OutputGenerateSpeed" in df.columns:
                _generate_speed = df["OutputGenerateSpeed"][0]
            elif "GenerateSpeed" in df.columns:
                _generate_speed = df["GenerateSpeed"][0]
            if _generate_speed:
                if isinstance(_generate_speed, str):
                    common_generate_speed = float(_generate_speed.split()[0])
                elif isinstance(_generate_speed,
                                (int, float, np.int64, np.int32, np.float64, np.float32, np.float16)):
                    common_generate_speed = _generate_speed
                else:
                    logger.error(TypeError(f"GenerateSpeed: {_generate_speed}, type: {type(_generate_speed)}"))
            if common_generate_speed:
                performance_index.generate_speed = common_generate_speed
            req_returnd = None
            if "Returned" in df.columns:
                req_returnd = df["Returned"][0]
            if req_returnd:
                _m_res = self.pattern.search(req_returnd)
                if _m_res:
                    performance_index.success_rate = float(_m_res.group(1)) / 100
            if "Throughput" in df.columns:
                _throughput = df["Throughput"][0]
                if _throughput:
                    performance_index.throughput = float(_throughput.split()[0])
        except ValueError as e:
            logger.error(e)


    def get_performance_index(self):
        output_path = Path(self.benchmark_config.command.save_path)
        first_token_time = None
        perf_generate_token_speed = None
        decode_time = None
        performance_index = PerformanceIndex()
        for file in output_path.iterdir():
            if "result_common" in file.name:
                self.update_result_common(file, performance_index)
            if "result_perf" in file.name:
                first_token_time, decode_time, perf_generate_token_speed = self.get_ttft_tpot(file)
        if first_token_time is None or decode_time is None:
            raise ValueError("Not Found first_token_time.")
        if self.throughput_type != "common":
            performance_index.generate_speed = perf_generate_token_speed
        time_to_first_token = first_token_time / MS_TO_S
        time_per_output_token = decode_time / MS_TO_S
        performance_index.time_to_first_token = time_to_first_token
        performance_index.time_per_output_token = time_per_output_token
        return performance_index

    def prepare(self):
        remove_file(Path(self.benchmark_config.output_path))


class ProfilerBenchmark(CustomProcess):
    def __init__(self, profile_config: ProfileConfig, benchmark_config: BenchMarkConfig, *args,
                 analyze_tool: AnalyzeTool = AnalyzeTool.default,
                 **kwargs):
        super().__init__(benchmark_config, *args, **kwargs)
        self.analyze_tool = analyze_tool
        self.profile_config = profile_config
        self.profiler_cmd = ["python", "-m", "ms_service_profiler.parse",
                             f"--input-path={self.profile_config.profile_input_path}",
                             f"--output-path={self.profile_config.profile_output_path}"]
        self.profiler_process = CustomProcess(command=self.profiler_cmd, bak_path=self.bak_path,
                                              work_path=self.work_path, print_log=self.print_log)

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

    def backup(self):
        super().backup()
        backup(self.profile_config.profile_input_path, self.bak_path, self.__class__.__name__)
        backup(self.profile_config.profile_output_path, self.bak_path, self.__class__.__name__)
        self.profiler_process.backup()

    def prepare(self):
        super().prepare()
        remove_file(Path(self.profile_config.profile_input_path))
        remove_file(Path(self.profile_config.profile_output_path))

    def get_performance_index(self):
        logger.debug("get_performance_index")
        try:
            self.profiler_process.run()
            logger.debug("wait profiler")
            timeout = 1800
            start_time = time.time()
            while True:
                if self.profiler_process.check_success():
                    break
                if time.time() - start_time > timeout:
                    self.profiler_process.stop()
                    raise TimeoutError(f"commmand did not finish within {timeout} seconds.")
                time.sleep(1)
        except Exception as e:
            logger.error(f"Failed in start profiler. command: {self.profiler_cmd} "
                         f"log: {self.profiler_process.run_log}")
            raise e
        collect_path = Path(settings.simulator_output)
        if self.analyze_tool == AnalyzeTool.profiler.value:
            res = self.extra_performance_index(self.benchmark_config.profile_output_path, collect_path)
            return res
        else:
            return super().get_performance_index()

    def stop(self, del_log: bool = False):
        super().stop(del_log)
        self.profiler_process.stop(del_log)


class VllmBenchMark(CustomProcess):
    def __init__(self, benchmark_config: VllmBenchmarkConfig, bak_path: Optional[Path] = None, print_log: bool = False):
        super().__init__(bak_path=bak_path, print_log=print_log, process_name=benchmark_config.process_name)
        self.benchmark_config = benchmark_config
        self.command = VllmBenchmarkCommand(self.benchmark_config.command).command

    def backup(self):
        backup(self.benchmark_config.output_path, self.bak_path, self.__class__.__name__)
        super().backup()

    def update_command(self):
        self.command = VllmBenchmarkCommand(self.benchmark_config.command).command

    def prepare(self):
        remove_file(Path(self.benchmark_config.output_path))

    def get_performance_index(self):
        output_path = Path(self.benchmark_config.command.result_dir)
        performance_index = PerformanceIndex()
        for file in walk_s(output_path):
            file = Path(file)
            if not file.name.endswith(".json"):
                continue
            with open_s(file, mode='r', encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Failed in parse vllm benchmark result. file: {file}")
                    continue

            performance_index.generate_speed = data.get("output_throughput", 0)
            ttft_metric = self.benchmark_config.performance_config.time_to_first_token.metric
            performance_index.time_to_first_token = data.get(ttft_metric, 0) / MS_TO_S
            tpot_metric = self.benchmark_config.performance_config.time_per_output_token.metric
            performance_index.time_per_output_token = data.get(tpot_metric, 0) / MS_TO_S
            num_prompts = data.get("num_prompts", 1)
            completed = data.get("completed", 0)
            performance_index.success_rate = 0
            if num_prompts > 0:
                performance_index.success_rate = completed / num_prompts
            performance_index.throughput = float(data.get("request_throughput", 3.0))
        return performance_index


    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField]] = None):
        self.update_command()
        super().before_run(run_params)
        Path(self.benchmark_config.command.result_dir).mkdir(parents=True, exist_ok=True, mode=0o750)

        if VLLM_CUSTOM_OUTPUT not in os.environ:
            os.environ[VLLM_CUSTOM_OUTPUT] = str(self.benchmark_config.command.result_dir)
        _var_name = f"${VLLM_CUSTOM_OUTPUT}"
        for i, item in enumerate(self.command):
            if item == _var_name:
                self.command[i] = str(self.benchmark_config.command.result_dir)