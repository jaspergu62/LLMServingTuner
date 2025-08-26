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
import re
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger

from msserviceprofiler.modelevalstate.config.config import AnalyzeTool, BenchMarkConfig, ProfileConfig, VllmBenchmarkConfig, \
    VLLM_CUSTOM_OUTPUT, MINDIE_BENCHMARK_PERF_COLUMNS, settings
from msserviceprofiler.modelevalstate.config.config import PerformanceIndex, OptimizerConfigField
from msserviceprofiler.modelevalstate.config.custom_command import BenchmarkCommand, VllmBenchmarkCommand
from msserviceprofiler.modelevalstate.optimizer.analyze_profiler import analyze as analyze_profiler
from msserviceprofiler.modelevalstate.optimizer.custom_process import CustomProcess
from msserviceprofiler.modelevalstate.optimizer.utils import backup, remove_file
from msserviceprofiler.msguard.security import open_s


_analyze_mapping = {
    AnalyzeTool.profiler.value: analyze_profiler
}


class BenchMark(CustomProcess):
    def __init__(self, benchmark_config: BenchMarkConfig, throughput_type: str = "common",
                 bak_path: Optional[Path] = None, print_log: bool = False):
        super().__init__(bak_path=bak_path, print_log=print_log, process_name=benchmark_config.process_name)
        self.benchmark_config = benchmark_config
        self.throughput_type = throughput_type
        self.process = None
        self.pattern = re.compile(r"\s*(\d+\.?\d*)\s*\%")
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
            for _file in output_path.iterdir():
                if _file.name.startswith("results_per_request"):
                    results_per_request_path = _file
                    break
        if results_per_request_path is None:
            raise FileNotFoundError(f"Not Found results_per_request_path: {results_per_request_path}")
        with open_s(results_per_request_path, "r", encoding="utf-8") as f:
            data = json.load(f)
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
        df = pd.read_csv(file)
        if self.benchmark_config.performance_config.time_to_first_token.metric in df.columns:
            if self.benchmark_config.performance_config.time_to_first_token.algorithm.lower() not in self.mindie_benchmark_perf_columns:
                raise ValueError(f"Only one of these is supported {self.mindie_benchmark_perf_columns}. "
                                 f"but {self.benchmark_config.performance_config.time_to_first_token.algorithm.lower()} was obtained."
                                 f"please check the configuration file and modify benchmark.performance_config.time_to_first_token.algorithm.")
            _index = self.mindie_benchmark_perf_columns.index(
                self.benchmark_config.performance_config.time_to_first_token.algorithm.lower())
            first_token_time = float(
                df[self.benchmark_config.performance_config.time_to_first_token.metric][_index].split()[0])
        if self.benchmark_config.performance_config.time_per_output_token.metric in df.columns:
            if self.benchmark_config.performance_config.time_per_output_token.algorithm.lower() not in self.mindie_benchmark_perf_columns:
                raise ValueError(f"Only one of these is supported {self.mindie_benchmark_perf_columns}. "
                                 f"but {self.benchmark_config.performance_config.time_per_output_token.algorithm.lower()} was obtained."
                                 f"please check the configuration file and modify benchmark.performance_config.time_per_output_token.algorithm.")
            _index = self.mindie_benchmark_perf_columns.index(
                self.benchmark_config.performance_config.time_per_output_token.algorithm.lower())
            decode_time = float(
                df[self.benchmark_config.performance_config.time_per_output_token.metric][_index].split()[0])
        perf_generate_token_speed = float(df["GeneratedTokenSpeed"][0].split()[0])
        return first_token_time, decode_time, perf_generate_token_speed

    def get_performance_metric(self, metric_name: str, algorithm: str = "average"):
        output_path = Path(self.benchmark_config.command.save_path)
        metric_name = metric_name.lower().strip()
        algorithm = algorithm.strip().lower()
        if algorithm not in self.mindie_benchmark_perf_columns:
            raise ValueError(f"The {algorithm} does not support it; "
                             f"only {self.mindie_benchmark_perf_columns} are supported.")
        algorithm_index = self.mindie_benchmark_perf_columns.index(algorithm)

        for file in output_path.iterdir():
            if "result_perf" not in file.name:
                continue
            df = pd.read_csv(file)
            _columns = [k.lower().strip() for k in df.columns]
            if metric_name not in _columns:
                continue
            _i = _columns.index(metric_name)
            _res = df.iloc[:, _i][algorithm_index]
            if isinstance(_res, str):
                if _res.split()[1].strip() == "ms":
                    return float(_res.split()[0]) / 10 ** 3
                elif _res.split()[1].strip() == "us":
                    return float(_res.split()[0]) / 10 ** 6
                return float(_res.split()[0])
            return _res

    def update_result_common(self, file, performance_index):
        df = pd.read_csv(file)
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
        time_to_first_token = first_token_time / 10 ** 3
        time_per_output_token = decode_time / 10 ** 3
        performance_index.time_to_first_token = time_to_first_token
        performance_index.time_per_output_token = time_per_output_token
        performance_index.ttft_max = self.get_performance_metric("FirstTokenTime", "max")
        performance_index.ttft_min = self.get_performance_metric("FirstTokenTime", "min")
        performance_index.ttft_p75 = self.get_performance_metric("FirstTokenTime", "p75")
        performance_index.ttft_p90 = self.get_performance_metric("FirstTokenTime", "p90")
        performance_index.ttft_p99 = self.get_performance_metric("FirstTokenTime", "p99")
        performance_index.tpot_max = self.get_performance_metric("DecodeTime", "max")
        performance_index.tpot_min = self.get_performance_metric("DecodeTime", "min")
        performance_index.tpot_p75 = self.get_performance_metric("DecodeTime", "p75")
        performance_index.tpot_p90 = self.get_performance_metric("DecodeTime", "p90")
        performance_index.tpot_p99 = self.get_performance_metric("DecodeTime", "p99")
        performance_index.prefill_batch_size = self.get_performance_metric("PrefillBatchsize", "average")
        performance_index.prefill_batch_size_min = self.get_performance_metric("PrefillBatchsize", "min")
        performance_index.prefill_batch_size_max = self.get_performance_metric("PrefillBatchsize", "max")
        performance_index.prefill_batch_size_p75 = self.get_performance_metric("PrefillBatchsize", "p75")
        performance_index.prefill_batch_size_p90 = self.get_performance_metric("PrefillBatchsize", "p90")
        performance_index.prefill_batch_size_p99 = self.get_performance_metric("PrefillBatchsize", "p99")
        performance_index.decoder_batch_size = self.get_performance_metric("DecoderBatchsize", "average")
        performance_index.decoder_batch_size_min = self.get_performance_metric("DecoderBatchsize", "min")
        performance_index.decoder_batch_size_max = self.get_performance_metric("DecoderBatchsize", "max")
        performance_index.decoder_batch_size_p75 = self.get_performance_metric("DecoderBatchsize", "p75")
        performance_index.decoder_batch_size_p90 = self.get_performance_metric("DecoderBatchsize", "p90")
        performance_index.decoder_batch_size_p99 = self.get_performance_metric("DecoderBatchsize", "p99")
        return performance_index

    def prepare(self):
        remove_file(Path(self.benchmark_config.output_path))


class ProfilerBenchmark(BenchMark):
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
        logger.info("get_performance_index")
        try:
            self.profiler_process.run()
            logger.info("wait profiler")
            while True:
                if self.profiler_process.check_success():
                    break
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
        generate_speed = None
        time_per_output_token = None
        time_to_first_token = None
        success_rate = None
        for file in output_path.iterdir():
            if not file.name.endswith(".json"):
                continue
            with open(file, mode='r', encoding="utf-8") as f:
                data = json.load(f)

            generate_speed = data.get("output_throughput", 0)
            time_to_first_token = data.get(self.benchmark_config.performance_config.time_to_first_token.metric,
                                           0) / 10 ** 3
            time_per_output_token = data.get(self.benchmark_config.performance_config.time_per_output_token.metric,
                                             0) / 10 ** 3
            num_prompts = data.get("num_prompts", 1)
            completed = data.get("completed", 0)
            success_rate = completed / num_prompts
        return PerformanceIndex(generate_speed=generate_speed,
                                time_to_first_token=time_to_first_token,
                                time_per_output_token=time_per_output_token,
                                success_rate=success_rate)

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField]] = None):
        self.update_command()
        super().before_run(run_params)
        Path(self.benchmark_config.command.result_dir).mkdir(parents=True, exist_ok=True)
        if VLLM_CUSTOM_OUTPUT not in os.environ:
            os.environ[VLLM_CUSTOM_OUTPUT] = str(self.benchmark_config.command.result_dir)
        _var_name = f"${VLLM_CUSTOM_OUTPUT}"
        if _var_name in self.command:
            _i = self.command.index(_var_name)
            self.command[_i] = str(self.benchmark_config.command.result_dir)
