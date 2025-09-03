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
import subprocess
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from loguru import logger

from msserviceprofiler.modelevalstate.common import get_train_sub_path
from msserviceprofiler.modelevalstate.config.config import PerformanceIndex, OptimizerConfigField, \
    map_param_with_value, CommunicationConfig
from msserviceprofiler.modelevalstate.config.base_config import FOLDER_LIMIT_SIZE
from msserviceprofiler.modelevalstate.optimizer.communication import CommunicationForFile, CustomCommand
from msserviceprofiler.modelevalstate.optimizer.simulator import Simulator
from msserviceprofiler.modelevalstate.optimizer.store import DataStorage
from msserviceprofiler.modelevalstate.optimizer.utils import get_folder_size


class Scheduler:
    def __init__(self, simulator: Simulator, benchmark, data_storage: DataStorage,
                 bak_path: Optional[Path] = None, retry_number: int = 3, wait_start_time=1800):
        self.simulator = simulator
        self.benchmark = benchmark
        self.data_storage = data_storage
        self.bak_path = bak_path
        self.retry_number = retry_number
        self.wait_time = wait_start_time
        self.current_back_path = None
        self.simulate_run_info = None
        self.performance_index = None
        self.error_info = None
        self.run_start_timestamp = None
        self.del_log = None

    def set_back_up_path(self):
        if self.bak_path:
            if get_folder_size(self.bak_path) > FOLDER_LIMIT_SIZE:
                self.simulator.bak_path = None
                self.benchmark.bak_path = None
            else:
                self.current_back_path = get_train_sub_path(self.bak_path)
                self.simulator.bak_path = self.current_back_path
                self.benchmark.bak_path = self.current_back_path

    def wait_simulate(self):
        logger.info("wait run simulator")
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

    def backup(self):
        self.simulator.backup()
        self.benchmark.backup()

    def monitoring_status(self):
        logger.info("monitor status")
        while True:
            if self.simulator.process.poll() is not None:
                raise subprocess.SubprocessError(f"Failed in run simulator. "
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
                logger.error(f"Failed in simulator Running. error: {e}，\n"
                             f"simulator log {self.simulator.run_log}. \n"
                             f"log last info \n{self.simulator.get_last_log()}")
                self.stop_target_server(False)
                continue
            time.sleep(1)
            try:
                self.benchmark.run(tuple(self.simulate_run_info))
            except Exception as e:
                logger.error(f"Failed in Benchmark Running. error: {e},\n"
                             f"benchmark log {self.benchmark.run_log},\n"
                             f"log last info \n{self.benchmark.get_last_log()}")
                self.stop_target_server(False)
                continue
            time.sleep(1)
            try:
                self.monitoring_status()
            except Exception as e:
                self.stop_target_server(False)
                logger.error(f"Failed in monitoring status. error: {e}, \n"
                             f"simulator log {self.simulator.run_log}, \n"
                             f"log last info {self.simulator.get_last_log()}.\n"
                             f"benchmark log {self.benchmark.run_log}, \n"
                             f"log last info \n{self.benchmark.get_last_log()}.")
                continue
            return
        raise ValueError(
            f"Failed in run_target_server")

    def stop_target_server(self, del_log: bool = False):
        self.simulator.stop(del_log)
        self.benchmark.stop(del_log)

    def save_result(self, **kwargs):
        duration = None
        if self.run_start_timestamp:
            duration = time.time() - self.run_start_timestamp
        self.data_storage.save(self.performance_index, tuple(self.simulate_run_info),
                               error=self.error_info, bakcup=self.current_back_path, duration=duration,
                               **kwargs)
        if self.bak_path:
            self.backup()
        self.stop_target_server()

    def run(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]) -> PerformanceIndex:
        """
        1. 启动mindie仿真
        2. 启动benchmark 测试
        3. 获取benchmark测试结果
        4. 关闭mindie仿真
        5. 返回benchmark测试结果
        params: 是一维数组，其值对应mindie 的相关配置。
        """
        self.run_start_timestamp = time.time()
        logger.info("Start run in scheduler.")
        self.set_back_up_path()
        self.simulate_run_info = map_param_with_value(params, params_field)
        logger.info("run param info {}", {v.name: v.value for v in self.simulate_run_info})
        self.error_info = None
        self.del_log = True
        self.performance_index = PerformanceIndex()
        try:
            self.run_target_server(params, params_field)
            time.sleep(1)
            self.performance_index = self.benchmark.get_performance_index()
        except Exception as e:
            logger.error(f"Failed running. bak path: {self.simulator.bak_path}. error {e}")
            self.error_info = e
            self.del_log = False
        return self.performance_index

    def run_with_request_rate(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]) -> PerformanceIndex:
        """
        运行服务，先运行最大并发，获取request rate，然后再根据并发和request rate运行，最后一组作为评估结果
        params: 是一维数组，其值对应mindie 的相关配置。
        """
        self.run_start_timestamp = time.time()
        logger.info("Start run in scheduler.")
        self.set_back_up_path()
        self.simulate_run_info = map_param_with_value(params, params_field)
        logger.info("run param info {}", {v.name: v.value for v in self.simulate_run_info})
        self.error_info = None
        self.del_log = True
        self.performance_index = PerformanceIndex()
        try:
            self.run_target_server(params, params_field)
            time.sleep(1)
            self.performance_index = self.benchmark.get_performance_index()
            for _field in self.simulate_run_info:
                if _field.name == "REQUESTRATE":
                    _field.min = _field.max = _field.value = int(self.performance_index.throughput) + 1
            self.benchmark.update_command()
            try:
                self.benchmark.prepare()
                self.benchmark.run(tuple(self.simulate_run_info))
            except Exception as e:
                logger.error(f"Failed in Benchmark Running. error: {e}, benchmark log {self.benchmark.run_log}")
                raise e
            try:
                self.monitoring_status()
            except Exception as e:
                self.stop_target_server(False)
                logger.error(f"Failed in monitoring status. error: {e}, simulator log {self.simulator.run_log}, "
                             f"benchmark log {self.benchmark.run_log}")
                raise e
            time.sleep(1)
            self.performance_index = self.benchmark.get_performance_index()
        except Exception as e:
            logger.error(f"Failed running. bak path: {self.simulator.bak_path}")
            self.error_info = e
            self.del_log = False
        return self.performance_index


class ScheduleWithMultiMachine(Scheduler):
    def __init__(self, communication_config: CommunicationConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.communication_config = communication_config
        self.communication = CommunicationForFile(self.communication_config.cmd_file,
                                                  self.communication_config.res_file,
                                                  )
        self.cmd = CustomCommand()
        _cmd = self.cmd.init
        self.communication.send_command(_cmd)
        self.communication.clear_command(_cmd)

    def set_back_up_path(self):
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
                self.communication.clear_command(_cmd)

    def monitoring_status(self):
        logger.info("Start monitoring")
        while True:
            _cmd = self.cmd.process_poll
            self.communication.send_command(_cmd)
            all_poll = [self.simulator.process.poll(), self.communication.clear_command(_cmd)]
            if any([_i is not None for _i in all_poll]):
                self.stop_target_server(del_log=False)
                raise subprocess.SubprocessError(
                    f"Failed in run simulator. all status: {all_poll}.")
            if self.benchmark.check_success():
                return
            time.sleep(1)

    def run_simulate(self, params: np.ndarray, params_field: Tuple[OptimizerConfigField]):
        self.benchmark.prepare()
        _cmd = f"{self.cmd.start} params:{params.tolist()}"
        self.cmd.history = _cmd
        self.communication.send_command(_cmd)
        self.communication.clear_command(_cmd)
        self.simulator.run(tuple(self.simulate_run_info))
        self.wait_simulate()
        # wait 其他服务器上的服务成功。
        _cmd = self.cmd.check_success
        self.cmd.history = _cmd
        self.communication.send_command(_cmd)
        self.communication.clear_command(_cmd)

    def stop_target_server(self, del_log: bool = True):
        super(ScheduleWithMultiMachine, self).stop_target_server(del_log)
        # wait 其他服务器上的服务成功。
        _cmd = f"{self.cmd.stop} params:{del_log}"
        self.communication.send_command(_cmd)
        self.communication.clear_command(_cmd)
        self.cmd.history = _cmd
