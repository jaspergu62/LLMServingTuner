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
import sys
import time
from typing import Tuple
from pathlib import Path

import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler

import numpy as np
from loguru import logger

from msserviceprofiler.modelevalstate.optimizer.optimizer import Simulator, remove_file
from msserviceprofiler.modelevalstate.config.config import settings, map_param_with_value


def get_file(target_path, parent_name: str = "", save_current_path: bool = False):
    # 获取该目录下所有文件,或者这个文件
    _work_dir = Path(target_path)
    if not _work_dir.exists():
        raise FileNotFoundError
    res = []

    if _work_dir.is_file():
        _file_name = parent_name + "/" + _work_dir.name if parent_name else _work_dir.name
        with open(_work_dir, "rb") as handle:
            res.append((_file_name, xmlrpc.client.Binary(handle.read())))
    else:
        if save_current_path:
            parent_name = parent_name + "/" + _work_dir.name if parent_name else _work_dir.name
        for child in _work_dir.iterdir():
            res.extend(get_file(child, parent_name, True))
    return res


# 限制为特定的路径。
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


class RemoteScheduler:
    def __init__(self):
        self.simulator = None

    def run_simulator(self, params: np.ndarray):
        # 更新服务器上的config文件
        self.simulator = Simulator(settings.mindie)
        _simulate_run_info = map_param_with_value(params, settings.target_field)
        logger.info(f"simulate run info {_simulate_run_info}")
        self.simulator.run(tuple(_simulate_run_info))

    def check_success(self):
        if not self.simulator:
            return None
        for _ in range(10):
            if self.simulator.check_success():
                return True
            time.sleep(10)
        raise Exception(f"Simulator run failed. please check log: {self.simulator.mindie_log}")

    def stop_simulator(self, del_log=True):
        if self.simulator:
            self.simulator.stop(del_log)

    def process_poll(self):
        if self.simulator:
            return self.simulator.process.poll()
        return None


