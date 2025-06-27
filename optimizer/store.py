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
import csv
import shlex
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
from loguru import logger

from msserviceprofiler.modelevalstate.config.config import (
    BenchMarkConfig,
    DataStorageConfig,
    RUN_TIME,
    PerformanceIndex,
    OptimizerConfigField
)
from msserviceprofiler.msguard.security.io import read_csv_s



class DataStorage:
    def __init__(self, config: DataStorageConfig):
        self.config = config
        if not self.config.store_dir.exists():
            self.config.store_dir.mkdir(parents=True)
        self.save_file = self.config.store_dir.joinpath(f"data_storage_{RUN_TIME}.csv")

    @staticmethod
    def load_history_position(load_dir: Path) -> Optional[List]:
        if not load_dir.exists():
            raise FileNotFoundError(f"file: {load_dir}")
        if not load_dir.is_dir():
            raise ValueError(f"Expect a directory, not a file.")
        history_data = []
        for file in sorted([f for f in load_dir.iterdir() if f.is_file()], key=lambda x: x.stat().st_ctime):
            if file.name.startswith("data_storage") and file.suffix == ".csv":
                data = read_csv_s(file).to_dict(orient="records")
                history_data.extend(data)
        if not history_data:
            return None
        return history_data

    def save(self, performance_index: PerformanceIndex, params: Tuple[OptimizerConfigField],
             bench_mark_config: BenchMarkConfig, **kwargs):
        from msserviceprofiler.msguard.security import open_s, sanitize_csv_value
        logger.info("Save result with DataStorage.")
        _column = []
        _value = []
        for k, v in performance_index.model_dump().items():
            _column.append(k)
            _value.append(v)
        for _p in params:
            _column.append(_p.name)
            _value.append(_p.value)
        benchmark_param = shlex.split(bench_mark_config.command)[2:]
        for i in range(0, len(benchmark_param), 2):
            if (i + 1) < len(benchmark_param):
                _column.append(benchmark_param[i].strip("--"))
                _value.append(benchmark_param[i + 1])
            else:
                logger.warning(f"IndexError. index: {i + 1}, list: {benchmark_param}")
        for k, v in kwargs.items():
            _column.append(k)
            _value.append(v)
        if self.save_file.exists():
            with open_s(self.save_file, "a+") as f:
                data_writer = csv.writer(f)
                data_writer.writerow([sanitize_csv_value(_v) for _v in _value])
        else:
            with open_s(self.save_file, "w") as f:
                data_writer = csv.writer(f)
                data_writer.writerow(_column)
                data_writer.writerow([sanitize_csv_value(_v) for _v in _value])
