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
import os
import stat
from pathlib import Path

import numpy as np

from msserviceprofiler.modelevalstate.config.config import settings


class BatchStage:
    PREFILL = "prefill"
    DECODE = "decode"


class Collect:
    batch_file = Path(settings.benchmark.custom_collect_output_path).joinpath(f"batch_info_{os.getpid()}.csv")
    requests_file = Path(settings.benchmark.custom_collect_output_path).joinpath(f"request_info_{os.getpid()}.csv")
    total_req_input_len = []
    req_info = []

    @staticmethod
    def generate_request(plugin_object, input_metadata, cached_idx):
        output_len_count = plugin_object.input_manager.cache.output_len_count[cached_idx]
        all_input_length = input_metadata.batch_seq_len
        all_need_blocks = np.count_nonzero(input_metadata.block_tables > -1, axis=-1)
        _total_req_input_len = []
        all_request_field = []
        req_info = []
        for _cache_id_index, _cache_id in enumerate(cached_idx):
            _req_input_len = all_input_length[_cache_id_index] - output_len_count[_cache_id_index]
            _total_req_input_len.append(_req_input_len)
            req_info.append(input_metadata.batch_request_ids[_cache_id_index])
            req_info.append(output_len_count[_cache_id_index])
            request_field = (
                input_metadata.batch_request_ids[_cache_id_index], _req_input_len, all_need_blocks[_cache_id_index],
                output_len_count[_cache_id_index])
            all_request_field.append(request_field)
        Collect.total_req_input_len = _total_req_input_len
        Collect.req_info = req_info
        return all_request_field

    @staticmethod
    def generate_batch(input_metadata, preprocess_time, forward_time,
                       sample_time, postprocess_time, execute_time):
        if input_metadata.is_prefill:
            batch_stage = BatchStage.PREFILL
        else:
            batch_stage = BatchStage.DECODE
        batch_field = (batch_stage, input_metadata.batch_size,
                       np.count_nonzero(input_metadata.block_tables > -1, axis=-1).sum(),
                       sum(Collect.total_req_input_len), max(Collect.total_req_input_len),
                       Collect.req_info, preprocess_time, forward_time,
                       sample_time, postprocess_time, execute_time)
        return batch_field

    @staticmethod
    def save(request_field, file_name, mul_row=True):
        if not Path(file_name).parent.exists():
            Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        modes = stat.S_IWUSR | stat.S_IRUSR
        if mul_row:
            with os.fdopen(os.open(file_name, flags, modes, ), 'a', newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(request_field)
        else:
            with os.fdopen(os.open(file_name, flags, modes, ), 'a', newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(request_field)
