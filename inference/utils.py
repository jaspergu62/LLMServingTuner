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
from functools import reduce, lru_cache
from typing import Tuple

import numpy as np
import pandas as pd

from msserviceprofiler.modelevalstate.inference.common import (
    OP_EXPECTED_FIELD_MAPPING,
    model_op_size,
    HistInfo,
    get_field_bins_count
)
from msserviceprofiler.modelevalstate.inference.constant import (
    ALL_OP,
    OP_EXECUTE_DELTA_FIELD,
    DTYPE_CATEGORY,
    UNDEFINED,
    ALL_ARCHITECTURE,
    ALL_ARCHITECTURE_MAPPING
)

OUTPUT_LENGTH_FIELD = "output_length"
TOTAL_OUTPUT_LENGTH = "total_output_length"
TOTAL_SEQ_LENGTH = "total_seq_length"
TOTAL_PREFILL_TOKEN = "total_prefill_token"


class PreprocessTool:

    @staticmethod
    @lru_cache(maxsize=32)
    def generate_data(row, column) -> Tuple:
        try:
            new_row = [float(i) for i in row]
        except ValueError:
            new_row = []
            for i in row:
                try:
                    new_row.append(float(i))
                except ValueError:
                    new_row.append(i)
        return tuple(new_row), column

    @staticmethod
    @lru_cache(maxsize=32)
    def generate_data_with_request_info(row, column) -> Tuple:
        new_index = []
        new_row = []
        for _index in column:
            if _index == OUTPUT_LENGTH_FIELD:
                new_index.append(TOTAL_OUTPUT_LENGTH)
                new_row.append(sum([int(getattr(k, OUTPUT_LENGTH_FIELD)) for k in row]))
            hist_index = getattr(HistInfo, _index)
            new_index.extend(hist_index["label"])
            _hist_value = get_field_bins_count(row, _index, hist_index["bins"])
            # check value
            for _k in _hist_value:
                if _k < 0:
                    raise ValueError(f"Unexpected value found. label: {hist_index}, value:{_hist_value}")
            new_row.extend(_hist_value)
        return tuple(new_row), tuple(new_index)

    @staticmethod
    @lru_cache(maxsize=32)
    def generate_data_with_request_info_by_df(row, column) -> Tuple:
        row_df = pd.concat([pd.Series([float(i) for i in _row], index=column) for _row in row], axis=1).T
        new_row = [row_df[OUTPUT_LENGTH_FIELD].sum()]
        new_index = [TOTAL_OUTPUT_LENGTH]
        for _index in column:
            hist_index = getattr(HistInfo, _index)
            new_index.extend(hist_index["label"])
            _value = row_df[_index].values
            if not _value:
                _hist_value = [0 for _ in range(len(hist_index["bins"]) - 1)]
            else:
                _hist_value, _ = np.histogram(_value, hist_index["bins"])
                # check value
                _check = [True for _k in _hist_value if _k < 0]
                if any(_check):
                    raise ValueError(f"Unexpected value found. label: {hist_index}, value:{_hist_value}")
            new_row.extend(_hist_value)
        return tuple(new_row), tuple(new_index)

    @staticmethod
    @lru_cache(maxsize=32)
    def get_all_op_execute_delta(input_data, input_index, field="execute_delta"):
        _op_count = {}
        _op_delta = {}  # op_name: [[param1size, param2size,],[第二次调用param1size,第二次调用param2size]]
        for _, _op_info in enumerate(input_data):
            _tmp_op_name = _op_info[input_index.index("op_name")]
            _tmp_op_count = int(_op_info[input_index.index("call_count")])
            if _tmp_op_name not in _op_count:
                _op_count[_tmp_op_name] = [_tmp_op_count]
            else:
                _op_count[_tmp_op_name].append(_tmp_op_count)
            _tmp_op_delta = float(_op_info[input_index.index(field)])
            if _tmp_op_name in _op_delta:
                _op_delta[_tmp_op_name].append(_tmp_op_delta)
            else:
                _op_delta[_tmp_op_name] = [_tmp_op_delta]
        # 获取期望
        _op_delta_expected = {}
        for _op, _op_execute_deltas in _op_delta.items():
            total_count = sum(_op_count[_op])
            _tmp_expected = 0
            for i, _op_execute_delta in enumerate(_op_execute_deltas):
                _cur_count = _op_count[_op][i]
                _tmp_expected += _cur_count / total_count * _op_execute_delta
            _op_delta_expected[_op] = _tmp_expected
        return _op_delta_expected

    @staticmethod
    @lru_cache(maxsize=32)
    def get_all_op_input_expected(input_data, input_index, field: str = "input_shape"):
        _op_count = {}
        _op_param_size = {}  # op_name: [[param1size, param2size,],[第二次调用param1size,第二次调用param2size]]
        for _, _op_info in enumerate(input_data):
            _tmp_op_name = _op_info[input_index.index("op_name")]
            _tmp_op_count = int(_op_info[input_index.index("call_count")])
            if _tmp_op_name not in _op_count:
                _op_count[_tmp_op_name] = [_tmp_op_count]
            else:
                _op_count[_tmp_op_name].append(_tmp_op_count)
            _tmp_op_input_shape = _op_info[input_index.index(field)].split(";")
            _tmp_op_input_shape = [[int(_dim) for _dim in _shape.split(",")] for _shape in _tmp_op_input_shape]
            _tmp_op_input_size = [reduce(lambda x, y=1: x * y, _shape) for _shape in _tmp_op_input_shape]
            if _tmp_op_name in _op_param_size:
                _op_param_size[_tmp_op_name].append(_tmp_op_input_size)
            else:
                _op_param_size[_tmp_op_name] = [_tmp_op_input_size]

        # 获取期望
        _op_param_expected = {}
        for _op, _op_param in _op_param_size.items():
            total_count = sum(_op_count[_op])
            _tmp_expected = {}
            for i, _param_i in enumerate(_op_param):
                _cur_count = _op_count[_op][i]
                for j, _param_j in enumerate(_param_i):
                    if j in _tmp_expected:
                        _tmp_expected[j] += _cur_count / total_count * _param_j
                    else:
                        _tmp_expected[j] = _cur_count / total_count * _param_j
            _op_param_expected[_op] = _tmp_expected
        return _op_param_expected

    @staticmethod
    @lru_cache(maxsize=32)
    def get_op_in_origin_row_index(input_data, input_index):
        _op_in_origin_row_index = {}  # 找到每个op和它在origin_row的索引
        for i, _op_info in enumerate(input_data):
            for j, _op_index in enumerate(input_index):
                if _op_index == "op_name":
                    if _op_info[j] not in ALL_OP:
                        raise ValueError(f"Not Found {_op_info[j]}. please update ALL_OP.")
                    if _op_info[j] in _op_in_origin_row_index:
                        _op_in_origin_row_index[_op_info[j]].append(i)
                    else:
                        _op_in_origin_row_index[_op_info[j]] = [i]
                    break

        return _op_in_origin_row_index

    @staticmethod
    @lru_cache(maxsize=32)
    def generate_data_with_op_info(origin_row, origin_index) -> Tuple:
        new_index = []
        new_row = []
        _op_in_origin_row_index = PreprocessTool.get_op_in_origin_row_index(origin_row, origin_index)
        _op_input_param_expected = PreprocessTool.get_all_op_input_expected(origin_row, origin_index, "input_shape")
        _op_output_expected = PreprocessTool.get_all_op_input_expected(origin_row, origin_index, "output_shape")
        _op_delta_expected = {}
        for _field in OP_EXECUTE_DELTA_FIELD:
            _op_delta_expected[_field] = PreprocessTool.get_all_op_execute_delta(origin_row, origin_index, _field)

        # 计算期望来生成数据
        for _op in ALL_OP:
            _cur_op_default = OP_EXPECTED_FIELD_MAPPING[_op]
            if _op not in _op_in_origin_row_index:
                for k, v in _cur_op_default.items():
                    new_index.append(k)
                    new_row.append(v)
            else:
                for k, v in _cur_op_default.items():
                    new_index.append(k)
                    _op_index_on_origin_rows = _op_in_origin_row_index[_op]
                    if "op_name" in k:
                        new_row.append(1)  # 使用该算子了，置为1
                    elif "call_count" in k:
                        _op_call_count = sum(
                            [int(origin_row[_i][origin_index.index("call_count")]) for _i in _op_index_on_origin_rows])
                        new_row.append(_op_call_count)
                    elif "input_dtype" in k:
                        # 根据每种dtype 类型，记录所有次数
                        for _dtype in DTYPE_CATEGORY:
                            if _dtype in k:
                                _cur_dtype_count = 0
                                for _i in _op_index_on_origin_rows:
                                    _cur_dtype_count += origin_row[_i][origin_index.index("input_dtype")].split(
                                        ";").count(_dtype)
                                new_row.append(_cur_dtype_count)
                                break
                    elif "input_size" in k:
                        # 计算期望
                        _op_param_index = int(k.split("__")[-1])
                        new_row.append(_op_input_param_expected[_op].get(_op_param_index, v))
                    elif "output_dtype" in k:
                        for _dtype in DTYPE_CATEGORY:
                            if _dtype in k:
                                _cur_dtype_count = 0
                                for _i in _op_index_on_origin_rows:
                                    _cur_dtype_count += origin_row[_i][origin_index.index("output_dtype")].split(
                                        ";").count(_dtype)
                                new_row.append(_cur_dtype_count)
                                break
                    elif "output_size" in k:
                        _op_param_index = int(k.split("__")[-1])
                        new_row.append(_op_output_expected[_op].get(_op_param_index, v))
                    else:
                        for _field in OP_EXECUTE_DELTA_FIELD:
                            if _field in k:
                                new_row.append(_op_delta_expected[_field][_op])

        return tuple(new_row), tuple(new_index)

    @staticmethod
    @lru_cache(maxsize=32)
    def get_all_op_input_ratio(input_data: Tuple[Tuple], input_index: Tuple, field: str = "input_shape"):
        # 计算该op的字段，在所有采集到的op的字段中的占比。
        _sum = 0
        _op_count = {}
        _op_param_size = {}  # op_name: [[param1size, param2size,],[第二次调用param1size,第二次调用param2size]]
        for _, _op_info in enumerate(input_data):
            _tmp_op_name = _op_info[input_index.index("op_name")]
            _tmp_op_count = int(_op_info[input_index.index("call_count")])
            if _tmp_op_name not in _op_count:
                _op_count[_tmp_op_name] = [_tmp_op_count]
            else:
                _op_count[_tmp_op_name].append(_tmp_op_count)
            _tmp_op_input_shape = _op_info[input_index.index(field)].split(";")
            _tmp_op_input_shape = [[int(_dim) for _dim in _shape.split(",")] for _shape in _tmp_op_input_shape]
            _tmp_op_input_size = [reduce(lambda x, y=1: x * y, _shape) for _shape in _tmp_op_input_shape]
            if _tmp_op_name in _op_param_size:
                _op_param_size[_tmp_op_name].append(_tmp_op_input_size)
            else:
                _op_param_size[_tmp_op_name] = [_tmp_op_input_size]
            _sum += sum(_tmp_op_input_size)

        # 获取ratio
        _op_param_size_ratio = {}  # op_name: 0:[ratio1, ratio2,ratio3]
        for _op, _op_delta in _op_param_size.items():
            _tmp_ratio = {}
            for i, _param_i in enumerate(_op_delta):
                _cur_count = _op_count[_op][i]
                for j, _param_j in enumerate(_param_i):
                    _cur_ratio = [_param_j / _sum * 100] * _cur_count
                    if j in _tmp_ratio:
                        _tmp_ratio[j].extend(_cur_ratio)
                    else:
                        _tmp_ratio[j] = _cur_ratio
            _op_param_size_ratio[_op] = _tmp_ratio
        return _op_param_size_ratio

    @staticmethod
    @lru_cache(maxsize=32)
    def get_all_op_execute_delta_ratio(input_data: Tuple[Tuple], input_index: Tuple, field="execute_delta"):
        _sum = 0
        _op_count = {}
        _op_delta = {}  # op_name: [[第一次执行时间，第二次执行时间]]
        for _, _op_info in enumerate(input_data):
            _tmp_op_name = _op_info[input_index.index("op_name")]
            _tmp_op_count = int(_op_info[input_index.index("call_count")])
            if _tmp_op_name not in _op_count:
                _op_count[_tmp_op_name] = [_tmp_op_count]
            else:
                _op_count[_tmp_op_name].append(_tmp_op_count)
            _tmp_op_delta = float(_op_info[input_index.index(field)])
            if _tmp_op_name in _op_delta:
                _op_delta[_tmp_op_name].append(_tmp_op_delta)
            else:
                _op_delta[_tmp_op_name] = [_tmp_op_delta]
            _sum += _tmp_op_delta

        # 获取ratio
        _op_delta_ratio = {}
        for _op, _op_execute_deltas in _op_delta.items():
            _tmp_ratio = []
            for i, _op_delta in enumerate(_op_execute_deltas):
                _cur_count = _op_count[_op][i]
                _tmp_ratio.extend([_op_delta / _sum] * _cur_count)
            _op_delta_ratio[_op] = _tmp_ratio

        return _op_delta_ratio

    @staticmethod
    @lru_cache(maxsize=32)
    def get_label_hist_value(input_ratio):
        _op_input_param_hist_ratio = {}
        for _tmp_op, _tmp_op_value in input_ratio.items():
            _tmp_hist_ratio = {}
            for _input_index, _input_value in _tmp_op_value.items():
                hist, _ = np.histogram(_input_value, model_op_size["bins"])
                for i, _label in enumerate(model_op_size["label"]):
                    _tmp_hist_ratio[f"{_input_index}__{_label}"] = hist[i]
            _op_input_param_hist_ratio[_tmp_op] = _tmp_hist_ratio
        return _op_input_param_hist_ratio

    @staticmethod
    @lru_cache(maxsize=32)
    def generate_data_with_op_info_use_ratio(origin_row, origin_index) -> Tuple:
        new_index = []
        new_row = []
        _op_in_origin_row_index = PreprocessTool.get_op_in_origin_row_index(origin_row, origin_index)
        _op_input_param_ratio = PreprocessTool.get_all_op_input_ratio(origin_row, origin_index, "input_shape")
        _op_output_ratio = PreprocessTool.get_all_op_input_ratio(origin_row, origin_index, "output_shape")
        _op_delta_ratio = PreprocessTool.get_all_op_execute_delta_ratio(origin_row, origin_index, "execute_delta")
        _op_input_param_hist_ratio = PreprocessTool.get_label_hist_value(_op_input_param_ratio)
        _op_output_hist_ratio = PreprocessTool.get_label_hist_value(_op_output_ratio)
        _op_delta_hist_ratio = PreprocessTool.get_label_hist_value(_op_delta_ratio)

        # 生成数据
        for _op in ALL_OP:
            _cur_op_default = OP_EXPECTED_FIELD_MAPPING[_op]
            if _op not in _op_in_origin_row_index:
                for k, v in _cur_op_default.items():
                    new_index.append(k)
                    new_row.append(v)
            else:
                for k, _ in _cur_op_default.items():
                    new_index.append(k)
                    _op_index_on_origin_rows = _op_in_origin_row_index[_op]
                    if "op_name" in k:
                        new_row.append(1)  # 使用该算子了，置为1
                    elif "input_dtype" in k:
                        # 根据每种dtype 类型，记录所有次数
                        for _dtype in DTYPE_CATEGORY:
                            if _dtype in k:
                                _cur_dtype_count = 0
                                for _i in _op_index_on_origin_rows:
                                    _cur_dtype_count += origin_row[_i][origin_index.index("input_dtype")].split(
                                        ";").count(_dtype)
                                new_row.append(_cur_dtype_count)
                                break
                    elif "input_size" in k:
                        # 计算期望
                        _op_param_index = int(k.split("__")[-1])
                        ratio_key = k.split("__")[-3:].join("__")
                        new_row.append(_op_input_param_hist_ratio[_op].get(f"{_op_param_index}__{ratio_key}", 0))
                    elif "output_dtype" in k:
                        for _dtype in DTYPE_CATEGORY:
                            if _dtype in k:
                                _cur_dtype_count = 0
                                for _i in _op_index_on_origin_rows:
                                    _cur_dtype_count += origin_row[_i][origin_index.index("output_dtype")].split(
                                        ";").count(_dtype)
                                new_row.append(_cur_dtype_count)
                                break
                    elif "output_size" in k:
                        _op_param_index = int(k.split("__")[-1])
                        ratio_key = k.split("__")[-3:].join("__")
                        new_row.append(_op_output_hist_ratio[_op].get(f"{_op_param_index}__{ratio_key}", 0))
                    elif "execute_delta" in k:
                        _op_param_index = int(k.split("__")[-1])
                        ratio_key = k.split("__")[-3:].join("__")
                        new_row.append(_op_delta_hist_ratio[_op].get(f"{_op_param_index}__{ratio_key}", 0))

        return tuple(new_row), tuple(new_index)

    @staticmethod
    @lru_cache(maxsize=32)
    def generate_data_with_struct_info(origin_row, origin_index) -> Tuple:
        new_row = []
        for i, _value in enumerate(origin_row):
            if "rate" in origin_index[i]:
                # 使用百分之显示
                new_row.append(float(_value) * 100)
            else:
                new_row.append(float(_value))
        return new_row, origin_index

    @staticmethod
    def generate_data_with_model_config(origin_row, origin_index) -> Tuple:
        quantization_prefix_config = "quantization_config"
        default_field = ["kv_quant_type", "group_size", "reduce_quant_type"]
        default_value = (UNDEFINED, UNDEFINED, UNDEFINED)
        new_index = []
        new_row = []
        for i, v in enumerate(origin_index):
            if v == quantization_prefix_config:
                if origin_row[i]:
                    for k, w in enumerate(default_field):
                        new_index.append(w)
                        _tmp_value = origin_row[i].get(w, default_value[k])
                        if not _tmp_value:
                            _tmp_value = default_value[k]
                        new_row.append(_tmp_value)
                else:
                    new_row.extend(default_value)
                    new_index.extend(default_field)
            elif v == "architectures":
                for _arch in ALL_ARCHITECTURE:
                    new_index.append(ALL_ARCHITECTURE_MAPPING.get(_arch.lower(), f"architecture__{_arch.lower()}"))
                    new_row.append([k.lower() for k in origin_row[i]].count(_arch))
            elif v == "quantize":
                new_index.append(v)
                if origin_row[i]:
                    new_row.append(origin_row[i])
                else:
                    new_row.append(UNDEFINED)

            else:
                new_index.append(v)
                try:
                    new_row.append(float(origin_row[i]))
                except ValueError:
                    new_row.append(origin_row[i])

        return tuple(new_row), tuple(new_index)
