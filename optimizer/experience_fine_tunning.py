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
from itertools import cycle
from math import isinf, isnan
from typing import Optional, Tuple

import numpy as np

from msserviceprofiler.modelevalstate.config.config import default_support_field, PerformanceIndex, \
    map_param_with_value


class StopFineTune(Exception):
    pass


class FineTune:
    def __init__(self, ttft_penalty: float = 0, tpot_penalty: float = 0, target_field: Optional[Tuple] = None,
                 ttft_slo: float = 0.5, tpot_slo: float = 0.05, slo_coefficient: float = 0.1, step_size: float = 0.5):
        self.ttft_penalty = ttft_penalty  # 优化算法中惩罚系数
        self.tpot_penalty = tpot_penalty
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self.slo_coefficient = slo_coefficient
        self.target_field = target_field if target_field else default_support_field
        self.fine_tune_target = ["REQUESTRATE"]
        self.fine_tune_type = cycle(self.fine_tune_target)
        self.step_size = step_size
        self.ttft_lower_bound = self.ttft_slo * (1 - self.slo_coefficient)
        self.ttft_upper_bound = self.ttft_slo * (1 + self.slo_coefficient)
        self.tpot_lower_bound = self.tpot_slo * (1 - self.slo_coefficient)
        self.tpot_upper_bound = self.tpot_slo * (1 + self.slo_coefficient)
        if self.ttft_penalty == 0 and self.tpot_penalty == 0:
            raise StopFineTune("No penalties, no need to fine-tune.")
        ttft_flag = self.ttft_penalty != 0 and self.ttft_slo == 0
        tpot_flag = self.tpot_penalty != 0 and self.tpot_slo == 0
        if ttft_flag or tpot_flag:
            raise ValueError("Penalty is set but SLO is zero.")

    @staticmethod
    def update_field(simulate_run_info, signed_factor, field_names=("REQUESTRATE",)) -> bool:
        if signed_factor == 0 or isinf(signed_factor) or isnan(signed_factor):
            return False
        for _field in simulate_run_info:
            if _field.name.upper().strip() in field_names:
                if _field.min == _field.max:
                    return False
                original_value = _field.value
                _new_value = _field.value * (1 + signed_factor)
                if isinf(_new_value) or isnan(_new_value):
                    return False
                _field.value = _new_value
                _new_value = max(_field.min, min(_field.max, _field.value))
                if isinf(_new_value) or isnan(_new_value):
                    _field.value = original_value
                    return False
                _field.value = _new_value
                # 检查值是否发生了有影响的变化(>=0.1)
                return abs(_field.value - original_value) >= 0.1
        return False
    
    def check_config_and_performance(self, performance_index: PerformanceIndex):
 
        if self.ttft_penalty == 0 and self.tpot_penalty == 0:
            raise StopFineTune("No penalties, no need to fine-tune.")
        ttft_flag = self.ttft_penalty != 0 and self.ttft_slo == 0
        tpot_flag = self.tpot_penalty != 0 and self.tpot_slo == 0
        if ttft_flag or tpot_flag:
            raise ValueError("Penalty is set but SLO is zero.")
        if performance_index.time_per_output_token is None:
            raise ValueError("Missing performance data for TPOT.")
        if self.ttft_penalty != 0 and performance_index.time_to_first_token is None:
            raise ValueError("Missing performance data for TTFT.")
    
    def fine_tune_with_concurrency(self, params: np.ndarray, performance_index: PerformanceIndex):
        # request rate 设为固定值，并发逐渐减小
        self.check_config_and_performance(performance_index)
        actual_tpot = performance_index.time_per_output_token
        actual_ttft = performance_index.time_to_first_token
        ttft_over_slo = False
        ttft_under_lower_bound = False
        tpot_over_slo = actual_tpot > self.tpot_upper_bound
        tpot_under_lower_bound = actual_tpot < self.tpot_lower_bound
        # 同时约束ttft
        if self.ttft_penalty != 0:
            ttft_over_slo = actual_ttft > self.ttft_upper_bound
            ttft_under_lower_bound = actual_ttft < self.ttft_lower_bound
        simulate_run_info = map_param_with_value(params, self.target_field)
 
        for _field in simulate_run_info:
            if _field.name.strip().upper() == "REQUESTRATE":
                _field.value = 0
        for _field in simulate_run_info:
            if _field.name.strip().upper() in ("CONCURRENCY", "MAXCONCURRENCY"):
                _new_value = None
                if ttft_over_slo or tpot_over_slo:
                    _new_value = _field.value / 2
                elif (self.ttft_penalty == 0 or ttft_under_lower_bound) and tpot_under_lower_bound:
                    _new_value = _field.value * 2
                else:
                    raise StopFineTune
                if _field.min < _new_value < _field.max:
                    _field.value = _new_value
                    return simulate_run_info
                elif _new_value < _field.min != _field.value:
                    _field.value = _field.min
                    return simulate_run_info
                elif _new_value > _field.max != _field.value:
                    _field.value = _field.max
                    return simulate_run_info
                else:
                    raise StopFineTune
        raise StopFineTune