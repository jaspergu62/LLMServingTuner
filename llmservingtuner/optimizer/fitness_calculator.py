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

"""
Fitness calculator for evaluating LLM serving configurations.
Computes a weighted cost function based on throughput, latency metrics, and SLO compliance.
"""

from math import exp, inf

from loguru import logger

from llmservingtuner.config.config import PerformanceIndex


class FitnessCalculator:
    """
    Calculate fitness scores for LLM serving configurations.

    The fitness function is a weighted sum of normalized costs for:
    - Generation speed (throughput)
    - Time to first token (TTFT)
    - Time per output token (TPOT)
    - Success rate

    Lower fitness values indicate better configurations.
    """

    # Default weights (must sum to 1.0)
    DEFAULT_W_GEN = 0.4
    DEFAULT_W_TTFT = 0.2
    DEFAULT_W_TPOT = 0.3
    DEFAULT_W_SUCC = 0.1

    def __init__(
            self,
            ttft_penalty: float = 3.0,
            tpot_penalty: float = 3.0,
            success_rate_penalty: float = 5.0,
            ttft_slo: float = 0.5,
            tpot_slo: float = 0.05,
            success_rate_slo: float = 1.0,
            generate_speed_target: float = 5300.0,
            w_gen: float = None,
            w_ttft: float = None,
            w_tpot: float = None,
            w_succ: float = None,
    ):
        """
        Initialize the fitness calculator.

        Args:
            ttft_penalty: Penalty coefficient for TTFT violations (k value)
            tpot_penalty: Penalty coefficient for TPOT violations
            success_rate_penalty: Penalty coefficient for success rate violations
            ttft_slo: TTFT SLO threshold in seconds
            tpot_slo: TPOT SLO threshold in seconds
            success_rate_slo: Success rate SLO threshold (0-1)
            generate_speed_target: Target generation speed (tokens/sec)
            w_gen: Weight for generation speed (default: 0.4)
            w_ttft: Weight for TTFT (default: 0.2)
            w_tpot: Weight for TPOT (default: 0.3)
            w_succ: Weight for success rate (default: 0.1)
        """
        # Weights
        self.w_gen = w_gen if w_gen is not None else self.DEFAULT_W_GEN
        self.w_ft = w_ttft if w_ttft is not None else self.DEFAULT_W_TTFT
        self.w_pot = w_tpot if w_tpot is not None else self.DEFAULT_W_TPOT
        self.w_succ = w_succ if w_succ is not None else self.DEFAULT_W_SUCC

        # SLO targets
        self.gen_speed_target = generate_speed_target
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self.success_rate_slo = success_rate_slo

        # Penalty coefficients
        self.ttft_penalty = ttft_penalty
        self.tpot_penalty = tpot_penalty
        self.success_rate_penalty = success_rate_penalty

    def get_fitness_value(self, performance_index: PerformanceIndex) -> float:
        """
        Calculate the fitness value (cost) for a configuration.

        The fitness function is:
        cost = w_gen * (target_speed / actual_speed)
             + w_ft * exp(k_ttft * (ttft / ttft_slo - 1))
             + w_pot * exp(k_tpot * (tpot / tpot_slo - 1))
             + w_succ * exp(k_succ * (success_slo / actual_success - 1))

        Lower values are better. Returns inf for invalid configurations.

        Args:
            performance_index: Performance metrics for the configuration

        Returns:
            Fitness value (cost). Lower is better.
        """
        total_cost = 0.0

        # Generation speed cost
        if performance_index.generate_speed is not None and performance_index.generate_speed > 0:
            cost_gen = self.gen_speed_target / performance_index.generate_speed
            total_cost += self.w_gen * cost_gen
        else:
            logger.warning(
                f"Invalid generate_speed metric. "
                f"performance_index: {performance_index}"
            )
            return inf

        # TTFT cost
        if performance_index.time_to_first_token is not None:
            try:
                cost_ft = exp(self.ttft_penalty * (
                        performance_index.time_to_first_token / self.ttft_slo - 1
                ))
                total_cost += self.w_ft * cost_ft
            except (OverflowError, ZeroDivisionError):
                logger.warning(
                    f"Invalid time_to_first_token metric. "
                    f"performance_index: {performance_index}"
                )
                return inf

        # TPOT cost
        if performance_index.time_per_output_token is not None:
            try:
                cost_pot = exp(self.tpot_penalty * (
                        performance_index.time_per_output_token / self.tpot_slo - 1
                ))
                total_cost += self.w_pot * cost_pot
            except (OverflowError, ZeroDivisionError):
                logger.warning(
                    f"Invalid time_per_output_token metric. "
                    f"performance_index: {performance_index}"
                )
                return inf

        # Success rate cost
        if performance_index.success_rate is not None and performance_index.success_rate > 0:
            try:
                cost_succ = exp(
                    self.success_rate_penalty * (
                            self.success_rate_slo / performance_index.success_rate - 1
                    )
                )
                total_cost += self.w_succ * cost_succ
            except (OverflowError, ZeroDivisionError):
                logger.warning(
                    f"Invalid success_rate metric. "
                    f"performance_index: {performance_index}"
                )
                return inf
        else:
            logger.warning(
                f"Invalid success_rate metric. "
                f"performance_index: {performance_index}"
            )
            return inf

        return total_cost

    def update_target(self, generate_speed_target: float):
        """Update the generation speed target dynamically."""
        self.gen_speed_target = generate_speed_target
