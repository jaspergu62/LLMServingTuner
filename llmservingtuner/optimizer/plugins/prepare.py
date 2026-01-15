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

from loguru import logger

from llmservingtuner.config.config import field_to_param
from llmservingtuner.optimizer.interfaces.prepare import PrepareInterface


class VerifyService(PrepareInterface):
    """
    Run a service scheduling cycle to verify the environment is working.
    This is the default prepare step that runs the service with default parameters.
    """

    def run(self):
        self.default_run_param = field_to_param(self.optimizer.target_field)
        self.default_res = self.optimizer.scheduler.run(self.default_run_param, self.optimizer.target_field)
        self.default_fitness = self.optimizer.minimum_algorithm(self.default_res)
        self.optimizer.scheduler.save_result(fitness=self.default_fitness)

        if self.optimizer.scheduler.error_info:
            raise ValueError(f"Failed to start the default service. "
                             "Please check if the service and the request to start it are correct.")

        if (self.default_res.generate_speed is None or self.default_res.time_to_first_token is None or
                self.default_res.time_per_output_token is None):
            logger.warning(f"Failed to obtain benchmark metric data. metric {self.default_res}"
                           "Please check if the benchmark is running successfully.")

        # Store default results in optimizer for later use
        self.optimizer.default_run_param = self.default_run_param
        self.optimizer.default_res = self.default_res
        self.optimizer.default_fitness = self.default_fitness

        # Update gen_speed_target if available
        if self.default_res.generate_speed:
            self.optimizer.gen_speed_target = 10 * self.default_res.generate_speed
