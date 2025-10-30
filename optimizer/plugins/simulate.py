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
from typing import Optional, Tuple
import shutil
from loguru import logger

from msserviceprofiler.modelevalstate.config.config import get_settings, OptimizerConfigField, VllmConfig
from msserviceprofiler.modelevalstate.config.custom_command import VllmCommand
from msserviceprofiler.modelevalstate.optimizer.interfaces.simulator import SimulatorInterface


class VllmSimulator(SimulatorInterface):
    def __init__(self, config: Optional[VllmConfig] = None, *args, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.vllm
        super().__init__(*args, process_name=self.config.process_name, **kwargs)

        self.command = VllmCommand(self.config.command).command

    @property
    def base_url(self) -> str:
        """
        获取服务的base url 属性
        Returns:

        """
        return f"http://127.0.0.1:{self.config.command.port}/health"

    def stop(self, del_log: bool = True):
        """
        运行时，其他的准备工作。
        Returns:

        """
        pkill_path = shutil.which("pkill")
        try:
            subprocess.run([pkill_path, "-15", "vllm"], stderr=subprocess.STDOUT, text=True)
        except subprocess.SubprocessError:
            logger.warning(f"Failed to stop vllm process with pkill.")
        super().stop(del_log)

    def update_command(self):
        self.command = VllmCommand(self.config.command).command
