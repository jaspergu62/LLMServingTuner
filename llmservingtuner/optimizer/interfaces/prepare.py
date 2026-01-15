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


class PrepareInterface:
    """
    Interface for pre-optimization preparation steps.
    Implementations can perform setup tasks before the optimization loop starts.
    """

    def __init__(self, optimizer):
        """
        Initialize the prepare interface.

        Args:
            optimizer: The optimizer instance that will run the optimization.
        """
        self.optimizer = optimizer

    def run(self):
        """
        Execute the preparation step.
        Subclasses must implement this method.
        """
        raise NotImplementedError
