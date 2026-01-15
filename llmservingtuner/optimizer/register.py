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
from typing import Type

from loguru import logger

simulates = {}
benchmarks = {}
prepares = {}


def register_simulator(model_arch: str,
                       model_cls
                       ) -> None:
    """
    Register a simulator implementation.

    Args:
        model_arch: Name identifier for the simulator (e.g., "vllm", "mindie")
        model_cls: A SimulatorInterface subclass
    """
    from llmservingtuner.optimizer.interfaces.simulator import SimulatorInterface
    if not isinstance(model_arch, str):
        msg = f"`model_arch` should be a string, not a {type(model_arch)}"
        raise TypeError(msg)

    if model_arch in simulates:
        logger.warning(
            f"Model architecture {model_arch} is already registered, and will be "
            "overwritten by the new model class {model_cls}.")
    if isinstance(model_cls, type) and issubclass(model_cls, SimulatorInterface):
        simulates[model_arch] = model_cls
    else:
        msg = ("`model_cls` should be a SimulatorInterface class, "
               f"not a {type(model_arch)}")
        raise TypeError(msg)


def register_benchmarks(model_arch: str,
                        model_cls
                        ) -> None:
    """
    Register a benchmark implementation.

    Args:
        model_arch: Name identifier for the benchmark (e.g., "vllm_benchmark", "ais_bench")
        model_cls: A BenchmarkInterface subclass
    """
    from llmservingtuner.optimizer.interfaces.benchmark import BenchmarkInterface
    if not isinstance(model_arch, str):
        msg = f"`model_arch` should be a string, not a {type(model_arch)}"
        raise TypeError(msg)

    if model_arch in benchmarks:
        logger.warning(
            f"Model architecture {model_arch} is already registered, and will be "
            "overwritten by the new model class {model_cls}.")
    if isinstance(model_cls, type) and issubclass(model_cls, BenchmarkInterface):
        benchmarks[model_arch] = model_cls
    else:
        msg = ("`model_cls` should be a BenchmarkInterface class, "
               f"not a {type(model_arch)}")
        raise TypeError(msg)


def register_prepare(name: str, prepare_cls) -> None:
    """
    Register a prepare step implementation.

    Args:
        name: Name identifier for the prepare step (e.g., "verify_service")
        prepare_cls: A PrepareInterface subclass
    """
    from llmservingtuner.optimizer.interfaces.prepare import PrepareInterface
    if not isinstance(name, str):
        msg = f"`name` should be a string, not a {type(name)}"
        raise TypeError(msg)

    if name in prepares:
        logger.warning(
            f"Prepare step {name} is already registered, and will be "
            f"overwritten by the new class {prepare_cls}.")
    if isinstance(prepare_cls, type) and issubclass(prepare_cls, PrepareInterface):
        prepares[name] = prepare_cls
    else:
        msg = ("`prepare_cls` should be a PrepareInterface class, "
               f"not a {type(prepare_cls)}")
        raise TypeError(msg)


def register_ori_functions():
    """Register built-in simulators, benchmarks, and prepare steps."""
    from llmservingtuner.optimizer.plugins.benchmark import VllmBenchMark, AisBench
    from llmservingtuner.optimizer.plugins.simulate import VllmSimulator, Simulator
    from llmservingtuner.optimizer.plugins.prepare import VerifyService

    register_benchmarks("vllm_benchmark", VllmBenchMark)
    register_benchmarks("ais_bench", AisBench)
    register_simulator("vllm", VllmSimulator)
    register_simulator("mindie", Simulator)
    register_prepare("verify_service", VerifyService)
