# !/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
"""
 通过设置环境变量 来决定是如何对代码打补丁。

# 直接修改原代码
1. 进行仿真 LLMSERVINGTUNER_SIMULATE
> export LLMSERVINGTUNER_SIMULATE=True

2. 进行寻优 LLMSERVINGTUNER_ALL
> export LLMSERVINGTUNER_ALL=True

"""
import os
import traceback

try:
    from loguru import logger
except ModuleNotFoundError as e:
    from logging import getLogger

    logger = getLogger(__name__)

LLMSERVINGTUNER_SIMULATE = "LLMSERVINGTUNER_SIMULATE"
LLMSERVINGTUNER_ALL = "LLMSERVINGTUNER_ALL"


def dispatch(target_env):
    target_flag = os.getenv(target_env) or os.getenv(target_env.lower())
    if target_flag and (target_flag.lower() == "true" or target_flag.lower() != "false"):
        from llmservingtuner.patch import enable_patch
        if enable_patch(target_env):
            logger.info("The collected patch is successfully installed.")
    else:
        logger.debug(f"{target_env}: {target_flag}")


try:
    dispatch(LLMSERVINGTUNER_SIMULATE)
    dispatch(LLMSERVINGTUNER_ALL)
except Exception as e:
    logger.error(f"Failed in Import patch. {e}")
    traceback.print_exc()
