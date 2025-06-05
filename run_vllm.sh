#!/usr/bin/env bash

pkill -15 'vllm'
ps -ef | grep 'python'
npu-smi info
vllm serve MODEL_PATH --served-model-name MODEL_NAME --host 127.0.0.1 --port 6379 \
--max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
--max-num-seqs $MAX_NUM_SEQS