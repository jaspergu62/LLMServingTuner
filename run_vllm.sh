#!/usr/bin/env bash

pkill -15 'vllm'
ps -ef | grep 'python'
npu-smi info
vllm serve MODEL_PATH --served-model-name MODEL_NAME --host 127.0.0.1 --port 6379
--max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
--max-num-seqs $MAX_NUM_SEQS \
--preemption-mode $PREEMPTION_MODE \
--num-scheduler-steps $NUM_SCHEDULER_STEPS \
--enable-prefix-caching $ENABLE_PREFIX_CACHING \
--prefix-caching-hash-algo $PREFIX_CACHING_HASH_ALGO