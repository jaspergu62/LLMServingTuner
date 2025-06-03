#!/usr/bin/bash
# benchmark文件权限可能有问题，需要设置
python vllm/benchmark/benchmark_serving.py \
    --backend vllm \
    --host 127.0.0.1 \
    --port 6379 \
    --model MODEL_PATH \
    --served-model-name MODEL_NAME \
    --dataset-name sonnets \
    --dataset-path vllm/benchmarks/sonnet.txt \
    --num-prompts 3000 \
    --max-concurrency $MAXCONCURRENCY \
    --request-rate $REQUESTRATE \
    --result-dir $MODEL_EVAL_STATE_CUSTOM_OUTPUT \
    --save-result