#!/bin/bash
PARENT_DIR=$(dirname $(realpath "$0"))
export PYTHONPATH=$PARENT_DIR:$PYTHONPATH
pkill -9 'mindie'
ps -ef | grep 'python'
ps -ef | grep 'mindie'
npu-smi info
cd /usr/local/Ascend/mindie/latest/mindie-service
./bin/mindieservice_daemon        