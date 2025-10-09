#!/bin/bash

JOB_NAME="doc2query_v5"

WORKER_IMAGE="registry.h.pjlab.org.cn/ailab/pytorch:22.04-pjlab-py3.10-torch2.2.0-cu12.1"
WORKER_COUNT="${WORKER_COUNT:-"4"}"
WORKER_GPU="${WORKER_GPU:-"8"}"
WORKER_CPU="${WORKER_CPU:-"128"}"
WORKER_MEMORY="${WORKER_MEMORY:-"1600000"}"

# [hx]
# WORKSPACE_ID="ailab-puyullmgpu"
# CHARGE_GROUP="puyullm_gpu"
WORKSPACE_ID="ailab-hx"
CHARGE_GROUP="hx_gpu"
RUN_CMD="/mnt/shared-storage-user/ailab-hx/tongjian/verl/examples/grpo_trainer/doc2query_v5/grpo_qwen3-30b-a3_general_doc2query_v5.sh"

chmod +x ${RUN_CMD}

rjob submit -e DISTRIBUTED_JOB=true \
    --image=${WORKER_IMAGE} \
    --host-network=true --namespace=${WORKSPACE_ID} --name ${JOB_NAME} -P ${WORKER_COUNT} --gpu ${WORKER_GPU} --cpu ${WORKER_CPU}  --memory ${WORKER_MEMORY} --charged-group ${CHARGE_GROUP} \
    --private-machine='group' \
    --gang-start=true \
    --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
    --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
    --mount=gpfs://gpfs1/ailab-hx:/mnt/shared-storage-user/ailab-hx \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    --health-check "" \
    -- bash -ecx ${RUN_CMD}