# #!/bin/bash

DLC_PATH="/cpfs01/shared/public/dlc"

JOB_NAME="internlm3-8b_qwq_fabricate_qa_0330"
PYARGS="${@:3}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
DLC_CONFIG_PATH="${DLC_CONFIG_PATH:-"/cpfs01/shared/llm_ddd/tongjian/dlc.config"}"
# [ddd]
WORKSPACE_ID="${WORKSPACE_ID:-"ws1ujefpjyfgqjwp"}"
# [he]
# WORKSPACE_ID="${WORKSPACE_ID:-"wso1cah3ytpgmaah"}"
# [hc]
# WORKSPACE_ID="${WORKSPACE_ID:-"ws1h2vgufjufr4jj"}"
# [h2]
# WORKSPACE_ID="wso1gao759kjyz4p"
DATA_SOURCES="datajrdc07nuo03o,dataui74zr3uig4f,datapcxxjb8czn7k,datajfxq87v3rx9k,data1otmepzybpqr"
PRIORITY="${PRIORITY:-"4"}"
WORKER_COUNT="${WORKER_COUNT:-"8"}"
WORKER_GPU="${WORKER_GPU:-"8"}"
WORKER_CPU="${WORKER_CPU:-"64"}"
WORKER_MEMORY="${WORKER_MEMORY:-"1024"}"
WORKER_IMAGE="pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/lishuaibin:lishuaibin-xpuyu-trainrlhf"
RUN_CMD="bash /cpfs01/shared/llm_ddd/tongjian/verl/examples/grpo_trainer/fabricate_qa/run_internlm3-8b_qwq_fabricate_qa_0330_late_stage_64gpu.sh"


dlcrun_cmd=$(
  cat <<EOF
/cpfs01/shared/public/dlc create job \\
  --config ${DLC_CONFIG_PATH} \\
  --kind PyTorchJob \\
  --name ${JOB_NAME} \\
  --priority ${PRIORITY} \\
  --resource_id=${RESOURCE_ID} \\
  --data_sources=${DATA_SOURCES} \\
  --workspace_id ${WORKSPACE_ID} \\
  --worker_count $WORKER_COUNT \\
  --worker_cpu $WORKER_CPU \\
  --worker_gpu $WORKER_GPU \\
  --worker_memory $WORKER_MEMORY \\
  --worker_shared_memory '80Gi' \\
  --worker_image ${WORKER_IMAGE} \\
  --command "${RUN_CMD}"
EOF
)

echo "Executing command:"
echo "$dlcrun_cmd" | sed 's/\\\s\+/\n  /g'

eval "$dlcrun_cmd"
