#!/bin/bash
# -*- coding: utf-8 -*-
set -euo pipefail

# ------------------------------
# Environment Configuration
# ------------------------------
setup_env() {
    export WANDB_API_KEY="2e3700316fecb744b594dff815d1b11fbe514d24"
    export VERL_PPO_LOGGING_LEVEL='DEBUG'
    export VLLM_ATTENTION_BACKEND="XFORMERS"
    export VLLM_USE_MODELSCOPE="False"
    export HOME="/cpfs01/shared/llm_ddd/tongjian"
}
setup_env

# ------------------------------
# Proxy Configuration
# ------------------------------
setup_proxy() {
    export PROXY_CREDENTIALS="tongjian:dazL5iB8mjDIOGtNj2uekzlsRCelVS38txIK98mWhKyoyLCBCCw9DNXlUOcX"
    export PROXY_URL="aliyun-proxy.pjlab.org.cn:13128"
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    export http_proxy="http://${PROXY_CREDENTIALS}@${PROXY_URL}"
    export https_proxy="https://${PROXY_CREDENTIALS}@${PROXY_URL}"
    export HTTP_PROXY="${https_proxy}"
    export HTTPS_PROXY="${https_proxy}"
}
# setup_proxy

# ------------------------------
# Conda Environment Setup
# ------------------------------
activate_conda() {
    # Hard Code: InternLM3-8B支持
    source /cpfs01/shared/llm_ddd/guoxu/public/verl_env/verl_env/bin/activate /cpfs01/shared/llm_ddd/tongjian/verl
}
activate_conda


# ------------------------------
# Path Configuration
# ------------------------------
setup_path() {
    YYMMDD=$(date +%Y-%m-%d)
    HHMMSS=$(date +%H-%M-%S)

    CUSTOM_CODE_DIR="/cpfs01/shared/llm_ddd/tongjian/verl"
    VERL_DIR="/cpfs01/shared/llm_ddd/tongjian/verl"
    BASE_MODEL_PATH="/cpfs01/shared/llm_ddd/opencompass/models/hf_hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/c79f47acaf303faabb7133b4b7b76f24231f2c8d/"
    TRAIN_DATA="/cpfs01/shared/llm_ddd/tongjian/rl/fabricate_qa/super_gpqa_aio_noneasy_train_0517.parquet"
    VAL_DATA="/cpfs01/shared/llm_ddd/tongjian/rl/fabricate_qa/super_gpqa_aio_noneasy_test_0517.parquet"

    experiment_name="qwen2_5-14b_qwq_fabricate_qa_supergpqa-dlc-${YYMMDD}-${HHMMSS}"
    project_name="verl_grpo_qwq_fabricate_qa_supergpqa"

    OUTPUT_DIR="/cpfs01/shared/llm_ddd/tongjian/ckpts/datareview_rl_test/verl/grpo/qwen2_5-14b_qwq_fabricate_qa/${YYMMDD}/${HHMMSS}"
    mkdir -p "${OUTPUT_DIR}"
}
setup_path


# ------------------------------
# Install Package
# ------------------------------
# setup_package() {
#     pip3 install -U torchdata
# }
# setup_package


# ------------------------------
# Main Training Command
# ------------------------------
run_training() {
    export PYTHONPATH="/cpfs01/shared/llm_ddd/tongjian/verl:${PYTHONPATH:-}"
    echo "PYTHONPATH: ${PYTHONPATH}"

    cd "${VERL_DIR}" || exit 1

    local num_gpus="${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}"
    local world_size="${WORLD_SIZE:-1}"
    local total_gpus=$((num_gpus * world_size))
    # self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
    # self.config.actor.ppo_mini_batch_size //= (self.device_mesh.size() // self.ulysses_sequence_parallel_size)
    # self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

    python3 -m verl.trainer.main_ppo \
        custom_reward_function.path="${CUSTOM_CODE_DIR}/rewards/fabricate_qa.py" \
        custom_reward_function.name=qwq_longcot_fabricate_qa_compute_score_train \
        +custom_valid_reward_function.path="${CUSTOM_CODE_DIR}/rewards/fabricate_qa.py" \
        +custom_valid_reward_function.name=qwq_longcot_fabricate_qa_compute_score_valid \
        algorithm.adv_estimator="grpo" \
        data.train_files="${TRAIN_DATA}" \
        data.val_files="${VAL_DATA}" \
        data.train_batch_size=128 \
        data.max_prompt_length=1024 \
        data.max_response_length=15360 \
        data.filter_overlong_prompts=True \
        trainer.default_local_dir="${OUTPUT_DIR}" \
        actor_rollout_ref.model.path="${BASE_MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.shuffle=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size=$((total_gpus)) \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.kl_loss_type="low_var_kl" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        +actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name="vllm" \
        actor_rollout_ref.rollout.max_num_batched_tokens=300000 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.temperature=1.1 \
        actor_rollout_ref.rollout.n=16 \
        +actor_rollout_ref.rollout.trust_remote_code=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        +actor_rollout_ref.rollout.n_val=1 \
        algorithm.kl_ctrl.kl_coef=0.000 \
        algorithm.lam=0.95 \
        trainer.logger='["console", "wandb"]' \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${experiment_name}" \
        +trainer.val_before_train=True \
        trainer.n_gpus_per_node="${num_gpus}" \
        trainer.nnodes="${world_size}" \
        trainer.save_freq=5 \
        trainer.test_freq=5 \
        trainer.total_epochs=10000 \
        reward_model.reward_manager="custom" "$@"
    local training_status=$?

    # 显式传递训练状态
    if [ $training_status -ne 0 ]; then
        echo "Training failed with exit code $training_status"
        exit $training_status  # 退出码传递给全局
    fi
}
# run_training "$@"


# ------------------------------
# Ray Cluster Setup
# ------------------------------
setup_ray() {
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
    export WORLD_SIZE=${WORLD_SIZE:-1}
    export RANK=${RANK:-0}

    echo "Ray Cluster Configuration:"
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "WORLD_SIZE: $WORLD_SIZE"
    echo "RANK: $RANK"

    if [ "$WORLD_SIZE" -le 1 ]; then
        echo "Single node training, starting without Ray cluster..."
        run_training "$@"
    else
        if [ "$RANK" -eq 0 ]; then
            ray start --head \
                --node-ip-address="$MASTER_ADDR" \
                --port="$MASTER_PORT"
        else
            ray start --address "${MASTER_ADDR}:${MASTER_PORT}" \
                --block
        fi
        sleep 10
        run_training "$@"
    fi

    ray stop
}

# ------------------------------
check_permissions() {
    echo "Updating permissions for output directories..."
    chmod -R 777 "${VERL_DIR}/outputs" || true
    chmod -R 777 "${VERL_DIR}/wandb" || true
}

# ------------------------------
# Main Execution Flow
# ------------------------------
check_permissions
setup_ray "$@"
chmod -R 777 "${OUTPUT_DIR}" || true
echo "Training completed successfully: $(basename "${0}")"
exit 0