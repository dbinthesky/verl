#!/bin/bash
# -*- coding: utf-8 -*-
set -euo pipefail

# ------------------------------
# Environment Configuration
# ------------------------------
setup_env() {
    export WANDB_API_KEY="2e3700316fecb744b594dff815d1b11fbe514d24"
    export WANDB_BASE_URL=https://api.bandw.top

    # export WANDB_MODE="offline"
    export VERL_PPO_LOGGING_LEVEL='DEBUG'
    export VLLM_ATTENTION_BACKEND="XFORMERS"
    export VLLM_USE_MODELSCOPE="False"
    export HOME="/cpfs01/shared/llm_ddd/tongjian"
    export HYDRA_FULL_ERROR=1
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
    source /cpfs01/shared/llm_ddd/tongjian/.bashrc
    conda activate /cpfs01/shared/llm_ddd/gaoxuan/anaconda3/envs/Verl
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
    BASE_MODEL_PATH="/cpfs01/shared/llm_ddd/tongjian/ckpts/datareview_rl_test/verl/grpo/archived/fabricate_aio_distillv16_s3_roll16_bsz32_dapo_wo_kl_coef_wo_entropy_t08_solver_qwq_bo8_grpo_step_80"
    TRAIN_DATA="/cpfs01/shared/llm_ddd/tongjian/rl/salt/runtu_error_collection_0703_train.parquet"
    VAL_DATA="/cpfs01/shared/llm_ddd/tongjian/rl/salt/runtu_error_collection_0703_test.parquet"

    experiment_name="distill-qwen-32b_salt-${YYMMDD}-${HHMMSS}"
    project_name="salt"

    OUTPUT_DIR="/cpfs01/shared/llm_ddd/tongjian/ckpts/datareview_rl_test/verl/grpo/salt/${experiment_name}/"
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

    python3 -m recipe.dapo.main_dapo \
        custom_reward_function.path="${CUSTOM_CODE_DIR}/rewards/fabricate_qa.py" \
        custom_reward_function.name=salt_default_compute_score_train \
        +custom_valid_reward_function.path="${CUSTOM_CODE_DIR}/rewards/fabricate_qa.py" \
        +custom_valid_reward_function.name=salt_default_compute_score_valid \
        algorithm.adv_estimator="grpo" \
        data.train_files="${TRAIN_DATA}" \
        data.val_files="${VAL_DATA}" \
        data.train_batch_size=32 \
        data.max_prompt_length=16386 \
        data.max_response_length=8192 \
        data.filter_overlong_prompts=True \
        trainer.default_local_dir="${OUTPUT_DIR}" \
        actor_rollout_ref.model.path="${BASE_MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.shuffle=False \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size=32 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24578 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        reward_model.overlong_buffer.enable=True \
        reward_model.overlong_buffer.len=$((1024 * 4)) \
        reward_model.overlong_buffer.penalty_factor=1.0 \
        algorithm.filter_groups.enable=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name="vllm" \
        actor_rollout_ref.rollout.max_num_batched_tokens=300000 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
        actor_rollout_ref.rollout.temperature=0.9 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
        +actor_rollout_ref.rollout.trust_remote_code=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        +actor_rollout_ref.rollout.n_val=1 \
        algorithm.kl_ctrl.kl_coef=0.000 \
        algorithm.lam=0.95 \
        reward_model.reward_manager=dapo_custom \
        trainer.logger='["console", "wandb"]' \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${experiment_name}" \
        trainer.n_gpus_per_node="${num_gpus}" \
        trainer.nnodes="${world_size}" \
        trainer.save_freq=5 \
        trainer.test_freq=10 \
        trainer.total_epochs=10000 \
        "$@"
    local training_status=$?

    # 显式传递训练状态
    if [ $training_status -ne 0 ]; then
        echo "Training failed with exit code $training_status"
        exit $training_status # 退出码传递给全局
    fi
}
# run_training "$@"

# ------------------------------
# Ray Cluster Setup
# ------------------------------
setup_ray() {
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export MASTER_PORT=29905
    export WORLD_SIZE=${WORLD_SIZE:-1}
    export RANK=${RANK:-0}
    # export no_proxy="localhost,127.0.0.1,*local,10.130.133.200"

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
            sleep 240
        else
            sleep 10
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
