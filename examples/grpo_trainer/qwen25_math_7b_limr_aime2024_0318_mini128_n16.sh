#!/bin/bash
# -*- coding: utf-8 -*-
set -euo pipefail


# ------------------------------
# Environment Configuration
# ------------------------------
setup_env() {
    export WANDB_API_KEY="2b944d922bcfa2701fde10c3cb5f99a04b197e05"
    export OPENAI_API_KEY="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"
    export OPENAI_API_URL="https://api.openai.com/v1"

    export VLLM_ATTENTION_BACKEND="XFORMERS"
    export VLLM_USE_MODELSCOPE="False"
    export VLLM_USE_V1=0

    export HOME="/cpfs01/shared/llm_ddd/zhangjin/public"


    export VERL_PPO_LOGGING_LEVEL='DEBUG'
    # 实时刷新日志，调试打开
    export PYTHONUNBUFFERED=1
    export HYDRA_FULL_ERROR=1
}
setup_env

# ------------------------------
# Proxy Configuration
# ------------------------------
setup_proxy() {
    export PROXY_CREDENTIALS="zhangjin:QiL6Vklber0wH6cYRB7TvP0opZrQBq4QjoXhg6NFyYBT1ysc6EiNbgPeaSJZ"
    export PROXY_URL="aliyun-proxy.pjlab.org.cn:13128"

    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
    # export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    
    export http_proxy="http://${PROXY_CREDENTIALS}@${PROXY_URL}"
    export https_proxy="https://${PROXY_CREDENTIALS}@${PROXY_URL}"
    export HTTP_PROXY="${https_proxy}"
    export HTTPS_PROXY="${https_proxy}"

    export OPENAI_API_URL="https://ai-proxy.shlab.tech/internal"
}
setup_proxy

# ------------------------------
# Conda Environment Setup
# ------------------------------
activate_conda() {
    local conda_env="verl_vllm"
    source "/cpfs01/shared/llm_ddd/zhangjin/.bashrc"
    conda activate "${conda_env}" || {
        echo "Failed to activate conda environment: ${conda_env}"
        exit 1
    }
}
activate_conda

# ------------------------------
# Path Configuration
# ------------------------------
setup_path() {
    YYMMDD=$(date +%Y-%m-%d)
    HHMMSS=$(date +%H-%M-%S)

    CUSTOM_CODE_DIR="/cpfs01/shared/llm_ddd/zhangjin/codehub/overthink/code_zj"
    VERL_DIR="/cpfs01/shared/llm_ddd/zhangjin/codehub/overthink/verl"

    BASE_MODEL_PATH="/cpfs01/shared/public/zhangjin/public/models/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e"
    TRAIN_DATA="/cpfs01/shared/llm_ddd/zhangjin/codehub/overthink/data/LIMR/train.parquet"
    VAL_DATA="/cpfs01/shared/llm_ddd/zhangjin/codehub/overthink/data/aime2024/train.parquet"
    # 倒数第二个/
    train_dataset_name=$(echo "${TRAIN_DATA}" | awk -F'/' '{print $(NF-1)}')

    experiment_name="qwen_math_7b_${train_dataset_name}_aime2024-dlc-${HHMMSS}"
    project_name="verl_grpo_${train_dataset_name}_aime2024"
    job_name=$(basename "${0}" .sh)

    OUTPUT_DIR="/cpfs01/shared/llm_ddd/zhangjin/public/ckpts/verl/grpo/${experiment_name}/${YYMMDD}/${HHMMSS}"
    sudo mkdir -p "${OUTPUT_DIR}"
    OUTPUT_LOG_DIR="/cpfs01/shared/llm_ddd/zhangjin/codehub/overthink/verl/outputs/exploration_data/${job_name}/${YYMMDD}/${HHMMSS}"
}
setup_path



# ------------------------------
# Main Training Command
# ------------------------------
run_training() {
    export PYTHONPATH="/cpfs01/shared/llm_ddd/zhangjin/codehub/overthink/:${PYTHONPATH:-}"
    echo "PYTHONPATH: ${PYTHONPATH}"

    cd "${VERL_DIR}" || exit 1

    local num_gpus="${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}"
    local world_size="${WORLD_SIZE:-1}"
    local total_gpus=$((num_gpus * world_size))
    echo "total_gpus: ${total_gpus}"
    # self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
    # self.config.actor.ppo_mini_batch_size //= (self.device_mesh.size() // self.ulysses_sequence_parallel_size)
    # self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

    python3 -m verl.trainer.main_ppo \
        custom_reward_function.path="${CUSTOM_CODE_DIR}/rewards/math_reward_w_remote.py" \
        custom_reward_function.name=custom_reward_fn \
        algorithm.adv_estimator="grpo" \
        data.train_files="${TRAIN_DATA}" \
        data.val_files="${VAL_DATA}" \
        data.train_batch_size=1024 \
        data.max_prompt_length=$((1024 * 2)) \
        data.max_response_length=$((1024 * 3)) \
        data.filter_overlong_prompts=True \
        data.truncation="error" \
        trainer.default_local_dir="${OUTPUT_DIR}" \
        actor_rollout_ref.model.path="${BASE_MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.shuffle=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size=$((total_gpus * 4)) \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.kl_loss_type="low_var_kl" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name="vllm" \
        actor_rollout_ref.rollout.max_num_batched_tokens=300000 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.temperature=1.0 \
        +actor_rollout_ref.rollout.val_temperature=0.6 \
        actor_rollout_ref.rollout.n=16 \
        +actor_rollout_ref.rollout.n_val=1 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        algorithm.lam=0.95 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${experiment_name}" \
        +trainer.val_before_train=True \
        trainer.n_gpus_per_node="${num_gpus}" \
        trainer.nnodes="${world_size}" \
        trainer.save_freq=5 \
        trainer.test_freq=1 \
        trainer.total_epochs=10000 \
        reward_model.reward_manager="custom" \
        +trainer.output_dir="${OUTPUT_LOG_DIR}" "$@"
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
    sudo chmod -R 777 "${VERL_DIR}/outputs" || true
    sudo chmod -R 777 "${VERL_DIR}/wandb" || true
}

# ------------------------------
# Main Execution Flow
# ------------------------------
check_permissions
setup_ray "$@"
sudo chmod -R 777 "${OUTPUT_DIR}" || true
echo "Training completed successfully: $(basename "${0}")"
exit 0