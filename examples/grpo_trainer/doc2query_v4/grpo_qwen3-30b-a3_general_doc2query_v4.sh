#!/bin/bash
# -*- coding: utf-8 -*-
set -euo pipefail

# ------------------------------
# Environment Configuration
# ------------------------------
setup_env() {
    export WANDB_API_KEY="2e3700316fecb744b594dff815d1b11fbe514d24"
    export WANDB_BASE_URL=https://api.bandw.top

    export WANDB_MODE="offline"
    export VERL_PPO_LOGGING_LEVEL='DEBUG'
    export VLLM_ATTENTION_BACKEND="XFORMERS"
    export VLLM_USE_MODELSCOPE="False"
    export HOME="/mnt/shared-storage-user/ailab-hx/tongjian"
    export CKPTS_DIR="${HOME}/ckpts"
    export HYDRA_FULL_ERROR=1
}
setup_env

# ------------------------------
# Conda Environment Setup
# ------------------------------
activate_conda() {
    source /mnt/shared-storage-user/ailab-hx/wulianyi/miniconda3/etc/profile.d/conda.sh
    conda activate /mnt/shared-storage-user/ailab-hx/wulianyi/miniconda3/envs/verl-0.4.1_conda
}
activate_conda

# ------------------------------
# Path Configuration
# ------------------------------
setup_path() {
    YYMMDD=$(date +%Y-%m-%d)
    HHMMSS=$(date +%H-%M-%S)

    local num_gpus="${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}"
    local world_size="${WORLD_SIZE:-1}"

    ROLLOUT_N=5
    TRAIN_BSZ=32
    KL_LOSS_COEF="0"
    KL_COEF="1.5"
    TEMPERATURE="1.0"
    USE_RM_PAD="True" # must be true 
    ULYSSES_SP="1" # must be 1
    USE_KL_IN_REWARD="True"

    HOME="/mnt/shared-storage-user/ailab-hx/tongjian"
    CUSTOM_CODE_DIR="${HOME}/verl"
    VERL_DIR="${HOME}/verl"
    # BASE_MODEL_PATH="/mnt/shared-storage-user/ailab-hx/tongjian/ckpts/datareview_sft_test/Qwen_30B_A3_instruct_20250425_s1_32k_S2_32k_doc2query_merge_1010/20251011035959/hf-153"
    BASE_MODEL_PATH="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/4a8a1645504d39f8c2b9eacfd6d72dac693d3488"
    TRAIN_DATA="/mnt/shared-storage-user/ailab-hx/tongjian/rl/doc2query_v4/kcle_diamond_8k_rl_inputs_test.parquet"
    VAL_DATA="/mnt/shared-storage-user/ailab-hx/tongjian/rl/doc2query_v4/pretrain_general_doc_8k_rl_inputs_test.parquet"

    experiment_name="doc2query_v4_30b_a3_think_kcle_${YYMMDD}_roll${ROLLOUT_N}_${TRAIN_BSZ}_kl_coef_${KL_LOSS_COEF}_wo_entropy_t${TEMPERATURE}"
    project_name="doc2query_v4"

    OUTPUT_DIR="/mnt/shared-storage-user/ailab-hx/tongjian/ckpts/datareview_rl_test/verl/grpo/doc2query_v4/${experiment_name}"
    mkdir -p "${OUTPUT_DIR}"
}
setup_path

# ------------------------------

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
        custom_reward_function.name=doc2query_v4_compute_score_train \
        +custom_valid_reward_function.path="${CUSTOM_CODE_DIR}/rewards/fabricate_qa.py" \
        +custom_valid_reward_function.name=doc2query_v4_compute_score_valid \
        algorithm.adv_estimator="grpo" \
        data.train_files="${TRAIN_DATA}" \
        data.val_files="${VAL_DATA}" \
        data.train_batch_size=${TRAIN_BSZ} \
        data.max_prompt_length=12288 \
        data.max_response_length=20480 \
        data.filter_overlong_prompts=True \
        data.filter_overlong_prompts_workers=256 \
        trainer.default_local_dir="${OUTPUT_DIR}" \
        trainer.val_before_train=False \
        actor_rollout_ref.model.path="${BASE_MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.model.use_remove_padding=${USE_RM_PAD} \
        actor_rollout_ref.actor.shuffle=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BSZ} \
        actor_rollout_ref.actor.ppo_micro_batch_size=${TRAIN_BSZ} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SP} \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
        actor_rollout_ref.rollout.name="vllm" \
        actor_rollout_ref.rollout.max_num_batched_tokens=300000 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
        actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
        actor_rollout_ref.rollout.n=${ROLLOUT_N} \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        +actor_rollout_ref.rollout.trust_remote_code=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        +actor_rollout_ref.rollout.n_val=1 \
        algorithm.use_kl_in_reward=${USE_KL_IN_REWARD} \
        algorithm.kl_ctrl.kl_coef=${KL_COEF} \
        algorithm.kl_ctrl.type="fixed" \
        algorithm.lam=0.95 \
        reward_model.reward_manager="custom" "$@" \
        trainer.logger='["console", "wandb"]' \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${experiment_name}" \
        trainer.n_gpus_per_node="${num_gpus}" \
        trainer.nnodes="${world_size}" \
        trainer.save_freq=20 \
        trainer.test_freq=100 \
        trainer.total_epochs=100 \
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
