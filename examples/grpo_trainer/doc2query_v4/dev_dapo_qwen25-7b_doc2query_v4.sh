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
    conda activate /mnt/shared-storage-user/ailab-hx/gaoxuan/miniconda3/envs/verl
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

    ROLLOUT_N=4
    TRAIN_BSZ=$((num_gpus * world_size))
    KL_LOSS_COEF="0.5"
    TEMPERATURE="1.0"

    CUSTOM_CODE_DIR="${HOME}/verl"
    VERL_DIR="${HOME}/verl"
    # [PLACEHOLDER]
    BASE_MODEL_PATH="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
    # BASE_MODEL_PATH="${CKPTS_DIR}/datareview_sft_test/Qwen_30B_A3_instruct_20250425_s1_32k_S2_32k_baseline/20250716000644/hf-30"

    # [PLACEHOLDER]
    # TRAIN_DATA='["/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v4/zhihu_article_and_qa_1_0_0_8k_rl_inputs_train/index0.parquet","/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v4/zhihu_article_and_qa_1_0_0_8k_rl_inputs_train/index1.parquet", "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v4/zhihu_article_and_qa_1_0_0_8k_rl_inputs_train/index2.parquet", "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v4/zhihu_article_and_qa_1_0_0_8k_rl_inputs_train/index3.parquet", "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v4/zhihu_article_and_qa_1_0_0_8k_rl_inputs_train/index4.parquet", "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v4/zhihu_article_and_qa_1_0_0_8k_rl_inputs_train/index5.parquet", "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v4/zhihu_article_and_qa_1_0_0_8k_rl_inputs_train/index6.parquet", "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v4/zhihu_article_and_qa_1_0_0_8k_rl_inputs_train/index7.parquet", "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v4/zhihu_article_and_qa_1_0_0_8k_rl_inputs_train/index8.parquet", "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v4/zhihu_article_and_qa_1_0_0_8k_rl_inputs_train/index9.parquet"]' 
    TRAIN_DATA='/mnt/shared-storage-user/ailab-hx/tongjian/rl/doc2query_v4/pretrain_general_doc_8k_rl_inputs_train_sample5k.parquet'
    VAL_DATA='/mnt/shared-storage-user/ailab-hx/tongjian/rl/doc2query_v4/pretrain_general_doc_8k_rl_inputs_test.parquet'

    experiment_name="doc2query_v4_30b_a3_zhihu_${YYMMDD}_roll${ROLLOUT_N}_${TRAIN_BSZ}_dapo_kl_coef_${KL_LOSS_COEF}_wo_entropy_t${TEMPERATURE}"
    project_name="doc2query_v4"

    OUTPUT_DIR="${CKPTS_DIR}/datareview_rl_test/verl/grpo/doc2query_v4/${experiment_name}/"
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

    python3 -m recipe.dapo.main_dapo \
        custom_reward_function.path="${CUSTOM_CODE_DIR}/rewards/fabricate_qa.py" \
        custom_reward_function.name=doc2query_v4_compute_score_train \
        +custom_valid_reward_function.path="${CUSTOM_CODE_DIR}/rewards/fabricate_qa.py" \
        +custom_valid_reward_function.name=doc2query_v4_compute_score_valid \
        algorithm.adv_estimator="grpo" \
        algorithm.use_kl_in_reward=False \
        data.train_files="${TRAIN_DATA}" \
        data.val_files="${VAL_DATA}" \
        data.train_batch_size=${TRAIN_BSZ} \
        data.max_prompt_length=8192 \
        data.max_response_length=12288 \
        data.filter_overlong_prompts=True \
        data.filter_overlong_prompts_workers=256 \
        trainer.default_local_dir="${OUTPUT_DIR}" \
        trainer.val_before_train=False \
        actor_rollout_ref.model.path="${BASE_MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.shuffle=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BSZ} \
        actor_rollout_ref.actor.ppo_micro_batch_size=${TRAIN_BSZ} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.3 \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        reward_model.overlong_buffer.enable=True \
        reward_model.overlong_buffer.len=$((1024 * 4)) \
        reward_model.overlong_buffer.penalty_factor=1.0 \
        algorithm.filter_groups.enable=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
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
        algorithm.kl_ctrl.kl_coef=0.001 \
        algorithm.kl_ctrl.type="fixed" \
        algorithm.lam=0.95 \
        reward_model.reward_manager=dapo_custom \
        trainer.logger='["console", "wandb"]' \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${experiment_name}" \
        trainer.n_gpus_per_node="${num_gpus}" \
        trainer.nnodes="${world_size}" \
        trainer.save_freq=10 \
        trainer.test_freq=100 \
        trainer.total_epochs=10 \
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