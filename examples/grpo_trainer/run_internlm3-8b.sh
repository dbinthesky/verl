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
setup_proxy

# ------------------------------
# Conda Environment Setup
# ------------------------------
activate_conda() {
    # Hard Code: InternLM3-8B支持
    source /cpfs01/shared/llm_ddd/guoxu/public/verl_env/verl_env/bin/activate /cpfs01/shared/llm_ddd/guoxu/public/verl
}
activate_conda


# ------------------------------
# Path Configuration
# ------------------------------
setup_path() {
    YYMMDD=$(date +%Y-%m-%d)
    HHMMSS=$(date +%H-%M-%S)

    # CUSTOM_CODE_DIR="/cpfs01/shared/llm_ddd/zhangjin/codehub/overthink/code_zj"
    # VERL_DIR="/cpfs01/shared/llm_ddd/zhangjin/codehub/overthink/verl"
    BASE_MODEL_PATH="/cpfs01/shared/llm_ddd/tongjian/ckpts/datareview_sft_test/DATAREVIEW_SFT_TEST_internlm3_dense8B_xml_cot_v19_253_open_source_hf"
    TRAIN_DATA="/cpfs01/shared/llm_ddd/tongjian/rl/hard_case_mixed/hard_case_mixed_v0_0_1_train.parquet"
    VAL_DATA="/cpfs01/shared/llm_ddd/tongjian/rl/hard_case_mixed/hard_case_mixed_v0_0_1_test.parquet"

    experiment_name="internlm3-8b_xml_cot-dlc-${HHMMSS}"
    project_name="verl_grpo_xml_cot"

    OUTPUT_DIR="/cpfs01/shared/llm_ddd/tongjian/ckpts/datareview_rl_test/verl/grpo/${experiment_name}/${YYMMDD}/${HHMMSS}"
    sudo mkdir -p "${OUTPUT_DIR}"
}
setup_path



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
        custom_reward_function.path="${CUSTOM_CODE_DIR}/rewards/math_reward_w_remote.py" \
        custom_reward_function.name=custom_reward_fn \
        algorithm.adv_estimator="grpo" \
        data.train_files="${TRAIN_DATA}" \
        data.val_files="${VAL_DATA}" \
        data.train_batch_size=768 \
        data.max_prompt_length=1892 \
        data.max_response_length=32768 \
        data.filter_overlong_prompts=True \
        data.truncation="error" \
        trainer.default_local_dir="${OUTPUT_DIR}" \
        actor_rollout_ref.model.path="${BASE_MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=2e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.shuffle=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size=$((total_gpus)) \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=36864 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type="low_var_kl" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name="vllm" \
        actor_rollout_ref.rollout.max_num_batched_tokens=300000 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.temperature=1.0 \
        +actor_rollout_ref.rollout.val_temperature=0.6 \
        actor_rollout_ref.rollout.n=8 \
        +actor_rollout_ref.rollout.n_val=1 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        algorithm.lam=0.95 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${experiment_name}" \
        +trainer.val_before_train=True \
        trainer.n_gpus_per_node="${num_gpus}" \
        trainer.nnodes="${world_size}" \
        trainer.save_freq=-1 \
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
run_training "$@"