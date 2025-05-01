INPUT="/cpfs01/shared/llm_ddd/tongjian/cargoflow/work_dirs/reason_pretrain_v3_8k_train_input_0427_2025-05-01-04-18-51_grpo_step_20/predictions"
PARALLEL=20
OUTPUT="reason_pretrain_v3_train_sft"


seq 0 19 | shuf | parallel --lb -j ${PARALLEL} \
   "python3 pt_refine_data_clean.py -i ${INPUT} -o ${OUTPUT} -n {}"

