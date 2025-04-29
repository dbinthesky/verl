INPUT="/cpfs01/shared/llm_ddd/tongjian/cargoflow/work_dirs/reason_pretrain_v3_train_input_0427_2025-04-27-17-45-03_grpo_step_30/predictions"
PARALLEL=20
OUTPUT="reason_pretrain_v3_train_sft"


seq 0 19 | shuf | parallel --lb -j ${PARALLEL} \
   "python3 pt_refine_data_clean.py -i ${INPUT} -o ${OUTPUT} -n {}"

