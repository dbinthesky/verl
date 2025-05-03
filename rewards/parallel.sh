INPUT="/cpfs01/shared/llm_ddd/tongjian/cargoflow/work_dirs/reason_pretrain_v3_8k_train_en_input_2025-05-02-00-14-04_grpo_step_20/predictions"
PARALLEL=20
OUTPUT="ttmp"


seq 0 19 | shuf | parallel --lb -j ${PARALLEL} \
   "python3 pt_refine_data_clean.py -i ${INPUT} -o ${OUTPUT} -n {}"



# python3 pt_refine_data_clean.py -i /cpfs01/shared/llm_ddd/tongjian/cargoflow/work_dirs/reason_pretrain_v3_8k_train_input_2025-05-02-00-14-04_grpo_step_20/predictions \
#     -o ttp -n 20
