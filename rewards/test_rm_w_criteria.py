import os
import json
import random
import unittest
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import asyncio as aio
from rm_w_criteria import (
    batchify,
    compute_rm_score,
    postprocess_solution,
    simple_tokenize,
    qwq_longcot_compute_score_train,
    ConclusionTooLongPenalty,
    FabricateQATooLongPenalty,
    FabricateQALengthPenalty,
    QwQLongCoTComputeScore,
    BleuSimilarity,
    QwQLongCoTFabricateQAComputeScore,
    QwQLongCoTFabricateQAComputeScoreV2,
    QwQLongCoTCriteriaEnvolveComputeScore,
    QwQLongCoTPretrainBackTranslationComputeScore,
    QwQLongCoTSFTBackTranslationComputeScore,
    CoTPretrainRefineComputeScore,
    CoTPretrainRLComputeScore,
    CoTPretrainAnnotationComputeScore,
    ROUGEScorer,
    ROUGEScorerForPretrainAnnotation,
    CorpusLengthPenalty,
    qwq_longcot_fabricate_qa_compute_score_train,
    qwq_longcot_fabricate_qa_compute_score_v2_valid
)


def generate_random_string(n):
    all_characters = string.ascii_letters + string.digits + " "
    return ''.join(random.choice(all_characters) for _ in range(n))


def load_xml_cot_data(num):
    TEST_CASE = "/cpfs01/shared/llm_ddd/tongjian/ddm/thought_xml/verify_enhance/xml_verify_enhance_v2.jsonl"

    batch_solution_str, batch_ground_truth = [], []
    correct_indices, wrong_indices = [], []
    with open(TEST_CASE, "rt") as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            try:
                prompt = example["self_improvement"]["prompt"]
                corrects = example["self_improvement"]["responses"]
                wrongs = example["self_improvement"]["wrong_responses"]
                if len(corrects) > 0 and len(wrongs) > 0:
                    correct = random.choice(corrects)
                    wrong = random.choice(wrongs)

                    batch_solution_str.append(correct["conclusion"])
                    batch_ground_truth.append({
                        "ground_truth": f'{prompt}\n\n\n\nJudge only by determining whether the final answer is correct.\n** Final Answer: {example["self_improvement"]["reference_meta"]["final_answer"]}',
                        "extra_info": {
                            "question_type": "object"
                        }
                    })
                    correct_indices.append(len(batch_ground_truth)-1)

                    batch_solution_str.append(wrong["conclusion"])
                    batch_ground_truth.append({
                        "ground_truth": f'{prompt}\n\n\n\nJudge only by determining whether the final answer is correct.\n** Final Answer: {example["self_improvement"]["reference_meta"]["final_answer"]}',
                        "extra_info": {
                            "question_type": "object"
                        }
                    })
                    wrong_indices.append(len(batch_ground_truth)-1)

            except Exception as err:
                continue
            if i > num:
                break
        return batch_solution_str, batch_ground_truth, correct_indices, wrong_indices


def load_qwq_data(num=100):
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/hard_case_mixed/hard_case_mixed_v0_0_1_aug_v1_finish.jsonl"
    batch_solution_str, batch_ground_truth = [], []

    with open(filename, "rt") as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            batch_solution_str.append(
                example["self_improvement"]["responses"][0]["response"])
            batch_ground_truth.append({
                "ground_truth": f'{example["self_improvement"]["prompt"]}\n\nFinal Answer: Unknown',
                "extra_info": {
                    "question_type": random.choice(["object", "subject"])
                }
            })
            if i > num:
                break
    return batch_solution_str, batch_ground_truth


def load_qwq_fabricate_qa_data(num=100):
    # filename = "/cpfs01/shared/llm_ddd/tongjian/rl/fabricate_qa/authentic_qa_aio_20250115_test_bugfix_0329.parquet"
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/sandbox_fabricate/sandbox_data_fabricate_qa_valid.parquet"
    batch_solution_str, batch_ground_truth = [], []

    df = pd.read_parquet(filename)
    for _, row in df.iterrows():
        row = row.to_dict()
        batch_ground_truth.append(row["reward_model"])
        gt = QwQLongCoTFabricateQAComputeScore.extract_gt_question(
            row["reward_model"])

        batch_solution_str.append(
            f'<think>\n{generate_random_string(100)}\n</think>\n\n<question>{gt}\n{generate_random_string(2000)}</question>')
    return batch_solution_str, batch_ground_truth


def load_cot_pretrain_rl(num=100):
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/pretrain_rl/pretrain_rl_pdf_knowledge_4k_finish_0421/part_0.parquet"
    batch_solution_str, batch_ground_truth = [], []

    df = pd.read_parquet(filename)
    for _, row in df.iterrows():
        row = row.to_dict()
        if _ > 100:
            break
        batch_ground_truth.append(row["reward_model"])
        gt = row["reward_model"]["ground_truth"]
        batch_solution_str.append(
            f'<chain-of-thought>\n{generate_random_string(100)}\n</chain-of-thought>\n\n<doc>\n{gt}\n</doc>')
    return batch_solution_str, batch_ground_truth


def load_pretrain_refinement(num=100):
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/pretrain_rl/nv-nemotron-cc-medium_100w/part_0.parquet"
    batch_solution_str, batch_ground_truth = [], []

    df = pd.read_parquet(filename)
    for _, row in df.iterrows():
        row = row.to_dict()
        if _ > 100:
            break
        batch_ground_truth.append(row["reward_model"])
        gt = row["reward_model"]["ground_truth"]
        batch_solution_str.append(
            f'<chain-of-thought>\n{generate_random_string(100)}\n</chain-of-thought>\n\n<doc>\n{gt}\n> [Note] xxxx \n>\n> xxx [/Note]\n</doc>')
        # batch_solution_str.append(
        #     f'<chain-of-thought>\n{generate_random_string(100)}\n</chain-of-thought>\n\n<doc>\n{gt}\n> [Note] xxxx \n>\n> xxx [/Note]\n{gt}\n</doc>')
        # batch_solution_str.append(
        #     f'<chain-of-thought>\n{generate_random_string(100)}\n</chain-of-thought>\n\n<doc>\n{gt[:1000]}\n> 【注】 xxxx \n>\n> xxx 【/注】\n</doc>')
    return batch_solution_str, batch_ground_truth


def load_qwq_criteria_envolve_data(num=100):
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/pretrain_bt/bt_seed_discipline_250426_4k_test/part_0.parquet"
    batch_solution_str, batch_ground_truth = [], []

    df = pd.read_parquet(filename)
    for _, row in df.iterrows():
        row = row.to_dict()
        batch_ground_truth.append(row["reward_model"])
        if "reference" not in row["reward_model"] or row["reward_model"]["reference"] == "":
            continue

        # batch_solution_str.append(
        #     f'<think>\n{generate_random_string(500)}\n</think>\n\n{row["reward_model"]["reference"]}')
        batch_solution_str.append(
            f'<think>\n{generate_random_string(500)}\n</think>\n\n{row["reward_model"]["reference"]}')
    return batch_solution_str, batch_ground_truth


def load_qwq_back_translation_data(num=100, mode="pretrain", filename=None):
    if filename is None:
        if mode == "pretrain":
            filename = "/cpfs01/shared/llm_ddd/tongjian/rl/pretrain_bt/pdf_2025_v1_zh_en_4k_test/part_0.parquet"
        else:
            filename = "/cpfs01/shared/llm_ddd/tongjian/rl/sft_bt/internlm3_s2_release_0213_8k_sample100_test/part_0.parquet"

    batch_solution_str, batch_ground_truth = [], []

    df = pd.read_parquet(filename)
    for _, row in df.iterrows():
        row = row.to_dict()
        batch_ground_truth.append(row["reward_model"])

        batch_solution_str.append(
            f'<think>\n{generate_random_string(500)}\n</think>\n\n<instruction>\n写一篇文档\n</instruction>')
        # batch_solution_str.append(
        #     f'<think>\n{generate_random_string(500)}\n</think>\n\n[PROMPT]\n# INSTRUCTION\{row["reward_model"]["ground_truth"]}\n# INPUT')
        # batch_solution_str.append(
        #     f'<think>\n{generate_random_string(500)}\n</think>\n\n\n写一篇文档')
    return batch_solution_str, batch_ground_truth


class TestRMReward(unittest.TestCase):
    def test_grid_search_rm_threshold(self):
        num = 10000
        threshold = 0.0
        batch_solution_str, batch_ground_truth, correct_indices, wrong_indices = load_xml_cot_data(
            num)

        precision, recall = [], []
        for i, score in enumerate(compute_rm_score(batch_solution_str, batch_ground_truth, postprocess_solution)):
            if score >= threshold:
                if i in correct_indices:
                    precision.append(1.)
                else:
                    precision.append(0.)
            else:
                if i in wrong_indices:
                    recall.append(1.)
                else:
                    recall.append(0.)
        print(f'Precision={np.mean(precision)*100:.2f}% ({len(precision)})')
        print(f'Recall={np.mean(recall)*100:.2f}% ({len(recall)})')

    def test_solution_len_analysis(self):
        num = 10000
        batch_solution_str, batch_ground_truth, correct_indices, wrong_indices = load_xml_cot_data(
            num)

        size = []
        for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            size.append(len(simple_tokenize(solution_str)))
        size = sorted(size)
        for _ in (0.85, 0.9, 0.95):
            print(f'{_}% Length={size[int(len(size)*_)]}')

    def test_conclusion_too_long_penalty(self):
        penalty_fn = ConclusionTooLongPenalty(
            postprocess_solution_fn=QwQLongCoTComputeScore.postprocess_solution)
        batch_solution_str, batch_ground_truth = load_qwq_data()

        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            self.assertTrue(penalty_fn.get_penalty_or_reward(
                solution_str, ground_truth) <= 0.0)

    def test_fabricate_qa_too_long_penalty(self):
        penalty_fn = FabricateQATooLongPenalty(
            postprocess_solution_fn=QwQLongCoTFabricateQAComputeScore.postprocess_solution_fn,
            postprocess_gt_fn=QwQLongCoTFabricateQAComputeScore.extract_gt_question,
        )
        batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data()
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty_fn.get_penalty_or_reward(solution_str, ground_truth))

    def test_qwq_long_cot_fabricate_qa_compute_score(self):
        batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
            num=100)

        qwq_longcot_fabricate_qa_compute_score_train(
            [None] * len(batch_solution_str),
            batch_solution_str,
            batch_ground_truth
        )

    def test_qwq_long_cot_pretrain_back_translation_compute_score(self):
        task = QwQLongCoTPretrainBackTranslationComputeScore(
            parse_result_failure_score=-2, split="valid"
        )
        batch_solution_str, batch_ground_truth = load_qwq_back_translation_data()
        batch_solution_str, batch_ground_truth = batch_solution_str[:100], batch_ground_truth[:100]

        print(task.compute_score(
            [None] * len(batch_solution_str),
            batch_solution_str,
            batch_ground_truth
        ))

    def test_qwq_long_cot_sft_back_translation_compute_score(self):
        task = QwQLongCoTSFTBackTranslationComputeScore(
            parse_result_failure_score=-2, split="valid"
        )
        batch_solution_str, batch_ground_truth = load_qwq_back_translation_data(
            mode="sft")
        print(task.compute_score(
            [None] * len(batch_solution_str),
            batch_solution_str,
            batch_ground_truth
        ))

    def test_qwq_long_cot_back_translation_stats(self):
        task = QwQLongCoTPretrainBackTranslationComputeScore(
            parse_result_failure_score=-2, split="valid"
        )
        batch_solution_str, batch_ground_truth = load_qwq_back_translation_data(
            filename="/cpfs01/shared/llm_ddd/tongjian/rl/sft_bt/internlm2_5_s2_release_8k_train/part_0.parquet")
        pbar = tqdm(total=len(batch_solution_str))
        info_of_prompt = []
        for batch in batchify(batch_ground_truth, 128):
            _batch_ground_truth = [_["ground_truth"] for _ in batch]
            info_of_prompt.extend(task.compute_bt_reward(
                [None] * len(_batch_ground_truth),
                _batch_ground_truth
            ))
            print(np.mean(info_of_prompt))
            pbar.update(len(_batch_ground_truth))
        print(np.mean(info_of_prompt))

    def test_qwq_long_cot_criteria_envolve_compute_score(self):
        task = QwQLongCoTCriteriaEnvolveComputeScore(
            parse_result_failure_score=0.)
        batch_solution_str, batch_ground_truth = load_qwq_criteria_envolve_data(
            num=100)

        print(task.compute_score(
            [None] * len(batch_solution_str),
            batch_solution_str,
            batch_ground_truth
        ))

    def test_fabricate_qa_length_penalty(self):
        penalty_fn = FabricateQALengthPenalty(
            postprocess_solution_fn=QwQLongCoTFabricateQAComputeScore.postprocess_solution_fn,
            postprocess_gt_fn=QwQLongCoTFabricateQAComputeScore.extract_gt_question,
        )
        batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data()
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty_fn.get_penalty_or_reward(solution_str, ground_truth))

    def test_qwq_long_cot_fabricate_qa_compute_score_v2(self):
        batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data(
            num=100)

        qwq_longcot_fabricate_qa_compute_score_v2_valid(
            [None] * len(batch_solution_str),
            batch_solution_str,
            batch_ground_truth
        )

    def test_cot_pretrain_rl_compute_score(self):
        batch_solution_str, batch_ground_truth = load_cot_pretrain_rl(
            num=100)
        task = CoTPretrainRLComputeScore(split="valid")
        task.compute_score(
            [None] * len(batch_solution_str),
            batch_solution_str,
            batch_ground_truth
        )

    def test_cot_pretrain_refinement_compute_score(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)

        task = CoTPretrainRefineComputeScore(split="valid")
        scorer = ROUGEScorer(
            postprocess_solution_fn=task.postprocess_for_rouge)
        len_penalty = CorpusLengthPenalty(
            postprocess_solution_fn=task.postprocess_for_rouge, postprocess_gt_fn=lambda x: x["ground_truth"])

        task.compute_score(
            [None] * len(batch_solution_str),
            batch_solution_str,
            batch_ground_truth
        )

    # def test_cot_pretrain_annotation_compute_score(self):
    #     gt = "# 5SBF\n#### PanDDA analysis group deposition of ground-state model of SARS-CoV-2 NendoU\nChanges made to a PDB entry after its initial release are considered to be either “major” or “minor”. The latest minor version of each major version is available as a file download. More information about the PDB versioning is available.\n| Version Number | Version Date | Version Type/Reason | Version Change | Revised CIF Category |  |\n|---|---|---|---|---|---|\n| 1.0 | 2022-02-09 | Initial release |  |  |  |\n| 1.1 | 2023-05-10 |  | Database references | citation, citation_author |  |\n| 1.2 | 2023-06-21 |  | Database references | citation |  |\n| 1.3 | 2024-05-22 |  | Data collection | chem_comp_atom, chem_comp_bond | Download |"
    #     solution = ""
    #     with open("/cpfs01/shared/llm_ddd/tongjian/verl/rewards/solution.txt", "rt") as f:
    #         for line in f:
    #             solution += line
    #     # batch_solution_str, batch_ground_truth = load_pretrain_refinement(
    #     #     num=100)

    #     task = CoTPretrainAnnotationComputeScore(split="valid")
    #     # scorer = ROUGEScorerForPretrainAnnotation(
    #     #     postprocess_solution_fn=task.postprocess_for_rouge)

    #     # # for solution_str, gt in zip(batch_solution_str, batch_ground_truth):
    #     # #     print(scorer.get_penalty_or_reward(solution_str, gt))
    #     # #     break
    #     task.compute_score(
    #         [None],
    #         [solution],
    #         [{"ground_truth": gt}]
    #     )


if __name__ == '__main__':
    unittest.main()
