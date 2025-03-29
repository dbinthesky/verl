import os
import json
import random
import unittest
import string
import numpy as np
import pandas as pd
from rm_w_criteria import (
    compute_rm_score,
    postprocess_solution,
    simple_tokenize,
    qwq_longcot_compute_score_train,
    ConclusionTooLongPenalty,
    FabricateQATooLongPenalty,
    QwQLongCoTComputeScore,
    QwQLongCoTFabricateQAComputeScore
)


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
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/fabricate_qa/authentic_qa_aio_20250115_test_bugfix_0329.parquet"
    batch_solution_str, batch_ground_truth = [], []

    def generate_random_string(n):
        all_characters = string.ascii_letters + string.digits
        return ''.join(random.choice(all_characters) for _ in range(n))

    df = pd.read_parquet(filename)
    for _, row in df.iterrows():
        row = row.to_dict()
        batch_ground_truth.append(row["reward_model"])
        gt = QwQLongCoTFabricateQAComputeScore.extract_gt_question(
            row["reward_model"])

        batch_solution_str.append(
            f'<think>\n{generate_random_string(100)}\n</think>\n\n<answer>{gt}\n{generate_random_string(500)}</answer>')
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
            self.assertTrue(penalty_fn.get_penalty(
                solution_str, ground_truth) <= 0.0)

    def test_fabricate_qa_too_long_penalty(self):
        penalty_fn = FabricateQATooLongPenalty(
            postprocess_solution_fn=QwQLongCoTFabricateQAComputeScore.postprocess_solution_fn,
            postprocess_gt_fn=QwQLongCoTFabricateQAComputeScore.extract_gt_question,
        )
        batch_solution_str, batch_ground_truth = load_qwq_fabricate_qa_data()
        print(len(batch_solution_str))
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty_fn.get_penalty(solution_str, ground_truth))

    def test_qwq_long_cot_compute_score(self):
        batch_solution_str, batch_ground_truth = load_qwq_data(num=100)

        # task = QwQLongCoTComputeScore()
        # task.compute_score(
        #     [None] * len(batch_solution_str),
        #     batch_solution_str,
        #     batch_ground_truth
        # )
        qwq_longcot_compute_score_train(
            [None] * len(batch_solution_str),
            batch_solution_str,
            batch_ground_truth
        )


if __name__ == '__main__':
    unittest.main()
