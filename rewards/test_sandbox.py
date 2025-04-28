import json
import string
import random
import unittest
import pandas as pd
from sandbox import (
    LengthDiffPenalty,
    TextSimilarity,
    fabricate_qa_postprocess_solution_fn,
    get_rm_rewards,
    stage1_qwq_longcot_fabricate_qa_compute_score_valid,
    stage1_qwq_longcot_fabricate_qa_compute_score_train,
)


def random_generate_doc():
    all_characters = string.ascii_letters + string.digits + " "
    n = 8
    doc_size = 10
    doc = []
    for i in range(doc_size):
        doc.append(''.join(random.choice(all_characters) for _ in range(n)))
    return " ".join(doc)


def load_fabricate_data(num=100):
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/sandbox_fabricate/sandbox_data_0428_fabricate_qa_test.parquet"
    batch_solution_str, batch_ground_truth = [], []

    df = pd.read_parquet(filename)
    for _, row in df.iterrows():
        row = row.to_dict()
        batch_ground_truth.append(row["reward_model"])
        if random.random() < 0.3:
            batch_solution_str.append(
                row["reward_model"]["ground_truth"] + random_generate_doc())
        else:
            batch_solution_str.append(
                f'<think>\n{random_generate_doc()}\n</think>\n\n<question>{row["reward_model"]["ground_truth"]+random_generate_doc()}</question>')
    return batch_solution_str, batch_ground_truth


class TestFabricateQA(unittest.TestCase):
    def test_length_diff_penalty(self):
        batch_solution_str, batch_ground_truth = load_fabricate_data(
            num=100)
        penalty = LengthDiffPenalty(
            postprocess_solution_fn=fabricate_qa_postprocess_solution_fn)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth))

    def test_text_similarity(self):
        batch_solution_str, batch_ground_truth = load_fabricate_data(
            num=100)
        penalty = TextSimilarity(
            postprocess_solution_fn=fabricate_qa_postprocess_solution_fn)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth))

    def test_rm_score(self):
        batch_solution_str, batch_ground_truth = load_fabricate_data(
            num=100)
        print(get_rm_rewards(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth
        ))

    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_fabricate_data(
            num=100)
        stage1_qwq_longcot_fabricate_qa_compute_score_train(
            [None] * len(batch_solution_str),
            batch_solution_str,
            batch_ground_truth
        )


if __name__ == '__main__':
    unittest.main()
