import json
import string
import random
import unittest
import pandas as pd
from pt_refine import (
    pretrain_postprocess,
    parse_doc_wo_notes,
    MainBodyRecall
)


def random_generate_doc():
    all_characters = string.ascii_letters + string.digits + " "
    n = 8
    doc_size = 500
    doc = []
    for i in range(doc_size):
        doc.append(''.join(random.choice(all_characters) for _ in range(n)))
    return " ".join(doc)


def load_pretrain_refinement(num=100):
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/pretrain_rl/reason_pretrain_v1_4k_train/part_0.parquet"
    batch_solution_str, batch_ground_truth = [], []

    df = pd.read_parquet(filename)
    for _, row in df.iterrows():
        row = row.to_dict()
        if _ > 100:
            break
        batch_ground_truth.append(row["reward_model"])
        gt = row["reward_model"]["ground_truth"]
        batch_solution_str.append(
            f'<chain-of-thought>\n{random_generate_doc()}\n</chain-of-thought>\n\n<doc>\n{gt}\n> [Note] xxxx \n>{random_generate_doc()}\n> xxx [/Note]\n</doc>')
        # batch_solution_str.append(
        #     f'<chain-of-thought>\n{generate_random_string(100)}\n</chain-of-thought>\n\n<doc>\n{gt}\n> [Note] xxxx \n>\n> xxx [/Note]\n{gt}\n</doc>')
        # batch_solution_str.append(
        #     f'<chain-of-thought>\n{generate_random_string(100)}\n</chain-of-thought>\n\n<doc>\n{gt[:1000]}\n> 【注】 xxxx \n>\n> xxx 【/注】\n</doc>')
    return batch_solution_str, batch_ground_truth


class TestRulebasedPostprocess(unittest.TestCase):
    def test_cot_pretrain_refinement_compute_score(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        recall = MainBodyRecall(
            postprocess_solution_fn=parse_doc_wo_notes)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(recall.get_penalty_or_reward(
                solution_str, ground_truth))


if __name__ == '__main__':
    unittest.main()
