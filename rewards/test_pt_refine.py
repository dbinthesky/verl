import json
import string
import random
import unittest
import pandas as pd
from pt_refine import (
    pretrain_postprocess,
    parse_doc_wo_notes,
    parse_doc_w_notes,
    MainBodyRecall,
    LengthDiffPenalty
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
    filename = "/cpfs01/shared/llm_ddd/tongjian/verl/rewards/pt_refine.json"
    batch_solution_str, batch_ground_truth = [], []

    with open(filename, "rt") as f:
        data = json.load(f)
    batch_solution_str, batch_ground_truth = data["batch_solution_str"], data["batch_ground_truth"]
    return batch_solution_str, batch_ground_truth


class TestRulebasedPostprocess(unittest.TestCase):
    def test_main_body_recall(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        recall = MainBodyRecall(
            postprocess_solution_fn=parse_doc_wo_notes)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            recall.get_penalty_or_reward(
                solution_str, ground_truth)

    def test_length_diff_penalty(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        penalty = LengthDiffPenalty(
            postprocess_solution_fn=parse_doc_w_notes)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth))


if __name__ == '__main__':
    unittest.main()
