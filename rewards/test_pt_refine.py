import os
import re
import json
import string
import random
import unittest
from tqdm import tqdm
import pandas as pd
from pt_refine import (
    contain_chinese,
    pretrain_postprocess,
    parse_doc_wo_notes,
    parse_doc_w_notes,
    parse_solution_fn,
    MainBodyRecall,
    LengthDiffPenalty,
    NotesFormatReward,
    NotesRepetitionPenalty,
    QwQLongCoTPretrainRefineComputeScore,
    qwq_longcot_pretrain_refine_compute_score_valid,
    qwq_longcot_pretrain_refine_compute_score_train
)


def batchify(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


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

    def tag_modify(s):
        return s.replace("<chain-of-thought>", "<think>").replace("</chain-of-thought>", "</think>")
    batch_solution_str = [tag_modify(_) for _ in batch_solution_str]
    return batch_solution_str, batch_ground_truth


class TestPretrainRefine(unittest.TestCase):
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
            postprocess_solution_fn=parse_doc_wo_notes)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth))

    def test_notes_format_reward(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        penalty = NotesFormatReward(
            postprocess_solution_fn=parse_doc_w_notes)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth))

    def test_notes_repetition_penalty(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        penalty = NotesRepetitionPenalty(
            postprocess_solution_fn=parse_doc_w_notes)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth))

    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_pretrain_refinement(
            num=100)
        task = QwQLongCoTPretrainRefineComputeScore(split="valid")
        qwq_longcot_pretrain_refine_compute_score_train(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth
        )


if __name__ == '__main__':
    unittest.main()
