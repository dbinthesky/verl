import os
import re
import json
import string
import random
import unittest
from tqdm import tqdm
import pandas as pd
import asyncio as aio
from pt_refine import (
    get_notes_and_conclusions,
)
from pt_mine import (
    parse_solution_fn,    CoTRecall,
    QwQLongCoTPretrainMiningComputeScore,
    QwQLongCoTQuestionRefineComputeScore,
    qwq_longcot_question_refine_compute_score_valid
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


def random_generate_doc(doc_size=500):
    all_characters = string.ascii_letters + string.digits + " "
    n = 8
    doc = []
    for i in range(doc_size):
        doc.append(''.join(random.choice(all_characters) for _ in range(n)))
    return " ".join(doc)


def load_pretrain_mine(num=100):
    filename = "/cpfs01/shared/llm_ddd/tongjian/verl/rewards/pt_refine.json"
    batch_solution_str, batch_ground_truth = [], []

    with open(filename, "rt") as f:
        data = json.load(f)
    batch_solution_str, batch_ground_truth = data["batch_solution_str"], data["batch_ground_truth"]
    batch_solution_str, batch_ground_truth = batch_solution_str[:num], batch_ground_truth[:num]

    def postprocess(s):
        notes = re.findall(r'\[Note\] Q: (.*) Think: (.*)\[/Note\]', s)
        return "```xml\n"+"\n".join([f'<question>\n{_[0]}\n</question>\n<cot-in-document>\n{_[1]}\n</cot-in-document>'for _ in notes]) + "\n```"

    batch_solution_str = [postprocess(_) for _ in batch_solution_str]
    return batch_solution_str, batch_ground_truth


def load_query_refine(num=100):
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/doc2query/doc2query_question_refine_test"

    df = pd.read_parquet(filename)
    batch_solution_str, batch_ground_truth = [], []

    for _, row in df.iterrows():
        row = row.to_dict()
        batch_ground_truth.append(row["reward_model"])
        batch_solution_str.append(
            f'[QUESTION REFINED]\n{row["reward_model"]["raw_question"]}\n[/QUESTION REFINED]')

    return batch_solution_str, batch_ground_truth


class TestPretrainMine(unittest.TestCase):
    def test_cot_recall(self):
        batch_solution_str, batch_ground_truth = load_pretrain_mine(num=100)
        penalty = CoTRecall(
            postprocess_solution_fn=parse_solution_fn)
        for solution_str, ground_truth in zip(batch_solution_str, batch_ground_truth):
            print(penalty.get_penalty_or_reward(
                solution_str, ground_truth))

    def test_get_single_question_judge_rm_rewards(self):
        async def main():
            batch_solution_str, batch_ground_truth = load_pretrain_mine(
                num=100)
            task = QwQLongCoTPretrainMiningComputeScore(split="valid")
            results = await task.bank_covery_rewards(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            print(results)
        aio.run(main())

    def test_compute_score(self):
        async def main():
            batch_solution_str, batch_ground_truth = load_pretrain_mine(
                num=10)
            task = QwQLongCoTPretrainMiningComputeScore(split="valid")
            results = await task._compute_score(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            print(results)
        aio.run(main())

    def test_question_validation(self):
        async def main():
            batch_solution_str, batch_ground_truth = load_pretrain_mine(
                num=10)
            task = QwQLongCoTPretrainMiningComputeScore(split="valid")
            results = await task.question_validation(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            print(results)
        aio.run(main())


class TestQuestionRefine(unittest.TestCase):
    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_query_refine(
            num=10)
        results = qwq_longcot_question_refine_compute_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth
        )
        # async def main():
        #     batch_solution_str, batch_ground_truth = load_query_refine(
        #         num=10)
        #     task = QwQLongCoTQuestionRefineComputeScore(split="valid")
        #     results = await task._compute_score(
        #         [None] *
        #         len(batch_solution_str), batch_solution_str, batch_ground_truth
        #     )
        #     print(results)
        # aio.run(main())


if __name__ == '__main__':
    unittest.main()
