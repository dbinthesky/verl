import os
import re
import json
import uuid
import string
import random
import unittest
import pandas as pd
import asyncio as aio
from tqdm import tqdm
from map_reduce import (
    MapReduceComputeScore,
    compute_score_valid
)


def load_data(num=100):
    batch_solution_str, batch_ground_truth = [], []
    with open("/cpfs01/shared/llm_ddd/tongjian/verl/rewards/math_rewards.jsonl", "rt") as f:
        for line in f:
            example = json.loads(line)
            batch_solution_str.append(example["solution_str"])
            batch_ground_truth.append(
                {"ground_truth": example["ground_truth"], "prompt": "<skip>", "uuid": uuid.uuid4().hex})
    return batch_solution_str, batch_ground_truth


class TestMapReduce(unittest.TestCase):
    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_data()
        task = MapReduceComputeScore(split="valid")

        # async def main():
        #     results = await task._compute_score(
        #         [None] *
        #         len(batch_solution_str), batch_solution_str, batch_ground_truth,
        #     )
        # aio.run(main())

        print(compute_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth
        ))
        print(compute_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth
        ))


if __name__ == '__main__':
    unittest.main()
