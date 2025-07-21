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
from criteria_rm import (
    Agent,
    RLVRVerify
)


UNITTEST_AGENT = Agent(**{
    "model": "qwen25_32B_instruct",
    "base_url": "http://10.130.142.154:8000/v1",
    "api_keys": "EMPTY",
    "request_kwargs": {
        "temperature": 0.7,
        "timeout": 360,
        "max_tokens": 16384,
    },
})


def load_data(num=100):
    batch_solution_str, batch_ground_truth = [], []
    with open("/cpfs01/shared/llm_ddd/tongjian/verl/rewards/math_rewards.jsonl", "rt") as f:
        for line in f:
            example = json.loads(line)
            batch_solution_str.append(example["solution_str"])
            batch_ground_truth.append(
                {"ground_truth": example["ground_truth"], "prompt": "<skip>", "uuid": uuid.uuid4().hex})
    return batch_solution_str, batch_ground_truth


class TestAutoPE(unittest.TestCase):

    def test_rlvr_verify(self):
        task = RLVRVerify()

        async def main():
            results = await task.do_job(
                agent=UNITTEST_AGENT,
                batch_inputs=[(
                    "54+45=89", {"prompt": "54+45=?", "ground_truth": "99"}
                )],
                max_concurrent_requests=64,
            )
            print(results)
        aio.run(main())

    def test_compute_score(self):
        pass
        # batch_solution_str, batch_ground_truth = load_data()
        # task = MapReduceComputeScore(split="valid")

        # # async def main():
        # #     results = await task._compute_score(
        # #         [None] *
        # #         len(batch_solution_str), batch_solution_str, batch_ground_truth,
        # #     )
        # # aio.run(main())

        # print(compute_score_valid(
        #     [None] *
        #     len(batch_solution_str), batch_solution_str, batch_ground_truth
        # ))
        # print(compute_score_valid(
        #     [None] *
        #     len(batch_solution_str), batch_solution_str, batch_ground_truth
        # ))


if __name__ == '__main__':
    unittest.main()
