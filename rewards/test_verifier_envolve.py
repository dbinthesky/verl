import tqdm
import asyncio as aio
import uuid
import os
import json
import random
import unittest
import string
import numpy as np
import pandas as pd
from verifier_envolve import (
    Agent,
    compute_score
)


def load_test_data():
    root = "/cpfs01/shared/llm_ddd/tongjian/rl/verifier/kaf_dataset_test.parquet"
    df = pd.read_parquet(root)

    examples, batch_ground_truth = [], []

    for _, row in df.iterrows():
        row = row.to_dict()
        output = row["reward_model"]
        examples.append(row)
        batch_ground_truth.append(output)

    return examples, batch_ground_truth


def load_mock_data():
    root = "/cpfs01/shared/llm_ddd/tongjian/verl/rewards/verifier_envolve_mock.jsonl"
    outputs = []
    with open(root, "rt") as f:
        for line in f:
            example = json.loads(line)
            outputs.append(example)
    return outputs


def mock_rollout_and_save():
    examples, batch_ground_truth = load_test_data()

    agent = Agent(**{
        "model": "qwen25_7B_instruct",
        "base_url": "http://10.130.247.138:8000/v1",
        "api_keys": "EMPTY",
        "request_kwargs": {
            "temperature": 0.7,
            "timeout": 30,
            "max_tokens": 2048
        },
    })

    async def main():
        prompts = [
            f'{example["prompt"][0]["content"]}\n\n{example["prompt"][1]["content"]}' for example in examples]
        max_concurrent_requests = 32
        results = await agent.run(prompts, max_concurrent_requests, desc="unittest", postprocess_fns=[lambda x: x]*len(prompts))

        with open("verifier_envolve_mock.jsonl", "wt") as f:
            for example, (prompt, response), gt in zip(examples, results, batch_ground_truth):
                if response.startswith("think>"):
                    response = f'<{response}'
                if "<answer_constraint>" not in response:
                    response = response.replace(
                        "</answer_constraint>", "[XXXYYYZZZAAABBBCCC]")
                    response = response.replace(
                        "answer_constraint>", "<answer_constraint>")
                    response = response.replace(
                        "[XXXYYYZZZAAABBBCCC]", "</answer_constraint>")
                if "<answer_extraction>" not in response:
                    response = response.replace(
                        "</answer_extraction>", "[XXXYYYZZZAAABBBCCC]")
                    response = response.replace(
                        "answer_extraction>", "<answer_extraction>")
                    response = response.replace(
                        "[XXXYYYZZZAAABBBCCC]", "</answer_extraction>")
                output = {
                    "solution_str": response,
                    "ground_truth": gt
                }
                f.write(f'{json.dumps(output, ensure_ascii=False)}\n')

    aio.run(main())


class TestVerifierEnvolve(unittest.TestCase):
    def test_compute_score(self):
        data = load_mock_data()
        batch_solution_str, batch_ground_truth = [], []
        for example in data:
            batch_solution_str.append(example["solution_str"])
            batch_solution_str.append(example["solution_str"])
            del example["solution_str"]
            batch_ground_truth.append(example)
            batch_ground_truth.append(example)

        # async def main():
        #     await compute_score([None]*len(batch_solution_str),
        #                         batch_solution_str, batch_ground_truth)
        # aio.run(main())
        print(compute_score([None]*len(batch_solution_str),
                                batch_solution_str, batch_ground_truth))


if __name__ == '__main__':
    # mock_rollout_and_save()
    # data = load_mock_data()
    unittest.main()

    # class TestAgent(unittest.TestCase):
    #         async def main():
    #             prompts = [
    #                 "Explain quantum computing in simple terms.",
    #                 "Write a haiku about artificial intelligence.",
    #                 "Describe the process of photosynthesis."
    #             ] * 5
    #             max_concurrent_requests = 2
    #             results = await agent.run(prompts, max_concurrent_requests, desc="unittest", postprocess_fns=[len]*len(prompts))
    #             for prompt, response in results:
    #                 # print(f"Prompt: {prompt}\nResponse: {response}\n")
    #                 self.assertTrue(f"Prompt: {prompt}\nResponse: {response}\n")

    #         aio.run(main())
