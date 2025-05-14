import json
import random
import unittest
import pandas as pd
import asyncio as aio
from fabricate_qa import (
    agent,
    criteria_parse_solution_fn,
    get_total_score,
    decode_to_question,
    criteria_get_score
)


def load_criteria():
    filename = "/cpfs01/shared/llm_ddd/tongjian/verl/rewards/fabricate_qa_criteria.json"
    batch_solution_str, batch_ground_truth = [], []

    with open(filename, "rt") as f:
        data = json.load(f)
    batch_solution_str, batch_ground_truth = data["batch_solution_str"], data["batch_ground_truth"]
    return batch_solution_str, batch_ground_truth


async def create_mock_data():
    df = pd.read_parquet(
        "/cpfs01/shared/llm_ddd/tongjian/rl/fabricate_qa_criteria/super_gpqa_aio_noneasy_train_0513_criteria_test_0514.parquet")

    prompt_mapper = {}
    for _, row in df.iterrows():
        row = row.to_dict()
        prompt = f'{row["prompt"][0]["content"]}\n\n\n{row["prompt"][1]["content"]}'
        prompt_mapper[prompt] = row

    prompts = prompt_mapper.keys()

    batch_solution_str, batch_ground_truth = [], []

    results = await agent.run(prompts, 32, desc="[MOCK DATASET]", postprocess_fns=[lambda x: x]*len(prompts))
    for prompt, response in results:
        example = prompt_mapper[prompt]
        rm = example["reward_model"]
        rm = {
            "positive": rm["positive"],
            "negatives": rm["negatives"].tolist(),
        }
        batch_ground_truth.append(rm)
        batch_solution_str.append(response)
    with open("fabricate_qa_criteria.json", "wt") as f:
        json.dump({"batch_ground_truth": batch_ground_truth,
                  "batch_solution_str": batch_solution_str}, f, ensure_ascii=False)


class TestCriteria(unittest.TestCase):
    def test_criteria_get_score(self):
        batch_solution_str, batch_ground_truth = load_criteria()

        x, y = [], []
        for solution_str, gt in zip(batch_solution_str, batch_ground_truth):
            criteria = criteria_parse_solution_fn(solution_str)
            if criteria is not None:
                x.append(random.choice(gt["negatives"]))
                y.append(criteria)

        async def main():
            print(await criteria_get_score(x, y))
        aio.run(main())

    def test_decode_to_question(self):
        batch_solution_str, batch_ground_truth = load_criteria()

        x, y = [], []
        for solution_str, gt in zip(batch_solution_str, batch_ground_truth):
            criteria = criteria_parse_solution_fn(solution_str)
            if criteria is not None:
                x.append(random.choice(gt["negatives"]))
                y.append(criteria)

        y = y[:10]

        async def main():
            print(await decode_to_question(y))
        aio.run(main())


if __name__ == '__main__':
    # async def main():
    #     await create_mock_data()
    # aio.run(main())
    unittest.main()
