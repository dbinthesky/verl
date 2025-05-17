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
    criteria_get_score,
    question_similarity,
    QwQLongCoTCreateCriteriaComputeScore,
    qwq_longcot_create_criteria_compute_score_valid
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

    def test_question_similarity(self):
        batch_solution_str, batch_ground_truth = load_criteria()

        x, y = [], []
        for solution_str, gt in zip(batch_solution_str, batch_ground_truth):
            criteria = criteria_parse_solution_fn(solution_str)
            if criteria is not None:
                x.append(gt["positive"])
                y.append(random.choice(gt["negatives"]))

        async def main():
            print(await question_similarity(x, y))
        aio.run(main())

    def test_calc_compression_ratio_reward(self):
        task = QwQLongCoTCreateCriteriaComputeScore()
        batch_solution_str, batch_ground_truth = load_criteria()

        async def main():
            print(await task.calc_compression_ratio_reward(
                [None]*len(batch_solution_str),
                batch_solution_str,
                batch_ground_truth))
        aio.run(main())

    def test_calc_classify_acc_reward(self):
        task = QwQLongCoTCreateCriteriaComputeScore(split="valid")
        batch_solution_str, batch_ground_truth = load_criteria()
        batch_solution_str = batch_solution_str[:10]
        batch_ground_truth = batch_ground_truth[:10]

        # async def main():
        #     print(await qwq_longcot_create_criteria_compute_score_valid(
        #         [None]*len(batch_solution_str),
        #         batch_solution_str,
        #         batch_ground_truth))
        # aio.run(main())
        qwq_longcot_create_criteria_compute_score_valid(
            [None]*len(batch_solution_str),
            batch_solution_str,
            batch_ground_truth)


def load_criteria_infer():
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/hard_case_mixed/gpqa/super_gpqa_aio_noneasy_train_0513_criteria_output.jsonl"

    def preprocess(batch):
        batch_solution_str, batch_ground_truth = [], []
        for elem in batch:
            batch_solution_str.append(
                elem["self_improvement"]["responses"][0]["response"]["text"])
            batch_ground_truth.append(
                {
                    "positive": elem["self_improvement"]["question"],
                    "negatives": elem["self_improvement"]["fabricate_questions"]
                }
            )
        return batch_solution_str, batch_ground_truth, batch

    batch = []
    with open(filename, "rt") as f:
        for line in f:
            example = json.loads(line)
            batch.append(example)

            if len(batch) == 1024:
                yield preprocess(batch)
                batch = []
    if len(batch):
        yield preprocess(batch)


async def offline_compute_score():
    with open("/cpfs01/shared/llm_ddd/tongjian/rl/hard_case_mixed/gpqa/_super_gpqa_aio_noneasy_train_0513_criteria_output2.jsonl", "wt") as g:
        max_concurrent_requests = 256
        task = QwQLongCoTCreateCriteriaComputeScore(split="train")
        for batch_solution_str, batch_ground_truth, batch_examples in load_criteria_infer():
            data_sources = [None] * len(batch_solution_str)
            score1, score2 = await task.calc_classify_acc_reward(
                data_sources,
                batch_solution_str, batch_ground_truth,
                max_concurrent_requests=max_concurrent_requests,
                return_single_score=False
            )
            calc_compression_ratio_reward = await task.calc_compression_ratio_reward(
                data_sources,
                batch_solution_str, batch_ground_truth,
                max_concurrent_requests=max_concurrent_requests
            )
            for example, _score1, _score2, _score3 in zip(batch_examples, score1, score2, calc_compression_ratio_reward):
                example["calc_compression_ratio_reward"] = _score3
                example["classify_acc_reward"] = _score1
                example["gt_match_score"] = _score2
                g.write(f'{json.dumps(example, ensure_ascii=False)}\n')


if __name__ == '__main__':
    # async def main():
    #     await create_mock_data()
    # aio.run(main())
    # unittest.main()

    async def main():
        await offline_compute_score()
    aio.run(main())
