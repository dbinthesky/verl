import os
import re
import uuid
import time
import json
import random
import string
import unittest
import aiohttp
import pandas as pd
import asyncio as aio
from tqdm import tqdm
from self_taught import (
    Agent,
    RewardModelAgent,
    JudgeTwoQuestionSimilarity,
    parse_question_solution_fn,
    general_qa_parse_solution_fn,
    Doc2QuerySelfTaughtComputeScore,
    DOC2QUERY_ST_DEFAULT_PARAMS,
    parse_doc_fn,
    Query2DocComputeScore,
    QUERY2DOC_DEFAULT_PARAMS,
    parse_doc_fn,
    LengthReward
)

UNITTEST_AGENT = Agent(**{
    "model": "Qwen2.5-32B-Instruct",
    "base_url": "https://sd262bskcm47j59r1292g.apigateway-cn-beijing.volceapi.com/v1",
    "api_keys": "caa6246b-afbe-4d9b-ab34-87bf9922032b",
    "request_kwargs": {
        "temperature": 0.7,
        "timeout": 360,
        "max_tokens": 16384,
    },
})


def load_doc2query_st_dataset(num=100, xml_cot=False):
    filename = "/mnt/shared-storage-user/ailab-hx/tongjian/rl/fabricate_qa_self_taught/fabricate_qa_self_taught_cot_rl_input_test.parquet"

    batch_solution_str, batch_ground_truth = [], []
    batch_data_sources = []

    df = pd.read_parquet(filename)
    count = 0
    for _, row in df.iterrows():
        row = row.to_dict()
        batch_data_sources.append(row["data_source"])
        batch_solution_str.append(
            f'<think>\nkakakkaka\n</think>\n\n\n<question>\nQuestion: {row["reward_model"]["question"]}\n\nAnswer: {row["reward_model"]["answer"]}\n</question>'
        )
        batch_ground_truth.append(row["reward_model"])
        if len(batch_ground_truth) == num:
            break
    return batch_solution_str, batch_ground_truth


def load_query2doc_dataset(num=100, xml_cot=False):
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/query2doc/query2doc_inputs_train/index0.parquet"

    batch_solution_str, batch_ground_truth = [], []
    batch_data_sources = []

    df = pd.read_parquet(filename)
    count = 0
    for _, row in df.iterrows():
        row = row.to_dict()
        batch_data_sources.append(row["data_source"])
        batch_solution_str.append(
            f'<think>\n\n</think>\n\n\n[LECTURE]\nFUCK DOC\nQuestion: {row["reward_model"]["question"]}\n\nAnswer: {row["reward_model"]["answer"]}\n[/LECTURE]'
        )
        batch_ground_truth.append(row["reward_model"])
        if len(batch_ground_truth) == num:
            break
    return batch_solution_str, batch_ground_truth


class TestSelfTaught(unittest.TestCase):

    def test_thought_similarity(self):
        task = Doc2QuerySelfTaughtComputeScore(
            parse_solution_fn=general_qa_parse_solution_fn,
            args=DOC2QUERY_ST_DEFAULT_PARAMS
        )
        batch_solution_str, batch_ground_truth = load_doc2query_st_dataset(
            num=20)
        print(task.thought_similarity_reward(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth,
        ))

    def test_reward_model_agent(self):
        task = Doc2QuerySelfTaughtComputeScore(
            parse_solution_fn=general_qa_parse_solution_fn,
            args=DOC2QUERY_ST_DEFAULT_PARAMS
        )
        agent = task.rm_agent
        batch_solution_str = [
            "<think>今天天气怎么样？</think><question>xxx</question>"]
        batch_ground_truth = [{"ground_truth": "今天天气怎么样？"}]
        for i, score in enumerate(agent.compute_rm_score(batch_solution_str, batch_ground_truth)):
            print(score)

    def test_batch_call_open_api(self):
        task = JudgeTwoQuestionSimilarity()

        async def main():
            results = await task.do_job(
                agent=UNITTEST_AGENT,
                batch_inputs=[(
                    "During a drama club meeting, a member erroneously attributed the Salem witch trials play \"The Crucible,\" focusing on family conflict in 1692 Massachusetts, to Eugene O'Neill. Name the correct playwright who explored societal pressures through this drama",
                    "Who is considered the most important American playwright of the 20th century?\nA) Shepard\nB) Albee\nC) Williams\nD) Wilder\nE) Mamet\nF) O'Neal\nG) Miller\nH) Wilson\nI) Hellman\nJ) Simon")],
                max_concurrent_requests=64,
            )
            print(results)
        aio.run(main())

    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_doc2query_st_dataset(
            num=20)
        task = Doc2QuerySelfTaughtComputeScore(
            parse_solution_fn=general_qa_parse_solution_fn,
            args=DOC2QUERY_ST_DEFAULT_PARAMS
        )

        async def main():
            results = await task._compute_score(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())


class TestQuery2Doc(unittest.TestCase):
    def test_parse_doc_fn(self):
        batch_solution_str, batch_ground_truth = load_query2doc_dataset(
            num=20)
        print(parse_doc_fn(batch_solution_str[0]))

    def test_compute_score(self):
        task = Query2DocComputeScore(
            parse_solution_fn=parse_doc_fn,
            args=QUERY2DOC_DEFAULT_PARAMS,
            thought_log_prob=1.0
        )
        batch_solution_str, batch_ground_truth = load_query2doc_dataset(
            num=20)
        # agent = task.rm_agent
        # for i, score in enumerate(agent.compute_rm_score(batch_solution_str, batch_ground_truth, judge_prompt_key="rm_judge_prompt")):
        #     print(score)

        async def main():
            results = await task._compute_score(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())
