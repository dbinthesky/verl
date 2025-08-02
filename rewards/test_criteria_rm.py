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
    RLVRVerify,
    JudgeWithCriteria,
    AutoPEComputeScore,
    CriteriaRMRecallComputeScore,
    parse_autope_solution_fn,
    AUTOPE_DEFAULT_PARAMS,
    CRITERIA_RM_RECALL_DEFAULT_PARAMS,
    CRITERIA_RM_RFT_DEFAULT_PARAMS,
    compute_score_valid,
    criteria_parse_solution_fn,
    criteria_recall_score_valid,
    criteria_rft_parse_solution_fn,
    CriteriaRFTComputeScore,
    rft_score_valid,
    xml_cot_translation_score_valid,
    Doc2QueryV3VerifierComputeScore,
    doc2query_verifier_parse_solution_fn,
    DOC2QUERY_V3_VERIFIER_DEFAULT_PARAMS
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


UNITTEST_AGENT_LONGCOT = Agent(**{
    "model": "distill_qwen25_7B",
    "base_url": "http://10.130.142.154:8000/v1",
    "api_keys": "EMPTY",
    "request_kwargs": {
        "temperature": 0.7,
        "timeout": 360,
        "max_tokens": 16384,
    },
})


def load_dataset(num=100, format="autope"):
    if format == "autope":
        filename = "/cpfs01/shared/llm_ddd/tongjian/rl/criteria_rm/autope_dapo_math_17k_bo32.parquet"
    elif format == "criteria_rm_recall":
        filename = "/cpfs01/shared/llm_ddd/tongjian/rl/criteria_rm/ultra_feedback_test.parquet"
    elif format == "criteria_rm_rft":
        filename = "/cpfs01/shared/llm_ddd/tongjian/rl/criteria_rm/ultra_feedback_rft_test.parquet"
    elif format == "xml_cot_rft":
        filename = '/cpfs01/shared/llm_ddd/tongjian/sft/self_improvement/dapo_math_17k_xml_cot_pass1_2@2_elite.jsonl'
    elif format == "xml_cot_translation":
        filename = '/cpfs01/shared/llm_ddd/tongjian/sft/self_improvement/xml_cot_translation_if_test_0731.jsonl'
    elif format == "doc2query_v3_verifier":
        filename = '/cpfs01/shared/llm_ddd/tongjian/rl/doc2query_v3_verifier/doc2query_v3_critique_annotations.parquet'

    batch_solution_str, batch_ground_truth = [], []
    batch_data_sources = []

    if format != "xml_cot_rft" and format != "xml_cot_translation":
        df = pd.read_parquet(filename)

        count = 0
        for _, row in df.iterrows():
            row = row.to_dict()
            batch_data_sources.append(row["data_source"])

            if format == "autope":
                prompt = row["reward_model"]["prompt"]
                batch_solution_str.append(
                    f'<think>\nUNITTEST_ONLY\n</think>\n\n<prompt_engineering>\nQuestion: {prompt}\n\nThink step by step.\n</prompt_engineering>'
                )
            elif format == "criteria_rm_recall":
                batch_solution_str.append(
                    f'```xml\n<think>\nUNITTEST_ONLY\n</think>\n\n<conclusion>\n# 评价标准\n有用性，真实性，安全性\n</conclusion>\n```xml'
                )
            elif format == "criteria_rm_rft":
                batch_solution_str.append(
                    f'```xml\n<think>\nUNITTEST_ONLY\n</think>\n\n<conclusion>\n{random.choice(row["reward_model"]["completions"])}\n</conclusion>\n```xml'
                )
            elif format == "xml_cot_rft":
                batch_solution_str.append(
                    f'```xml\n<think>\nUNITTEST_ONLY\n</think>\n\n<conclusion>\n{row["reward_model"]["criteria"]}\n</conclusion>\n```xml'
                )
            elif format == "doc2query_v3_verifier":
                if row["reward_model"]["critique"] is None:
                    batch_solution_str.append(
                        f'<think>\nUNITTEST_ONLY\n</think>\n\n<conclusion>\n"qualified": True\n</conclusion>\n```xml'
                    )
                else:
                    critique = {
                        "qualified": "False",
                        "reason": row["reward_model"]["critique"]
                    }
                    critique = json.dumps(
                        critique, ensure_ascii=False, indent="  ")
                    batch_solution_str.append(
                        f'<think>\nUNITTEST_ONLY\n</think>\n\n<conclusion>\n{critique}\n</conclusion>\n```xml'
                    )
            batch_ground_truth.append(row["reward_model"])
            if len(batch_ground_truth) == num:
                break
    elif format == "xml_cot_rft":
        with open(filename, "rt") as f:
            for i, line in enumerate(f):
                example = json.loads(line)
                batch_solution_str.append(
                    example["self_improvement"]["responses"][0]["response"]["text"]
                )
                batch_solution_str.append(
                    example["self_improvement"]["responses"][0]["response"]["text"]
                )
                batch_ground_truth.append({
                    "source":  "dapo",
                    "instruction": example["self_improvement"]["prompt"],
                    "criteria": f'Answer: {example["self_improvement"]["answer"]}',
                    "extra_info": {
                        "uuid": example["uuid"],
                    },
                })
                batch_ground_truth.append({
                    "source":  "dapo",
                    "instruction": example["self_improvement"]["prompt"],
                    "criteria": f'Answer: {example["self_improvement"]["answer"]}',
                    "extra_info": {
                        "uuid": example["uuid"],
                    },
                })
                if len(batch_ground_truth) == num:
                    break
    elif format == "xml_cot_translation":
        with open(filename, "rt") as f:
            for i, line in enumerate(f):
                example = json.loads(line)
                resp = example["self_improvement"]["responses"][0]["response"]
                conclusion = resp[resp.index("</think>"):].strip()
                batch_solution_str.append(
                    f"```xml\n<think><approach><mutter></mutter></approach></think>\n<conclusion>NO ANSWER</conclusion>\n```")

                batch_ground_truth.append({
                    "source":  "dapo",
                    "instruction": example["self_improvement"]["prompt"],
                    "criteria": f'Answer: {conclusion}',
                    "meta_cognition": ["<mutter> unittest </mutter>"],
                    "extra_info": {
                        "uuid": example["uuid"],
                    },
                })
                if len(batch_ground_truth) == num:
                    break
    return batch_solution_str, batch_ground_truth


class TestCriteriaRMRecall(unittest.TestCase):
    def test_criteria_parse_solution_fn(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            num=6, format="criteria_rm_recall")
        for sol in batch_solution_str:
            print(criteria_parse_solution_fn(sol))

    def test_compute_score(self):
        task = CriteriaRMRecallComputeScore(
            criteria_parse_solution_fn, split="valid", args=CRITERIA_RM_RECALL_DEFAULT_PARAMS)
        batch_solution_str, batch_ground_truth = load_dataset(
            num=6, format="criteria_rm_recall")
        criteria_recall_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth,
        )
        # async def main():
        #     results = await task._compute_score(
        #         [None] *
        #         len(batch_solution_str), batch_solution_str, batch_ground_truth,
        #     )
        #     print(results)
        # aio.run(main())


class TestCriteriaRFT(unittest.TestCase):
    def test_judge_with_criteria(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            num=6, format="criteria_rm_rft")
        task = JudgeWithCriteria()

        async def main():
            results = await task.do_job(
                agent=UNITTEST_AGENT_LONGCOT,
                batch_inputs=list(zip(batch_solution_str, batch_ground_truth)),
                max_concurrent_requests=64,
            )
            print(results)
        aio.run(main())

    def test_compute_score(self):
        task = CriteriaRFTComputeScore(
            criteria_rft_parse_solution_fn, split="valid", args=CRITERIA_RM_RFT_DEFAULT_PARAMS)
        batch_solution_str, batch_ground_truth = load_dataset(
            num=6, format="criteria_rm_rft")

        async def main():
            results = await task._compute_score(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth,
            )
            print(results)
        aio.run(main())


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
        # task = AutoPEComputeScore(
        #     parse_autope_solution_fn, split="valid", args=AUTOPE_DEFAULT_PARAMS)
        batch_solution_str, batch_ground_truth = load_dataset(num=6)
        compute_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth,
        )
        # async def main():
        #     results = await task._compute_score(
        #         [None] *
        #         len(batch_solution_str), batch_solution_str, batch_ground_truth,
        #     )
        #     print(results)
        # aio.run(main())


class TestRFT(unittest.TestCase):
    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            num=100, format="xml_cot_rft")
        rft_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth,
        )


class TestXMLCoTTranslation(unittest.TestCase):
    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            num=4, format="xml_cot_translation")
        xml_cot_translation_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth,
        )


class TestDoc2QueryV3Verifier(unittest.TestCase):
    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_dataset(
            num=4, format="doc2query_v3_verifier")
        task = Doc2QueryV3VerifierComputeScore(
            doc2query_verifier_parse_solution_fn, args=DOC2QUERY_V3_VERIFIER_DEFAULT_PARAMS
        )
        print(task.compute_score(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth,
        ))


if __name__ == '__main__':
    unittest.main()
