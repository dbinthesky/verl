import os
import re
import json
import string
import random
import unittest
import asyncio as aio
from xml_cot import (
    xml_cot_parse_solution_fn,
    tree_depth, tree_width,
    XMLCoTComputeScore
)


def load_xml_cot():
    batch_solution_str, batch_ground_truth = [], []
    with open("/cpfs01/shared/llm_ddd/tongjian/sft/self_improvement/xml_cot_if_enhance_0528.jsonl", "rt") as f:
        for line in f:
            example = json.loads(line)

            batch_solution_str.append(
                example["self_improvement"]["responses"][0]["response"]
            )
            batch_ground_truth.append(
                {
                    "ground_truth": example["self_improvement"]["reference_meta"]["final_answer"],
                }
            )
            if len(batch_ground_truth) == 128:
                break

    return batch_solution_str, batch_ground_truth


class TestXMLCoT(unittest.TestCase):
    def test_compute_score(self):
        async def main():
            batch_solution_str, batch_ground_truth = load_xml_cot()
            task = XMLCoTComputeScore(split="valid")
            results = await task.get_accuracy(
                [None] *
                len(batch_solution_str), batch_solution_str, batch_ground_truth
            )
            print(results)
        aio.run(main())

        # for sol in batch_solution_str:
        #     root = xml_cot_parse_solution_fn(sol)
        #     print(sol)
        #     print(tree_depth(root), tree_width(root))
        #     print('='*80)
        #     break
        #     # task = CoTEnhanceComputeScore(
        #     #     split="valid", parse_result_failure_score=-10.)
        #     # task.compute_score(
        #     #     [None] *
        #     #     len(batch_solution_str), batch_solution_str, batch_ground_truth
        #     # )


if __name__ == '__main__':
    unittest.main()
