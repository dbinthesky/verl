import json
import unittest
from pt_refine import (
    pretrain_postprocess
)


class TestRulebasedPostprocess(unittest.TestCase):
    def test_remove_identifiers(self):
        import random
        x = []
        with open("/cpfs01/shared/llm_ddd/tongjian/pretrain/reason_pretrain_v1_4k_train.jsonl", "rt") as f:
            for line in f:
                example = json.loads(line)
                x.append(example)
        random.shuffle(x)
        print(x[0]["content"])
        print("="*80)
        print(pretrain_postprocess(x[0]["content"]))


        # x.append(line)
        # print(x)
if __name__ == '__main__':
    unittest.main()
