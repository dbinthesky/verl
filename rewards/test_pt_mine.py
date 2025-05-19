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
    get_notes_and_conclusions
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

    def postprocess(s):
        notes = re.findall(r'\[Note\] Q: (.*) Think: (.*)\[/Note\]', s)
        return "```xml\n"+"\n".join([f'<question>\n{_[0]}\n</question>\n<cot-in-document>\n{_[1]}\n</cot-in-document>'for _ in notes]) + "\n```"

    batch_solution_str = [postprocess(_) for _ in batch_solution_str]
    return batch_solution_str, batch_ground_truth


class TestPretrainMine(unittest.TestCase):
    def test_load_data(self):
        batch_solution_str, batch_ground_truth = load_pretrain_mine(num=100)


if __name__ == '__main__':
    unittest.main()
