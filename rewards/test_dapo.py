import os
import re
import json
import string
import random
import unittest
from tqdm import tqdm
import pandas as pd
from dapo import (
    compute_score,
    compute_score_valid
)


def load_data(num=100):
    filename = "/cpfs01/shared/llm_ddd/tongjian/rl/dapo/filtered_dapo_limo_skywork_test.parquet"
    batch_solution_str, batch_ground_truth = [], []

    df = pd.read_parquet(filename)
    batch_solution_str, batch_ground_truth = [], []

    for _, row in df.iterrows():
        row = row.to_dict()
        batch_ground_truth.append(row["reward_model"])
        if random.random() < 0.5:
            batch_solution_str.append(
                f'Final Answer:{row["reward_model"]["ground_truth"]}+1')
        else:
            batch_solution_str.append(
                f'Final Answer:{row["reward_model"]["ground_truth"]}')
    return batch_solution_str, batch_ground_truth


class TestDapo(unittest.TestCase):
    def test_compute_score(self):
        batch_solution_str, batch_ground_truth = load_data()
        print(compute_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth
        ))


if __name__ == '__main__':
    unittest.main()
