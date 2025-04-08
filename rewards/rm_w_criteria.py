import re
import uuid
import time
import random
import requests
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
from typing import Dict, Any, Callable
import xml.etree.ElementTree as ET
from functools import partial
from collections import namedtuple, defaultdict


URLS = [
    "http://10.130.1.205:5001",
    "http://10.130.1.44:5002"
]

DEFAULT_PARSE_FAILURE_REWARD = -2.
DEFAULT_RM_REWARD_CLIP = 0.1
DEFAULT_RM_REWARD_CLIP_AMPLIFY = 1.0


def contain_chinese(string):
    try:
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if re.search(pattern, string):
            return True
        return False
    except Exception as err:
        return False


def simple_tokenize(s):
    if contain_chinese(s):
        return list(s)
    else:
        return s.split(" ")


def batchify(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def is_subject_question(ground_truth):
    if "# JUDGE CRITERIA" in ground_truth or "# 评价标准" in ground_truth:
        return True
    else:
        return False


def post_with_retry(urls, data, max_retries=3, retry_delay=1):
    retries = 0
    while retries < max_retries:
        try:
            url = random.choice(urls)
            response = requests.post(f'{url}/reward', json=data, timeout=30)
            response.raise_for_status()  # 如果状态码不是 200，抛出异常
            return response.json()
        except requests.RequestException as e:
            print(f"请求(数据总量={len(data)})失败，错误信息: {e}，重试第 {retries + 1} 次...")
            retries += 1
            if retries < max_retries:
                time.sleep(retry_delay)
    print("达到最大重试次数，请求失败。")
    return None


def postprocess_solution(solution_str):
    if "<|im_end|>" in solution_str:
        return solution_str[:solution_str.index("<|im_end|>")]
    return solution_str


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------

class Penalty(object):
    @abstractmethod
    def get_penalty(self, solution_str, ground_truth):
        raise NotImplementedError


class ConclusionTooLongPenalty(Penalty):
    def __init__(self,
                 postprocess_solution_fn,
                 conclusion_limit=600,
                 penalty_base=-0.1):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.conclusion_limit = conclusion_limit
        self.penalty_base = penalty_base

    def get_penalty(self, solution_str, ground_truth):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        return self.penalty_base * min(max(len(simple_tokenize(solution_str))-self.conclusion_limit, 0) / self.conclusion_limit, 5.)


def compute_rm_score(
        batch_solution_str,
        batch_ground_truth,
        postprocess_solution_fn,
        parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD,
):
    input_datas = []
    rewards = {}

    for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
        solution_str = postprocess_solution_fn(solution_str)
        if solution_str is None:
            rewards[i] = parse_result_failure_score
            continue
        if ground_truth is None:
            rewards[i] = parse_result_failure_score
            continue

        input_data = {
            "prompt": ground_truth["ground_truth"], "response": solution_str, "id": i
        }
        input_datas.append(input_data)

    if len(input_datas) > 0:
        for batch in tqdm(batchify(input_datas, n=32), desc='[RM] batchify inference'):
            output_datas = post_with_retry(URLS, batch)
            for _ in output_datas['reward']:
                _id = int(_["id"])
                rewards[_id] = _["rm_score"]

    final_results = []
    for i in range(len(batch_solution_str)):
        if i in rewards:
            final_results.append(rewards[i])
        else:
            final_results.append(0.)
    return final_results


class ComputeScoreBase(object):
    def __init__(self,
                 split="train",
                 parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD,
                 ):
        self.split = split
        self.parse_result_failure_score = parse_result_failure_score

    def clip_string(self, s: str):
        if len(s) > 2000:
            return f'{s[:1000]}... ...{s[-1000:]}'
        return s

    @abstractmethod
    def get_penalties(self) -> Dict[str, Callable]:
        raise NotImplementedError

    def get_question_type(self, ground_truth):
        if "extra_info" in ground_truth and "question_type" in ground_truth["extra_info"]:
            return ground_truth["extra_info"]["question_type"]
        else:
            return "unknown"

    @abstractmethod
    def postprocess_solution_fn(self, solution_str: str):
        raise NotImplementedError

    def get_rm_rewards(self,
                       batch_data_sources,
                       batch_solution_str,
                       batch_ground_truth):
        return compute_rm_score(
            batch_solution_str=batch_solution_str,
            batch_ground_truth=batch_ground_truth,
            postprocess_solution_fn=self.postprocess_solution_fn,
            parse_result_failure_score=self.parse_result_failure_score
        )

    def log_solution(self, solution):
        norm = self.postprocess_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return repr(self.clip_string(norm))

    def log_ground_truth(self, ground_truth):
        return repr(ground_truth["ground_truth"])

    def get_uuid(self, ground_truth):
        try:
            return ground_truth["extra_info"]["uuid"]
        except Exception as err:
            return ""

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      ):

        penalty = defaultdict(dict)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            for key, fn in self.get_penalties().items():
                penalty[key][i] = fn(solution_str, ground_truth)

        base_rewards = self.get_rm_rewards(
            batch_data_sources, batch_solution_str, batch_ground_truth)

        final_results = []
        for i in range(len(batch_solution_str)):
            penalty_log_str = []
            _reward = base_rewards[i]
            for name, _penalty in penalty.items():
                if i in _penalty:
                    _reward += _penalty[i]
                    penalty_log_str.append(f'{name} Penalty={_penalty[i]:.2f}')

            final_results.append(_reward)

            if self.split == "valid":
                print(
                    f"--------------------------------[VALID]--------------------------------")
                print(
                    f"【Solution】 `{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Ground Truth】({self.get_question_type(batch_ground_truth[i])}) `{self.log_ground_truth(batch_ground_truth[i])}`")
                print(f'Reward={_reward:.3f};{";".join(penalty_log_str)}\n')
            elif self.split == "train" and random.random() < 0.2:
                print(
                    f"--------------------------------[TRAIN]--------------------------------")
                print(
                    f"【Solution】`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Ground Truth】({self.get_question_type(batch_ground_truth[i])}) `{self.log_ground_truth(batch_ground_truth[i])}`")
                print(
                    f'【{self.get_uuid(batch_ground_truth[i])}】Reward={_reward:.3f};{";".join(penalty_log_str)}\n')

        return final_results


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# QwQ LongCoT Reward
# ------------------------------------------------------------------------------------------------------------------------------------------------------

class QwQLongCoTComputeScore(ComputeScoreBase):
    def __init__(self, split="train"):
        super().__init__(split=split)
        self.c_length_penalty = ConclusionTooLongPenalty(
            postprocess_solution_fn=QwQLongCoTComputeScore.postprocess_solution_fn)

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "CONCLUSION_LENGTH": self.c_length_penalty.get_penalty
        }

    @classmethod
    def postprocess_solution_fn(cls, solution_str: str):
        solution_str = postprocess_solution(solution_str)
        try:
            thought = re.findall(r'<think>.*</think>',
                                 solution_str, re.DOTALL)[0]
        except Exception as err:
            return None

        conclusion = solution_str.replace(thought, "").strip()

        return conclusion

    def get_rm_rewards(self,
                       batch_data_sources,
                       batch_solution_str,
                       batch_ground_truth):
        rewards = compute_rm_score(
            batch_solution_str=batch_solution_str,
            batch_ground_truth=batch_ground_truth,
            postprocess_solution_fn=self.postprocess_solution_fn,
            parse_result_failure_score=self.parse_result_failure_score
        )
        reshape_rewards = []
        for reward, ground_truth in zip(rewards, batch_ground_truth):
            cate = self.get_question_type(ground_truth)
            if cate == "object":
                if reward >= 0.115:
                    reward = 1.0
                if reward <= 0.0 and reward >= -1.0:
                    reward = -1.0
            reshape_rewards.append(reward)
        return reshape_rewards


_qwq_longcot_compute_score_train = QwQLongCoTComputeScore(split="train")
_qwq_longcot_compute_score_valid = QwQLongCoTComputeScore(split="valid")
qwq_longcot_compute_score_train = _qwq_longcot_compute_score_train.compute_score
qwq_longcot_compute_score_valid = _qwq_longcot_compute_score_valid.compute_score


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# XML CoT Reward
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def get_thought(solution_str: str):
    thought = re.findall(r'```xml.*```', solution_str, re.DOTALL)[0]
    return thought


def get_conclusion(solution_str: str):
    thought = get_thought(solution_str)
    return solution_str[solution_str.index(thought)+len(thought):].strip()


def parse_xml_cot_solution_score(solution_str):
    try:
        thought = get_thought(solution_str)
    except Exception as err:
        return (-0.1, None, None)

    try:
        conclusion = get_conclusion(solution_str).strip()
        if len(conclusion) == 0:
            return (-0.1, None, None)
    except Exception as err:
        return (-0.1, None, None)

    try:
        thought_content = re.findall(r'```xml(.*)```', thought, re.DOTALL)[0]

    except Exception as err:
        return (-0.1, None, None)

    thought_content = f'<doc> {thought_content} </doc>'
    try:
        root = ET.fromstring(thought_content)
        min_threshold = -0.05
    except Exception as err:
        min_threshold = -0.1

    return (None, min_threshold, conclusion)


def xml_cot_compute_score(batch_data_sources, batch_solution_str, batch_ground_truth):
    input_datas = []
    rewards, reward_min_threshold = {}, {}
    for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
        solution_str = postprocess_solution(solution_str)
        reward, min_reward_threshold, conclusion = parse_xml_cot_solution_score(
            solution_str)
        if reward is not None:
            rewards[i] = reward
        else:
            reward_min_threshold[i] = min_reward_threshold
            input_data = {
                "prompt": ground_truth, "response": conclusion, "id": i
            }
            input_datas.append(input_data)

    if len(input_datas) > 0:
        for batch in tqdm(batchify(input_datas, n=32), desc='[RM] batchify inference'):
            output_datas = post_with_retry(URLS, batch)
            for _ in output_datas['reward']:
                _id = int(_["id"])
                rewards[_id] = _["rm_score"]
    final_results = []
    for i in range(len(batch_solution_str)):
        if i in rewards:
            if i in reward_min_threshold:
                min_threshold = reward_min_threshold[i]
                final_results.append(max(min_threshold, rewards[i]))
            else:
                final_results.append(rewards[i])
            rewards[i]
        else:
            final_results.append(0.)

    return final_results


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Fabricate QA Reward
# ------------------------------------------------------------------------------------------------------------------------------------------------------


class FabricateQATooLongPenalty(ConclusionTooLongPenalty):
    def __init__(self,
                 postprocess_solution_fn,
                 postprocess_gt_fn,
                 conclusion_limit=600,
                 penalty_base=-0.1):
        super().__init__(
            postprocess_solution_fn=postprocess_solution_fn,
            conclusion_limit=conclusion_limit,
            penalty_base=penalty_base
        )
        self.postprocess_gt_fn = postprocess_gt_fn

    def get_penalty(self, solution_str, ground_truth):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        limit = len(simple_tokenize(self.postprocess_gt_fn(ground_truth)))
        solution_size = len(simple_tokenize(solution_str))

        return self.penalty_base * min(max(solution_size-limit, 0) / limit, 5.)


class QwQLongCoTFabricateQAComputeScore(ComputeScoreBase):
    def __init__(self, split="train"):
        super().__init__(split=split)
        self.c_length_penalty = FabricateQATooLongPenalty(
            postprocess_solution_fn=QwQLongCoTFabricateQAComputeScore.postprocess_solution_fn,
            postprocess_gt_fn=QwQLongCoTFabricateQAComputeScore.extract_gt_question
        )

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "CONCLUSION_LENGTH": self.c_length_penalty.get_penalty
        }

    @classmethod
    def postprocess_solution_fn(cls, solution_str: str):
        solution_str = postprocess_solution(solution_str)
        try:
            thought = re.findall(r'<think>.*</think>',
                                 solution_str, re.DOTALL)[0]
        except Exception as err:
            return None
        try:
            conclusion = re.findall(r'<question>(.*)</question>',
                                    solution_str, re.DOTALL)[0]
        except Exception as err:
            return None

        return conclusion.strip()

    @classmethod
    def extract_gt_question(cls, ground_truth):
        ground_truth = ground_truth["ground_truth"]
        bg_flag = "Your response (the created question) must be the following:"
        ed_flag = "Respond only with the created question directly"
        ground_truth = ground_truth[ground_truth.index(bg_flag)+len(bg_flag):]
        ground_truth = ground_truth[:ground_truth.index(ed_flag)]

        conclusion = re.findall(r'```(.*)```',
                                ground_truth, re.DOTALL)[0].strip()
        return conclusion

    def log_ground_truth(self, ground_truth):
        return repr(self.extract_gt_question(ground_truth))

    def log_solution(self, solution):
        norm = self.postprocess_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return repr(self.clip_string(norm))

    def get_rm_rewards(self,
                       batch_data_sources,
                       batch_solution_str,
                       batch_ground_truth):
        rewards = compute_rm_score(
            batch_solution_str=batch_solution_str,
            batch_ground_truth=batch_ground_truth,
            postprocess_solution_fn=self.postprocess_solution_fn,
            parse_result_failure_score=self.parse_result_failure_score
        )
        return rewards


_qwq_longcot_fabricate_qa_compute_score_train = QwQLongCoTFabricateQAComputeScore(
    split="train")
_qwq_longcot_fabricate_qa_compute_score_valid = QwQLongCoTFabricateQAComputeScore(
    split="valid")
qwq_longcot_fabricate_qa_compute_score_train = _qwq_longcot_fabricate_qa_compute_score_train.compute_score
qwq_longcot_fabricate_qa_compute_score_valid = _qwq_longcot_fabricate_qa_compute_score_valid.compute_score


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Criteria Envolve Reward
# ------------------------------------------------------------------------------------------------------------------------------------------------------

class QwQLongCoTCriteriaEnvolveComputeScore(ComputeScoreBase):
    def __init__(self, split="train"):
        super().__init__(split=split)

    def get_penalties(self) -> Dict[str, Callable]:
        return {}

    @classmethod
    def postprocess_solution_fn(cls, solution_str: str):
        solution_str = postprocess_solution(solution_str)
        try:
            thought = re.findall(r'<think>.*</think>',
                                 solution_str, re.DOTALL)[0]
        except Exception as err:
            return None
        try:
            conclusion = solution_str.replace(thought, "")
            if "# JUDGE CRITERIA" not in conclusion and "# 评价标准" not in conclusion:
                return None
            if "# JUDGE CRITERIA" in conclusion:
                conclusion = conclusion[conclusion.index(
                    "# JUDGE CRITERIA"):].strip()
                return conclusion
            elif "# 评价标准" in conclusion:
                conclusion = conclusion[conclusion.index("# 评价标准"):].strip()
                return conclusion
            else:
                return None
        except Exception as err:
            return None

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      ):

        penalty = defaultdict(dict)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            for key, fn in self.get_penalties().items():
                penalty[key][i] = fn(solution_str, ground_truth)

        base_rewards = self.get_rm_rewards(
            batch_data_sources, batch_solution_str, batch_ground_truth)
        final_results = []
        for i in range(len(batch_solution_str)):
            penalty_log_str = []
            _reward = base_rewards[i]
            for name, _penalty in penalty.items():
                if i in _penalty:
                    _reward += _penalty[i]
                    penalty_log_str.append(f'{name} Penalty={_penalty[i]:.2f}')

            final_results.append(_reward)

            if self.split == "valid":
                print(
                    f"--------------------------------[VALID]--------------------------------")
                print(
                    f"【Solution】 `{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Ground Truth】`{self.log_ground_truth(batch_ground_truth[i])}`")
                print(f'Reward={_reward:.3f}\n')
            elif self.split == "train" and random.random() < 0.01:
                print(
                    f"--------------------------------[TRAIN]--------------------------------")
                print(
                    f"【Solution】`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Ground Truth】`{self.log_ground_truth(batch_ground_truth[i])}`")
                print(
                    f'Reward={_reward:.3f}\n')
        return final_results

    def log_ground_truth(self, ground_truth):
        if "reference" in ground_truth:
            return repr(ground_truth["reference"])
        else:
            return repr("[no reference]")

    def log_solution(self, solution):
        norm = self.postprocess_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return repr(self.clip_string(norm))

    def get_rm_rewards(self,
                       batch_data_sources,
                       batch_solution_str,
                       batch_ground_truth):

        new_batch_solution_str, new_batch_ground_truth = [], []
        for gt, sol in zip(batch_ground_truth, batch_solution_str):
            prompt, chosen, rejected = gt["prompt"], gt["chosen"], gt["rejected"]

            criteria = self.postprocess_solution_fn(sol)

            if criteria is not None:
                judge_prompt = {"ground_truth": f'{prompt}\n\n\n\n{criteria}'}
            else:
                judge_prompt = None

            # Chosen
            new_batch_solution_str.append(chosen)
            new_batch_ground_truth.append(judge_prompt)

            # Rejected
            new_batch_solution_str.append(rejected)
            new_batch_ground_truth.append(judge_prompt)

        def postprocess_solution_fn(x): return x

        rewards = compute_rm_score(
            batch_solution_str=new_batch_solution_str,
            batch_ground_truth=new_batch_ground_truth,
            postprocess_solution_fn=postprocess_solution_fn,
            parse_result_failure_score=self.parse_result_failure_score
        )
        def split_array(arr):
            odd = []
            even = []
            for num, elem in enumerate(arr):
                if num % 2 == 0:
                    even.append(elem)
                else:
                    odd.append(elem)
            return odd, even

        rejected_scores, chosen_scores = split_array(rewards)
        acc = []
        for c, r in zip(chosen_scores, rejected_scores):
            if c == self.parse_result_failure_score or r == self.parse_result_failure_score:
                acc.append(self.parse_result_failure_score)
            else:
                if c > r:
                    acc.append(1.0)
                else:
                    acc.append(.0)

        return acc


_qwq_longcot_criteria_envolve_compute_score_train = QwQLongCoTCriteriaEnvolveComputeScore(
    split="train")
_qwq_longcot_criteria_envolve_compute_score_valid = QwQLongCoTCriteriaEnvolveComputeScore(
    split="valid")
qwq_longcot_criteria_envolve_compute_score_train = _qwq_longcot_criteria_envolve_compute_score_train.compute_score
qwq_longcot_criteria_envolve_compute_score_valid = _qwq_longcot_criteria_envolve_compute_score_valid.compute_score

if __name__ == "__main__":
    pass
