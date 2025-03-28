import re
import uuid
import time
import random
import requests
from tqdm import tqdm
from abc import abstractmethod
from typing import Dict, Any, Callable
import xml.etree.ElementTree as ET
from functools import partial
from collections import namedtuple, defaultdict


URLS = [
    "http://10.130.1.205:5001"
]

DEFAULT_PARSE_FAILURE_REWARD = -2.
DEFAULT_RM_REWARD_CLIP = 0.1
DEFAULT_RM_REWARD_CLIP_AMPLIFY = 1.0


def contain_chinese(string):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    if re.search(pattern, string):
        return True
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
            return self.clip_string(solution)
        return self.clip_string(norm)

    def log_ground_truth(self, ground_truth):
        return ground_truth["ground_truth"]

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
            batch_solution_str, batch_ground_truth)

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
                print(f'Reward={_reward:.3f};{";".join(penalty_log_str)}\n')

        return final_results


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
                if reward <= 0.0:
                    reward = -1.0
            reshape_rewards.append(reward)
        return reshape_rewards


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# QwQ LongCoT Reward
# ------------------------------------------------------------------------------------------------------------------------------------------------------


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


# ------------------------------
# Fabricate QA Reward
# ------------------------------

def length_penalty(solution_str, ground_truth):
    return -0.05 * min(abs(len(simple_tokenize(solution_str))-len(simple_tokenize(ground_truth))) / len(simple_tokenize(ground_truth)), 5.)


def fabricate_qa_format_penalty(solution_str):
    if solution_str.startswith('"') and solution_str.endswith('"'):
        return solution_str[1:-1].strip(), 0.
    else:
        return solution_str, -0.1


def fabricate_qa_task_postprocess(solution_str):
    if "[CONCLUSION BEGIN]" not in solution_str or "[CONCLUSION END]" not in solution_str:
        return None
    solution_str = solution_str[solution_str.index(
        "[CONCLUSION BEGIN]")+len("[CONCLUSION BEGIN]"):solution_str.index("[CONCLUSION END]")]

    if "The constructed question is: " in solution_str:
        solution_str = solution_str.replace(
            "The constructed question is: ", "").strip()

    solution_str = solution_str.strip()
    if not solution_str.startswith("**Question:**"):
        return None
    solution_str = solution_str.replace("**Question:**", "").strip()

    return solution_str


def fabricate_qa_compute_score_nothink(batch_data_sources, batch_solution_str, batch_ground_truth, split="train"):
    input_datas = []
    rewards = {}
    len_penalty, format_penalty = {}, {}

    logs = {}

    for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
        solution_str = postprocess_solution(solution_str)
        if data_source == "fabricate_qa":
            raw_solution_str = solution_str
            solution_str = fabricate_qa_task_postprocess(solution_str)

            show_ground_truth = ground_truth
            flag = "# Final Anwer (Authentic Exam)"
            if flag in show_ground_truth:
                show_ground_truth = show_ground_truth[show_ground_truth.index(
                    flag)+len(flag):].strip()
            flag = "## Note"
            if flag in show_ground_truth:
                show_ground_truth = show_ground_truth[:show_ground_truth.index(
                    "## Note")].strip()

            if solution_str is None:
                rewards[i] = -1.
                logs[i] = (raw_solution_str, show_ground_truth)
                continue
            else:
                _len_penalty = length_penalty(solution_str, show_ground_truth)
                len_penalty[i] = _len_penalty
                logs[i] = (solution_str, show_ground_truth)
                solution_str, _format_penalty = fabricate_qa_format_penalty(
                    solution_str)
                format_penalty[i] = _format_penalty

        input_data = {
            "prompt": ground_truth, "response": solution_str, "id": i
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
            _reward = rewards[i]
            if i in len_penalty:
                _reward += len_penalty[i]
                _reward += format_penalty[i]
            final_results.append(_reward)
            if split == "valid" and batch_data_sources[i] == "fabricate_qa" and i in logs:
                print(f"--------------------------------")
                print(f"【Solution】 `{repr(logs[i][0])}`")
                print(f"【Ground Truth】 `{repr(logs[i][1])}`")
                print(
                    f'Reward={_reward};Length Penalty={len_penalty.get(i, 0.)};Format Penalty={format_penalty.get(i, 0.)}')
        else:
            final_results.append(0.)

    return final_results


fabricate_qa_compute_score_nothink_train = partial(
    fabricate_qa_compute_score_nothink, split="train")
fabricate_qa_compute_score_nothink_valid = partial(
    fabricate_qa_compute_score_nothink, split="train")


if __name__ == "__main__":
    pass
