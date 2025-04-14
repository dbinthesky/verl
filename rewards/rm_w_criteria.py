import os
import re
import uuid
import time
import math
import jieba
import random
import asyncio
import requests
import sacrebleu
import numpy as np
from tqdm import tqdm
import tqdm.asyncio
import asyncio as aio
from abc import abstractmethod
from typing import Dict, Any, Callable
import xml.etree.ElementTree as ET
from functools import partial
from collections import namedtuple, defaultdict


from openai import OpenAI, RateLimitError, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

RATE_LIMIT_RETRY_DELAY = 60
RATE_LIMIT_RETRY_ATTEMPTS = 10
WORKFLOW_AGENT_LOGFILE = os.getenv("WORKFLOW_AGENT_LOGFILE", None)

URLS = [
    "http://10.130.1.101:5001",
]


DEFAULT_PARSE_FAILURE_REWARD = -2.
DEFAULT_RM_REWARD_CLIP = 0.1
DEFAULT_RM_REWARD_CLIP_AMPLIFY = 1.0
VERIFIER_MODEL_NAME = "qwen25_7B_instruct"
VERIFIER_MODEL_PATH = "http://10.130.247.138:8000/v1"


class APIError(Exception):
    pass


class PostprocessError(Exception):
    pass


class Agent:
    def __init__(
        self,
        system: str | None = None,
        model: str = "gpt-3.5-turbo",
        base_url: str | None = None,
        api_keys: str | list[str] | None = None,
        request_kwargs: dict[str, Any] = None,
    ):
        self.system = system
        if self.system is None:
            self.history = []
        else:
            self.history = [{"role": "system", "content": self.system}]
        self.model = model
        self.base_url = base_url

        if api_keys is not None:
            pass
        else:
            api_keys = [os.getenv("OPENAI_API_KEY", "EMPTY")]
        self.api_keys = api_keys

        self.request_kwargs = {
            "max_tokens": 1024,
            "temperature": 0.6,
            "top_p": 0.95,
            "seed": 100745534,
        }
        if request_kwargs is not None:
            self.request_kwargs.update(request_kwargs)

    async def run(self, messages, max_concurrent, desc, postprocess_fns):
        semaphore = asyncio.Semaphore(max_concurrent)
        async with AsyncOpenAI(api_key=self.api_keys, base_url=self.base_url) as client:
            results = []
            tasks = [self.process_prompt(client, message, semaphore, postprocess_fn)
                     for message, postprocess_fn in zip(messages, postprocess_fns)]

            if desc is not None:
                for f in tqdm.asyncio.tqdm.as_completed(tasks, dynamic_ncols=True, desc=desc):
                    results.append(await f)
            else:
                try:
                    results = await asyncio.gather(*tasks)
                except Exception as err:
                    print(f'[ERROR] asyncio.gather failed: {err}')
                    return None
            return results

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=5, max=20))
    async def chat_completion(self, client, messages, postprocess_fn) -> str | None:
        response = None
        try:
            response = await client.chat.completions.create(
                model=self.model, messages=[{
                    "role": "user", "content": messages
                }], **self.request_kwargs,
            )
            return postprocess_fn(response.choices[0].message.content)
        except PostprocessError as e:
            print(
                f"[ERROR] failure occurred when parse result: (response={response}), {e}")
            raise PostprocessError("Failed to generate text")
        except Exception as e:
            print(
                f"[ERROR] failure occurred when call API: {e} (response={response})")
            raise APIError("Failed to generate text")

    async def process_prompt(self, client, messages, semaphore, postprocess_fn):
        async with semaphore:
            try:
                result = await self.chat_completion(client, messages, postprocess_fn)
                return messages, result
            except Exception as err:
                return messages, None


agent = Agent(**{
    "model": VERIFIER_MODEL_NAME,
    "base_url": VERIFIER_MODEL_PATH,
    "api_keys": "EMPTY",
    "request_kwargs": {
        "temperature": 0.7,
        "timeout": 30,
        "max_tokens": 2048
    },
})


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
        return list(jieba.cut(s.lower()))
    else:
        return s.lower().strip().split(" ")


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

class PenaltyOrReward(object):
    @abstractmethod
    def get_penalty_or_reward(self, solution_str, ground_truth):
        raise NotImplementedError


class ConclusionTooLongPenalty(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 conclusion_limit=600,
                 penalty_base=-0.1):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.conclusion_limit = conclusion_limit
        self.penalty_base = penalty_base

    def get_penalty_or_reward(self, solution_str, ground_truth):
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
            "CONCLUSION_LENGTH": self.c_length_penalty.get_penalty_or_reward
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
                 penalty_base=-0.1,
                 no_penalty_range=0.2,
                 ):
        super().__init__(
            postprocess_solution_fn=postprocess_solution_fn,
            conclusion_limit=conclusion_limit,
            penalty_base=penalty_base
        )
        self.no_penalty_range = no_penalty_range
        self.postprocess_gt_fn = postprocess_gt_fn

    def get_penalty_or_reward(self, solution_str, ground_truth):
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
            "CONCLUSION_LENGTH": self.c_length_penalty.get_penalty_or_reward
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
    def __init__(self, split="train", parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD):
        super().__init__(split=split, parse_result_failure_score=parse_result_failure_score)

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
            conclusion = solution_str.replace(thought, "").strip()
            if ("# JUDGE CRITERIA" not in conclusion) and ("# 评价标准" not in conclusion):
                return None
            return conclusion
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
                if self.split == "train":
                    # Bradley–Terry
                    acc.append(math.exp(c)/(math.exp(c)+math.exp(r)))
                else:
                    if c > r:
                        acc.append(1.0)
                    else:
                        acc.append(.0)
        return acc


_qwq_longcot_criteria_envolve_compute_score_train = QwQLongCoTCriteriaEnvolveComputeScore(
    split="train", parse_result_failure_score=0.)
_qwq_longcot_criteria_envolve_compute_score_valid = QwQLongCoTCriteriaEnvolveComputeScore(
    split="valid", parse_result_failure_score=0.)
qwq_longcot_criteria_envolve_compute_score_train = _qwq_longcot_criteria_envolve_compute_score_train.compute_score
qwq_longcot_criteria_envolve_compute_score_valid = _qwq_longcot_criteria_envolve_compute_score_valid.compute_score


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Fabricate QA Reward V2
# ------------------------------------------------------------------------------------------------------------------------------------------------------

class FabricateQALengthPenalty(ConclusionTooLongPenalty):
    def __init__(self,
                 postprocess_solution_fn,
                 postprocess_gt_fn,
                 conclusion_limit=600,
                 penalty_base=-0.5):
        super().__init__(
            postprocess_solution_fn=postprocess_solution_fn,
            conclusion_limit=conclusion_limit,
            penalty_base=penalty_base
        )
        self.postprocess_gt_fn = postprocess_gt_fn

    def get_penalty_or_reward(self, solution_str, ground_truth):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        gt = self.postprocess_gt_fn(ground_truth)

        gt_size = len(simple_tokenize(gt))
        solution_size = len(simple_tokenize(solution_str))

        if abs(solution_size-gt_size)/gt_size < 0.2:
            return 0.

        return self.penalty_base * min(abs(solution_size-gt_size) / gt_size, 5.)


class BleuSimilarity(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 postprocess_gt_fn):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.postprocess_gt_fn = postprocess_gt_fn

    def get_penalty_or_reward(self, solution_str, ground_truth):
        try:
            solution_str = self.postprocess_solution_fn(solution_str)
            if solution_str is None:
                return 0.

            gt = self.postprocess_gt_fn(ground_truth)
            gt_tokens = " ".join(simple_tokenize(gt))
            sl_tokens = " ".join(simple_tokenize(solution_str))
            bleu = sacrebleu.sentence_bleu(sl_tokens, [gt_tokens]).score
            return bleu / 100
        except Exception as err:
            return 0.


class QwQLongCoTFabricateQAComputeScoreV2(QwQLongCoTFabricateQAComputeScore):
    JUDGE_QUESTION_SIMILARITY_FEWSHOTS = '任务: 判断两道问题的相似程度，相似程度用0-4分进行区分，0表示题目完全不同，4表示题目完全一样。\n\n下面是一个具体的例子，向你展示如何对不同问题进行判断相似度。\n\n\n\n# 原始问题\nA solid rectangular block is formed by gluing together $N$ congruent 1-cm cubes face to face. When the block is viewed so that three of its faces are visible, exactly 231 of the 1-cm cubes cannot be seen. Find the smallest possible value of $N.$\n\n\n# 相似程度评分表（含示例）\n## 一、几何组合问题专项评分表\n| 评价维度       | 评分等级 | 具体标准                                                                 | 示例（与原题对比）                                                                 | 得分 |\n|----------------|----------|--------------------------------------------------------------------------|----------------------------------------------------------------------------------|------|\n| **题目类型**   | 0分      | 非几何问题                                                               | *解方程 \\(x^3 - 6x^2 + 11x - 6 = 0\\)*（代数题，无立方体组合概念）                  | 0    |\n|                 | 1分      | 二维几何问题                                                             | *用瓷砖铺成矩形，求隐藏瓷砖数*（二维拼接，非三维立方体）                          | 1    |\n|                 | 2分      | 三维几何但非立方体组合                                                   | *球体堆积中不可见球体计数*（三维但基本单元为球体，非立方体）                      | 2    |\n|                 | 3分      | 三维立方体组合但目标不同                                                 | *求立方体组合体的表面积*（目标为表面积计算，非隐藏块计数）                        | 3    |\n|                 | 4分      | 完全一致题型：三维立方体拼接隐藏块计数问题                               | 原题：3个可见面隐藏231个立方体，求最小N                                          | 4    |\n| **已知条件**   | 0分      | 无核心要素重合（如二维、非单位立方体、可见面≠3、无隐藏块数值、目标不同） | *二维瓷砖问题，求周长*（维度、单元、可见面、目标均不同）                          | 0    |\n|                 | 1-2分    | 20%-40%要素重合（如三维+单位立方体，但可见面≠3或隐藏块数值无关）          | *4个可见面隐藏120个立方体*（三维+单位立方体，但可见面数不同，隐藏块数值不同）      | 1.5  |\n|                 | 3-4分    | 60%-80%要素重合（如三维+3可见面+求N，但隐藏块数值不同）                   | *3个可见面隐藏120个立方体，求最小N*（仅隐藏块数值231→120，其他条件一致）          | 3.5  |\n|                 | 5-6分    | 80%以上要素重合（仅数值精度或符号不同）                                   | 原题微调：隐藏块数231→231，表述完全一致                                          | 6    |\n| **求解目标**   | 0分      | 目标无关（如求表面积、棱长和）                                           | *求立方体组合体的总棱长*                                                          | 0    |\n|                 | 1-2分    | 目标相关但维度不同（如求最大N、已知N求K）                                 | *已知N=300，求隐藏块数K*（目标为反向求解，与原题求最小N不同）                     | 1.5  |\n|                 | 3-4分    | 目标完全一致（仅数据不同）                                               | *3个可见面隐藏120个立方体，求最小N*（同求最小N，仅K值不同）                        | 3.5  |\n| **解题步骤**   | 0分      | 无核心步骤重合（如用代数方法而非因数分解）                               | *用微积分求最优解*（完全不同解题方法）                                             | 0    |\n|                 | 1-2分    | 1-2个核心步骤重合（如设边长但未建立隐藏块公式）                           | *设边长a,b,c但直接计算体积*（缺少隐藏块公式推导）                                  | 1.5  |\n|                 | 3-4分    | 3-4个核心步骤重合（公式建立+因数分解，但未优化最小值）                     | *建立\\((a-1)(b-1)(c-1)=231\\)但未按有序分解求最小值*                               | 3    |\n|                 | 5分      | 所有核心步骤一致（设边长+公式+因数分解+优化最小值）                       | 原题解题步骤：分解231=3×7×11→边长4×8×12→N=384                                   | 5    |\n\n## 二、综合评分计算表（以原题为例）\n| 维度       | 权重 | 专项得分 | 加权得分 | 示例对比说明                                                                 |\n|------------|------|----------|----------|------------------------------------------------------------------------------|\n| 题目类型   | 20%  | 4        | 0.8      | 完全一致题型（三维立方体隐藏块计数）                                         |\n| 已知条件   | 30%  | 5        | 1.5      | 隐藏块数值不同（231→120），其他条件（3可见面、单位立方体、求最小N）一致       |\n| 求解目标   | 20%  | 3        | 0.6      | 同求最小N，仅隐藏块数值不同                                                   |\n| 解题步骤   | 30%  | 4        | 1.2      | 包含设边长、公式建立、因数分解，步骤顺序一致但数值不同                         |\n| **总分**   | 100% | -        | **4.1**  | 四舍五入后对应相似等级3分（7-8分区间，因10分制转换为4.1×2.5≈10.25，此处按原题标准修正为3分） |\n\n## 三、各相似等级示例对照表\n| 相似等级 | 得分区间 | 判定标准                                                                 | 示例题目                                                                       | 与原题差异点                                                                 |\n|----------|----------|--------------------------------------------------------------------------|----------------------------------------------------------------------------|--------------------------------------------------------------------------|\n| **0分**  | 0-1      | 无任何几何组合要素，目标/条件/步骤完全无关                               | *求等差数列前n项和*                                                         | 类型：代数；条件：无立方体；目标：求和；步骤：等差公式                      |\n| **1分**  | 2-3      | 单一维度相关（如三维几何），但核心模型/步骤不重合                        | *长方体体积120，求整数边长组合*                                             | 类型：三维几何；条件：无可见面/隐藏块；目标：求边长；步骤：体积分解，无隐藏块公式 |\n| **2分**  | 4-6      | 两个以上维度部分相似（条件+步骤），但关键公式不同                        | *2个可见面隐藏120个立方体，求N*                                              | 可见面数不同（2→3），隐藏块公式变为\\((a-1)(b-1)c=K\\)，步骤需调整维度           |\n| **3分**  | 7-8      | 所有核心要素一致，仅数据/参数不同                                         | *3个可见面隐藏120个立方体，求最小N*                                          | 隐藏块数值不同（231→120），解题步骤完全一致（分解120=3×4×10→边长4×5×11→N=220） |\n| **4分**  | 9-10     | 题目内容完全一致（含数据、表述、目标）                                   | 原题复现："231个立方体不可见，求最小N"                                       | 无差异，数据与表述完全相同                                                   |\n\n## 四、评分表使用说明\n1. **要素拆解**：将题目分解为**类型、条件、目标、步骤**四大维度，每个维度按专项标准（如几何题关注空间维度、立方体单元、可见面数）打分。  \n2. **加权计算**：类型（20%）+条件（30%）+目标（20%）+步骤（30%），单项满分按专项细化标准（如条件维度最多6分，对应30%权重即1.8分）。  \n3. **示例对照**：通过表格右侧示例快速定位题目差异点，例如"可见面数不同"对应条件维度扣分，"解题步骤一致"对应步骤维度加分。  \n4. **特殊处理**：若题目包含相同数学模型（如隐藏块公式\\((a-1)(b-1)(c-1)=K\\)），即使数据不同，步骤维度自动加1分。\n\n\n好，现在让我们具体来进行问题与问题之间相似度的比较，下面是具体的例子。你的回答中，先进行详尽的思考，然后需要按照下面的格式展示最终的判断结果\n\\"\\"\\"\n[CONCLUSION START]\nSIMILARITY=*\n[CONCLUSION END]\n\\"\\"\\"\n\n\n[例子一]\n\n# 原问题\nGiven a sequence $a_1,$ $a_2,$ $a_3,$ $\\\\dots,$ let $S_n$ denote the sum of the first $n$ terms of the sequence.\\n\\nIf $a_1 = 1$ and\\n\\\\[a_n = \\\\frac{2S_n^2}{2S_n - 1}\\\\]for all $n \\\\ge 2,$ then find $a_{100}.$\n\n# 待评价问题\nWhat is the value of \\\\(a_{10}\\\\) in the sequence defined recursively by \\\\(a_1 = 1\\\\), \\\\(a_2 = 2\\\\), and \\\\(a_n = 3a_{n-1} - 2a_{n-2}\\\\) for \\\\(n \\\\geq 3\\\\)?\n\n# 输出\n首先，从题目类型来看，原问题是通过数列中\\(a_n\\)与\\(S_n\\)的关系求数列某一项的值，属于数列中\\(a_n\\)与\\(S_n\\)关系的问题；而待评价问题是通过数列的递推公式（二阶递推关系\\(a_n = 3a_{n - 1}-2a_{n - 2}\\)）求数列某一项的值，属于数列递推公式求项的问题，两者类型不同，题目类型维度得分为0分。\n\n从已知条件来看，原问题的已知条件是\\(a_1 = 1\\)以及\\(a_n=\\frac{2S_n^2}{2S_n - 1}(n\\geq2)\\)；待评价问题的已知条件是\\(a_1 = 1\\)，\\(a_2 = 2\\)以及\\(a_n = 3a_{n - 1}-2a_{n - 2}(n\\geq3)\\)，两者的已知条件完全不同，已知条件维度得分为0分。\n\n从求解目标来看，原问题是求\\(a_{100}\\)，待评价问题是求\\(a_{10}\\)，虽然都是求数列某一项的值，但由于数列本身的定义不同，所以求解目标也不能简单认为相似，求解目标维度得分为0分。\n\n从解题步骤来看，原问题需要利用\\(a_n=S_n - S_{n - 1}(n\\geq2)\\)将\\(a_n=\\frac{2S_n^2}{2S_n - 1}\\)转化为关于\\(S_n\\)的递推关系，进而求出\\(S_n\\)，再求\\(a_{100}\\)；待评价问题需要根据递推公式\\(a_n = 3a_{n - 1}-2a_{n - 2}\\)求出数列的通项公式（如通过特征方程等方法），然后求\\(a_{10}\\)，两者的解题步骤完全不同，解题步骤维度得分为0分。\n\n综合计算，根据权重：题目类型（20%）+已知条件（30%）+求解目标（20%）+解题步骤（30%），加权得分 = \\(0\\times20\\%+0\\times30\\%+0\\times20\\%+0\\times30\\% = 0\\)分，对应相似等级0分。\n\n[CONCLUSION START]\nSIMILARITY=0\n[CONCLUSION END]\n\n\n\n[例子二]\n\n# 原问题\nFind the minimum value of\\n\\\\[x^2 + 2xy + 3y^2 - 6x - 2y,\\\\]over all real numbers $x$ and $y.$\n\n# 待评价问题\nFind the minimum value of the quadratic function \\\\( f(x) = x^2 - 6x + 7 \\\\).\n\n# 输出\n首先，从题目类型来看，原问题是求二元二次多项式\\(x^2 + 2xy + 3y^2 - 6x - 2y\\)的最小值，属于二元二次函数求最值问题；而待评价问题是求一元二次函数\\(f(x)=x^2 - 6x + 7\\)的最小值，属于一元二次函数求最值问题，两者类型不同，题目类型维度得分为1分（因为都是函数求最值问题，有一定相关性，但维度不同）。\n\n从已知条件来看，原问题的已知条件是二元二次多项式\\(x^2 + 2xy + 3y^2 - 6x - 2y\\)；待评价问题的已知条件是一元二次函数\\(f(x)=x^2 - 6x + 7\\)，两者的已知条件不同，已知条件维度得分为1分（都是二次函数形式，有一定相似性，但变量个数不同）。\n\n从求解目标来看，原问题是求给定表达式的最小值，待评价问题也是求给定函数的最小值，求解目标完全一致，求解目标维度得分为3分。\n\n从解题步骤来看，原问题求二元二次函数最小值，可通过配方法（将式子变形为含有完全平方的形式）等方法，且要考虑两个变量\\(x\\)和\\(y\\)；待评价问题求一元二次函数最小值，通过配方法将\\(f(x)=x^2 - 6x + 7\\)变形为\\(f(x)=(x - 3)^2 - 2\\)等方法，只考虑一个变量\\(x\\)，两者有部分核心步骤重合（都可使用配方法），解题步骤维度得分为2分。\n\n综合计算，根据权重：题目类型（20%）+已知条件（30%）+求解目标（20%）+解题步骤（30%），加权得分 = \\(1\\times20\\% + 1\\times30\\%+3\\times20\\%+2\\times30\\% = 0.2 + 0.3+0.6 + 0.6 = 1.7\\)分，四舍五入后对应相似等级1分。\n\n[CONCLUSION START]\nSIMILARITY=1\n[CONCLUSION END]\n\n\n\n[例子三]\n\n# 原问题\nFind the inverse of the matrix\\n\\\\[\\\\begin{pmatrix} 5 & -4 \\\\\\\\ 0 & 1 \\\\end{pmatrix}.\\\\]If the inverse does not exist, then enter the zero matrix.\n\n# 待评价问题\nWhat is the inverse of the matrix \\\\(\\\\begin{pmatrix} 2 & 1 \\\\\\\\ 3 & 4 \\\\end{pmatrix}\\\\)?\n\n# 输出\n首先，从题目类型来看，原问题是求给定矩阵\\(\\begin{pmatrix} 5 & -4 \\\\ 0 & 1 \\end{pmatrix}\\)的逆矩阵（若不存在则输出零矩阵），属于矩阵求逆的问题；待评价问题是求给定矩阵\\(\\begin{pmatrix} 2 & 1 \\\\ 3 & 4 \\end{pmatrix}\\)的逆矩阵，同样属于矩阵求逆的问题，两者类型完全一致，题目类型维度得分为4分。\n\n从已知条件来看，原问题的已知条件是矩阵\\(\\begin{pmatrix} 5 & -4 \\\\ 0 & 1 \\end{pmatrix}\\)，待评价问题的已知条件是矩阵\\(\\begin{pmatrix} 2 & 1 \\\\ 3 & 4 \\end{pmatrix}\\)，两者矩阵元素不同，但都是矩阵求逆的已知条件形式，已知条件维度得分为3分（仅矩阵元素不同，其他条件一致）。\n\n从求解目标来看，原问题是求给定矩阵的逆矩阵（若不存在则输出零矩阵），待评价问题是求给定矩阵的逆矩阵，求解目标完全一致，求解目标维度得分为3分。\n\n从解题步骤来看，原问题和待评价问题都需要使用矩阵求逆的方法，比如通过公式\\(A^{-1}=\\frac{1}{\\det(A)}\\text{adj}(A)\\)（\\(\\det(A)\\)为矩阵\\(A\\)的行列式，\\(\\text{adj}(A)\\)为伴随矩阵）来求解逆矩阵，核心步骤完全重合，解题步骤维度得分为4分。\n\n综合计算，根据权重：题目类型（20%）+已知条件（30%）+求解目标（20%）+解题步骤（30%），加权得分 = \\(4\\times20\\% + 3\\times30\\%+3\\times20\\%+4\\times30\\% = 0.8 + 0.9+0.6 + 1.2 = 3.5\\)分，四舍五入后对应相似等级3分。\n\n[CONCLUSION START]\nSIMILARITY=3\n[CONCLUSION END]'
    JUDGE_QUESTION_SIMILARITY_TEMPLATE = """
现在需要你对下面的问题相似度进行判断

# 原问题
{authentic_question}

# 待评价问题
{fabricate_question}

# 输出
"""

    def __init__(self, split="train"):
        super().__init__(split=split)
        self.length_penalty = FabricateQALengthPenalty(
            postprocess_solution_fn=QwQLongCoTFabricateQAComputeScoreV2.postprocess_solution_fn,
            postprocess_gt_fn=QwQLongCoTFabricateQAComputeScoreV2.extract_gt_question
        )
        self.bleu_similarity = BleuSimilarity(
            postprocess_solution_fn=QwQLongCoTFabricateQAComputeScoreV2.postprocess_solution_fn,
            postprocess_gt_fn=QwQLongCoTFabricateQAComputeScoreV2.extract_gt_question
        )

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "LengthPenalty": self.length_penalty.get_penalty_or_reward,
            "BLEU": self.bleu_similarity.
        }

    def parse_llm_judge_sim_result(self, result):
        if "[CONCLUSION START]" in result and "[CONCLUSION END]" in result:
            result = result[result.index(
                "[CONCLUSION START]") + len("[CONCLUSION START]"):result.index("[CONCLUSION END]")].strip()
            if "SIMILARITY=" not in result:
                raise PostprocessError(
                    f'parse failure: do not comply `SIMILARITY=*` format')
            result = result.replace("SIMILARITY=", "").strip()
            try:
                score = eval(result)
                assert (isinstance(score, int) and score in (0, 1, 2, 3, 4))
                return score
            except Exception as err:
                raise PostprocessError(
                    f'parse failure: do not comply `SIMILARITY=*` format')
        else:
            raise PostprocessError(
                f'`[CONCLUSION START]` or `[CONCLUSION END]` flag not found.')

    async def llm_judge_similarity(self,
                                   batch_data_sources,
                                   batch_solution_str,
                                   batch_ground_truth,
                                   ):

        base_rewards = [0.0] * len(batch_ground_truth)

        prompt_mapper = defaultdict(list)
        for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            solution_str = self.postprocess_solution_fn(solution_str)
            if solution_str is None:
                continue

            gt = self.extract_gt_question(ground_truth)

            sim_judge_prompt = self.JUDGE_QUESTION_SIMILARITY_FEWSHOTS + "\n\n\n" + self.JUDGE_QUESTION_SIMILARITY_TEMPLATE.format(
                authentic_question=gt,
                fabricate_question=solution_str,
            ).strip()

            prompt_mapper[sim_judge_prompt].append(i)

            max_concurrent_requests = 48
            prompts = list(prompt_mapper.keys())

        results = await agent.run(prompts, max_concurrent_requests, desc="[Verify Constraint]", postprocess_fns=[self.parse_llm_judge_sim_result]*len(prompts))
        for prompt, response in results:
            if response is not None:
                indices = prompt_mapper[prompt]
                for index in indices:
                    try:
                        score = response
                        base_rewards[index] += score / 4.0

                    except Exception as err:
                        continue
        return base_rewards

    async def compute_score(self,
                            batch_data_sources,
                            batch_solution_str,
                            batch_ground_truth,
                            ):
        penalty = defaultdict(dict)
        # for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
        #     for key, fn in self.get_penalties().items():
        #         penalty[key][i] = fn(solution_str, ground_truth)

        # base_rewards = self.get_rm_rewards(
        #     batch_data_sources, batch_solution_str, batch_ground_truth)
        # final_results = []
        # for i in range(len(batch_solution_str)):
        #     penalty_log_str = []
        #     _reward = base_rewards[i]
        #     for name, _penalty in penalty.items():
        #         if i in _penalty:
        #             _reward += _penalty[i]
        #             penalty_log_str.append(f'{name} Penalty={_penalty[i]:.2f}')

        #     final_results.append(_reward)

        #     if self.split == "valid":
        #         print(
        #             f"--------------------------------[VALID]--------------------------------")
        #         print(
        #             f"【Solution】 `{self.log_solution(batch_solution_str[i])}`")
        #         print(
        #             f"【Ground Truth】`{self.log_ground_truth(batch_ground_truth[i])}`")
        #         print(f'Reward={_reward:.3f}\n')
        #     elif self.split == "train" and random.random() < 0.01:
        #         print(
        #             f"--------------------------------[TRAIN]--------------------------------")
        #         print(
        #             f"【Solution】`{self.log_solution(batch_solution_str[i])}`")
        #         print(
        #             f"【Ground Truth】`{self.log_ground_truth(batch_ground_truth[i])}`")
        #         print(
        #             f'Reward={_reward:.3f}\n')
        # return final_results

        # class QuestionSimilarityJudgeByLLM(PenaltyOrReward):

        #     def __init__(self,
        #                  postprocess_solution_fn,
        #                  postprocess_gt_fn):
        #         self.postprocess_solution_fn = postprocess_solution_fn
        #         self.postprocess_gt_fn = postprocess_gt_fn

        #     def get_penalty_or_reward(self, solution_str, ground_truth):
        #         try:
        #             solution_str = self.postprocess_solution_fn(solution_str)
        #             if solution_str is None:
        #                 return 0.

        #             gt = self.postprocess_gt_fn(ground_truth)
        #             print(solution_str)
        #             print(gt)
        #             # gt = self.postprocess_gt_fn(ground_truth)
        #             # gt_tokens = " ".join(simple_tokenize(gt))
        #             # sl_tokens = " ".join(simple_tokenize(solution_str))
        #             # bleu = sacrebleu.sentence_bleu(sl_tokens, [gt_tokens]).score
        #             # return bleu / 100
        #         except Exception as err:
        #             return 0.

        # async def compute_answer_constraint_format_score(
        #     batch_parsed_results, batch_ground_truth
        # ):
        #     base_rewards = [0.0] * len(batch_ground_truth)

        #     prompt_mapper = defaultdict(list)
        #     for i, (parsed, gt) in enumerate(zip(batch_parsed_results, batch_ground_truth)):
        #         if parsed is None:
        #             pass
        #         else:
        #             _, constraint, _ = parsed
        #             if contain_chinese(gt["prompt"]):  # zh
        #                 if contain_chinese(constraint):
        #                     base_rewards[i] += 0.1
        #             else:  # en
        #                 if contain_chinese(constraint):
        #                     pass
        #                 else:
        #                     base_rewards[i] += 0.1
        #             constraint_verify_prompt = CONSTRAINT_VERIFY_TEMPLATE.format(
        #                 constraint=constraint).strip()
        #             prompt_mapper[constraint_verify_prompt].append(i)

        #     max_concurrent_requests = 16
        #     prompts = list(prompt_mapper.keys())

        #     results = await agent.run(prompts, max_concurrent_requests, desc="[Verify Constraint]", postprocess_fns=[partial(parse_json, best_effort="valid")]*len(prompts))
        #     for prompt, response in results:
        #         if response is not None:
        #             indices = prompt_mapper[prompt]
        #             for index in indices:
        #                 try:
        #                     if isinstance(response["valid"], bool):
        #                         if response["valid"]:
        #                             base_rewards[index] += 1.0
        #                         else:
        #                             base_rewards[index] -= 1.0
        #                     elif isinstance(response["valid"], str):
        #                         if response["valid"] == "True":
        #                             base_rewards[index] += 1.0
        #                         else:
        #                             base_rewards[index] -= 1.0
        #                 except Exception as err:
        #                     continue
        #     return base_rewards

        # class QwQLongCoTComputeScore(ComputeScoreBase):
        #     def __init__(self, split="train"):
        #         super().__init__(split=split)
        #         self.c_length_penalty = ConclusionTooLongPenalty(
        #             postprocess_solution_fn=QwQLongCoTComputeScore.postprocess_solution_fn)

        #     def get_penalties(self) -> Dict[str, Callable]:
        #         return {
        #             "CONCLUSION_LENGTH": self.c_length_penalty.get_penalty_or_reward
        #         }

        #     @classmethod
        #     def postprocess_solution_fn(cls, solution_str: str):
        #         solution_str = postprocess_solution(solution_str)
        #         try:
        #             thought = re.findall(r'<think>.*</think>',
        #                                  solution_str, re.DOTALL)[0]
        #         except Exception as err:
        #             return None
        #         conclusion = solution_str.replace(thought, "").strip()
        #         return conclusion
        #     def get_rm_rewards(self,
        #                        batch_data_sources,
        #                        batch_solution_str,
        #                        batch_ground_truth):
        #         rewards = compute_rm_score(
        #             batch_solution_str=batch_solution_str,
        #             batch_ground_truth=batch_ground_truth,
        #             postprocess_solution_fn=self.postprocess_solution_fn,
        #             parse_result_failure_score=self.parse_result_failure_score
        #         )
        #         reshape_rewards = []
        #         for reward, ground_truth in zip(rewards, batch_ground_truth):
        #             cate = self.get_question_type(ground_truth)
        #             if cate == "object":
        #                 if reward >= 0.115:
        #                     reward = 1.0
        #                 if reward <= 0.0 and reward >= -1.0:
        #                     reward = -1.0
        #             reshape_rewards.append(reward)
        #         return reshape_rewards
        # _qwq_longcot_compute_score_train = QwQLongCoTComputeScore(split="train")
        # _qwq_longcot_compute_score_valid = QwQLongCoTComputeScore(split="valid")
        # qwq_longcot_compute_score_train = _qwq_longcot_compute_score_train.compute_score
        # qwq_longcot_compute_score_valid = _qwq_longcot_compute_score_valid.compute_score
if __name__ == "__main__":
    pass
