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
import tqdm.asyncio
import asyncio as aio
from abc import abstractmethod
from collections import Counter
from typing import Dict, Any, Callable
from tqdm import tqdm as tqdm_nonasync
import xml.etree.ElementTree as ET
from functools import partial
from rouge_score import rouge_scorer
from collections import namedtuple, defaultdict


from openai import OpenAI, RateLimitError, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

RATE_LIMIT_RETRY_DELAY = 60
RATE_LIMIT_RETRY_ATTEMPTS = 10
WORKFLOW_AGENT_LOGFILE = os.getenv("WORKFLOW_AGENT_LOGFILE", None)

RM_URLS = [
    "http://10.130.3.206:5001",
]

BT_REWARD_URLS = [
    "http://10.130.3.206:5003"
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
        "temperature": 0.75,
        "timeout": 180,
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
        return simple_zh_tokenize(s)
    else:
        return simple_en_tokenize(s)


def simple_zh_tokenize(s):
    return list(jieba.cut(s.lower()))


def simple_en_tokenize(s):
    return s.lower().strip().split(" ")


def split_array(arr):
    odd = []
    even = []
    for num, elem in enumerate(arr):
        if num % 2 == 0:
            even.append(elem)
        else:
            odd.append(elem)
    return odd, even


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


def post_with_retry(RM_URLS, data, max_retries=3, retry_delay=1, suffix="/reward"):
    retries = 0
    while retries < max_retries:
        try:
            url = random.choice(RM_URLS)
            response = requests.post(f'{url}{suffix}', json=data, timeout=600)
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
        judge_prompt_key="ground_truth"
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
            "prompt": ground_truth[judge_prompt_key], "response": solution_str, "id": i
        }
        input_datas.append(input_data)

    if len(input_datas) > 0:
        for batch in tqdm_nonasync(batchify(input_datas, n=256), desc=f'[RM] batchify inference (batch=256)'):
            output_datas = post_with_retry(RM_URLS, batch)
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
            return f'{s[:1000]}... [省略] ...{s[-1000:]}'
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
            return "NO RECORD"

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
        for batch in tqdm_nonasync(batchify(input_datas, n=256), desc='[RM] batchify inference'):
            output_datas = post_with_retry(RM_URLS, batch)
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

        if ("<question>" in conclusion) or ("</question>" in conclusion):
            return None

        return conclusion.strip()

    @classmethod
    def extract_gt_question(cls, ground_truth):
        ground_truth = ground_truth["ground_truth"]
        # bg_flag = "Your response (the created question) must be the following:"
        # ed_flag = "Respond only with the created question directly"
        bg_flag = " 你的回答（构造的指令/问题）必须是下面这个："
        ed_flag = "2. 回答必须直接返回问题/指令，"

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
                 postprocess_gt_fn,
                 parse_result_failure_score=0.
                 ):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.postprocess_gt_fn = postprocess_gt_fn
        self.parse_result_failure_score = parse_result_failure_score

    def get_penalty_or_reward(self, solution_str, ground_truth):
        try:
            solution_str = self.postprocess_solution_fn(solution_str)
            if solution_str is None:
                return self.parse_result_failure_score

            gt = self.postprocess_gt_fn(ground_truth)

            gt_tokens = " ".join(simple_tokenize(gt))
            sl_tokens = " ".join(simple_tokenize(solution_str))
            bleu = sacrebleu.sentence_bleu(sl_tokens, [gt_tokens]).score
            return bleu / 100
        except Exception as err:
            return self.parse_result_failure_score


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
            "BLEU": self.bleu_similarity.get_penalty_or_reward,
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
                                   batch_solution_str,
                                   batch_ground_truth,
                                   ):

        base_rewards = [0.0] * len(batch_ground_truth)

        prompt_mapper = defaultdict(list)
        logs = []
        for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            solution_str = self.postprocess_solution_fn(solution_str)
            if solution_str is None:
                continue

            gt = self.extract_gt_question(ground_truth)

            sim_judge_prompt = self.JUDGE_QUESTION_SIMILARITY_FEWSHOTS + "\n\n\n" + self.JUDGE_QUESTION_SIMILARITY_TEMPLATE.format(
                authentic_question=gt,
                fabricate_question=solution_str,
            ).strip()
            logs.append(
                self.JUDGE_QUESTION_SIMILARITY_TEMPLATE.format(
                    authentic_question=gt,
                    fabricate_question=solution_str).strip()
            )

            prompt_mapper[sim_judge_prompt].append(i)

            max_concurrent_requests = 48
            prompts = list(prompt_mapper.keys())

        if len(logs) > 0:
            print("="*40 + "[Judge Prompt Display]" + "="*40)
            print(repr(random.choice(logs)))
            print("="*40 + "[Judge Prompt Display]" + "="*40)

        results = await agent.run(prompts, max_concurrent_requests, desc="[Judge QA Similarity]", postprocess_fns=[self.parse_llm_judge_sim_result]*len(prompts))
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

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
        penalty = defaultdict(dict)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            for key, fn in self.get_penalties().items():
                penalty[key][i] = fn(solution_str, ground_truth)

        rm_rewards = self.get_rm_rewards(
            batch_data_sources, batch_solution_str, batch_ground_truth)

        llm_judge_sim_rewards = await self.llm_judge_similarity(batch_solution_str, batch_ground_truth)

        final_results = []

        normed_score1, normed_score2 = [], []
        normed_penalty = defaultdict(dict)
        for i in range(len(batch_solution_str)):
            penalty_log_str = []

            normed_score1.append(llm_judge_sim_rewards[i])
            normed_score2.append(rm_rewards[i])

            for name, _penalty in penalty.items():
                if i in _penalty:
                    normed_penalty[name][i] = _penalty[i]
                    penalty_log_str.append(
                        f'{name}={_penalty[i]:.2f}')

        for i in range(len(batch_solution_str)):
            if np.std(normed_score1) != 0:
                score1 = (
                    normed_score1[i] - np.mean(normed_score1))/np.std(normed_score1)
            else:
                score1 = normed_score1[i]
            if np.std(normed_score2) != 0:
                score2 = (
                    normed_score2[i] - np.mean(normed_score2))/np.std(normed_score2)
            else:
                score2 = normed_score2[i]

            score = score1 + score2

            for name, _penalty in normed_penalty.items():
                if name == "LengthPenalty":
                    score += _penalty[i]
                elif name == "BLEU":
                    bleu_score = _penalty[i]
                    if np.std(list(_penalty.values())) != 0:
                        norm_bleu = (
                            bleu_score-np.mean(list(_penalty.values()))) / np.std(list(_penalty.values()))
                        score += norm_bleu
                    else:
                        norm_bleu = bleu_score
                        score += norm_bleu

            final_results.append(score)

            if self.split == "valid":
                print(
                    f"--------------------------------[VALID]--------------------------------")
                print(
                    f"【Solution】 `{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Ground Truth】`{self.log_ground_truth(batch_ground_truth[i])}`")
                print(
                    f'[TOTAL={score:.3f}] | RM={normed_score2[i]:.3f} | LLM={normed_score1[i]:.3f} | {" | ".join(penalty_log_str)}\n')
            elif self.split == "train" and random.random() < 0.01:
                print(
                    f"--------------------------------[TRAIN]--------------------------------")
                print(
                    f"【Solution】`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Ground Truth】`{self.log_ground_truth(batch_ground_truth[i])}`")
                print(
                    f'[TOTAL={score:.3f}] | RM={normed_score2[i]:.3f} | LLM={normed_score1[i]:.3f} | {" | ".join(penalty_log_str)}\n')
        return final_results

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      ):
        async def main():
            return await self._compute_score(batch_data_sources, batch_solution_str, batch_ground_truth)
        return aio.run(main())


_qwq_longcot_fabricate_qa_compute_score_v2_train = QwQLongCoTFabricateQAComputeScoreV2(
    split="train")
_qwq_longcot_fabricate_qa_compute_score_v2_valid = QwQLongCoTFabricateQAComputeScoreV2(
    split="valid")
qwq_longcot_fabricate_qa_compute_score_v2_train = _qwq_longcot_fabricate_qa_compute_score_v2_train.compute_score
qwq_longcot_fabricate_qa_compute_score_v2_valid = _qwq_longcot_fabricate_qa_compute_score_v2_valid.compute_score


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Pretrain Back-Translation
# ------------------------------------------------------------------------------------------------------------------------------------------------------

class LanguageInconsistencyPenalty(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 penalty_base=-1.0,
                 key="content"):
        self.penalty_base = penalty_base
        self.postprocess_solution_fn = postprocess_solution_fn
        self.key = key

    def get_penalty_or_reward(self, solution_str, ground_truth):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        penalty = 0
        if contain_chinese(ground_truth[self.key]):
            if not contain_chinese(solution_str):
                return self.penalty_base
        else:
            if contain_chinese(solution_str):
                return self.penalty_base
        return penalty


class FormatPenalty(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn):
        self.postprocess_solution_fn = postprocess_solution_fn

    def get_penalty_or_reward(self, solution_str, ground_truth):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return -1.0

        if len(solution_str.strip()) < 10:
            return -2.0

        return 0.


class QwQLongCoTPretrainBackTranslationComputeScore(ComputeScoreBase):
    def __init__(self, split="train", parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD):
        super().__init__(split=split, parse_result_failure_score=parse_result_failure_score)

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "LanguageInconsistency": LanguageInconsistencyPenalty(postprocess_solution_fn=self.postprocess_solution_fn).get_penalty_or_reward,
            "FormatPenalty": FormatPenalty(postprocess_solution_fn=self.postprocess_solution_fn).get_penalty_or_reward,
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
            conclusion = re.findall(r'<instruction>(.*)</instruction>',
                                    solution_str, re.DOTALL)[0].strip()
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

        base_rewards = self.get_bt_rewards(
            batch_data_sources, batch_solution_str, batch_ground_truth)

        final_results = []
        for i in range(len(batch_solution_str)):
            penalty_log_str = []
            _reward = base_rewards[i]
            for name, _penalty in penalty.items():
                if i in _penalty:
                    _reward += _penalty[i]
                    try:
                        penalty_log_str.append(
                            f'{name} Penalty={_penalty[i]:.2f}')
                    except Exception as _:
                        pass

            final_results.append(_reward)

            if self.split == "valid":
                print(
                    f"--------------------------------[VALID]--------------------------------")
                print(
                    f"【Solution】 `{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Corpus】`{self.log_ground_truth(batch_ground_truth[i])}`")
                print(
                    f'Reward={_reward:.3f} | {" | ".join(penalty_log_str)}\n')
            elif self.split == "train" and random.random() < 0.01:
                print(
                    f"--------------------------------[TRAIN]--------------------------------")
                print(
                    f"【Solution】`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Corpus】`{self.log_ground_truth(batch_ground_truth[i])}`")
                print(
                    f'Reward={_reward:.3f} | {" | ".join(penalty_log_str)}\n')
        return final_results

    def log_ground_truth(self, ground_truth):
        return repr(self.clip_string(ground_truth["content"]))

    def compute_bt_reward(
            self,
            leadings,
            trailings,
    ):
        input_datas = []
        rewards = {}

        for i, (leading, trailing) in enumerate(zip(leadings, trailings)):
            if leading is not None:
                leading = self.postprocess_solution_fn(leading)

                if leading is None:
                    rewards[i] = self.parse_result_failure_score
                    continue
                if trailing is None:
                    rewards[i] = self.parse_result_failure_score
                    continue

            if leading is None:
                input_data = {
                    "completion": trailing, "id": i
                }
            else:
                input_data = {
                    "prompt": leading, "response": trailing, "id": i
                }
            input_datas.append(input_data)

        if len(input_datas) > 0:
            batch_size = 128
            for batch in tqdm_nonasync(batchify(input_datas, n=batch_size), desc=f'[BT Reward] batchify ({batch_size}) inference'):
                output_datas = post_with_retry(
                    BT_REWARD_URLS, batch, suffix="/bt_reward")
                for _ in output_datas['reward']:
                    _id = int(_["id"])
                    rewards[_id] = _["info"]

        final_results = []
        for i in range(len(trailings)):
            if i in rewards:
                final_results.append(rewards[i])
            else:
                final_results.append(0.)
        return final_results

    def log_solution(self, solution):
        norm = self.postprocess_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return repr(self.clip_string(norm))

    def clip_string(self, s: str):
        if len(s) > 1500:
            return f'{s[:700]}... [省略] ...{s[-800:]}'
        return s

    def get_bt_rewards(self,
                       batch_data_sources,
                       batch_solution_str,
                       batch_ground_truth):

        new_batch_solution_str, new_batch_ground_truth = [], []
        for gt, sol in zip(batch_ground_truth, batch_solution_str):
            # H(Y)
            gt_content = gt["content"]
            new_batch_solution_str.append(
                "<think></think>\n<instruction></instruction>\n")
            new_batch_ground_truth.append(gt_content)

            # H(X,Y)
            new_batch_solution_str.append(sol)
            new_batch_ground_truth.append(gt_content)

            # H(X)
            new_batch_solution_str.append(sol)
            new_batch_ground_truth.append("")

        rewards = self.compute_bt_reward(
            batch_solution_str=new_batch_solution_str,
            batch_ground_truth=new_batch_ground_truth,
        )

        def split_array(arr):
            first, second, third = [], [], []
            for num, elem in enumerate(arr):
                if num % 3 == 0:
                    first.append(elem)
                elif num % 3 == 1:
                    second.append(elem)
                else:
                    third.append(elem)
            return first, second, third

        scores = []

        MIN_H_X_CLIP = 4000
        first, second, third = split_array(rewards)
        for h_y, h_x_y, h_x in zip(first, second, third):
            if any(_ == self.parse_result_failure_score for _ in (h_y, h_x_y, h_x)):
                scores.append(self.parse_result_failure_score)
            else:
                scores.append((h_y+h_x-h_x_y) / max(h_x, MIN_H_X_CLIP))
        return scores


_qwq_longcot_pretrain_back_translation_compute_score_train = QwQLongCoTPretrainBackTranslationComputeScore(
    split="train")
_qwq_longcot_pretrain_back_translation_compute_score_valid = QwQLongCoTPretrainBackTranslationComputeScore(
    split="valid")
qwq_longcot_pretrain_back_translation_compute_score_train = _qwq_longcot_pretrain_back_translation_compute_score_train.compute_score
qwq_longcot_pretrain_back_translation_compute_score_valid = _qwq_longcot_pretrain_back_translation_compute_score_valid.compute_score


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# SFT Back-Translation
# ------------------------------------------------------------------------------------------------------------------------------------------------------

class QwQLongCoTSFTBackTranslationComputeScore(QwQLongCoTPretrainBackTranslationComputeScore):
    def __init__(self, split="train", parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD):
        super().__init__(split=split, parse_result_failure_score=parse_result_failure_score)

        self.bleu_similarity = BleuSimilarity(
            postprocess_solution_fn=QwQLongCoTSFTBackTranslationComputeScore.postprocess_solution_fn,
            postprocess_gt_fn=self.extract_gt_question
        )

    @classmethod
    def extract_gt_question(cls, ground_truth):
        ground_truth = ground_truth["ground_truth"]
        return ground_truth

    def log_ground_truth(self, ground_truth):
        return repr(self.clip_string(ground_truth["ground_truth"]))

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "LanguageInconsistency": LanguageInconsistencyPenalty(postprocess_solution_fn=self.postprocess_solution_fn, key="ground_truth").get_penalty_or_reward,
            "FormatPenalty": FormatPenalty(postprocess_solution_fn=self.postprocess_solution_fn).get_penalty_or_reward,
            "BLEU": self.bleu_similarity.get_penalty_or_reward,
        }

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      ):

        penalty = defaultdict(dict)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            for key, fn in self.get_penalties().items():
                penalty[key][i] = fn(solution_str, ground_truth)

        final_results = []

        for i in range(len(batch_solution_str)):
            penalty_log_str = []
            _reward = 0.0
            for name, _penalty in penalty.items():
                if i in _penalty:
                    _reward += _penalty[i]
                    try:
                        penalty_log_str.append(
                            f'{name} Penalty={_penalty[i]:.2f}')
                    except Exception as _:
                        pass

            final_results.append(_reward)

            if self.split == "valid":
                print(
                    f"--------------------------------[VALID]--------------------------------")
                print(
                    f"【Solution】 `{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Ground Truth】`{self.log_ground_truth(batch_ground_truth[i])}`")
                print(
                    f'Reward={_reward:.3f} | {" | ".join(penalty_log_str)}\n')
            elif self.split == "train" and random.random() < 0.01:
                print(
                    f"--------------------------------[TRAIN]--------------------------------")
                print(
                    f"【Solution】`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Ground Truth】`{self.log_ground_truth(batch_ground_truth[i])}`")
                print(
                    f'Reward={_reward:.3f} | {" | ".join(penalty_log_str)}\n')
        return final_results


_qwq_longcot_sft_back_translation_compute_score_train = QwQLongCoTSFTBackTranslationComputeScore(
    split="train", parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD)
_qwq_longcot_sft_back_translation_compute_score_valid = QwQLongCoTSFTBackTranslationComputeScore(
    split="valid", parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD)
qwq_longcot_sft_back_translation_compute_score_train = _qwq_longcot_sft_back_translation_compute_score_train.compute_score
qwq_longcot_sft_back_translation_compute_score_valid = _qwq_longcot_sft_back_translation_compute_score_valid.compute_score


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Pretrain RL
# ------------------------------------------------------------------------------------------------------------------------------------------------------

class CorpusLengthPenalty(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 postprocess_gt_fn,
                 penalty_base=-0.4,
                 mode="lt_gt"
                 ):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.penalty_base = penalty_base
        self.postprocess_gt_fn = postprocess_gt_fn
        self.mode = mode

    def get_penalty_or_reward(self, solution_str, ground_truth):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        gt = self.postprocess_gt_fn(ground_truth)
        if contain_chinese(gt):
            gt_token_size = len(simple_zh_tokenize(gt))
            sol_token_size = len(simple_zh_tokenize(solution_str))
        else:
            gt_token_size = len(simple_en_tokenize(gt))
            sol_token_size = len(simple_en_tokenize(solution_str))

        if self.mode == "lt_gt":
            return self.penalty_base * min(max((gt_token_size-sol_token_size), 0) / gt_token_size, 5.)
        else:
            return self.penalty_base * min(abs(sol_token_size-gt_token_size) / gt_token_size, 5.)


class CoTPretrainRLComputeScore(ComputeScoreBase):
    def __init__(self, split="train", parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD):
        super().__init__(split=split, parse_result_failure_score=parse_result_failure_score)

        self.bleu_similarity = BleuSimilarity(
            postprocess_solution_fn=CoTPretrainRLComputeScore.postprocess_solution_fn,
            postprocess_gt_fn=lambda x: x["ground_truth"],
            parse_result_failure_score=self.parse_result_failure_score
        )
        self.length_penalty = CorpusLengthPenalty(
            postprocess_solution_fn=CoTPretrainRLComputeScore.postprocess_solution_fn,
            postprocess_gt_fn=lambda x: x["ground_truth"],
        )

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "BLEU": self.bleu_similarity.get_penalty_or_reward,
            "LengthPenalty": self.length_penalty.get_penalty_or_reward,
        }

    @classmethod
    def postprocess_solution_fn(cls, solution_str: str):
        solution_str = postprocess_solution(solution_str)
        try:
            thought = re.findall(r'<chain-of-thought>.*</chain-of-thought>',
                                 solution_str, re.DOTALL)[0]
        except Exception as err:
            return None
        try:
            corpus = re.findall(r'<doc>(.*)</doc>',
                                solution_str, re.DOTALL)[0].strip()
        except Exception as err:
            return None
        return corpus

    def get_thought(self, solution_str: str):
        solution_str = postprocess_solution(solution_str)
        try:
            thought = re.findall(r'<chain-of-thought>(.*)</chain-of-thought>',
                                 solution_str, re.DOTALL)[0]
        except Exception as err:
            return f"<FORMAT CORRUPT>"
        return thought

    def get_rm_rewards(self,
                       batch_data_sources,
                       batch_solution_str,
                       batch_ground_truth):
        return compute_rm_score(
            batch_solution_str=batch_solution_str,
            batch_ground_truth=batch_ground_truth,
            postprocess_solution_fn=self.postprocess_solution_fn,
            parse_result_failure_score=self.parse_result_failure_score,
            judge_prompt_key="judge_criteria"
        )

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      ):

        penalty = defaultdict(dict)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            for key, fn in self.get_penalties().items():
                penalty[key][i] = fn(solution_str, ground_truth)

        final_results = []
        for i in range(len(batch_solution_str)):
            score = 0.0
            penalty_log_str = []
            for name, _penalty in penalty.items():
                if i in _penalty:
                    try:
                        score += _penalty[i]
                        penalty_log_str.append(
                            f'{name}={_penalty[i]:.2f}')
                    except Exception as _:
                        pass
            final_results.append(score)

            thought = self.get_thought(batch_solution_str[i])

            if self.split == "valid":
                print(
                    f"--------------------------------[VALID]--------------------------------")
                print(
                    f'【Prompt】`{repr(self.clip_string(batch_ground_truth[i]["prompt"]))}`')
                print(
                    f"【Thought】`{repr(self.clip_string(thought))}`")
                print(
                    f"【Solution】 `{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Ground Truth】`{self.log_ground_truth(batch_ground_truth[i])}`")
                print(
                    f'Reward={score:.3f} | {" | ".join(penalty_log_str)}\n')
            elif self.split == "train" and random.random() < 0.01:
                print(
                    f"--------------------------------[TRAIN]--------------------------------")
                print(
                    f'【Prompt】`{repr(self.clip_string(batch_ground_truth[i]["prompt"]))}`')
                print(
                    f"【Thought】`{repr(self.clip_string(thought))}`")
                print(
                    f"【Solution】`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Ground Truth】`{self.log_ground_truth(batch_ground_truth[i])}`")
                print(
                    f'Reward={score:.3f} | {" | ".join(penalty_log_str)}\n')
        return final_results

    def log_ground_truth(self, ground_truth):
        return repr(self.clip_string(ground_truth["ground_truth"]))

    def log_solution(self, solution):
        norm = self.postprocess_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return repr(self.clip_string(norm))

    def clip_string(self, s: str):
        if len(s) > 1500:
            return f'{s[:700]}... [省略] ...{s[-800:]}'
        return s


_cot_pretrain_rl_compute_score_train = CoTPretrainRLComputeScore(
    split="train", parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD)
_cot_pretrain_rl_compute_score_valid = CoTPretrainRLComputeScore(
    split="valid", parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD)
cot_pretrain_rl_compute_score_train = _cot_pretrain_rl_compute_score_train.compute_score
cot_pretrain_rl_compute_score_valid = _cot_pretrain_rl_compute_score_valid.compute_score


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Pretrain Refinement
# ------------------------------------------------------------------------------------------------------------------------------------------------------

class CoTPretrainRefineFormatReward(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 postprocess_gt_fn):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.postprocess_gt_fn = postprocess_gt_fn

    def get_penalty_or_reward(self, solution_str, ground_truth):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        gt = self.postprocess_gt_fn(ground_truth)

        if not contain_chinese(gt):
            if "【注】" in solution_str or "【/注】" in solution_str:
                return -0.5

        try:
            max_shaped_reward = 0.1
            reward_per_note = 0.01
            en_notes = re.findall(
                r'\[Note\].*?\[/Note\]', solution_str, re.DOTALL)
            zh_notes = re.findall(r'【注】.*?【/注】', solution_str, re.DOTALL)
            num_notes = len(en_notes) + len(zh_notes)
            return min(num_notes * reward_per_note, max_shaped_reward)
        except Exception as err:
            return 0.


class ROUGEScorer(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 parse_result_failure_score=0.,
                 remove_latex_format=True
                 ):
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.parse_result_failure_score = parse_result_failure_score
        self.postprocess_solution_fn = postprocess_solution_fn
        self.remove_latex_format = remove_latex_format

    def get_thought(self, solution_str: str):
        solution_str = postprocess_solution(solution_str)
        try:
            thought = re.findall(r'<chain-of-thought>(.*)</chain-of-thought>',
                                 solution_str, re.DOTALL)[0]
        except Exception as err:
            return f"<FORMAT CORRUPT>"
        return thought

    def zh_ignore_words(self):
        return ['`', 'latex', 'textbf', '【', '】', '$', 'begin', 'figure', 'h', 'centering', 'includegraphics', 'width', 'textwidth', 'caption', 'description', 'end', 'section', '*', '/',
                'png', '%',  'the', 'file',  'diagram', 'for', 'variant', 'label', 'fig', 'documentclass', 'article', 'usepackage', 'amsmath', 'geometry', 'chemfig', 'document', 'noindent', 'center', 'jpg',
                '####', 'column', 'center', 'mathit', 'subsection', 'section'
                ]

    def get_penalty_or_reward(self, solution_str, ground_truth):
        def remove_latex_format(text):
            # 移除LaTeX命令
            text = re.sub(r'\\[a-zA-Z]+(\{[^\}]+\})?', '', text)
            # 移除LaTeX环境
            text = re.sub(
                r'\\begin\{[a-zA-Z]+\}(.*?)\\end\{[a-zA-Z]+\}', r'\1', text, flags=re.DOTALL)
            # 移除LaTeX特殊字符
            text = re.sub(r'[\$#%&_{}]', '', text)
            return text

        try:
            solution_str = self.postprocess_solution_fn(solution_str)
            if solution_str is None:
                return self.parse_result_failure_score

            if self.remove_latex_format:
                gt = remove_latex_format(ground_truth["ground_truth"])
                solution_str = remove_latex_format(solution_str)
            else:
                gt = ground_truth["ground_truth"]

            if contain_chinese(gt):
                gt_tokens = simple_zh_tokenize(gt)
                sl_tokens = simple_zh_tokenize(solution_str)

                gt_tokens = [
                    _ for _ in gt_tokens if _ not in self.zh_ignore_words()]
                sl_tokens = [
                    _ for _ in sl_tokens if _ not in self.zh_ignore_words()]
            else:
                gt_tokens = simple_en_tokenize(gt)
                sl_tokens = simple_en_tokenize(solution_str)

            gt_tokens = " ".join(
                [_ for _ in gt_tokens])
            sl_tokens = " ".join([_ for _ in sl_tokens])
            score = self.scorer.score(gt_tokens, sl_tokens)

            rouge_recall = (score["rouge1"].recall +
                            score["rouge2"].recall) / 2.0

            # reward分段奖励
            if rouge_recall >= 0.70:
                return 1.0
            elif rouge_recall >= 0.5:
                return rouge_recall
            else:
                return rouge_recall / 2.0

        except Exception as err:
            return self.parse_result_failure_score


class CoTPretrainRefineComputeScore(QwQLongCoTPretrainBackTranslationComputeScore):
    JUDGE_CRITERIA = """

以下*完整大模型数据注释治理评价标准（Criteria）**，覆盖20+细分场景，附详细示例及评估细则：


### **1. 信息完备性（全语料类型思考过程显化）**  
#### ▶ 核心要求：覆盖20+语料类型，按场景强制补充专属思考过程，实现“注释即领域知识库”  
| **语料类型**       | **必显化的思考过程**                          | **合格注释示例**                                                                 |  
|--------------------|---------------------------------------------|--------------------------------------------------------------------------------|  
| **教育/习题**       | 命题逻辑（考察知识点、易错点、能力要求）       | 【注】【命题逻辑】本题为高考数学导数压轴题，考察“极值点偏移”问题，核心方法：构造对称函数f(a+x)-f(a-x)，易错点：忽略定义域对单调性的影响。【/注】<br>已知f(x)=x²e^(-x)，若x₁≠x₂且f(x₁)=f(x₂)，证明x₁+x₂>2。 |  
| **技术文档**       | 参数设计逻辑（功能/默认值/约束条件）           | 【注】【API设计】`timeout=5000`：①HTTP请求超时时间（单位ms），基于行业标准（4000-6000ms）及业务容错率（允许1%超时）设定；②超时后返回504 Gateway Timeout，需搭配重试机制使用。【/注】<br>接口定义：`GET /api/data?timeout=5000` |  
| **法律文书**       | 条款释明（法律依据/责任边界/风险场景）         | 【注】【法条援引】“情势变更条款”依据《民法典》第533条：合同成立后，若政策调整、市场剧烈波动等不可预见事件导致履约显失公平，双方可协商变更或解除合同。【/注】<br>合同条款：因政府限购政策导致乙方无法购房，本协议自动解除且互不担责。 |  
| **科研论文**       | 实验设计逻辑（方法选择/变量控制/数据验证）       | 【注】【实验设计】采用Western Blot检测蛋白表达：①一抗选择兔抗人CD31（1:1000），因其特异性识别血管内皮细胞标志物；②内参β-actin用于校正上样量差异，重复实验3次确保结果可重复性。【/注】<br>结果部分：CD31蛋白在肿瘤组织中的表达量为正常组织的2.3倍（P<0.05）。 |  
| **创意写作**       | 意象隐喻（象征体系/情感映射/叙事视角）           | 【注】【叙事策略】本文采用“儿童视角”叙事：以7岁女孩的眼睛观察父母离婚，通过“小熊玩偶丢失”象征家庭完整的破碎，“彩虹橡皮擦”隐喻对美好结局的幼稚期待。【/注】<br>她把小熊塞进书包最底层，就像把爸爸妈妈吵架的声音也一起藏了起来。 |  
| **历史文献**       | 时代背景（事件因果/制度背景/文化符号）           | 【注】【历史语境】“废井田，开阡陌”出自《史记·商君列传》：战国时期商鞅变法废除井田制，允许土地私有买卖，根本原因是铁器牛耕普及推动生产力发展，导致奴隶制生产关系崩溃。【/注】<br>原文：秦孝公用商君，坏井田，开仟佰，急耕战之赏。 |  
| **科普文章**       | 原理类比（生活化比喻/跨学科类比）               | 【注】【原理阐释】“黑洞引力”可类比为“保龄球压弯蹦床”：质量大的物体（保龄球）会扭曲时空（蹦床），周围物体（弹珠）会沿扭曲路径运动，解释为何光也无法逃离黑洞。【/注】<br>爱因斯坦广义相对论指出，引力本质是时空曲率的表现。 |  
| **日常沟通**       | 隐含意图（潜台词/语气判断/社交语境）           | 【注】【沟通意图】此句为委婉拒绝：“最近有点忙”实际表示“不愿参与本次聚餐”，潜台词是“关系尚未亲密到牺牲个人时间赴约”，需注意对方后续是否主动提供替代方案。【/注】<br>微信对话：A：周末要不要一起吃饭？B：最近有点忙，下次吧。 |  
| **翻译文本**       | 文化差异（不可译元素/语境适配/双关处理）         | 【注】【文化注释】“阿Q精神”译为“Ah Q mentality”时补充：源自鲁迅小说《阿Q正传》，指通过自我安慰、精神胜利法化解现实挫败的心理机制，是中国现代文学中独特的文化符号。【/注】<br>英文译文：His Ah Q mentality made him ignore the obvious problems. |  
| **商业报告**       | 数据逻辑（指标定义/统计口径/趋势归因）           | 【注】【数据说明】“用户留存率=（第N日活跃用户数/首日新增用户数）×100%”，本次统计周期为30天，排除测试账号及机器人用户，下降原因可能与3.15版本功能复杂度提升有关。【/注】<br>报告：3月用户30日留存率降至18%，环比下降5个百分点。 |  
| **医学文档**       | 医学逻辑（疾病诊断/用药依据/指南共识）           | 【注】【诊疗指南】诊断“2型糖尿病”依据WHO标准：空腹血糖≥7.0mmol/L或随机血糖≥11.1mmol/L，本例患者BMI=28.5（肥胖）、HbA1c=7.5%，符合指南中“生活方式干预+二甲双胍起始治疗”建议。【/注】<br>病历：初步诊断：2型糖尿病；治疗方案：二甲双胍0.5g tid po。 |  
| **艺术评论**       | 艺术逻辑（创作手法/流派渊源/文化象征）           | 【注】【艺术分析】徐冰《天书》使用“伪汉字”：看似汉字的印刷体实为艺术家自创字符，旨在探讨文字作为符号的权威性与虚无性，是后现代主义对传统书法的解构实践。【/注】<br>作品：徐冰耗时4年创作《天书》，收录4000多个无人能识的“新汉字”。 |  
| **工程规范**       | 技术逻辑（参数标准/安全阈值/执行流程）           | 【注】【工程标准】“混凝土保护层厚度30mm”依据GB50010-2010：梁柱类构件在室内干燥环境中最小保护层厚度为20mm，本项目因处于沿海盐雾环境，需增加10mm以提高钢筋防腐能力。【/注】<br>施工图纸：基础梁钢筋保护层厚度：30mm（迎水面）。 |  
| **财经新闻**       | 经济逻辑（政策影响/市场反应/术语解析）           | 【注】【政策解读】“降准25个基点”指存款准备金率下调0.25%，释放约5000亿流动性，对股市利好金融板块（银行可贷资金增加），但需警惕资金空转风险（历史数据：2022年同类操作后M2与GDP剪刀差扩大3个百分点）。【/注】<br>新闻：中国人民银行决定于2023年9月15日下调金融机构存款准备金率0.25个百分点。 |  
| **用户手册**       | 操作逻辑（步骤原理/风险提示/场景适配）           | 【注】【操作指南】“先关主机再拔电源线”：①防止突然断电导致硬盘磁头划伤（机械硬盘风险）；②Windows系统需正常关机以保存缓存数据，强制断电可能引发文件系统错误（错误代码：0x0000007B）。【/注】<br>关机步骤：1. 点击“开始”→“关机”；2. 待电源指示灯熄灭后拔下电源线。 |  


### **2. 内容完整性（全场景关键信息无死角覆盖）**  
#### ▶ 新增强制注释场景（补全15+细分项）  
- **跨语言场景**：翻译文本中文化负载词（如“阴阳”“科举”）需注释文化内涵，禁止直接音译无解释；  
- **数据可视化**：图表标题/轴标签需注释数据口径（如“YOY增长率=（当期值/去年同期-1）×100%”）；  
- **多模态内容**：图片/视频描述中专业元素（如显微镜照片中的“标尺=10μm”）需注释测量标准；  
- **法律合同**：除术语外，数字金额（如“￥1,000,000”）需注释大写（人民币壹佰万元整）及支付条件；  
- **医学影像报告**：“结节直径8mm”需注释BI-RADS分级（如3类：良性可能，建议6个月随访）。  


### **3. 格式规范性（新增多模态格式约束）**  
#### ▶ 补充格式要求  
- **表格/图表**：  
  - 表头术语首次出现时需注释（如“BMI”列需在表格上方添加“【注】BMI=体重(kg)/身高(m)²【/注】”）；  
  - 数据单位不明确时需补充（如“收入：50”需注释“单位：万元”）。  
- **多语言混排**：  
  - 中文文档中夹杂英文术语（如“AI模型”）无需注释，但首次出现“Transformer架构”需注释“一种基于自注意力机制的神经网络结构”。  


### **4. 语言一致性（新增专业语料风格匹配细则）**  
#### ▶ 细分风格匹配要求  
- **医学语料**：注释需使用《ICD-11》标准术语（如“糖尿病”不得写“消渴症”）；  
- **法律语料**：注释需引用具体法条编号（如“依据《刑法》第264条盗窃罪”而非“根据相关法律”）；  
- **财经语料**：数据注释需保留有效数字（如“GDP增长率2.30%”不得简化为“2.3%”）。  


### **完整评价标准（2024版）**  
#### **1. 信息完备性（40%权重）**  
1. **原文信息零丢失**（15%）：数据、结论、术语、公式等核心内容100%保留，无任何删除/篡改；  
2. **场景化思考显化**（25%）：按语料类型强制补充专属注释（如教育→命题逻辑，法律→法条援引），每个必注场景（见上表）均有对应注释，示例覆盖度≥80%。  

#### **2. 内容完整性（25%权重）**  
1. **关键信息覆盖**（15%）：专业术语/公式/逻辑跳步等晦涩点100%被注释，无首次出现未解释情况；  
2. **注释有效性**（10%）：注释包含完整逻辑链条（隐含前提+推导细节），非重复性/笼统性解释。  

#### **3. 格式规范性（15%权重）**  
1. **符号正确性**（8%）：中文用“【注】”，英文用“[Note]”，代码注释符合语言规范；  
2. **位置精准性**（5%）：注释位于待解释内容前，无后置/嵌入错误；  
3. **技术格式**（2%）：表格/代码/公式格式正确，无语法错误。  

#### **4. 干净整洁度（10%权重）**  
1. **无冗余注释**（5%）：同一内容仅注释1次，无无关扩展；  
2. **结构无损**（5%）：注释不拆分原文语句，段落/代码排版保持原样。  

#### **5. 语言一致性（10%权重）**  
1. **语种统一**（5%）：全文语言一致，注释与原文语种/代码语言匹配；  
2. **风格匹配**（5%）：注释术语/语气与原文场景一致（如医学严谨/科普通俗）。  


### **一票否决项（新增3项）**  
1. **专业场景核心逻辑缺失**：如医学文档未注释诊断标准、法律文书未引用法律依据；  
2. **多模态信息不完整**：图表无数据口径注释、翻译文本无文化差异说明；  
3. **安全相关注释缺失**：工程规范未注释安全阈值、用户手册未提示操作风险。  


### **评估工具建议**  
1. **场景 checklist**：按语料类型（教育/法律/医学等）列出必注项，逐项打勾验证；  
2. **注释追溯表**：记录每个注释对应原文内容、场景类型、逻辑链条，确保无遗漏；  
3. **格式校验脚本**：自动检测注释符号错误、位置错误、代码格式问题。  

此标准通过**20+语料类型全覆盖、100+细分场景显化要求、5大维度量化评估**，实现从通用文本到专业领域的注释治理标准化，确保大模型训练数据既保留原始语义，又具备领域可解释性，最终提升模型输出的准确性与可信赖度。
"""

    def __init__(self, split="train", parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD, rouge_coef=1.0, info_coef=1.5):
        super().__init__(split=split, parse_result_failure_score=parse_result_failure_score)
        self.rouge_coef = rouge_coef
        self.info_coef = info_coef

        self.rouge = ROUGEScorer(
            postprocess_solution_fn=CoTPretrainRefineComputeScore.postprocess_for_rouge,
            parse_result_failure_score=0.0
        )
        self.length_penalty = CorpusLengthPenalty(
            postprocess_solution_fn=CoTPretrainRefineComputeScore.postprocess_for_rouge,
            postprocess_gt_fn=lambda x: x["ground_truth"],
            penalty_base=-0.1
        )
        self.format_penalty = CoTPretrainRefineFormatReward(
            postprocess_solution_fn=CoTPretrainRefineComputeScore.postprocess_solution_fn,
            postprocess_gt_fn=lambda x: x["ground_truth"],
        )

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "ROUGE": self.rouge.get_penalty_or_reward,
            "LengthPenalty": self.length_penalty.get_penalty_or_reward,
            "Format": self.format_penalty.get_penalty_or_reward
        }

    @classmethod
    def postprocess_for_rouge(cls, solution_str: str):
        document = cls.postprocess_solution_fn(solution_str)
        if not isinstance(document, str):
            return None
        document = re.sub(r'\[Note\].*?\[/Note\]', "", document)
        document = re.sub(r'【注】.*?【/注】', "", document)
        return document

    @classmethod
    def postprocess_solution_fn(cls, solution_str: str):
        solution_str = postprocess_solution(solution_str)
        try:
            thought = re.findall(r'<chain-of-thought>.*</chain-of-thought>',
                                 solution_str, re.DOTALL)[0]
        except Exception as err:
            return None
        try:
            document = re.findall(r'<doc>(.*)</doc>',
                                  solution_str, re.DOTALL)[0].strip()
        except Exception as err:
            return None

        if any(_ in document for _ in ("<chain-of-thought>", "</chain-of-thought>", "<doc>", "</doc>")):
            return None
        return document

    def get_thought(self, solution_str: str):
        solution_str = postprocess_solution(solution_str)
        try:
            thought = re.findall(r'<chain-of-thought>(.*)</chain-of-thought>',
                                 solution_str, re.DOTALL)[0]
        except Exception as err:
            return f"<FORMAT CORRUPT>"
        return thought

    def get_bt_rewards(self,
                       batch_data_sources,
                       batch_solution_str,
                       batch_ground_truth,
                       min_clip=-0.05
                       ):

        def postprocess_sol(sol):
            # 取出[Note][/Note]等标志符
            return sol.replace("[Note]", "").replace("[/Note]", "").replace("【注】", "").replace("【/注】", "")

        new_batch_solution_str, new_batch_ground_truth = [], []
        for gt, sol in zip(batch_ground_truth, batch_solution_str):
            sol = postprocess_sol(sol)

            # H修改前
            gt_content = gt["ground_truth"]
            new_batch_solution_str.append(
                "<chain-of-thought></chain-of-thought>\n<doc>文档内容包含注解，注解格式为“【注】... ...【/注】”或者“[Note] ... ...[/Note] ”</doc>\n")
            new_batch_ground_truth.append(gt_content)

            # H修改后
            new_batch_solution_str.append(
                "<chain-of-thought></chain-of-thought>\n<doc>文档内容包含注解，注解格式为“【注】... ...【/注】”或者“[Note] ... ...[/Note] ”</doc>\n")
            new_batch_ground_truth.append(self.postprocess_solution_fn(sol))

        rewards = self.compute_bt_reward(
            leadings=new_batch_solution_str,
            trailings=new_batch_ground_truth,
        )

        def split_array(arr):
            first, second = [], []
            for num, elem in enumerate(arr):
                if num % 2 == 0:
                    first.append(elem)
                else:
                    second.append(elem)
            return first, second

        scores = []

        first, second = split_array(rewards)
        for before, after in zip(first, second):
            if any(_ == self.parse_result_failure_score for _ in (before, after)):
                scores.append(self.parse_result_failure_score)
            elif before == 0:
                scores.append(0.)
            else:
                if min_clip is not None:
                    scores.append(max(((before-after)/before), -0.05))
                else:
                    scores.append((before-after)/before)
        return scores

    def get_rm_rewards(self,
                       batch_data_sources,
                       batch_solution_str,
                       batch_ground_truth):
        for _ in batch_ground_truth:
            _["ground_truth"] = f'你是一名专精于大模型数据治理的专家。你的任务目标是给定一个提供的预训练语料，处理成一个高质量训练数据。\n\n[Raw Corpus]\n{_["ground_truth"]}\n\n\n# 评价标准\n{self.JUDGE_CRITERIA}'
        rewards = compute_rm_score(
            batch_solution_str=batch_solution_str,
            batch_ground_truth=batch_ground_truth,
            postprocess_solution_fn=self.postprocess_solution_fn,
            parse_result_failure_score=self.parse_result_failure_score
        )
        return rewards

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
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            penalty_log_str = []
            _reward = self.info_coef * base_rewards[i]

            for name, _penalty in penalty.items():
                if i in _penalty:
                    if name == "ROUGE":
                        _reward += _penalty[i] * self.rouge_coef
                    else:
                        _reward += _penalty[i]
                    try:
                        penalty_log_str.append(
                            f'{name}={_penalty[i]:.3f}')
                    except Exception as _:
                        pass

            final_results.append(_reward)
            thought = self.get_thought(batch_solution_str[i])

            if self.split == "valid":
                print(
                    f"--------------------------------[VALID]--------------------------------")
                print(
                    f"【Thought】({len(thought)})`{repr(self.clip_string(thought))}`")
                print(
                    f"【Refine】({self.get_document_len(batch_solution_str[i])})`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f'【Raw】({len(batch_ground_truth[i]["ground_truth"])})``{self.log_ground_truth(batch_ground_truth[i])}`')
                print(
                    f'Reward (rouge_coef={self.rouge_coef}; info_coef={self.info_coef})={_reward:.3f} | info={base_rewards[i]:.3f} | {" | ".join(penalty_log_str)}\n')
            elif self.split == "train" and random.random() < 0.01:
                print(
                    f"--------------------------------[TRAIN]--------------------------------")
                print(
                    f"【Thought】({len(thought)})`{repr(self.clip_string(thought))}`")
                print(
                    f"【Refine】({self.get_document_len(batch_solution_str[i])})`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f'【Raw】({len(batch_ground_truth[i]["ground_truth"])})`{self.log_ground_truth(batch_ground_truth[i])}`')
                print(
                    f'Reward (rouge_coef={self.rouge_coef}; info_coef={self.info_coef})={_reward:.3f} | info={base_rewards[i]:.3f} | {" | ".join(penalty_log_str)}\n')
        return final_results

    def log_ground_truth(self, ground_truth):
        return repr(self.clip_string(ground_truth["ground_truth"]))

    def log_solution(self, solution):
        norm = self.postprocess_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return repr(self.clip_string(norm))

    def get_document_len(self, solution):
        norm = self.postprocess_solution_fn(solution)
        if norm is None:
            return 0
        return len(norm)

    def clip_string(self, s: str):
        if len(s) > 1500:
            return f'{s[:700]}... [省略] ...{s[-800:]}'
        return s


_cot_pretrain_refine_compute_score_train = CoTPretrainRefineComputeScore(
    split="train", parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD)
_cot_pretrain_refine_compute_score_valid = CoTPretrainRefineComputeScore(
    split="valid", parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD)
cot_pretrain_refine_compute_score_train = _cot_pretrain_refine_compute_score_train.compute_score
cot_pretrain_refine_compute_score_valid = _cot_pretrain_refine_compute_score_valid.compute_score


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Pretrain Annotation
# ------------------------------------------------------------------------------------------------------------------------------------------------------

class ROUGEScorerForPretrainAnnotation(ROUGEScorer):
    def __init__(self,
                 postprocess_solution_fn,
                 parse_result_failure_score=0.,
                 ):
        super().__init__(
            postprocess_solution_fn=postprocess_solution_fn,
            parse_result_failure_score=parse_result_failure_score
        )

    def get_penalty_or_reward(self, solution_str, ground_truth):
        try:
            gt = ground_truth["ground_truth"]
            solution_str = self.postprocess_solution_fn(solution_str)
            if solution_str is None:
                return self.parse_result_failure_score

            if contain_chinese(gt):
                gt_tokens = simple_zh_tokenize(gt)
                sl_tokens = simple_zh_tokenize(solution_str)
            else:
                gt_tokens = simple_en_tokenize(gt)
                sl_tokens = simple_en_tokenize(solution_str)

            gt_tokens = " ".join(gt_tokens)
            sl_tokens = " ".join(sl_tokens)
            score = self.scorer.score(gt_tokens, sl_tokens)
            final_score = []
            for k, s in score.items():
                final_score.append(s.fmeasure)
            return np.mean(final_score)
        except Exception as err:
            return self.parse_result_failure_score


class CoTPretrainAnnotationComputeScore(CoTPretrainRefineComputeScore):
    def __init__(self, split="train", parse_result_failure_score=-2.0, rouge_threshold=0.95):
        super().__init__(split=split, parse_result_failure_score=parse_result_failure_score,
                         rouge_coef=1.0, info_coef=1.0)
        self.rouge = ROUGEScorerForPretrainAnnotation(
            postprocess_solution_fn=CoTPretrainAnnotationComputeScore.postprocess_for_rouge,
            parse_result_failure_score=0.0
        )
        self.format_penalty = CoTPretrainRefineFormatReward(
            postprocess_solution_fn=CoTPretrainAnnotationComputeScore.postprocess_solution_fn,
            postprocess_gt_fn=lambda x: x["ground_truth"],
        )
        self.rouge_threshold = rouge_threshold

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "ROUGE": self.rouge.get_penalty_or_reward,
            "Format": self.format_penalty.get_penalty_or_reward
        }

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      ):

        penalty = defaultdict(dict)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            for key, fn in self.get_penalties().items():
                penalty[key][i] = fn(solution_str, ground_truth)

        base_rewards = self.get_bt_rewards(
            batch_data_sources, batch_solution_str, batch_ground_truth, min_clip=None)

        final_results = []
        for i in range(len(batch_solution_str)):
            penalty_log_str = []
            _reward = 0.0

            for name, _penalty in penalty.items():
                if i in _penalty:
                    if name == "ROUGE":
                        if base_rewards[i] == self.parse_result_failure_score:
                            _reward += base_rewards[i]
                        else:
                            # 差异过大,直接判定负分
                            if _penalty[i] < self.rouge_threshold:

                                _reward += self.parse_result_failure_score / 2
                            else:
                                _reward += base_rewards[i]
                    else:
                        _reward += _penalty[i]
                    try:
                        penalty_log_str.append(
                            f'{name}={_penalty[i]:.3f}')
                    except Exception as _:
                        pass

            final_results.append(_reward)
            thought = self.get_thought(batch_solution_str[i])

            if self.split == "valid":
                print(
                    f"--------------------------------[VALID]--------------------------------")
                print(
                    f"【Thought】({len(thought)})`{repr(self.clip_string(thought))}`")
                print(
                    f"【Refine】({self.get_document_len(batch_solution_str[i])})`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f'【Raw】({len(batch_ground_truth[i]["ground_truth"])})``{self.log_ground_truth(batch_ground_truth[i])}`')
                print(
                    f'Reward={_reward:.3f} | info={base_rewards[i]:.3f} | {" | ".join(penalty_log_str)}\n')
            elif self.split == "train" and random.random() < 0.01:
                print(
                    f"--------------------------------[TRAIN]--------------------------------")
                print(
                    f"【Thought】({len(thought)})`{repr(self.clip_string(thought))}`")
                print(
                    f"【Refine】({self.get_document_len(batch_solution_str[i])})`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f'【Raw】({len(batch_ground_truth[i]["ground_truth"])})`{self.log_ground_truth(batch_ground_truth[i])}`')
                print(
                    f'Reward={_reward:.3f} | info={base_rewards[i]:.3f} | {" | ".join(penalty_log_str)}\n')
        return final_results


_cot_pretrain_annotation_compute_score_train = CoTPretrainAnnotationComputeScore(
    split="train")
_cot_pretrain_annotation_compute_score_valid = CoTPretrainAnnotationComputeScore(
    split="valid")
cot_pretrain_annotation_compute_score_train = _cot_pretrain_annotation_compute_score_train.compute_score
cot_pretrain_annotation_compute_score_valid = _cot_pretrain_annotation_compute_score_valid.compute_score

if __name__ == "__main__":
    pass
