import re
import jieba
import random
import requests
import sacrebleu
from functools import partial
from abc import abstractmethod
from typing import Any, Dict, Callable
from sacremoses import MosesTokenizer, MosesDetokenizer
from tqdm import tqdm as tqdm_nonasync
from collections import namedtuple, defaultdict


from openai import OpenAI, RateLimitError, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

en_mt = MosesTokenizer(lang='en')

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------

RM_URLS = [
    "http://10.130.0.174:5020"
]
VERIFIER_MODEL_NAME = "qwen25_7B_instruct"
VERIFIER_MODEL_PATH = "http://10.130.247.138:8000/v1"
DEFAULT_PARSE_FAILURE_REWARD = -2.


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


class PenaltyOrReward(object):
    @abstractmethod
    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        raise NotImplementedError


def batchify(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def contain_chinese(string):
    try:
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if re.search(pattern, string):
            return True
        return False
    except Exception as err:
        return False


def postprocess_solution(solution_str):
    if "<|im_end|>" in solution_str:
        return solution_str[:solution_str.index("<|im_end|>")]
    return solution_str


def rm_request_with_retry(RM_URLS, data, max_retries=3, retry_delay=1, suffix="/reward"):
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
        for batch in tqdm_nonasync(batchify(input_datas, n=128), desc=f'[RM][{RM_URLS}] batchify inference (batch=128)'):
            output_datas = rm_request_with_retry(RM_URLS, batch)
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

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 沙盒问题合成（一阶段）
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def fabricate_qa_postprocess_solution_fn(solution_str: str):
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


class LengthDiffPenalty(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 penalty_base=-0.1):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.penalty_base = penalty_base

    def get_penalty_or_reward(self, solution_str, ground_truth):
        solution_str = self.postprocess_solution_fn(solution_str)
        if solution_str is None:
            return 0.

        gt = ground_truth["ground_truth"]
        lang_code = ground_truth["lang_code"]

        if lang_code == "en":
            gt_tokenized = en_mt.tokenize(gt.lower())
            sol_tokenized = en_mt.tokenize(solution_str.lower())
        elif lang_code == "zh":
            gt_tokenized = list(jieba.cut(gt))
            sol_tokenized = list(jieba.cut(solution_str))

        return self.penalty_base * min(abs(len(sol_tokenized)-len(gt_tokenized)) / len(gt_tokenized), 10.)


class TextSimilarity(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 parse_result_failure_score=0.
                 ):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.parse_result_failure_score = parse_result_failure_score

    def get_penalty_or_reward(self, solution_str, ground_truth):
        try:
            solution_str = self.postprocess_solution_fn(solution_str)
            if solution_str is None:
                return self.parse_result_failure_score

            gt = ground_truth["ground_truth"]
            lang_code = ground_truth["lang_code"]

            if lang_code == "en":
                gt_tokenized = en_mt.tokenize(gt.lower())
                sol_tokenized = en_mt.tokenize(solution_str.lower())
            elif lang_code == "zh":
                gt_tokenized = list(jieba.cut(gt))
                sol_tokenized = list(jieba.cut(solution_str))

            gt_tokenized = " ".join(gt_tokenized)
            sol_tokenized = " ".join(sol_tokenized)

            bleu = sacrebleu.sentence_bleu(sol_tokenized, [gt_tokenized]).score
            return bleu / 100
        except Exception as err:
            return self.parse_result_failure_score


def get_rm_rewards(
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
        parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD):
    judge_template_zh = """ 直接回答我构造好的问题。

# 评价标准
1. 你的回答（构造的指令/问题）必须是下面这个：
```
{question}
```
2. 回答必须直接返回问题/指令，不能包含其他不相关的信息，比如问题分析或者问题解答等**任何**不是问题内容本身的信息。
3. 问题语言必须是中文
"""
    judge_template_en = """ Answer my well-constructed question directly.

# Evaluation Criteria
1. Your answer (constructed instruction/question) must be the following:
```
{question}
```
2. The answer must directly return the question/instruction, and must not contain any other irrelevant information, such as question analysis or question solution, and **any** information that is not the content of the question itself.
3. The language of the question must be English. 
"""

    new_batch_ground_truth = []
    for _ in batch_ground_truth:
        if _["lang_code"] == "zh":
            judge = judge_template_zh.format(question=_["ground_truth"])
        else:
            judge = judge_template_en.format(question=_["ground_truth"])
        new_batch_ground_truth.append({
            "ground_truth": judge
        })

    return compute_rm_score(
        batch_solution_str=batch_solution_str,
        batch_ground_truth=new_batch_ground_truth,
        postprocess_solution_fn=fabricate_qa_postprocess_solution_fn,
        parse_result_failure_score=parse_result_failure_score
    )


class Stage1QwQLongCoTFabricateQAComputeScore(object):
    def __init__(self, split="train", parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD):
        self.split = split
        self.parse_result_failure_score = parse_result_failure_score
        self.length_diff_penalty = LengthDiffPenalty(
            postprocess_solution_fn=fabricate_qa_postprocess_solution_fn)
        self.text_similarity = TextSimilarity(
            postprocess_solution_fn=fabricate_qa_postprocess_solution_fn)

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "LengthDiffPenalty": self.length_diff_penalty.get_penalty_or_reward,
            "TextSimilarity":  self.text_similarity.get_penalty_or_reward,
        }

    @classmethod
    def extract_gt_question(cls, ground_truth):
        ground_truth = ground_truth["ground_truth"]
        return ground_truth

    def log_ground_truth(self, ground_truth):
        return repr(self.extract_gt_question(ground_truth))

    def clip_string(self, s: str):
        if len(s) > 1500:
            return f'{s[:700]}... [省略] ...{s[-800:]}'
        return s

    def log_solution(self, solution):
        norm = fabricate_qa_postprocess_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return repr(self.clip_string(norm))

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      ):

        penalty = defaultdict(dict)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            for key, fn in self.get_penalties().items():
                penalty[key][i] = fn(solution_str, ground_truth)

        base_rewards = get_rm_rewards(
            batch_data_sources, batch_solution_str, batch_ground_truth)

        final_results = []
        for i in range(len(batch_solution_str)):
            penalty_log_str = []
            _reward = base_rewards[i]
            for name, _penalty in penalty.items():
                if i in _penalty:
                    _reward += _penalty[i]
                    penalty_log_str.append(f'{name}={_penalty[i]:.2f}')

            final_results.append(_reward)

            if self.split == "valid":
                print(
                    f"--------------------------------[VALID]--------------------------------")
                print(
                    f"【Fabricate】 `{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Authentic】`{self.log_ground_truth(batch_ground_truth[i])}`")
                print(
                    f'[Final Reward]={_reward:.3f}|{"|".join(penalty_log_str)}\n')
            elif self.split == "train" and random.random() < 0.1:
                print(
                    f"--------------------------------[TRAIN]--------------------------------")
                print(
                    f"【Fabricate】`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f"【Authentic】`{self.log_ground_truth(batch_ground_truth[i])}`")
                print(
                    f'[Final Reward]={_reward:.3f}|{"|".join(penalty_log_str)}\n')

        return final_results


_stage1_qwq_longcot_fabricate_qa_compute_score_train = Stage1QwQLongCoTFabricateQAComputeScore(
    split="train")
_stage1_qwq_longcot_fabricate_qa_compute_score_valid = Stage1QwQLongCoTFabricateQAComputeScore(
    split="valid")
stage1_qwq_longcot_fabricate_qa_compute_score_train = _stage1_qwq_longcot_fabricate_qa_compute_score_train.compute_score
stage1_qwq_longcot_fabricate_qa_compute_score_valid = _stage1_qwq_longcot_fabricate_qa_compute_score_valid.compute_score

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 沙盒问题合成（一阶段）
# ------------------------------------------------------------------------------------------------------------------------------------------------------
