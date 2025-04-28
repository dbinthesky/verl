import re
import jieba
import sacrebleu
from typing import Any
from functools import partial
from abc import abstractmethod
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
    "http://10.130.0.154:5020"
]
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


class PenaltyOrReward(object):
    @abstractmethod
    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        raise NotImplementedError


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


#     judge_template = """ 直接回答我构造好的问题。


# # 评价标准
# 1. 你的回答（构造的指令/问题）必须是下面这个：
# ```
# {question}
# ```
# 2. 回答必须直接返回问题/指令，不能包含其他不相关的信息，比如问题分析或者问题解答等等。
# """

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 沙盒问题合成（一阶段）
# ------------------------------------------------------------------------------------------------------------------------------------------------------
