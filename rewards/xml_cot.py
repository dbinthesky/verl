import re
import sys
import json
import jieba
import random
import aiohttp
import requests
import sacrebleu
import numpy as np
import tqdm.asyncio
import asyncio as aio
from functools import partial
from asyncio import Semaphore
from abc import abstractmethod
import xml.etree.ElementTree as ET
from typing import Any, Dict, Callable, List
from tqdm import tqdm as tqdm_nonasync
from collections import namedtuple, defaultdict
from sacremoses import MosesTokenizer, MosesDetokenizer


from openai import OpenAI, RateLimitError, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------


VERIFIER_MODEL_NAME = "qwen25_7B_fabricate_qa_criteria_judge_ehance_0518"
VERIFIER_MODEL_PATH = "http://10.130.133.200:8000/v1"
DEFAULT_PARSE_FAILURE_REWARD = -2.
MAX_CONCURRENT = 128


def tokenize(s, lang_code):
    if lang_code == "en":
        tokenized_text = en_mt.tokenize(s.lower())
    elif lang_code == "zh":
        tokenized_text = list(jieba.cut(s))
    return tokenized_text


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
        semaphore = aio.Semaphore(max_concurrent)
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
            # FIXME: Agent Logger
            with open("/cpfs01/shared/llm_ddd/tongjian/rl/hard_case_mixed/gpqa/agent2.log", "a+") as f:
                f.write(
                    f'{json.dumps({"message": messages, "result": response.choices[0].message.content}, ensure_ascii=False)}\n')
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
        "timeout": 360,
        "max_tokens": 16384,
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
        return solution_str[:solution_str.index("<|im_end|>")].strip()
    if "<｜end▁of▁sentence｜>" in solution_str:
        return solution_str[:solution_str.index("<｜end▁of▁sentence｜>")].strip()
    return solution_str


class PenaltyOrReward(object):
    @abstractmethod
    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# XML CoT
# ------------------------------------------------------------------------------------------------------------------------------------------------------

def get_thought(solution_str: str):
    thought = re.findall(r'```xml.*```', solution_str, re.DOTALL)[0]
    return thought


def get_conclusion(solution_str: str):
    thought = get_thought(solution_str)
    return solution_str[solution_str.index(thought)+len(thought):].strip()


# def recursive_traverse(element, depth=0):
#     # 打印当前元素（缩进显示层级）
#     indent = "  " * depth
#     print(f"{indent}标签: {element.tag}, 属性: {element.attrib}, 文本: {element.text.strip() if element.text else None}")

#     # 递归遍历子元素
#     for child in element:
#         recursive_traverse(child, depth + 1)

def xml_cot_parse_solution_fn(solution_str):
    try:
        thought = get_thought(solution_str)
    except Exception as err:
        return None

    try:
        conclusion = get_conclusion(solution_str).strip()
    except Exception as err:
        return None

    if any(_ in conclusion for _ in ("```xml", "<think>", "</think>", "<conclusion>", "</conclusion>")):
        return None

    try:
        thought_content = re.findall(r'```xml(.*)```', thought, re.DOTALL)[0]
    except Exception as err:
        return None

    thought_content = f'<doc> {thought_content} </doc>'
    try:
        root = ET.fromstring(thought_content)
    except Exception as err:
        return None

    if not all(tag in [child.tag for child in root]
               for tag in ("think", "conclusion")):
        return None

    return root
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# XML CoT
# ------------------------------------------------------------------------------------------------------------------------------------------------------
