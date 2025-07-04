import os
import re
import json
import time
import random
import asyncio
import tqdm.asyncio
import asyncio as aio
from functools import partial
from typing import Sequence
from collections import defaultdict


from time import sleep
from typing import Any

from openai import OpenAI, RateLimitError, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

RATE_LIMIT_RETRY_DELAY = 60
RATE_LIMIT_RETRY_ATTEMPTS = 10
WORKFLOW_AGENT_LOGFILE = os.getenv("WORKFLOW_AGENT_LOGFILE", None)


def parse_json(text: str, best_effort: str | None = None) -> dict | Sequence[dict]:
    raw = text[:]

    def json_recovery(s):
        bg = s.index("{")
        ed = len(s) - 1 - s[::-1].index("}")
        s = s[bg:ed+1]
        if s.count("}") < s.count("{"):
            return s + "}"
        return s

    try:
        if "```json" in text:
            contents = re.findall(r'```json(.*?)```', raw, re.DOTALL)
            contents = [json_recovery(_.strip()) for _ in contents]

            results = []
            for content in contents:
                try:
                    result = json.loads(
                        content,
                        strict=False,
                    )
                    results.append(result)
                except Exception as err:
                    continue
            if len(results) == 0:
                raise ValueError(f"```json exists but format corrupt")
            if len(results) == 1:
                return results[0]
            return results
        else:
            text = "{" + \
                text.split("{", 1)[-1].strip().rsplit("}", 1)[0].strip() + "}"
            text = json_recovery(text)
            result = json.loads(
                text,
                strict=False,
            )
            return result
    except Exception as e:
        if best_effort:
            extract = raw[raw.index(best_effort):]
            if "True" in extract:
                return {best_effort: True}
            else:
                return {best_effort: False}
        raise ValueError(f"Failed to parse JSON: “{repr(raw)}”") from e


def contain_chinese(string):
    try:
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if re.search(pattern, string):
            return True
        return False
    except Exception as err:
        return False


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
    "model": "qwen25_7B_instruct",
    "base_url": "http://10.130.247.138:8000/v1",
    "api_keys": "EMPTY",
    "request_kwargs": {
        "temperature": 0.7,
        "timeout": 30,
        "max_tokens": 2048
    },
})


def postprocess_solution(solution_str):
    if "<|im_end|>" in solution_str:
        return solution_str[:solution_str.index("<|im_end|>")]
    return solution_str


def coarse_format_parse(solution_str):
    solution_str = postprocess_solution(solution_str)
    try:
        thought = re.findall(r'<think>(.*)</think>',
                             solution_str, re.DOTALL)[0].strip()
    except Exception as err:
        return None

    try:
        answer_constraint = re.findall(r'<answer_constraint>(.*)</answer_constraint>',
                                       solution_str, re.DOTALL)[0].strip()
    except Exception as err:
        return None

    try:
        answer_extraction = re.findall(r'<answer_extraction>(.*)</answer_extraction>',
                                       solution_str, re.DOTALL)[0].strip()
    except Exception as err:
        return None
    return thought, answer_constraint, answer_extraction


async def compute_code_format_score(
    batch_parsed_results, batch_ground_truth, verify_times=2
):
    base_rewards = [0.0] * len(batch_ground_truth)
    normed_codes = []
    for i, (parsed, gt) in enumerate(zip(batch_parsed_results, batch_ground_truth)):
        if parsed is None:
            normed_codes.append(None)
        else:
            _, _, code = parsed
            code = code.strip()
            if code.startswith("```python\ndef extract_answer(response: str) -> str:") and code.endswith("```"):
                base_rewards[i] += 0.1
                normed_codes.append(code[len("```python"):-len("```")].strip())
            else:
                normed_codes.append(None)
                continue

    valid_indices = []
    prompt_mapper = defaultdict(list)
    for i, code in enumerate(normed_codes):
        if code is not None:
            valid_indices.append(i)
            prompt_mapper[batch_ground_truth[i]["prompt"]].append(i)
    max_concurrent_requests = 32

    prompts = list(prompt_mapper.keys()) * verify_times
    results = await agent.run(prompts, max_concurrent_requests, desc="[Code Runnable]", postprocess_fns=[lambda x: x]*len(prompts))

    for prompt, response in results:
        if response is not None:
            indices = prompt_mapper[prompt]
            for index in indices:
                code = normed_codes[index]
                try:
                    exec_code = f'{code}\n\nresponse = {repr(response)}\n_ANSWER = extract_answer(response)'
                    exec(exec_code)
                except Exception as err:
                    continue
                base_rewards[index] += 0.1

    return base_rewards


REVISE_RESPONSE_TEMPLATE = """
Rewrite the above answer to conform to the following instruction requirements. The rewrite should only meet the format requirements and do not change the main content of the original answer.

# QUESTION
{question}

# RESPONSE
{response}

# QUESTION
{question}
[Format Requirement] {constraint}

# RESPONSE
"""


async def compute_extract_answer_score(
    batch_parsed_results, batch_ground_truth, split
):
    base_rewards = [0.0] * len(batch_ground_truth)
    normed_codes = []
    thoughts, constraints = [], []
    for i, (parsed, gt) in enumerate(zip(batch_parsed_results, batch_ground_truth)):
        if parsed is None:
            normed_codes.append(None)
            constraints.append(None)
            thoughts.append(None)
        else:
            thought, constraint, code = parsed
            code = code.strip()
            thoughts.append(thought)
            constraints.append(constraint)
            if code.startswith("```python\ndef extract_answer(response: str) -> str:") and code.endswith("```"):
                normed_codes.append(code[len("```python"):-len("```")].strip())
            else:
                normed_codes.append(None)
                continue

    valid_indices = []
    prompt_mapper = defaultdict(list)

    for i, code in enumerate(normed_codes):
        if code is not None:
            valid_indices.append(i)
            _prompt = REVISE_RESPONSE_TEMPLATE.format(
                question=batch_ground_truth[i]["prompt"],
                response=batch_ground_truth[i]["llm_output"],
                constraint=constraints[i]
            ).strip()
            prompt_mapper[_prompt].append(i)

    max_concurrent_requests = 16
    prompts = list(prompt_mapper.keys())
    results = await agent.run(prompts, max_concurrent_requests, desc="[Refine Original Response]", postprocess_fns=[lambda x: x]*len(prompts))
    for prompt, response in results:
        if response is not None:
            indices = prompt_mapper[prompt]
            for index in indices:
                code = normed_codes[index]

                gt = batch_ground_truth[index]["gold_label"].strip(
                )

                success = False
                try:
                    # exec_code = f'{code}\n\nresponse = {repr(response)}\nprint(extract_answer(response))'
                    exec_code = f'{code}\n\nresponse = {repr(response)}\nassert extract_answer(response).strip() == {repr(gt)}'
                    # exec_code = f'{code}\n\nresponse = {repr(response)}\nprint(extract_answer(response).strip() == {repr(gt)})'
                    exec(exec_code)
                    success = True
                except Exception as err:
                    pass
                if success:
                    base_rewards[index] += 0.5
                if split == "valid" and random.random() < 1.0:
                    print(
                        f"--------------------------------[VALID]--------------------------------")
                    print(
                        f'【Thought】`{repr(thoughts[index])}`')
                    print(
                        f'【Question】({repr(constraints[index])})`{repr(batch_ground_truth[index]["prompt"])}`')
                    print(
                        f'【Response】`{repr(batch_ground_truth[index]["llm_output"])}`')
                    print(
                        f'【Modified Response】`{repr(response)}`')
                    print(f"【Solution Code】 `{repr(normed_codes[index])}`")
                    print(f'Reward={1.0 if success else 0.}')
    return base_rewards


CONSTRAINT_VERIFY_TEMPLATE = """
你需要判断下面的这段话是否一个“指令格式约束条件”，且不包含关于答案的信息（比如泄漏或者暗示正确答案是什么）

说明：指令格式约束条件指的是对于一个问题的补充，约束回答时需要满足的格式要求等

# INSTRUCT CONSTRAINT
{constraint}

参考下面样子返回标准json格式的回答
# 注意
# 1. 约束不应该透露或者暗示答案的具体内容，只应当针对答案格式进行限制
# 2. 这里提供的是指令限制或者约束，而没有提供问题本身，你不需要关心原问题是什么
# 3. 这里的判断需要检查约束是否包含题目信息，如果包含则不被允许，需要判断valid=False

```json
{{
  "reason": ***,
  "valid": True/False
}}
``
"""


async def compute_answer_constraint_format_score(
    batch_parsed_results, batch_ground_truth
):
    base_rewards = [0.0] * len(batch_ground_truth)

    prompt_mapper = defaultdict(list)
    for i, (parsed, gt) in enumerate(zip(batch_parsed_results, batch_ground_truth)):
        if parsed is None:
            pass
        else:
            _, constraint, _ = parsed
            if contain_chinese(gt["prompt"]):  # zh
                if contain_chinese(constraint):
                    base_rewards[i] += 0.1
            else:  # en
                if contain_chinese(constraint):
                    pass
                else:
                    base_rewards[i] += 0.1
            constraint_verify_prompt = CONSTRAINT_VERIFY_TEMPLATE.format(
                constraint=constraint).strip()
            prompt_mapper[constraint_verify_prompt].append(i)

    max_concurrent_requests = 16
    prompts = list(prompt_mapper.keys())

    results = await agent.run(prompts, max_concurrent_requests, desc="[Verify Constraint]", postprocess_fns=[partial(parse_json, best_effort="valid")]*len(prompts))
    for prompt, response in results:
        if response is not None:
            indices = prompt_mapper[prompt]
            for index in indices:
                try:
                    if isinstance(response["valid"], bool):
                        if response["valid"]:
                            base_rewards[index] += 1.0
                        else:
                            base_rewards[index] -= 1.0
                    elif isinstance(response["valid"], str):
                        if response["valid"] == "True":
                            base_rewards[index] += 1.0
                        else:
                            base_rewards[index] -= 1.0
                except Exception as err:
                    continue
    return base_rewards


async def _compute_score(batch_data_sources,
                         batch_solution_str,
                         batch_ground_truth,
                         split
                         ):
    base_rewards = [0.0] * len(batch_solution_str)
    parsed_results = []

    for i, (gt, solution_str) in enumerate(zip(batch_ground_truth, batch_solution_str)):
        parsed = coarse_format_parse(solution_str)
        if parsed is not None:
            base_rewards[i] += 0.1
        parsed_results.append(parsed)

    constraint_format_rewards = await compute_answer_constraint_format_score(
        parsed_results, batch_ground_truth)

    code_format_rewards = await compute_code_format_score(parsed_results, batch_ground_truth)
    extract_answer_rewards = await compute_extract_answer_score(parsed_results, batch_ground_truth, split)

    final_rewards = []
    assert len(base_rewards) == len(code_format_rewards)
    assert len(extract_answer_rewards) == len(code_format_rewards)
    for w, x, y, z in zip(base_rewards, constraint_format_rewards, extract_answer_rewards, code_format_rewards):
        if split == "train":
            if x > 0:
                final_rewards.append(w+x+y+z)
            else:
                final_rewards.append(w+x)
        else:
            final_rewards.append(y * 2)
    assert len(final_rewards) == len(batch_solution_str)

    return final_rewards


def compute_score(batch_data_sources,
                  batch_solution_str,
                  batch_ground_truth,
                  split="train"
                  ):
    async def main():
        return await _compute_score(batch_data_sources, batch_solution_str, batch_ground_truth, split)
    return aio.run(main())


compute_score_train = partial(compute_score, split="train")
compute_score_valid = partial(compute_score, split="valid")
