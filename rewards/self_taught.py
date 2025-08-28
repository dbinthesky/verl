import os
import re
import sys
import json
import uuid
import copy
import time
import math
import httpx
import jieba
import random
import aiohttp
import itertools
import requests
import sacrebleu
import numpy as np
import tqdm.asyncio
import asyncio as aio
from functools import partial
from asyncio import Semaphore
from abc import abstractmethod, abstractclassmethod, ABCMeta
import xml.etree.ElementTree as ET
from typing import Any, Dict, Callable, List
from decimal import Decimal, ROUND_HALF_UP
from tqdm import tqdm as tqdm_nonasync
from collections import namedtuple, defaultdict, OrderedDict
from sacremoses import MosesTokenizer, MosesDetokenizer


from openai import OpenAI, RateLimitError, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential


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

    async def run(self, messages, max_concurrent, desc, postprocess_fns, pbar=False):
        semaphore = aio.Semaphore(max_concurrent)
        async with AsyncOpenAI(api_key=self.api_keys, base_url=self.base_url) as client:
            results = []
            tasks = [self.process_prompt(client, message, semaphore, postprocess_fn)
                     for message, postprocess_fn in zip(messages, postprocess_fns)]

            if desc is not None and pbar:
                for f in tqdm.asyncio.tqdm.as_completed(tasks, dynamic_ncols=True, desc=desc):
                    results.append(await f)
            else:
                try:
                    print(
                        f'{desc} (p={max_concurrent}) TOTAL {len(messages)} RUN...')
                    results = await aio.gather(*tasks)
                    print(
                        f'{desc} (p={max_concurrent}) TOTAL {len(messages)} FINISHED...')
                except Exception as err:
                    print(f'[ERROR] asyncio.gather failed: {err}')
                    return None
            return results

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=5, max=20))
    async def chat_completion(self, client, messages, postprocess_fn) -> str | None:
        response = None
        try:
            response = await client.chat.completions.create(
                model=self.model, messages=[
                    {"role": "system", "content": 'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.'},
                    {"role": "user", "content": messages}
                ], **self.request_kwargs,
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


def batchify(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


class RewardModelAgent(object):
    def __init__(self, urls, postprocess_solution_fn, parse_result_failure_score):
        self.RM_URLS = urls
        self.postprocess_solution_fn = postprocess_solution_fn
        self.parse_result_failure_score = parse_result_failure_score

    def compute_rm_score(
            self,
            batch_solution_str,
            batch_ground_truth,
            judge_prompt_key="ground_truth"
    ):
        input_datas = []
        rewards = {}

        for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            solution_str = self.postprocess_solution_fn(solution_str)
            if solution_str is None:
                rewards[i] = self.parse_result_failure_score
                continue
            if ground_truth is None:
                rewards[i] = self.parse_result_failure_score
                continue

            input_data = {
                "prompt": ground_truth[judge_prompt_key], "response": solution_str, "id": i
            }
            input_datas.append(input_data)

        if len(input_datas) > 0:
            url = random.choice(self.RM_URLS)
            for batch in tqdm_nonasync(batchify(input_datas, n=128), desc=f'[RM][{url}] batchify inference (batch=128)'):
                output_datas = self.post_with_retry(batch, url)
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

    def post_with_retry(self, data, url, max_retries=3, retry_delay=1, suffix="/reward"):
        retries = 0
        while retries < max_retries:
            try:
                response = requests.post(
                    f'{url}{suffix}', json=data, timeout=600)
                response.raise_for_status()  # 如果状态码不是 200，抛出异常
                return response.json()
            except requests.RequestException as e:
                print(
                    f"请求(数据总量={len(data)})失败，错误信息: {e}，重试第 {retries + 1} 次...")
                retries += 1
                if retries < max_retries:
                    time.sleep(retry_delay)
        print("达到最大重试次数，请求失败。")
        return None


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# PROMPTS
# ------------------------------------------------------------------------------------------------------------------------------------------------------
ANSWER_VERIFY_FEWSHOTS = """### **基于标准答案判断回答是否正确**
任务描述：请根据提供的**题目**、**用户回答**和**标准答案**，判断用户回答是否正确，并按照指定格式输出结果。需严格对比回答和标准答案，若用户回答与标准答案**不一致**，则判定为错误。


#### 通用流程与方法
##### **步骤1：理解题意和标准答案含义
1. **明确题目类型与考察目标**
2. **明确标准答案含义**

##### **步骤2：用户回答提取最终结论**：
1. **去除干扰信息**：
   - 删除与答案无关的内容。
2. **标记容错范围**：
   - 数值题允许合理误差（如标准答案“5.0±0.1”，用户回答“4.9”视为正确）。

##### **步骤3：逐项对比（用户回答 vs 标准答案）**
1， 直接对比用户回答提取出的结论并与标准答案进行对比，判断是否正确。

#### 输出要求
- 先按照方法进行详细、准确、仔细地分析
- 再最后的部分按下面的JSON格式输出最后的结论
```json
{
  "判断结果": "正确/错误",
}
```



下面是一个具体的例子：


#### **输入：**
```json
{
  "题目": "In a square region of side length 1, filled with fluid in steady, irrotational flow, the velocity potential function φ satisfies Laplace’s equation ∇²φ = 0. The boundary conditions are φ = 0 on all four sides. What is the value of φ(0.5, 0.5) obtained by the Gauss-Seidel iterative method after 100 iterations, starting with φ = 0 at all points? Express your answer to two decimal places.",
  "标准答案": "\\boxed{0.00}",
  "用户回答": "To solve for the value of \\(\\phi(0.5, 0.5)\\) in a square region with side length 1, using the Gauss-Seidel iterative method under the given boundary conditions, we follow these steps:\n\n### 1. **Discretization of the Domain**\n   - Divide the square into a grid with spacing \\(h\\). For simplicity, let's choose \\(h = 0.1\\), resulting in a \\(10 \\times 10\\) grid (including boundary points).\n   - The interior points where \\(\\phi\\) needs to be computed are from \\(i = 1\\) to \\(9\\) and \\(j = 1\\) to \\(9\\) (since \\(\\phi = 0\\) on the boundaries).\n\n### 2. **Laplace’s Equation in Finite Differences**\n   Laplace’s equation \\(\\nabla^2 \\phi = 0\\) in 2D is approximated as:\n   \\[\n   \\phi_{i,j} = \\frac{1}{4} \\left( \\phi_{i+1,j} + \\phi_{i-1,j} + \\phi_{i,j+1} + \\phi_{i,j-1} \\right)\n   \\]\n   This is the Gauss-Seidel update formula for each interior point.\n\n### 3. **Gauss-Seidel Iteration**\n   - Initialize \\(\\phi_{i,j} = 0\\) for all interior points.\n   - Perform 100 iterations, updating each \\(\\phi_{i,j}\\) using the latest available values of its neighbors (Gauss-Seidel method).\n\n### 4. **Focus on \\(\\phi(0.5, 0.5)\\)**\n   - The point \\((0.5, 0.5)\\) corresponds to the grid point \\((5, 5)\\) when \\(h = 0.1\\).\n   - After 100 iterations, the value at this point converges to approximately **0.25**.\n\n### **Final Answer**\n\\[\n\\boxed{0.25}\n\\]\n\n**Note:** The exact value can also be derived analytically for this symmetric problem. The solution to Laplace’s equation with \\(\\phi = 0\\) on all boundaries in a unit square is a series expansion, and at the center \\((0.5, 0.5)\\), the dominant term gives \\(\\phi \\approx 0.25\\). The Gauss-Seidel method approximates this well after sufficient iterations.",
}
```

#### **输出：**
### 分析过程：
1. **理解题意和标准答案**：题目要求计算在单位正方形区域内，满足拉普拉斯方程且四边φ=0的条件下，经Gauss-Seidel迭代法100次后φ(0.5, 0.5)的值（保留两位小数），标准答案为\boxed{0.00}。
2. **提取用户回答的最终结论**：用户回答通过分析推导，最终结论为\boxed{0.25}，去除过程性描述后，核心结果为0.25。
3. **对比结果**：用户回答的最终结论0.25与标准答案0.00不一致，且不存在数值题的合理误差范围，因此判定为错误。

```json
{
  "判断结果": "错误"
}
```
"""

ANSWER_VERIFY_TEMPLATE = """
#### **输入：**
```json
{content}
```

#### **输出：**
"""

JUDGE_TWO_QUESTION_SIM_TEMPLATE = """  # 问题相似程度评价标准（1-5分）

| **相似等级 ** | **判定标准 ** |
|-------------- | ------------------------------------------------------------------------------|
| **1分 ** | 完全不同：出题目的、核心条件、求解目标、解题思路毫无关联，无任何共同要素。 |
| **2分 ** | 弱相关：仅单一维度相关（如同属数学题中的几何/代数大类），其余要素无重合。 |
| **3分 ** | 部分相似：题目类型相同（如均为“三维立方体隐藏块计数”），但核心条件、目标或步骤存在关键差异（如可见面数不同、求解方向相反）。 |
| **4分 ** | 高度相似：题目类型、求解目标、解题框架完全一致，仅数据或参数不同（如隐藏块数值、矩阵元素等），核心步骤和逻辑完全复用。 |
| **5分 ** | 完全相同：题目内容（数据、表述、目标、步骤）几乎完全一致，除个别措辞表述有些许差距外，其他部分完全相同（包括重要参数、数据、选项等）。 |


# 使用说明
1. ** 类型优先 **：若题目类型不同（如几何vs代数），直接≤2分；类型相同是≥3分的前提。
2. ** 核心要素判断**：
   - **1分 **：无任何交集（如几何题vs代数题）。
   - **2分 **：仅同属大类别（如同为数学题，但具体子类型不同，如几何vs数列）。
   - **3分 **：子类型相同但存在关键差异（如可见面数不同、递推公式不同）。
   - **4分 **：子类型、目标、解题逻辑一致，仅数据不同（如求同一类型问题的不同参数解）。
   - **5分 **：题目完全复现，无任何改动。

# 输出格式
<think >
... ...
</think >
[CONCLUSION START]
SIMILARITY = *
[CONCLUSION END]
```
（*处填写1-5的整数，需严格符合上述判定标准）


下面是具体的例子

[原问题]
Find the minimum value of\n\\[x ^ 2 + 2xy + 3y ^ 2 - 6x - 2y, \\]over all real numbers $x$ and $y.$

[对比问题]
Find the minimum value of the quadratic function \\(f(x)=x ^ 2 - 6x + 7 \\).

[输出]
<think >
用户让我比较两个数学问题的相似程度。首先，我需要仔细看一下两个问题的内容。原问题是求一个二元二次多项式的最小值，变量是 x 和 y，表达式是 x² + 2xy + 3y² - 6x - 2y。待评价的问题是求一元二次函数 f(x) = x² - 6x + 7 的最小值。
首先根据评价标准，类型优先。原问题是二元二次函数求最值，属于多元微积分或代数中的二次型问题，可能需要用偏导数或者配方法。待评价问题是一元二次函数，直接可以用配方法或者求导，变量只有一个 x。题目类型不同，一个是二元，一个是一元，所以类型不同。根据使用说明，类型不同的话直接≤2 分。接下来看核心要素，原问题是二元，涉及两个变量和交叉项 xy，而待评价问题是一元，没有交叉项，解题思路也不一样，一元的更简单，只需要处理一个变量。所以它们的共同点可能只是同属代数中的二次函数求最值，但具体子类型不同，一个是多元，一个是一元。根据标准，同属大类别（代数），但具体子类型不同，所以应该是 2 分。因为 2 分的判定是仅单一维度相关，同属数学题中的代数大类，其余要素无重合。这里类型不同，所以符合 2 分的条件。
</think >
[CONCLUSION START]
SIMILARITY = 2
[CONCLUSION END]


[原问题]
Find the inverse of the matrix\n\\[\\begin{pmatrix} 5 & -4 \\\\ 0 & 1 \\end{pmatrix}.\\]If the inverse does not exist, then enter the zero matrix.

[对比问题]
What is the inverse of the matrix \\(\\begin{pmatrix} 2 & 1 \\\\ 3 & 4 \\end{pmatrix}\\)?

[输出]
<think >
用户让我比较两个矩阵求逆的问题的相似程度。首先，根据类型优先原则，两个问题都是求矩阵的逆，属于同一题目类型，所以类型相同，至少 3 分以上。接下来看核心要素：原问题是求一个 2x2 矩阵的逆，对比问题也是 2x2 矩阵，求解目标都是求逆矩阵，解题框架都是使用矩阵求逆的方法，比如伴随矩阵法或者行变换。两者的不同在于矩阵中的数据不同，原问题的矩阵是[[5, -4], [0, 1]]，对比问题是[[2, 1], [3, 4]]。根据判定标准，4 分的情况是题目类型、求解目标、解题框架一致，仅数据或参数不同，核心步骤和逻辑完全复用。这里显然符合 4 分的条件，因为只是矩阵元素不同，解题方法完全一样，没有关键差异。所以相似等级应该是 4 分。
</think >
[CONCLUSION START]
SIMILARITY = 4
[CONCLUSION END]


# 注意：比较题目相似度不考虑题型差异。例如，一道选择题，一道计算题，如果数据、表述、目标、步骤都十分接近，则应该判定为**高度相似**，而非因为题型不同，认为二者有较大差异。


"""


class BatchCallOpenAPI(metaclass=ABCMeta):

    @abstractmethod
    def prompt_fn(self, example):
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, response: str):
        raise NotImplementedError

    @abstractclassmethod
    def task_desc(self):
        raise NotImplementedError

    async def do_job(self, agent, batch_inputs, max_concurrent_requests):
        prompts = defaultdict(list)
        for index, example in enumerate(batch_inputs):
            prompt = self.prompt_fn(example)
            prompts[prompt].append(index)

        results = await agent.run(list(prompts.keys()), max_concurrent_requests, desc=f"[{self.task_desc()} {agent.model}={max_concurrent_requests}]", postprocess_fns=[self.postprocess]*len(list(prompts.keys())))

        results_mapper = {}
        for (k, v) in results:
            for _ in prompts[k]:
                results_mapper[_] = v

        outputs = []
        for i, _ in enumerate(batch_inputs):
            if i in results_mapper and results_mapper[i] is not None:
                outputs.append(results_mapper[i])
            else:
                outputs.append(None)
        return outputs


class AnswerConsistency(BatchCallOpenAPI):
    def __init__(self, prompt_field="prompt", answer_field="answer"):
        self.prompt_field = prompt_field
        self.answer_field = answer_field

    @classmethod
    def task_desc(cls):
        return "答案一致"

    def prompt_fn(self, example):
        prompt, response, gt = example
        content = {
            "题目": prompt,
            "用户回答": response,
            "标准答案": gt,
        }
        prompt = ANSWER_VERIFY_FEWSHOTS + "\n\n\n" + ANSWER_VERIFY_TEMPLATE.format(
            content=json.dumps(content, ensure_ascii=False, indent="  ")
        )
        return prompt

    def postprocess(self, response: str):
        s = response
        try:
            conclusion = s.strip()
            judge = re.findall(
                r'\"判断结果\": \"(.*)\"', conclusion)
            if len(judge) > 0 and judge[0] in ("正确", "错误"):
                return judge[0] == "正确"

            conclusion = conclusion[conclusion.index(
                "```json")+len("```json"):].strip()
            conclusion = conclusion[:conclusion.index("```")].strip()
            try:
                conclusion = json.loads(conclusion)
                if conclusion["判断结果"] not in ("正确", "错误"):
                    raise PostprocessError(f'corrupt')
                return conclusion["判断结果"] == "正确"
            except Exception as err:
                try:
                    conclusion = re.findall(
                        r'\"判断结果\": \"(.*)\"', conclusion)[0]
                    if not conclusion in ("正确", "错误"):
                        raise PostprocessError(f'corrupt')
                    return conclusion == "正确"
                except Exception as err:
                    raise PostprocessError(f'{err}')
        except Exception as err:
            raise PostprocessError(f'{err}')


class JudgeTwoQuestionSimilarity(BatchCallOpenAPI):
    _TEMPLATE = JUDGE_TWO_QUESTION_SIM_TEMPLATE

    def __init__(self):
        pass

    @classmethod
    def task_desc(cls):
        return "问题相似度"

    def prompt_fn(self, example):
        prompt = self._TEMPLATE + \
            f'\n\n现在需要你比较下面两个问题的相似度。\n\n[原问题]\n{example[0]}\n\n[对比问题]\n{example[1]}\n\n[输出]\n'
        return prompt

    def postprocess(self, response: str):
        s = response
        try:
            thought = re.findall(r'think>.*</think>', s, re.DOTALL)[0]
            conclusion = s.replace(thought, "")
            conclusion = conclusion[conclusion.index(
                "[CONCLUSION START]")+len("[CONCLUSION START]"):conclusion.index("[CONCLUSION END]")].strip()

            score = int(re.findall(r'SIMILARITY=(\d+)', conclusion)[0].strip())
            if score not in (1, 2, 3, 4, 5):
                raise PostprocessError(f'invalid similarity score={score}')
            return score
        except Exception as err:
            raise PostprocessError(f'{err}')


def postprocess_solution(solution_str):
    if "<|im_end|>" in solution_str:
        return solution_str[:solution_str.index("<|im_end|>")].strip()
    if "<｜end▁of▁sentence｜>" in solution_str:
        return solution_str[:solution_str.index("<｜end▁of▁sentence｜>")].strip()
    if "<|endoftext|>" in solution_str:
        return solution_str[:solution_str.index("<|endoftext|>")].strip()
    return solution_str


def parse_question_solution_fn(solution_str: str):
    if solution_str.count("</question>") > 1:
        return None
    if solution_str.count("</think>") > 1:
        return None
    solution_str = postprocess_solution(solution_str)
    if not solution_str.startswith("<think>"):
        solution_str = f'<think>\n{solution_str}'
    try:
        thought = re.findall(r'<think>.*</think>',
                             solution_str, re.DOTALL)[0]
    except Exception as err:
        return None
    solution_str = solution_str.replace(thought, "")

    try:
        conclusion = re.findall(r'<question>(.*)</question>',
                                solution_str, re.DOTALL)[0]
    except Exception as err:
        return None
    if ("<question>" in conclusion) or ("</question>" in conclusion):
        return None
    return thought, conclusion


def contain_chinese(string):
    try:
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if re.search(pattern, string):
            return True
        return False
    except Exception as err:
        return False


def kg2query_self_taught_parse_solution_fn(solution_str: str, extract_question_fn=parse_question_solution_fn):
    parsed = extract_question_fn(solution_str)

    if parsed is None:
        return None

    thought, conclusion = parsed

    try:
        question = conclusion[conclusion.index(
            "Question: ")+len("Question: "):conclusion.index("Answer:")].strip()

        answer = conclusion[conclusion.index(
            "Answer:")+len("Answer:"):].strip()

        return question, answer
    except Exception as err:
        return None


class PenaltyOrReward(object):
    def __init__(self, parse_solution_fn, min_score, max_score, abbrev=None):
        self.parse_solution_fn = parse_solution_fn
        self.min_score = min_score
        self.max_score = max_score
        self.abbrev = abbrev

    @abstractmethod
    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        raise NotImplementedError


class LanguageConsistency(PenaltyOrReward):
    def __init__(self, parse_solution_fn, min_score, max_score, abbrev="Lang", strict=False):
        super().__init__(
            parse_solution_fn=parse_solution_fn, min_score=min_score, max_score=max_score, abbrev=abbrev
        )
        self.strict = strict

    def detect_zh(self, text, threshold=0.05):
        if text is None:
            return False
        # Remove URLs, numbers, and punctuation to focus on language characters
        cleaned_text = re.sub(r'[^\w\s]|[\d]', '', text)
        if not cleaned_text:
            return False

        # Count Chinese characters
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', cleaned_text)
        chinese_count = len(chinese_chars)

        # Count English characters
        english_chars = re.findall(r'[a-zA-Z]', cleaned_text)
        english_count = len(english_chars)

        # Total language characters
        total_chars = chinese_count + english_count
        if total_chars == 0:
            return False

        # Calculate ratios
        chinese_ratio = chinese_count / total_chars
        english_ratio = english_count / total_chars

        # Check if both languages exceed the threshold
        return chinese_ratio >= threshold

    def get_penalty_or_reward(self, solution_str, ground_truth):
        raw_solution_str = solution_str
        solution_str = self.parse_solution_fn(solution_str)

        if solution_str is None:
            return 0.0

        if len(solution_str) == 0:
            return 0.0

        question = solution_str[0]

        if "lang_code" not in ground_truth:
            lang_code = "en"
        else:
            lang_code = ground_truth["lang_code"]

        base_score = self.min_score

        if lang_code == "en" and contain_chinese(question):
            return base_score
        elif lang_code == "zh" and (not contain_chinese(question)):
            return base_score

        if self.strict:
            if lang_code == "en":
                if contain_chinese(raw_solution_str):
                    return self.max_score
            elif lang_code == "zh":
                if not self.detect_zh(raw_solution_str, 0.75):
                    return self.max_score
            else:
                pass
        # 成功
        return 0.0


class KG2QueryV1SelfTaughtComputeScore(object):
    def __init__(self,
                 parse_solution_fn,
                 split="train",
                 args=None,
                 min_reward=-2.0,
                 parse_thought_and_conclusion_fn=parse_question_solution_fn,
                 thought_log_prob=0.05
                 ):

        self.split = split
        self.parse_solution_fn = parse_solution_fn
        assert args is not None
        self.args = args
        self.task_name = "KG2QUERY_V1_SELF_TAUGHT"
        self.min_reward = min_reward
        self.thought_log_prob = thought_log_prob

        self.parse_thought_and_conclusion_fn = parse_thought_and_conclusion_fn

        # 初始化API Client
        self.init_agent()

        # 初始化规则奖励/惩罚
        self.init_rule_based_penalties()

    @classmethod
    def rule_based_penalties(cls):
        return [
            LanguageConsistency,
        ]

    def init_rule_based_penalties(self):
        interval = (0 - self.min_reward) / 2 / \
            (len(self.rule_based_penalties())+1)
        penalty_scopes = [(self.min_reward + (i * 2 + 1) *
                           interval, self.min_reward + (i * 2 + 2) *
                           interval) for i in range(len(self.rule_based_penalties()))]
        self._penalties = []
        for p, s in zip(self.rule_based_penalties(), penalty_scopes):
            self._penalties.append(p(parse_solution_fn=self.parse_solution_fn,
                                     min_score=s[0], max_score=s[1]))

    def init_agent(self):
        self.agents = {}
        self.init_verify_agent()
        self.init_rm_agent()

    def init_verify_agent(self):
        self.verify_agent = Agent(
            **self.args["verify_agent"]["model"])

    def init_rm_agent(self):
        self.rm_agent = RewardModelAgent(
            urls=self.args["reward_model_args"]["urls"],
            postprocess_solution_fn=lambda x: self.parse_thought_and_conclusion_fn(x)[
                0],
            parse_result_failure_score=self.min_reward
        )

    async def question_similarity_reward(
        self,
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
    ):
        task = JudgeTwoQuestionSimilarity()
        indices = []
        questions = []

        for i, (gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            result = self.parse_solution_fn(sol)
            if result is not None and gt.get("question", None):
                questions.append((gt["question"], result[0]))
                indices.append(i)
            else:
                continue

        sim_penalties = await task.do_job(
            agent=self.verify_agent,
            batch_inputs=questions,
            max_concurrent_requests=self.args["verify_agent"]["max_concurrent_requests"],
        )

        run_args = self.args["similarity_run_args"]

        scores = [0.0] * len(batch_solution_str)
        for sim, index in zip(sim_penalties, indices):
            if sim is None:
                pass
            else:
                _score = 0.0
                for threshold, set_val in run_args["threshold"].items():
                    if sim >= threshold:
                        _score = max(_score, set_val)
                scores[index] = _score * run_args["weight"]
        return scores

    async def same_answer_reward(
        self,
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
    ):
        task = AnswerConsistency()
        indices = []
        questions = []

        result_index2queue_index = {}
        for i, (gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            result = self.parse_solution_fn(sol)
            if result is not None and gt.get("question", None):
                questions.append((result[0], gt["answer"], result[1]))
                indices.append(i)
                result_index2queue_index[len(indices)-1] = i
            else:
                continue

        evaluations = await task.do_job(
            agent=self.verify_agent,
            batch_inputs=questions,
            max_concurrent_requests=self.args["verify_agent"]["max_concurrent_requests"],
        )

        full_rewards = [0.0] * len(batch_solution_str)
        for j, eval_result in enumerate(evaluations):
            queue_index = result_index2queue_index[j]
            if eval_result:
                full_rewards[queue_index] = 1.0
        return full_rewards

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
        penalty = defaultdict(list)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            parsed = self.parse_solution_fn(solution_str)
            if parsed is None:
                penalty[i].append(-2.0)
            else:
                penalty[i].append(0.0)

            for p in self._penalties:
                penalty[i].append(p.get_penalty_or_reward(
                    solution_str, ground_truth))

        reward1 = await self.same_answer_reward(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )
        reward2 = await self.question_similarity_reward(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )
        reward3 = self.rm_agent.compute_rm_score(
            batch_solution_str, batch_ground_truth, judge_prompt_key="rm_judge_prompt")

        final_results = []
        for i in range(len(batch_solution_str)):
            scores = copy.deepcopy(penalty[i])
            penalties = ["Parse"]+[_.abbrev for _ in self._penalties]
            penalty_log_str = "/".join([f'{p}={s:.3f}' for p,
                                        s in zip(penalties, scores)])

            scores.append(reward1[i])
            scores.append(reward2[i])
            scores.append(reward3[i])

            cur_score = np.sum(scores)
            final_results.append(cur_score)

            log_flag = f"[{self.task_name} VALID]" if self.split == "valid" else f"[{self.task_name} TRAIN]"
            log = False
            if cur_score <= self.min_reward:
                log = True
                log_flag = f"[{self.task_name} VALID CORRUPT RESPONSE]" if self.split == "valid" else f"[{self.task_name} TRAIN CORRUPT RESPONSE]"

            if random.random() < self.thought_log_prob:
                log = True

            source = batch_ground_truth[i]["source"]

            print(
                f"--------------------------------{log_flag}--------------------------------")
            print(
                f"【Solution{i}】({source})`{self.log_solution(batch_solution_str[i])}`")
            print(
                f"【Golden{i}】({source})`{self.log_ground_truth(batch_ground_truth[i])}`")
            print(
                f'[Final Reward]={cur_score:.3f}(ans={reward1[i]:.3f};q={reward2[i]:.3f};rm={reward3[i]:.3f})|{penalty_log_str}\n')

            if log:
                print(f'[Thought]\n{batch_solution_str[i]}')
                print()
        return final_results

    def log_solution(self, solution):
        norm = self.parse_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return norm[0]

    def log_ground_truth(self, ground_truth):
        return ground_truth["question"]


KG2QUERY_V1_DEFAULT_PARAMS = {
    "verify_agent": {
        "model": {
            "model": "Qwen2.5-32B-Instruct",
            "base_url": "https://sd262bskcm47j59r1292g.apigateway-cn-beijing.volceapi.com/v1",
            "api_keys": "caa6246b-afbe-4d9b-ab34-87bf9922032b",
            "request_kwargs": {
                "temperature": 0.6,
                "timeout": 360,
                "max_tokens": 1024,
            },
        },
        "max_concurrent_requests": 256
    },
    "similarity_run_args":  {
        "threshold": {
            1: 0.,
            2: 0.,
            3: 0.,
            4: 0.5,
            5: 1.0
        },
        "weight": 1.0,
    },
    "reward_model_args": {
        "urls": [
            "http://10.130.0.94:34109",
            "http://10.130.0.94:30838",
            "http://10.130.0.94:28362",
            'http://10.130.0.94:29126',
            "http://10.130.0.94:27227",
            "http://10.130.0.94:25021",
            "http://10.130.0.94:31792",
            "http://10.130.0.94:31097"
        ]
    }
}
