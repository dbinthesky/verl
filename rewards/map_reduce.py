import re
import json
import random
import aiohttp
import requests
import tqdm.asyncio
import asyncio as aio


from typing import Any
from collections import defaultdict

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

        # FIXME
        if self.model == "QwQ_32B":
            suffix = "\n<think>\n"
        else:
            suffix = ""
        try:
            response = await client.chat.completions.create(
                model=self.model, messages=[
                    {"role": "system", "content": 'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.'},
                    {"role": "user", "content": messages + suffix}
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


def postprocess_solution(solution_str):
    if "<|im_end|>" in solution_str:
        return solution_str[:solution_str.index("<|im_end|>")].strip()
    if "<｜end▁of▁sentence｜>" in solution_str:
        return solution_str[:solution_str.index("<｜end▁of▁sentence｜>")].strip()
    if "<|endoftext|>" in solution_str:
        return solution_str[:solution_str.index("<|endoftext|>")].strip()
    return solution_str


def parse_solution_fn(solution_str: str):
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
    return solution_str.replace(thought, "").strip()


def parse_thought_fn(solution_str: str, remove_option_letter=True):
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
    return thought


EVAL_PROMPT = """### **基于标准答案判断回答是否正确**
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


现在你需要按相同的步骤判断下面的回答是否正确：

"""

EVAL_TEMPLATE_TEMPLATE = """
#### **输入：**
```json
{content}
```

#### **输出：**
"""


class MapReduceComputeScore(object):
    def __init__(self, split="train"):
        self.split = split
        self.task_name = "MapReduce"

    @classmethod
    def get_verify_agent(cls):
        return Agent(**{
            "model": "qwen25_32B_instruct",
            "base_url": "http://10.130.142.154:8000/v1",
            "api_keys": "EMPTY",
            "request_kwargs": {
                "temperature": 0.6,
                "timeout": 360,
                "max_tokens": 4096,
            },
        })

    async def get_accuracy(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth, max_concurrent_requests=64, majority_vote=3):
        def postprocess(s):
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

        results_mapper = defaultdict(list)

        for i, (solution_str, gt) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            solution = parse_solution_fn(solution_str)

            if solution is not None:
                prompt = EVAL_PROMPT + EVAL_TEMPLATE_TEMPLATE.format(
                    content=json.dumps({"题目": gt["prompt"], "标准答案": gt["ground_truth"], "用户回答": solution}, ensure_ascii=False, indent="  "))

                results_mapper[prompt].append(i)

        prompts = list(results_mapper.keys()) * majority_vote

        results = await self.get_verify_agent().run(
            prompts, max_concurrent_requests,
            desc="[VERIFY]", postprocess_fns=[postprocess]*len(prompts))

        judges = defaultdict(list)
        for prompt, conclusion in results:
            for index in results_mapper[prompt]:
                judges[index].append(conclusion)

        full_rewards = []
        for i in range(len(batch_solution_str)):
            if i in judges:
                scores = [_ for _ in judges[i] if _ is not None]
                correct_votes = scores.count(True)
                wrong_votes = scores.count(False)
                if correct_votes > wrong_votes:
                    full_rewards.append(1.0)
                else:
                    full_rewards.append(0.0)
            else:
                full_rewards.append(0.0)
        return full_rewards

    def clip_string(self, s: str):
        if len(s) > 1500:
            return f'{s[:700]}... [省略] ...{s[-800:]}'
        return s

    def log_solution(self, solution):
        norm = parse_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return repr(norm)

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      max_concurrent_requests=64,
                      ):
        async def main():
            return await self._compute_score(batch_data_sources, batch_solution_str, batch_ground_truth,  max_concurrent_requests=max_concurrent_requests)
        return aio.run(main())

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             max_concurrent_requests=64,
                             ):
        accuracy = await self.get_accuracy(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            _reward = accuracy[i]
            final_results.append(_reward)

            if _reward > 0 or (self.split == "valid" and random.random() < 0.5) or (self.split == "train" and random.random() < 0.1):
                log = True
                log_flag = f"[{self.task_name} VALID]" if self.split == "valid" else f"[{self.task_name} TRAIN]"
            else:
                log = False

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f"【Solution】`{self.log_solution(batch_solution_str[i])}`")
                try:
                    print(
                        f'【Ground Truth】`{batch_ground_truth[i]["ground_truth"]}`')
                except Exception as err:
                    pass
                print(
                    f'[Final Reward]={_reward:.3f}\n')

                thought = parse_thought_fn(batch_solution_str[i])

                if random.random() < 0.5 and thought is not None and _reward > 0:
                    print(f'[Thought]\n{thought}')
                    print()


compute_score_train = MapReduceComputeScore(split="train").compute_score
compute_score_valid = MapReduceComputeScore(split="valid").compute_score
