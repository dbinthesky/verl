import re
import os
import copy
import json
import uuid
import jieba
import random
import aiohttp
import requests
import tqdm.asyncio
import numpy as np
import asyncio as aio
from functools import partial
import xml.etree.ElementTree as ET
from collections import namedtuple, defaultdict
from sacremoses import MosesTokenizer, MosesDetokenizer
from abc import ABCMeta, abstractmethod, abstractclassmethod


from typing import Any
from collections import defaultdict

from openai import OpenAI, RateLimitError, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

en_mt = MosesTokenizer(lang='en')


RLVR_VERIFY_FEWSHOTS = """### **基于标准答案判断回答是否正确**
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


RLVR_VERIFY_TEMPLATE = """
#### **输入：**
```json
{content}
```

#### **输出：**
"""

CRITIQUE_RECALL_FEWSHOTS = """
下面是对于同一用户提问的相同回复的两条不同评论，第一个是人类评论员，第二个是大模型评论员；

任务：现在需要你按照要求帮我分析**大模型评论员**的评论内容
分析**大模型评论员**的评论中是否覆盖**人类评论员**指出的批评？覆盖度1-5分，5分是最高分，分数仅包含1、2、3、4、5不包含小数点。

#### 输出要求
{分析内容}
```json
{{
    "批评覆盖度": *** # int (1-5)
}}
```

下面是一些例子

#### 用户提问
```
Can you provide a comprehensive list of stress management techniques using a bullet-point format in a [Mark down] document? Please include both physical and mental techniques that can be used to manage stress. Additionally, it would be helpful to categorize the techniques as short-term or long-term solutions. Thank you.
```

#### **人类评论员**
```
Your answer was well-organized and comprehensive, covering a variety of short-term and long-term stress management techniques. You also followed the instruction to use bullet-points and categorize the techniques. However, the instruction asked for a Markdown document, which requires specific formatting.

To improve, use proper Markdown formatting, such as adding asterisks or dashes for bullet points and using double asterisks for bold text. Additionally, the note in the instruction asks for a blend of accuracy, positivity, and intrigue, while keeping the user engaged. Your response is accurate and positive, but adding a touch of intrigue or a fun fact could make it more engaging.
```

#### **大模型评论员**
```
[分析开始]
用户的请求明确要求提供一个结构化的压力管理技巧列表，并以Markdown格式呈现。用户还特别指出要将技巧分为短期和长期，并包含物理和心理方法。这意味着我需要确保内容全面且易于理解，同时满足分类要求。

首先，我需要确定哪些技巧属于短期和长期。短期技巧通常是一些快速应用的方法，如深呼吸、冥想或写作，这些方法可以在短时间内缓解压力。长期技巧则涉及更系统的行为，如规律的锻炼、健康的饮食习惯或良好的睡眠，这些需要更持续的努力。

其次，用户强调了物理和心理方法的结合，因此我需要确保每个类别中都涵盖这两种方法。例如，深呼吸和冥想是物理和心理的结合，而写作或绘画则是心理表达的方式。

此外，考虑到用户可能需要个性化的建议，我提醒用户根据自身情况选择最适合的方法，并建议在长期压力情况下寻求专业帮助。这不仅增加了回复的实用性，也体现了对用户需求的尊重。

最后，我组织了内容，确保每个类别下的技巧清晰易读，并使用了Markdown格式的列表结构，以提高可读性和专业性。
[分析结束]
```

#### 输出
我现在需要分析大模型评论员的评论是否覆盖了人类评论员指出的批评。首先，先看人类评论员的批评点：

人类评论员指出的问题有两个：1. 没有使用要求的 Markdown 格式（比如项目符号用星号或破折号，粗体用双星号）；2. 虽然准确和积极，但缺乏吸引力或趣味点，没有达到指令中提到的 “融合准确性、积极性和吸引力，保持用户参与”。

接下来看大模型评论员的评论，但这里提供的大模型评论员内容是 “[分析开始]”，看起来不完整，没有实际的评论内容。由于大模型评论员的评论没有包含任何内容，自然无法覆盖人类评论员指出的任何批评点。

所以，批评覆盖度应该是 1 分？不，因为完全没有覆盖，按照评分标准，1-5 分，完全不覆盖的话应该是 1 分吗？是的，因为没有覆盖任何一点，所以分数是 1。

``` json
{{
    "批评覆盖度": 1
}}
```

#### 用户提问
```
Can you recommend a car service that offers private day tours to Versailles from Paris?
```

#### **人类评论员**
```
Your answer is well-structured, informative, and it adheres to the instructions given. You provided two options, which gives the user a choice, and you included the important details about each service. Your reminder for the user to conduct further research and compare options is also valuable. However, you could improve by providing more sources for the user to verify the information. For example, you could suggest checking the companies' websites or other reliable platforms for reviews and up-to-date information.
```

#### **大模型评论员**
```

[分析开始]
模型响应中提供了两个具体的车服务公司：Cityrama 和 Versailles Express。这两个公司都详细介绍了其服务亮点，包括行程内容、语言支持和交通安排，这在满足用户需求方面做得很好，属于有用性高的内容。此外，两个公司都是知名且值得信赖的，因此安全性较高。信息真实可靠，没有虚构内容，因此真实性评分也较高。虽然模型没有直接比较两家公司的优劣，但提供了足够的信息供用户参考，因此在有用性和可靠性方面得分高。
[分析结束]
```

#### 输出
我需要分析大模型评论员的评论是否覆盖人类评论员指出的批评。人类评论员指出的批评点是：回复虽有价值，但可改进之处在于提供更多让用户验证信息的来源（如公司网站、可靠平台的评论和最新信息）。

而大模型评论员的评论仅分析了响应中两个车服务公司的服务亮点、有用性、安全性、真实性，以及未直接比较但提供了参考信息，完全没有提及人类评论员指出的 “提供更多信息来源” 这一批评点。

``` json
{{
    "批评覆盖度": 1
}}
```
"""

CRITIQUE_RECALL_TEMPLATE = """
#### 用户提问
```
{instruction}
```

#### **人类评论员**
```
{human}
```

#### **大模型评论员**
```
{llm}
```

#### 输出
"""


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


def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))


def tokenize(s, lang_code):
    if lang_code == "en":
        tokenized_text = en_mt.tokenize(s.lower())
    elif lang_code == "zh":
        tokenized_text = list(jieba.cut(s))
    return tokenized_text


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

    async def do_job(self, agent, batch_inputs, max_concurrent_requests, repeats=3):
        prompts = defaultdict(list)
        for index, example in enumerate(batch_inputs):
            prompt = self.prompt_fn(example)
            prompts[prompt].append(index)

        _prompts = list(prompts.keys()) * repeats
        results = await agent.run(_prompts, 64, desc=f"[{self.task_desc()} {agent.model}={max_concurrent_requests}]", postprocess_fns=[self.postprocess]*len(_prompts))

        results_mapper = defaultdict(list)
        for (k, v) in results:
            for _ in prompts[k]:
                results_mapper[_].append(v)

        outputs = []
        for i in range(len(batch_inputs)):
            if i in results_mapper:
                scores = [_ for _ in results_mapper[i] if _ is not None]
                correct_votes = scores.count(True)
                wrong_votes = scores.count(False)
                if correct_votes > wrong_votes:
                    outputs.append(1.0)
                else:
                    outputs.append(0.0)
            else:
                outputs.append(0.0)
        return outputs


class RLVRVerify(BatchCallOpenAPI):
    def __init__(self):
        pass

    @classmethod
    def task_desc(cls):
        return "RLVR验证"

    def prompt_fn(self, example):
        solver_response, gt = example
        content = {
            "题目": gt["prompt"],
            "用户回答": solver_response,
            "标准答案": gt["ground_truth"],
        }
        prompt = RLVR_VERIFY_FEWSHOTS + "\n\n\n" + RLVR_VERIFY_TEMPLATE.format(
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


class CritiqueRecall(BatchCallOpenAPI):
    def __init__(self):
        pass

    @classmethod
    def task_desc(cls):
        return "批评覆盖度"

    def prompt_fn(self, example):
        solver_response, gt = example
        prompt = CRITIQUE_RECALL_FEWSHOTS + "\n\n\n" + CRITIQUE_RECALL_TEMPLATE.format(
            llm=solver_response,
            human=gt["critique"],
            instruction=gt["instruction"]
        )
        return prompt

    def postprocess(self, response: str):
        s = response
        try:
            conclusion = s.strip()

            recall = re.findall(
                r'\"批评覆盖度\": (\d+)', conclusion)[0]
            if isinstance(recall, str):
                recall = float(recall)
            assert isinstance(recall, float)
            assert recall in (0, 1, 2, 3, 4, 5)
            return recall

        except Exception as err:
            raise PostprocessError(f'{err}')

    async def do_job(self, agent, batch_inputs, max_concurrent_requests, repeats=3):
        prompts = defaultdict(list)
        for index, example in enumerate(batch_inputs):
            prompt = self.prompt_fn(example)
            prompts[prompt].append(index)

        _prompts = list(prompts.keys()) * repeats
        results = await agent.run(_prompts, 64, desc=f"[{self.task_desc()} {agent.model}={max_concurrent_requests}]", postprocess_fns=[self.postprocess]*len(_prompts))

        results_mapper = defaultdict(list)
        for (k, v) in results:
            for _ in prompts[k]:
                results_mapper[_].append(v)

        outputs = []
        for i in range(len(batch_inputs)):
            if i in results_mapper:
                scores = [_ for _ in results_mapper[i] if _ is not None]
                outputs.append(np.mean(scores) / 5.0)
            else:
                outputs.append(0.0)
        return outputs

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# AUTOPE
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def parse_autope_solution_fn(solution_str: str, max_prompt_size=4096):
    solution_str = postprocess_solution(solution_str)
    if not solution_str.startswith("<think>"):
        solution_str = f'<think>\n{solution_str}'

    if solution_str.count("</prompt_engineering>") > 1 or solution_str.count("<prompt_engineering>") > 1:
        return None

    if solution_str.count("</think>") > 1 or solution_str.count("<think>") > 1:
        return None

    if not solution_str.endswith("</prompt_engineering>"):
        return None

    try:
        thought = re.findall(r'<think>.*</think>',
                             solution_str, re.DOTALL)[0]
    except Exception as err:
        return None

    solution_str = solution_str.replace(thought, "")

    try:
        conclusion = re.findall(r'<prompt_engineering>(.*)</prompt_engineering>',
                                solution_str, re.DOTALL)[0]
    except Exception as err:
        return None

    if ("<prompt_engineering>" in conclusion) or ("</prompt_engineering>" in conclusion):
        return None

    if ("<think>" in conclusion) or ("</think>" in conclusion):
        return None

    if contains_chinese(conclusion):
        tokens = tokenize(conclusion, "zh")
    else:
        tokens = tokenize(conclusion, "en")

    if len(tokens) > max_prompt_size:
        return None

    return thought, conclusion


VerifyInfo = namedtuple("VerifyInfo", "index,tag,response,extra,ground_truth")


class AutoPEComputeScore(object):
    def __init__(self,
                 parse_solution_fn,
                 split="train",
                 args=None,
                 min_reward=-2.0
                 ):

        self.split = split
        self.parse_solution_fn = parse_solution_fn
        assert args is not None
        self.args = args
        self.task_name = "AUTOPE"
        self.min_reward = min_reward

        # 初始化API Client
        self.init_agent()

        self.initialized_save_rollout = False

    def init_save_rollouts(self):
        if self.initialized_save_rollout:
            return

        # 保存Rollouts数据
        default_local_dir = self.args["save_rollouts"]["default_local_dir"]

        save_path = os.path.join(
            default_local_dir, f'{self.task_name}_{uuid.uuid4().hex}.jsonl')
        self.save_rollouts_path = save_path

        print(
            f'[INFO] SAVE ROLLOUTS {self.task_name}: {self.save_rollouts_path}')

        self.rollout_cache = []
        self.initialized_save_rollout = True

    def init_agent(self):
        self.agents = {}
        self.init_verify_agent()
        self.init_solver_agent()

    def init_verify_agent(self):
        self.verify_agent = Agent(
            **self.args["verify_agent"]["model"])

    def init_solver_agent(self):
        self.solver_agent = Agent(
            **self.args["solver_agent"]["model"])
        self.agents["solver_agent"] = self.solver_agent

    def clip_string(self, s: str):
        if len(s) > 1500:
            return f'{s[:700]}... [省略] ...{s[-800:]}'
        return s

    async def simulate_respondent(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            skip_run=None
    ):
        verify_task = RLVRVerify()

        result = await self._simulate_respondent(
            batch_data_sources=batch_data_sources,
            batch_solution_str=batch_solution_str,
            batch_ground_truth=batch_ground_truth,
            skip_run=skip_run,
            run_args={"judge_agent": self.args["judge_agent"]},
            batch_verify_fn=partial(
                self.batch_verify_results, verify_task=verify_task),
            resp_postprocess_fn=self.response_postprocess,
            prompt_contexts=None
        )
        return result["solver_agent"]

    def postprocess(self, s):
        try:
            thought = re.findall(r'think>.*</think>', s, re.DOTALL)
            if len(thought) > 0:
                thought = thought[0]
                conclusion = s.replace(thought, "")
                return conclusion
            else:
                return s
        except Exception as err:
            raise PostprocessError(f'{err}')

    async def batch_verify_results(self,
                                   verify_queue,
                                   max_concurrent_requests,
                                   group_names,
                                   verify_task,
                                   return_input_response=False,
                                   ):
        correctness = {name: defaultdict(list) for name in group_names}

        result_index2queue_index = {}

        batch_eval_inputs = []
        for queue_index, example in enumerate(verify_queue):
            solver_response = example.response
            if solver_response is None:
                correctness[example.tag][example.index].append(0.0)
            else:
                batch_eval_inputs.append(
                    (solver_response, example.ground_truth))

                result_index2queue_index[len(
                    batch_eval_inputs)-1] = queue_index

        task = verify_task
        evaluations = await task.do_job(
            agent=self.verify_agent,
            batch_inputs=batch_eval_inputs,
            max_concurrent_requests=max_concurrent_requests,
        )

        for j, (eval_result, eval_input) in enumerate(zip(evaluations, batch_eval_inputs)):
            queue_index = result_index2queue_index[j]
            queue_elem = verify_queue[queue_index]
            if eval_result is not None:
                if return_input_response:
                    correctness[queue_elem.tag][queue_elem.index].append(
                        (eval_result, eval_input[0]))
                else:
                    correctness[queue_elem.tag][queue_elem.index].append(
                        eval_result)
        return correctness

    def autope_template(self, result, gt):
        thought, new_pe = result
        return new_pe

    async def _simulate_respondent(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            run_args,
            batch_verify_fn,
            resp_postprocess_fn,
            skip_run=None,
            prompt_contexts=None
    ):
        prompt2index = {_: defaultdict(list) for _ in run_args.keys()}
        extra = {}

        if prompt_contexts is None:
            prompt_contexts = [None] * len(batch_ground_truth)

        for i, (solution_str, gt, extra_ctx) in enumerate(zip(batch_solution_str, batch_ground_truth, prompt_contexts)):
            result = self.parse_solution_fn(solution_str)
            if result is not None:
                extra[i] = result
                if skip_run is not None and i in skip_run:
                    continue

                for name, v in run_args.items():
                    fn = getattr(self, v["fn"])
                    if extra_ctx is None:
                        _prompt = fn(result, gt)
                    else:
                        _prompt = fn(result, gt, extra_ctx)
                    prompt2index[name][_prompt].append(i)
        tasks = []
        task_names = []
        for name, v in prompt2index.items():
            prompts = list(v.keys()) * run_args[name]["repeat"]
            tasks.append(self.agents[name].run(
                prompts,
                run_args[name]["max_concurrent_requests"],
                desc=f'[{run_args[name]["desc"]} {run_args[name]["model"]["model"]} Solver]',
                pbar=False,
                postprocess_fns=[resp_postprocess_fn] * len(prompts)
            ))
            task_names.append(name)

        respond_questions = await aio.gather(*tasks)

        # 验证答案正确性
        verify_queue = []
        for name, results in zip(task_names, respond_questions):
            for (p, r) in results:
                for index in prompt2index[name][p]:
                    verify_queue.append(VerifyInfo(
                        index=index,
                        tag=name,
                        response=r,
                        extra=extra[index],
                        ground_truth=batch_ground_truth[index]))

        correctness = await batch_verify_fn(
            verify_queue=verify_queue,
            max_concurrent_requests=self.args["verify_agent"]["max_concurrent_requests"],
            group_names=task_names
        )
        return correctness

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      ):
        async def main():
            return await self._compute_score(batch_data_sources, batch_solution_str, batch_ground_truth)
        return aio.run(main())

    def log_solution(self, solution):
        norm = self.parse_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution.strip()))
        return repr(norm[1])

    def update_rollout_info(self, solution_str, ground_truth, score):
        inst_id = ground_truth["extra_info"]["uuid"]
        args = copy.deepcopy(self.args)

        self.rollout_cache.append({
            "prompt_generation_process": solution_str,
            "score": score,
            "uuid": inst_id,
        })

    def save_rollout_info(self):
        """将缓存保存为JSON文件"""
        if os.path.exists(self.save_rollouts_path):
            flag = "a+"
        else:
            flag = "wt"

        with open(self.save_rollouts_path, flag) as f:
            for _ in self.rollout_cache:
                f.write(f'{json.dumps(_, ensure_ascii=False)}\n')
        self.rollout_cache = []

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
        self.init_save_rollouts()

        accuracy = await self.simulate_respondent(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            if i in accuracy:
                _reward = np.mean(accuracy[i])
            else:
                _reward = 0.0
            final_results.append(_reward)

            # 保存Rollout信息
            if self.split == "train":
                self.update_rollout_info(
                    solution_str=batch_solution_str[i],
                    ground_truth=batch_ground_truth[i],
                    score=_reward,
                )

            if _reward > 0.5 or (self.split == "valid" and random.random() < 0.5) or (self.split == "train" and random.random() < 0.1):
                log = True
                log_flag = f"[{self.task_name} VALID]" if self.split == "valid" else f"[{self.task_name} TRAIN]"
            else:
                log = False

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f'【Question】`{repr(batch_ground_truth[i]["prompt"])}`')
                print(
                    f"【Solution】`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f'【Ground Truth】`{batch_ground_truth[i]["ground_truth"]}`')
                print(
                    f'[Final Reward]={_reward:.3f}\n')

                parsed = self.parse_solution_fn(batch_solution_str[i])

                if parsed is not None and random.random() < 0.5:
                    print(f'[Thought]\n{parsed[0]}')
                    print()

        self.save_rollout_info()
        return final_results

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# AUTOPE
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# CRITERIA_RM_RECALL
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def xml_cot_parse_solution_fn(solution_str):
    def get_thought(solution_str: str):
        thought = re.findall(r'```xml.*```', solution_str, re.DOTALL)[0]
        return thought

    def get_conclusion(solution_str: str):
        thought = get_thought(solution_str)
        return solution_str[solution_str.index(thought)+len(thought):].strip()

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


def criteria_parse_solution_fn(solution_str: str):
    solution_str = postprocess_solution(solution_str)
    if not solution_str.startswith("<think>"):
        solution_str = f'<think>\n{solution_str}'

    try:
        root = xml_cot_parse_solution_fn(solution_str)
    except Exception as err:
        return None

    if root is not None:
        try:
            conclusion = [
                child for child in root if child.tag == "conclusion"][0]

            conclusion = conclusion.text.strip()
        except Exception as err:
            return None
    else:
        return None

    if "# 评价标准" not in conclusion:
        return None
    conclusion = conclusion[conclusion.index("# 评价标准"):]
    return conclusion


class CriteriaRMRecallComputeScore(AutoPEComputeScore):
    def __init__(self,
                 parse_solution_fn,
                 split="train",
                 args=None,
                 min_reward=-2.0
                 ):
        super().__init__(
            split=split, parse_solution_fn=parse_solution_fn, args=args, min_reward=min_reward
        )

    def init_agent(self):
        self.agents = {}
        self.init_verify_agent()
        self.init_judge_agent()

    def init_verify_agent(self):
        self.verify_agent = Agent(
            **self.args["verify_agent"]["model"])

    def init_judge_agent(self):
        self.judge_agent = Agent(
            **self.args["judge_agent"]["model"])
        self.agents["judge_agent"] = self.judge_agent

    def response_postprocess(self, s, debug=False):
        if "</think>" in s:
            s = s[s.index("</think>")+len("</think>"):].strip()

        if "[分析开始]" in s and "[结论结束]" in s:
            s = s[s.index("[分析开始]"):s.index("[结论结束]")+len("[结论结束]")].strip()
        return s

    def judge_with_criteria(self, result, gt):
        format_template = """

基于上面的评价标准对“[模型响应]”进行评价，找出回答的不足。

你最终的回答部分需要包含**分析**和**结论**两部分
- 分析：详细的分析与思考过程
- 结论：对于模型响应的评价，给出全面的评价。

按照下面的格式
[分析开始]
... ...
[分析结束]

[结论开始]
{得分}
[结论结束]
"""

        prompt = f'[用户指令]\n{gt["response"]}\n\n[模型响应]\n{gt["instruction"]}\n\n[评价标准]\n{result}\n\n\n' + format_template
        return prompt

    async def simulate_respondent(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            skip_run=None
    ):
        verify_task = CritiqueRecall()

        result = await self._simulate_respondent(
            batch_data_sources=batch_data_sources,
            batch_solution_str=batch_solution_str,
            batch_ground_truth=batch_ground_truth,
            skip_run=skip_run,
            run_args={"judge_agent": self.args["judge_agent"]},
            batch_verify_fn=partial(
                self.batch_verify_results, verify_task=verify_task),
            resp_postprocess_fn=self.response_postprocess,
            prompt_contexts=None
        )
        return result["judge_agent"]

    def log_solution(self, solution):
        norm = self.parse_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution.strip()))
        return repr(norm)

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
        self.init_save_rollouts()

        recall = await self.simulate_respondent(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            if i in recall:
                _reward = np.mean(recall[i])
            else:
                _reward = 0.0
            final_results.append(_reward)

            # 保存Rollout信息
            if self.split == "train":
                self.update_rollout_info(
                    solution_str=batch_solution_str[i],
                    ground_truth=batch_ground_truth[i],
                    score=_reward,
                )

            if _reward > 0.5 or (self.split == "valid" and random.random() < 0.5) or (self.split == "train" and random.random() < 0.1):
                log = True
                log_flag = f"[{self.task_name} VALID]" if self.split == "valid" else f"[{self.task_name} TRAIN]"
            else:
                log = False

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f'【Question】`{repr(batch_ground_truth[i]["instruction"])}`')
                print(
                    f"【Critique LLM】`{self.log_solution(batch_solution_str[i])}`")
                print(
                    f'【Critique Human】`{repr(batch_ground_truth[i]["critique"])}`')
                print(
                    f'[Final Reward]={_reward:.3f}\n')

                parsed = self.parse_solution_fn(batch_solution_str[i])

                if parsed is not None and random.random() < 0.5:
                    print(f'[Thought]\n{parsed}')
                    print()

        self.save_rollout_info()
        return final_results
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# CRITERIA_RM_RECALL
# ------------------------------------------------------------------------------------------------------------------------------------------------------


AUTOPE_DEFAULT_PARAMS = {
    "verify_agent": {
        "model": {
            "model": "qwen25_32B_instruct",
            "base_url": "http://10.130.142.154:8000/v1",
            "api_keys": "EMPTY",
            "request_kwargs": {
                "temperature": 0.7,
                "timeout": 360,
                "max_tokens": 512,
            },
        },
        "max_concurrent_requests": 64
    },
    "solver_agent": {
        "model": {
            "model": "qwen25_32B_instruct",
            "base_url": "http://10.130.142.154:8000/v1",
            "api_keys": "EMPTY",
            "request_kwargs": {
                "temperature": 0.9,
                "timeout": 360,
                "max_tokens": 8192,
            },
        },
        "repeat": 8,
        "fn": "autope_template",
        "desc": 'AUTOPE',
        "max_concurrent_requests": 128
    },
    "save_rollouts": {
        "default_local_dir": "/cpfs01/shared/llm_ddd/tongjian/ckpts/datareview_rl_test/verl/grpo/autope_rollouts"
    }
}

CRITERIA_RM_RECALL_DEFAULT_PARAMS = {
    "verify_agent": {
        "model": {
            "model": "qwen25_32B_instruct",
            "base_url": "http://10.130.142.154:8000/v1",
            "api_keys": "EMPTY",
            "request_kwargs": {
                "temperature": 0.7,
                "timeout": 360,
                "max_tokens": 512,
            },
        },
        "max_concurrent_requests": 64
    },
    "judge_agent": {
        "model": {
            "model": "qwen25_32B_instruct",
            "base_url": "http://10.130.142.154:8000/v1",
            "api_keys": "EMPTY",
            "request_kwargs": {
                "temperature": 0.6,
                "timeout": 360,
                "max_tokens": 4096,
            },
        },
        "repeat": 1,
        "fn": "judge_with_criteria",
        "desc": 'JUDGE W CRITERIA',
        "max_concurrent_requests": 128
    },
    "save_rollouts": {
        "default_local_dir": "/cpfs01/shared/llm_ddd/tongjian/ckpts/datareview_rl_test/verl/grpo/autope_rollouts"
    }
}


compute_score_train = AutoPEComputeScore(
    parse_autope_solution_fn, split="train", args=AUTOPE_DEFAULT_PARAMS).compute_score
compute_score_valid = AutoPEComputeScore(
    parse_autope_solution_fn, split="valid", args=AUTOPE_DEFAULT_PARAMS).compute_score


criteria_recall_score_train = CriteriaRMRecallComputeScore(
    criteria_parse_solution_fn, split="train", args=CRITERIA_RM_RECALL_DEFAULT_PARAMS).compute_score
criteria_recall_score_valid = CriteriaRMRecallComputeScore(
    criteria_parse_solution_fn, split="valid", args=CRITERIA_RM_RECALL_DEFAULT_PARAMS).compute_score
