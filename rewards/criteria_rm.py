import re
import os
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

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# AUTOPE
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def parse_autope_solution_fn(solution_str: str, max_prompt_size=4096):
    if solution_str.count("</prompt_engineering>") > 1:
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
        conclusion = re.findall(r'<prompt_engineering>(.*)</prompt_engineering>',
                                solution_str, re.DOTALL)[0]
    except Exception as err:
        return None
    if ("<prompt_engineering>" in conclusion) or ("</prompt_engineering>" in conclusion):
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

        return await self._simulate_respondent(
            batch_data_sources=batch_data_sources,
            batch_solution_str=batch_solution_str,
            batch_ground_truth=batch_ground_truth,
            skip_run=skip_run,
            run_args={"solver_agent": self.args["solver_agent"]},
            batch_verify_fn=partial(
                self.batch_verify_results, verify_task=verify_task),
            resp_postprocess_fn=self.postprocess,
            prompt_contexts=None
        )

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
                pbar=True,
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
        return correctness["solver_agent"]

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

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
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

            if _reward > 0 or (self.split == "valid" and random.random() < 0.5) or (self.split == "train" and random.random() < 0.1):
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

                if parsed is not None and random.random() < 0.2:
                    print(f'[Thought]\n{parsed[0]}')
                    print()
        return final_results

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# AUTOPE
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

compute_score_train = AutoPEComputeScore(
    parse_autope_solution_fn, split="train", args=AUTOPE_DEFAULT_PARAMS).compute_score
compute_score_valid = AutoPEComputeScore(
    parse_autope_solution_fn, split="valid", args=AUTOPE_DEFAULT_PARAMS).compute_score
