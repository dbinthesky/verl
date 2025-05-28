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
from collections import namedtuple, defaultdict, deque
from sacremoses import MosesTokenizer, MosesDetokenizer


from openai import OpenAI, RateLimitError, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------


VERIFIER_MODEL_NAME = "qwen25_32B_instruct"
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


def tree_depth(element):
    # 如果没有子节点，深度为1
    if len(element) == 0:
        return 1
    # 否则，深度为子树的最大深度加1
    max_child_depth = max(tree_depth(child) for child in element)
    return max_child_depth + 1


def tree_width(root):
    if not root:
        return 0
    max_width = 0
    queue = deque([root])
    while queue:
        level_size = len(queue)
        max_width = max(max_width, level_size)
        for _ in range(level_size):
            current = queue.popleft()
            queue.extend(current)
    return max_width


def collect_tags(element, tags=None):
    if tags is None:
        tags = set()
    tags.add(element.tag)
    for child in element:
        collect_tags(child, tags)
    return tags


EVAL_PROMPT = """### **基于标准答案判断回答是否正确**  
任务描述：请根据提供的**题目**、**用户回答**和**标准答案**，判断用户回答是否正确，并按照指定格式输出结果。需严格对比答案核心要点，若用户回答与标准答案**完全一致**，则判定为正确，否则为错误。


#### 通用流程与方法
##### **步骤1：理解题意和标准答案含义
1. **明确题目类型与考察目标**：  
   - 分析题目是**事实性问题**（如“中国首都是哪里”）、**概念理解题**（如“解释光合作用原理”）还是**逻辑推理题**（如“根据数据推导结论”）。  
   - 确定考察重点：是**精确记忆**（客观题）、**要点完整性**（主观题）还是**逻辑合理性**（推理题）。  
2. **深度解析标准答案**：  
   - **客观题**：提取**唯一性核心答案**（如“北京”“5×3=15”）。  
   - **主观题**：拆解为**得分要点**并标注权重（如“①原因1（3分）；②影响2（2分）”）。  
   - **逻辑题**：梳理标准答案的**推理链条**（如“条件A→结论B→依据C”）。  
3. **识别隐含要求**：  
   - 例如：题目要求“用公式作答”，则需判断用户回答是否包含指定公式；  
   - 计算题要求“保留两位小数”，需检查用户回答的格式是否符合。  

##### **步骤2：用户回答提取最终结论**：
1. **去除干扰信息**：  
   - 删除与答案无关的内容。  
2. **统一基础格式**：  
   - 转换单位（如“千克”→“kg”）、大小写（如“USA”→“usa”）、简繁字体。  
3. **标记容错范围**：  
   - 数值题允许合理误差（如标准答案“5.0±0.1”，用户回答“4.9”视为正确）。  
   - 同义词替换允许语义等效（如“母亲”与“妈妈”视为一致）。 

##### **步骤3：逐项对比（用户回答 vs 标准答案）**  

| **题目类型**       | **对比方法**                                                                 | **判断标准**                                                                 |  
|--------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|  
| **客观题（精确答案）** | 直接对比字符或语义是否完全一致（允许数值误差等 **有限容错**，但核心答案需唯一）。 | **必须与标准答案完全一致**（含字符、单位、符号等），仅数值题允许极小误差（如±0.1），否则判错。 |  
| **主观题（要点型）**   | 将用户回答拆解为关键词/要点，与标准答案的得分点逐一匹配，计算匹配率。            | 核心要点（权重≥70%）**必须全部命中**，次要要点匹配率≥80%（可自定义阈值）→正确；否则判错。 |  
| **计算题**         | 拆分回答为 **公式、数值、单位** 三要素，逐项校验正确性。                          | **公式、数值、单位需全部正确，缺一不可**（公式错误则直接判错，数值误差超范围或单位错误均判错）。 |  
| **语义理解题**     | 使用语义相似度工具（如BERT模型）计算回答与标准答案的语义匹配度。            | 相似度≥阈值（如0.7）→正确；否则→错误。                                      |  


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
为了判断用户的回答是否正确，我们对问题进行分析并与标准答案对比：  

1. **问题分析**：  
   该问题要求在边长为1的正方形区域内求解拉普拉斯方程（∇²φ = 0），其中流体作稳定无旋流动，速度势函数φ满足**齐次狄利克雷边界条件**（所有四条边上φ = 0）。在偏微分方程理论中，这是一个基本结论：当所有边界值均为零时，拉普拉斯方程的唯一解是全域内的**平凡解**φ = 0。  

2. **用户回答评估**：  
   用户声称通过高斯-赛德尔迭代法进行100次迭代后，φ(0.5, 0.5) ≈ 0.25。这一结果与齐次边界条件下的理论预期矛盾。误差可能源于对边界条件的误读（例如，错误假设边界值非零，如非齐次边界条件问题）。但题目明确所有边界φ = 0，因此唯一有效解为全域φ = 0。  

3. **与标准答案的关键对比**：  
   标准答案为**0.00**，而用户答案为**0.25**。由于问题要求严格遵循边界条件及拉普拉斯方程解的唯一性，用户的结果显然错误。  

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


class XMLCoTComputeScore(object):
    def __init__(self,
                 split="train", parse_result_failure_score=-1.0):
        self.split = split
        self.parse_result_failure_score = parse_result_failure_score

    def f(self, n):
        return 1 / (1 + np.exp(-n))

    def thought_reward(self,
                       batch_data_sources,
                       batch_solution_str,
                       batch_ground_truth
                       ):

        roots = []
        for i in range(len(batch_solution_str)):
            root = xml_cot_parse_solution_fn(batch_solution_str[i])
            roots.append(root)

        widths, depths = [], []
        for root in roots:
            if root is not None:
                width, depth = tree_width(root), tree_depth(root)
                widths.append(width)
                depths.append(depth)

        base_score = 0.025

        full_rewards = []
        for i in range(len(batch_solution_str)):
            root = roots[i]
            if root is not None:
                width, depth = tree_width(root), tree_depth(root)
                if len(widths) == 0:
                    full_rewards.append(0.0)
                else:
                    full_rewards.append(
                        base_score *
                        self.f((width-np.mean(widths))/np.std(widths)) +
                        base_score *
                        self.f((depth-np.mean(depths))/np.std(depths))
                    )
            else:
                full_rewards.append(0.0)
        return full_rewards

    async def get_accuracy(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth, max_concurrent_requests=256, majority_vote=3):
        def postprocess(s):
            try:
                conclusion = s.strip()

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

        results_mapper = {}
        prompts = []

        for i, (solution_str, gt) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            root = xml_cot_parse_solution_fn(solution_str)

            if root is not None:
                conclusion = [
                    child for child in root if child.tag == "conclusion"][0]

                conclusion = conclusion.text.strip()
                prompt = EVAL_PROMPT + EVAL_TEMPLATE_TEMPLATE.format(
                    content=json.dumps({"题目": gt["prompt"], "标准答案": gt["ground_truth"], "用户回答": conclusion}, ensure_ascii=False, indent="  "))

                results_mapper[prompt] = i
                prompts.append(prompt)

        prompts = prompts * majority_vote

        results = await agent.run(prompts, max_concurrent_requests, desc="[VERIFY]", postprocess_fns=[postprocess]*len(prompts))

        judges = defaultdict(list)
        for prompt, conclusion in results:
            judges[results_mapper[prompt]].append(conclusion)

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
                full_rewards.append(self.parse_result_failure_score)
        return full_rewards

    def log_ground_truth(self, gt):
        return repr(gt["ground_truth"])

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      ):
        async def main():
            return await self._compute_score(batch_data_sources, batch_solution_str, batch_ground_truth)
        return aio.run(main())

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):

        thought_reward = self.thought_reward(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )
        accuracy = await self.get_accuracy(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            _reward = accuracy[i] + thought_reward[i]

            if (self.split == "valid" and random.random() < 0.05) or (self.split == "train" and random.random() < 0.01):
                log = True
                log_flag = "[VALID]" if self.split == "valid" else "[TRAIN]"
            else:
                log = False

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f'【Prompt】`{repr(batch_ground_truth[i]["prompt"])}`')
                print(
                    f'【Answer】`{self.log_ground_truth(batch_ground_truth[i])}`')
                print(
                    f'[Final Reward]={_reward:.3f}|ACC={accuracy[i]:.3f}|THOUGHT={thought_reward[i]:.3f}\n')
                print(f'【Solution】\n{batch_solution_str[i]}')


_xml_cot_compute_score_train = XMLCoTComputeScore(
    split="train")
_xml_cot_compute_score_valid = XMLCoTComputeScore(
    split="valid")
xml_cot_compute_score_train = _xml_cot_compute_score_train.compute_score
xml_cot_compute_score_valid = _xml_cot_compute_score_valid.compute_score
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# XML CoT
# ------------------------------------------------------------------------------------------------------------------------------------------------------
