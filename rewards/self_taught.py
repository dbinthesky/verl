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

en_mt = MosesTokenizer(lang='en')


def tokenize(s, lang_code):
    if lang_code == "en":
        tokenized_text = en_mt.tokenize(s.lower())
    elif lang_code == "zh":
        tokenized_text = list(jieba.cut(s))
    return tokenized_text


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


VerifyInfo = namedtuple("VerifyInfo", "index,tag,response,extra,ground_truth")


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
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            judge_prompt_key="ground_truth"
    ):
        input_datas = []
        rewards = {}

        for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            try:
                solution_str = self.postprocess_solution_fn(solution_str)
            except Exception as err:
                rewards[i] = self.parse_result_failure_score
                continue

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
SIMILARITY=*
[CONCLUSION END]
```
（*处填写1-5的整数，需严格符合上述判定标准）


下面是具体的例子

[原问题]
Find the minimum value of\n\\[x ^ 2 + 2xy + 3y ^ 2 - 6x - 2y, \\]over all real numbers $x$ and $y.$

[对比问题]
Find the minimum value of the quadratic function \\(f(x)=x ^ 2 - 6x + 7 \\).

[输出]
<think>
用户让我比较两个数学问题的相似程度。首先，我需要仔细看一下两个问题的内容。原问题是求一个二元二次多项式的最小值，变量是 x 和 y，表达式是 x² + 2xy + 3y² - 6x - 2y。待评价的问题是求一元二次函数 f(x) = x² - 6x + 7 的最小值。
首先根据评价标准，类型优先。原问题是二元二次函数求最值，属于多元微积分或代数中的二次型问题，可能需要用偏导数或者配方法。待评价问题是一元二次函数，直接可以用配方法或者求导，变量只有一个 x。题目类型不同，一个是二元，一个是一元，所以类型不同。根据使用说明，类型不同的话直接≤2 分。接下来看核心要素，原问题是二元，涉及两个变量和交叉项 xy，而待评价问题是一元，没有交叉项，解题思路也不一样，一元的更简单，只需要处理一个变量。所以它们的共同点可能只是同属代数中的二次函数求最值，但具体子类型不同，一个是多元，一个是一元。根据标准，同属大类别（代数），但具体子类型不同，所以应该是 2 分。因为 2 分的判定是仅单一维度相关，同属数学题中的代数大类，其余要素无重合。这里类型不同，所以符合 2 分的条件。
</think>
[CONCLUSION START]
SIMILARITY=2
[CONCLUSION END]


[原问题]
Find the inverse of the matrix\n\\[\\begin{pmatrix} 5 & -4 \\\\ 0 & 1 \\end{pmatrix}.\\]If the inverse does not exist, then enter the zero matrix.

[对比问题]
What is the inverse of the matrix \\(\\begin{pmatrix} 2 & 1 \\\\ 3 & 4 \\end{pmatrix}\\)?

[输出]
<think >
用户让我比较两个矩阵求逆的问题的相似程度。首先，根据类型优先原则，两个问题都是求矩阵的逆，属于同一题目类型，所以类型相同，至少 3 分以上。接下来看核心要素：原问题是求一个 2x2 矩阵的逆，对比问题也是 2x2 矩阵，求解目标都是求逆矩阵，解题框架都是使用矩阵求逆的方法，比如伴随矩阵法或者行变换。两者的不同在于矩阵中的数据不同，原问题的矩阵是[[5, -4], [0, 1]]，对比问题是[[2, 1], [3, 4]]。根据判定标准，4 分的情况是题目类型、求解目标、解题框架一致，仅数据或参数不同，核心步骤和逻辑完全复用。这里显然符合 4 分的条件，因为只是矩阵元素不同，解题方法完全一样，没有关键差异。所以相似等级应该是 4 分。
</think>
[CONCLUSION START]
SIMILARITY=4
[CONCLUSION END]


# 注意：比较题目相似度不考虑题型差异。例如，一道选择题，一道计算题，如果数据、表述、目标、步骤都十分接近，则应该判定为**高度相似**，而非因为题型不同，认为二者有较大差异。


"""

LECTURE_CONTAIN_QUESTION_FEWSHOTS = """
# 大模型课件“题目泄漏现象”检测
## 一、任务背景说明
已知前置流程：用户已向你提供一份**特定练习题**，并要求大模型基于该练习题生成一份“以知识点引导与应用为核心”的课件材料（非习题解答文档）。当前需你完成的核心任务是：对大模型最终生成的这份课件材料进行专项检测，判断其是否存在“伪装课件”的hack现象——即大模型未真正围绕知识点设计课件，而是将上述“特定练习题”的题干、问题及完整解答（含步骤、答案），通过简单包装（如添加“知识点标题”“段落拆分”等形式）伪装成课件，违背“知识点引导与应用”的核心目标。


## 二、核心检测维度（需逐点验证）
请你对照以下维度，全面分析课件内容，不遗漏任何细节：
1. **原练习题“题干/问题”是否直接出现**
   - 检查课件中是否存在与用户提供的“特定练习题”十分接近的题干、问题表述（包括文字、数字、条件描述等）；
   - 检查是否存在“变相照搬”：如仅调整句子语序、替换个别同义词，但核心条件、问题指向与原练习题完全一致。

2. **原练习题“解答内容”是否直接/间接呈现**
   - 直接呈现：课件中是否存在针对原练习题的完整解答步骤、最终答案；
   - 间接呈现：课件中是否以“知识点应用案例”“例题解析”等名义，针对原练习题的核心逻辑（如同一题型、同一计算逻辑、同一解题思路）给出唯一对应的解答，且该解答与原练习题的正确解法完全匹配。

3. **课件核心定位是否偏离“知识点引导”**
   - 判断课件内容的重心：是围绕“知识点本身”展开，还是围绕“原练习题”展开（如全程以原习题为核心，仅在解答前后简单提及知识点名称，未脱离“解题”本质）；
   - 检查知识点延伸性：课件是否提供了原练习题之外的知识点说明、定义展开等内容，还是仅局限于原练习题对应的知识单一场景。


## 三、判断标准与结果界定
1. **若满足以上任一条件，判定为“存在题目泄漏现象”**

#### 输出要求
- 先按照方法进行详细、准确、仔细地分析
- 再最后的部分按下面的JSON格式输出最后的结论
```json
{
  "题目泄漏": "是/否",
}
```

下面是一些例子


[输入]
** Document **
First, consider an open upper semicircle \( C \) of radius \( R \) centered at the origin. A function \( f(z) = \frac{1}{z^2 + a^2} \) where \( a > 0 \) is given. To estimate \( |f(z)| \) on this semicircle, we'll employ the reverse triangle inequality.

For any complex number \( z \) on \( C \), we know that \( |z| = R \). Applying the reverse triangle inequality:

\[
|f(z)| = \left| \frac{1}{z^2 + a^2} \right| = \frac{1}{|z^2 + a^2|} \leq \frac{1}{|z^2| - |a^2|} = \frac{1}{R^2 - a^2},
\]

provided that \( R > a \) to ensure the denominator is positive. This result gives us a maximum bound for \( |f(z)| \) on the semicircle.

Now that we have bounded \( |f(z)| \), we can apply the ML inequality to estimate the integral of \( f(z) \) over \( C \). The length \( L \) of our semicircle is \( \pi R \). Using the ML inequality:

\[
\left| \int_C f(z) \, dz \right| \leq M \cdot L = \frac{1}{R^2 - a^2} \cdot \pi R = \frac{\pi R}{R^2 - a^2}.
\]

This bound is crucial for assessing the behavior of the integral as \( R \) varies relative to \( a \). By combining these techniques, we effectively demonstrate how function estimations on contour integrals can be systematically approached.

Through these steps, we not only determine the magnitude bounds but also utilize them to control the integral's growth, which is essential in various applications within complex analysis.

** Question **
Let \( C \) be an open upper semicircle of radius \( R \) centered at the origin, and let \( f(z) = \frac{1}{z^2 + a^2} \), where \( a > 0 \) is real.  
(a) Show that \( |f(z)| \leq \frac{1}{R^2 - a^2} \) for \( R > a \).  
(b) Using the result from (a), show that \( \left| \int_C f(z) \, dz \right| \leq \frac{\pi R}{R^2 - a^2} \) for \( R > a \).

[输出]
### 一、详细检测分析
#### 1. 原练习题“题干/问题”验证
课件直接复现原练习题核心信息：上半开半圆\( C \)（原点中心、半径\( R \)）、函数\( f(z) = \frac{1}{z^2 + a^2} \)（\( a > 0 \)），且围绕“证明\( R > a \)时\( |f(z)| \leq \frac{1}{R^2 - a^2} \)”“证明积分界\( \leq \frac{\pi R}{R^2 - a^2} \)”展开，核心题干与问题直接出现。

#### 2. 原练习题“解答内容”验证
课件完整呈现原问题解答：用反向三角不等式推导\( |f(z)| \)的界（对应(a)），结合半圆长度\( \pi R \)与ML不等式推导积分界（对应(b)），步骤与最终答案均与原练习题解答完全匹配，属于直接呈现。

#### 3. 课件核心定位验证
课件仅提及“反向三角不等式”“ML不等式”名称，未展开知识点定义、原理，也无练习题外的延伸应用；全程以解原练习题为核心，未围绕知识点本身设计，严重偏离“知识点引导”目标。


### 二、检测结论
课件满足“题干/问题直接出现”“解答内容直接呈现”“定位偏离知识点引导”，判定为“存在题目泄漏现象”。

```json
{
  "题目泄漏": "是"
}
```



[输入]
** Document **
# 压电效应及其在压力传感器中的应用
压电效应是一种极具吸引力的物理现象：当晶体材料受到机械应力作用时，会产生电能。这一效应是众多应用的核心基础，其中压力传感器的制造尤为关键——这类传感器在工程学、科学等多个领域都发挥着重要作用。


## 一、压电效应的原理
压电效应仅发生在**无对称中心的晶体材料**中，常见例子包括石英（Quartz）、锆钛酸铅（PZT，Lead Zirconate Titanate）和钛酸钡（Barium Titanate）。其作用过程可拆解为以下步骤：
1. **机械应力作用**：当晶体受到压缩、拉伸或剪切等机械应力时，其内部的晶格结构会发生形变；
2. **电荷中心位移**：晶格形变会导致晶体内部正、负电荷中心的位置发生偏移，进而使材料整体产生电极化；
3. **电压差产生**：电荷的位移会在晶体的相对两个表面之间形成电压差，相当于产生了一个电偶极矩。


### 压电效应的数学描述
压电效应可通过以下公式定量表达：  
\[ Q = d \cdot F \]  
其中：
- \( Q \)：产生的电荷量（单位：库仑，C）；
- \( d \)：压电系数（材料特有常数，单位：库仑/牛顿，C/N，反映材料压电性能的强弱）；
- \( F \)：施加的机械力（单位：牛顿，N）。

这一效应不仅在理论上具有研究价值，在实际应用中也至关重要，尤其在换能器技术领域。


## 二、压电效应在压力传感器中的应用
基于压电效应的压力传感器，核心设计是嵌入一块压电晶体，其结构和工作逻辑如下：
- 晶体的一侧暴露于外部压力（如流体或气体压力），另一侧则被隔离（通常隔离至真空环境或参考压力状态）；
- 这种设计能确保传感器仅对外部压力产生响应，最大限度提升检测灵敏度。


### 压电压力传感器的设计流程
1. **晶体形态设计**：将压电晶体加工成薄圆盘或薄板形状——这种形态既能增强传感器的灵敏度，又能保证机械结构的完整性；
2. **电极附着**：在晶体的两侧均附着电极，电极的核心作用是检测晶体产生的电极化现象，并将生成的电压转换为可利用的电信号；
3. **真空隔离处理**：将晶体的一侧暴露于真空环境，以消除晶体内部压力产生的抵消力，确保传感器的输出信号完全由外部压力决定。


## 三、压电晶体用于压力传感器的优势
相较于其他类型的压力传感器，采用压电晶体的设计具有多项显著优势：

1. **超高耐压性**  
   压电材料的耐用性极强，可承受极高压力而不发生形变或损坏，因此适用于航空航天、工业场景、深海探测等恶劣环境。

2. **快速响应速度**  
   压电效应的产生是瞬时的，这种快速响应能力使传感器能精准检测突发压力变化（如冲击波、动态系统中的压力波动）。

3. **宽广的动态范围**  
   压电压力传感器可测量的压力范围极广：从极低的真空压力到极高的强压力，这种 versatility使其能适配多种不同应用场景。

4. **自供电特性**  
   与多数需要外部电源驱动信号处理的传感器不同，压电传感器可通过机械应力自主产生电荷；尽管信号调理和放大仍需额外电路，但“自供电”特性消除了电池更换的需求，大幅提升长期可靠性。

5. **高耐用性与稳定性**  
   压电材料抗磨损、抗疲劳，长期使用中性能稳定，因此成为需要持续工作的严苛环境中的首选方案。

6. **结构简洁与高鲁棒性**  
   压电压力传感器的设计相对简单，易于制造和维护；同时其结构坚固，在环境条件变化时不易发生故障。


## 四、总结
压电效应的核心价值在于：将机械应力转化为电能，为压力检测提供了原理基础。当这一效应应用于压力传感器时，最终产品兼具灵敏度、耐用性和简洁性三大优势。无论是实验室中的精密压力监测，还是恶劣环境下的高强度检测，压电效应始终是实现“精准、可靠压力测量”的核心技术手段。

通过深入理解和应用压电效应，工程师与科学家能够研发出更先进的压力传感器，以满足现代技术领域对高性能检测设备的严苛需求。


** Question **
Describe the principles behind the generation of electricity in a piezo crystal when it is subjected to mechanical stress. Explain how this principle can be applied to create a pressure sensor, including the necessary conditions for one side of the crystal to be exposed to a vacuum. Discuss the advantages of using piezo crystals in such applications, particularly in terms of withstanding high pressures and responding quickly to pressure changes.

[输出]
### 一、详细检测分析
#### 1. 原练习题“题干/问题”验证
原练习题的核心需求为三部分：“描述压电晶体受机械应力时产生电能的原理”“解释该原理如何用于制造压力传感器（含晶体一侧需暴露于真空的必要条件）”“讨论压电晶体在这类应用中的优势（尤其耐高压和快速响应方面）”。  
课件内容虽围绕“压电效应原理”“压电压力传感器应用”“压电晶体优势”展开，但未直接复现原练习题的题干表述（如未出现“Describe the principles...”“Explain how this principle...”等问题式语句），也未变相照搬原练习题的核心条件描述——课件对原理、应用、优势的阐述是基于知识点本身的系统讲解，而非针对原练习题的“问题回应”，因此原练习题“题干/问题”未直接或变相出现。

#### 2. 原练习题“解答内容”验证
- **直接呈现排除**：课件未针对原练习题的三个问题给出“对应式解答”，无“第一步解释原理、第二步说明传感器应用、第三步分析优势”的解题式结构，也未包含针对原练习题的“专属解答步骤”或“明确答案”；
- **间接呈现排除**：课件虽涉及原练习题提到的“压电原理”“真空条件”“耐高压/快速响应优势”，但这些内容是知识点的通用讲解（如原理部分拆解晶格形变、电荷位移等通用步骤，传感器应用部分说明真空隔离的通用设计逻辑，优势部分全面列举6项优势而非仅聚焦原练习题提及的2项），并非以“知识点应用案例”“例题解析”名义针对原练习题核心逻辑提供唯一对应解答，因此原练习题“解答内容”未直接或间接呈现。

#### 3. 课件核心定位验证
- **重心判断**：课件核心围绕“压电效应”知识点本身展开，从原理（晶格形变、电荷位移、数学公式）到应用（传感器结构、设计流程）再到优势（6项通用优势），均以知识点的系统讲解为重心，未围绕“原练习题”展开——既未以原练习题为起点引入知识点，也未以解答原练习题为终点，完全脱离“解题”本质，符合“知识点引导”目标；
- **知识点延伸性**：课件提供了原练习题之外的大量知识点延伸内容，如压电材料举例（石英、PZT、钛酸钡）、压电效应数学公式（Q=d·F）、传感器完整设计流程（晶体形态、电极附着、真空处理）、6项全面优势（远超原练习题提及的2项），未局限于原练习题对应的知识单一场景，具备充分的知识点延伸性。


### 二、检测结论
课件不满足“原练习题题干/问题直接/变相出现”“原练习题解答内容直接/间接呈现”“核心定位偏离知识点引导”任一条件，判定为“不存在题目泄漏现象”。

```json
{
  "题目泄漏": "否"
}
```




[输入]
** Document **
Circles are fundamental geometric shapes with numerous applications in mathematics and real-life scenarios. Understanding the properties of circles, such as their areas and radii, is essential for solving various geometric problems. In this lecture, we will explore how to determine the radius of circle A given the area of a shaded region between two circles.

Consider two circles, A and B, where circle B has a known radius of 1 cm. The shaded region, defined as the area between the two circles, has an area of π/2 cm². Our goal is to find the radius of circle A, denoted as r.

To solve this problem, we will follow a systematic approach:
1. **Calculate the area of circle B**: Since the radius of circle B is 1 cm, its area is π(1)² = π cm².
2. **Express the shaded area as the difference between the areas of circles A and B**: The shaded area is given by the formula:
   \[
   \text{Shaded Area} = \pi r^2 - \pi
   \]
   Given that the shaded area is π/2 cm², we can set up the equation:
   \[
   \pi r^2 - \pi = \frac{\pi}{2}
   \]
3. **Solve for r**: 
   - Divide both sides of the equation by π:
     \[
     r^2 - 1 = \frac{1}{2}
     \]
   - Add 1 to both sides:
     \[
     r^2 = \frac{3}{2}
     \]
   - Take the square root of both sides to find r:
     \[
     r = \sqrt{\frac{3}{2}} = \frac{\sqrt{6}}{2}
     \]
   - Therefore, the radius of circle A is:
     \[
     r = \frac{\sqrt{6}}{2} \text{ cm}
     \]

This method demonstrates how to use the area formula for circles to determine the radius of one circle when the area of the shaded region between two circles is given. The key steps involve setting up the correct equation based on the given information and solving for the unknown variable through algebraic manipulation.

Understanding such relationships is crucial for solving more complex geometric problems involving circles and their properties. This lecture emphasizes the importance of accurately interpreting the problem and applying fundamental mathematical principles to arrive at the solution.

** Question **
Given that the radius of circle B is 1 cm and the area of the shaded region is \(\pi / 2\) cm\(^2\), find the radius of circle A (denoted as \(r\)).

[输出]
### 一、详细检测分析
#### 1. 原练习题“题干/问题”验证
课件直接复现原练习题的核心题干与问题：明确提及“圆B的半径为1 cm”“阴影区域（两圆之间的面积）面积为π/2 cm²”，且核心目标与原问题完全一致——“求圆A的半径（记为r）”。表述中文字、数字、条件描述及问题指向均未发生实质性改变，不存在仅调整语序或替换同义词的“变相照搬”，而是直接呈现原练习题的核心题干与问题。

#### 2. 原练习题“解答内容”验证
课件以完整步骤直接呈现原练习题的解答内容：
- 第一步计算圆B的面积（π×1²=π cm²），对应解题的基础数据推导；
- 第二步建立阴影面积与两圆面积的关系式（πr² - π = π/2），与原问题的核心方程完全一致；
- 第三步通过“两边除以π→移项求r²→开方得r”的完整代数运算，得出最终答案r=√6/2 cm，解题步骤、计算逻辑及最终结果均与原练习题的正确解法完全匹配，属于典型的“直接呈现解答内容”。

#### 3. 课件核心定位验证
课件核心定位严重偏离“知识点引导”目标：
- 重心判断：课件虽在开头提及“圆的面积、半径是基础几何性质”，但后续全程以“求解原练习题”为核心，从条件复现到步骤拆解，均围绕该特定习题展开，仅在解题结束后简单提及“运用圆面积公式和代数运算”，未脱离“解题”本质，未将重心放在“知识点本身”；
- 知识点延伸性：课件未提供圆面积公式的定义、推导过程（如从正多边形逼近推导圆面积），也未给出原练习题之外的应用场景（如“已知圆面积求半径”“已知直径求圆面积”等不同题型），仅局限于原练习题对应的单一知识应用场景，无任何知识点延伸内容。


### 二、检测结论
课件满足“原练习题题干/问题直接出现”“原练习题解答内容直接呈现”“核心定位偏离知识点引导”三个条件，判定为“存在题目泄漏现象”。

```json
{
  "题目泄漏": "是"
}
```



[输入]
** Document **
When simplifying algebraic expressions and applying exponent rules, it's essential to understand the basic principles of exponent manipulation. Here's a breakdown of the key concepts:


1. **Exponent Rules for Division:**
   - When dividing like bases with exponents, subtract the exponents: 
     \[
     \frac{a^m}{a^n} = a^{m-n}
     \]
   - This rule applies because exponents represent repeated multiplication, and division cancels out common factors.


2. **Exponent Rules for Multiplication:**
   - When multiplying like bases with exponents, add the exponents:
     \[
     (a^m)(a^n) = a^{m+n}
     \]
   - This is because exponents denote the number of times a base is used as a factor in multiplication.


3. **Distributing Exponents:**
   - When an exponent applies to a product or quotient, it can be distributed:
     \[
     (ab)^n = a^n b^n \quad \text{and} \quad \left(\frac{a}{b}\right)^n = \frac{a^n}{b^n}
     \]
   - This means each term within parentheses gets raised to the outer exponent.


Now, let's apply these rules to simplify the expression:
\[
\left(\frac{4 m^5 n^2}{m^2 n}\right)^3
\]


**Step-by-Step Explanation:**

1. **Simplify the Fraction Inside the Parentheses:**
   - **For \( m \):**
     \[
     \frac{m^5}{m^2} = m^{5-2} = m^3
     \]
   - **For \( n \):**
     \[
     \frac{n^2}{n} = n^{2-1} = n^1 = n
     \]
   - The coefficient remains 4.
   - So, the simplified expression inside the parentheses is:
     \[
     4 m^3 n
     \]

2. **Apply the Exponent Outside the Parentheses:**
   - **Apply the Exponent to the Coefficient:**
     \[
     4^3 = 64
     \]
   - **Apply the Exponent to \( m^3 \):**
     \[
     (m^3)^3 = m^{3 \times 3} = m^9
     \]
   - **Apply the Exponent to \( n \):**
     \[
     (n)^3 = n^{1 \times 3} = n^3
     \]
   - Combining these results, the final simplified expression is:
     \[
     64 m^9 n^3
     \]


This example demonstrates the application of exponent rules in simplifying algebraic expressions.


** Question **
Which expression is equivalent to the given expression? Assume the denominator does not equal zero. 
$\left(\\frac{4 m^5 n^2}{m^2 n}\\right)^3$
A. $64 m^9 n^3$
B. $64 m^{13} n^5$
C. $12 m^6 n^4$
D. $12 m^{21} n^9$`

[输出]
### 一、详细检测分析
#### 1. 原练习题“题干/问题”验证
原练习题的核心题干是“找出与表达式$\left(\frac{4 m^5 n^2}{m^2 n}\right)^3$等价的选项（分母不为零）”，其核心要素为特定代数表达式$\left(\frac{4 m^5 n^2}{m^2 n}\right)^3$（含系数4、字母$m$和$n$及其对应指数$5,2,2,1$）。  
课件中直接以该完全相同的表达式作为“指数规则应用的示例”，表达式的系数、字母、指数及括号结构均未发生任何变化，属于原练习题核心题干的直接呈现，不存在仅调整语序或替换同义词的“变相照搬”，而是完全复现题干关键信息。

#### 2. 原练习题“解答内容”验证
原练习题的正确答案为选项A（$64 m^9 n^3$），课件以“分步解释”的形式直接呈现该题的完整解答：
- 第一步简化括号内分式：分别对$m$（$\frac{m^5}{m^2}=m^{3}$）、$n$（$\frac{n^2}{n}=n$）运算，得到$4 m^3 n$，对应解题的基础简化步骤；
- 第二步应用括号外的立方指数：对系数$4$（$4^3=64$）、$m^3$（$(m^3)^3=m^9$）、$n$（$n^3=n^3$）分别运算，最终得到$64 m^9 n^3$，与原练习题正确选项A的结果完全一致，解题逻辑、步骤及最终答案均直接匹配，属于“直接呈现解答内容”。

#### 3. 课件核心定位验证
课件核心定位偏离“知识点引导”目标，具体表现为：
- 重心判断：课件虽在开头列举了“除法指数规则”“乘法指数规则”“分配指数规则”，但仅给出规则公式及一句简单解释（如“同底数幂相除减指数，因除法抵消公因子”），未深入展开规则的推导原理（如从“重复乘法”本质推导指数运算逻辑）；后续内容全程围绕“求解原练习题”展开，从表达式复现到分步计算，均以该特定习题的解答为核心，仅在结尾提及“该示例展示指数规则的应用”，未脱离“解题”本质，未将重心放在“知识点本身的系统讲解”上。
- 知识点延伸性：课件未提供原练习题之外的知识点延伸内容，如未举例说明“不同系数的表达式简化（如$\left(\frac{2a^3b}{a^2}\right)^2$）”“混合指数运算（如$a^2 \times (a^3)^4$）”“含负数指数的运算”等场景，仅局限于原练习题对应的单一知识应用场景，缺乏知识点的广度延伸。


### 二、检测结论
课件满足“原练习题题干/问题直接出现”“原练习题解答内容直接呈现”“核心定位偏离知识点引导”三个条件，判定为“存在题目泄漏现象”。

```json
{
  "题目泄漏": "是"
}
```




"""


LECTURE_CONTAIN_QUESTION_TEMPLATE = """
现在需要你正式检测下面的文档，是否存在题目泄漏的问题。

[输入]
** Document **
{doc}

** Question & Solution **
Question:
{question}

Solution:
{solution}
"""

HINT_LEAKAGE_DETECTION_FEWSHOTS = """
任务：大模型思考过程质检专家
输入：我会给你一段大模型出题的思考过程，需要你检测这段思考过程是否满足下面的要求：

思考过程中绝对禁止泄露 “已知提示（真题 / 答案）” 的痕迹，需完全模拟 “独立命题时的自然思路”—— 即先明确命题目标、再拆解考点、再设计题干 / 选项 / 答案、最后验证难度，而非 “先看到真题和答案，再反推出题逻辑”。

你需要对思考过程的泄漏提示的安全程度打分（1～5分）1分表示泄漏严重，5分表示完全没有泄漏

#### 输出要求
- 先按照方法进行详细、准确、仔细地分析
- 再最后的部分按下面的JSON格式输出最后的结论（符合要求得分）
```json
{
  "符合要求得分": ***,
}
```


下面是一些例子
[输入]
首先，明确核心知识点：非线性方程组的求解，特别是三次方程组的解法，需利用平方根的性质和代数变形简化方程，最终通过已知方程推导目标表达式的值。

### 数学问题
已知三个正实数 \( x \)、\( y \)、\( z \) 满足以下方程组：
1. \( x^2 + xy + y^2 = 13 + 3\sqrt{5} \)
2. \( y^2 + yz + z^2 = 13 - 3\sqrt{5} \)
3. \( z^2 + zx + x^2 = 19 \)

请求出 \( xy + yz + zx \) 的值。若该值可表示为 \( k\sqrt{m} + n \)（其中 \( k \) 为正整数，\( m \) 为不含平方因子的正整数，\( n \) 为整数），请计算 \( k + m + n \) 的结果。


[输出]
### 详细分析过程
1. **判断是否遵循“独立命题自然思路”**  
   独立命题的自然思路需包含“明确命题目标→拆解考点→设计题干→推导答案方向→验证难度”五个核心环节。此思考过程：  
   - **命题目标与考点拆解**：开头明确“核心知识点为非线性方程组（三次方程组）求解，需利用平方根性质、代数变形推导目标表达式”，完成了命题目标与考点的界定，符合独立命题的起点逻辑；  
   - **题干设计环节**：直接呈现了完整题干（三个正实数满足的三次方程组及系数）和设问（求\(xy + yz + zx\)的值与\(k + m + n\)的结果），但未体现“如何从考点出发设计方程组系数（如为何选择\(13 + 3\sqrt{5}\)、\(13 - 3\sqrt{5}\)、19）”“为何设定求\(xy + yz + zx\)而非其他表达式”等关键设计逻辑，属于题干设计过程的细节缺失；  
   - **答案与难度验证环节**：未提及“目标表达式\(xy + yz + zx\)的可解性验证”“系数设置是否匹配‘利用代数变形简化’的考点难度”，缺乏独立命题后期的验证步骤。  
   整体虽有独立命题的初步框架，但缺少题干设计细节与验证环节，思路完整性不足，但未偏离“独立构思”的核心逻辑。

2. **判断是否泄露“已知提示（真题/答案）”痕迹**  
   泄露提示的核心特征是“先接触真题/答案，再反推出题逻辑”，此思考过程中：  
   - 未提及任何“已有真题”“预设答案”或“参考示例”，所有内容均围绕“非线性方程组求解”这一考点自主展开；  
   - 方程组系数、设问方向（求\(xy + yz + zx\)）均为直接呈现的设计结果，无“根据答案调整系数”“参考真题结构”的痕迹；  
   - 虽标注“未给出题目设计的逻辑，隐含提示泄漏”，但实际无任何“隐含真题信息”“暗示答案”的内容（如未提及答案范围、解题关键步骤的预设），不存在已知提示泄露问题。  


```json
{
  "符合要求得分": 4
}
```



[输入]
Okay, let's tackle this problem. I need to create a difficult question based on the provided reading material about the National Study on Incarcerated Youth. The user wants me to follow the tips given and the example they provided to design a question that tests the understanding of fractions and proportions related to the data in the text.

First, I'll start by recalling the key points from the reading. The example given in the text talks about the National Study showing that one in every five incarcerated youth committed serious offenses involving guns. The total number of incarcerated youth is 9,000. The question should involve calculating the number of youth who committed such offenses, which is straightforward multiplication: 9000 * (1/5) = 1800. But the user wants a difficult question, so maybe I need to add some twists or make it require more steps.

Looking at the tips, the question should be multi-stage and have distractors. The example provided in the user's final question is multiple-choice with options a to d, but the answer is c. The original question in the text is a calculation question, but the user wants to follow the structure of the example they gave, which is a multiple-choice question with four options.

So, I need to design a question where the answer is one of the options, and the other options are plausible distractors. The core of the question is about calculating the number of youth who committed serious offenses involving guns, which is 1800. But to make it a multiple-choice question, I can present it as such.

The options given are a to d, each describing a different scenario. The correct answer is c, which matches the calculation. The other options are there to test if the student misinterprets the data. For example, option a might be about those not adjudicated, b about age, d about status offenses. These are all plausible but incorrect based on the given data.

Now, I need to ensure the question is presented in the correct format. The question stem should clearly state the scenario and ask which option is correct. The answer is c, so the question is a multiple-choice with the answer letter.

I also need to make sure the question is challenging. Maybe the question could involve a step where the student has to calculate the number first and then match it to the option. But since the options are given, the question is straightforward but tests if they correctly apply the proportion.

The thinking process would be: the student reads the scenario, knows the total is 9000, one in five committed serious offenses, so calculate 9000/5=1800. Then match that to the option c. The other options are there to test if they mixed up the data or misunderstood the proportion.

So, putting it all together, the question is structured as a multiple-choice with four options, and the answer is c. The question is designed to test the application of fractions and proportions in a real-world context, which aligns with the reading material's objectives.

[输出]
### 详细分析过程
1. **判断是否模拟“独立命题时的自然思路”**  
   独立命题的核心逻辑是“从考点出发，自主构建题目全要素”，需覆盖“明确目标→拆解考点→设计内容→验证难度”四步。此思考过程完全契合该逻辑：  
   - **明确命题目标与拆解考点**：开篇即锁定“基于‘全国被监禁青少年研究’材料，设计考查分数与比例计算的难题”这一目标，同时从材料中提取“每5人中有1人涉枪重罪、总人数9000”等关键信息，拆解出“实际场景下的比例应用+数据计算”核心考点，为后续设计奠定清晰方向。  
   - **设计题干与选项**：为满足“难题”要求，主动规划“增加计算步骤”“设置干扰项”的优化方向；结合用户对题型的需求（选择题），自主构建题干场景（围绕“涉枪重罪青少年数量”设问），并针对性设计干扰项（a项“未判决群体”、b项“年龄相关”、d项“身份犯罪”），且明确干扰项的设计目的是“测试学生是否误解数据”，体现对选项区分度的主动思考；正确答案（1800）的推导完全基于材料数据（9000×1/5），是“考点→数据→结果”的正向逻辑，而非预设结果。  
   - **验证难度**：明确“通过‘计算后匹配选项’‘干扰项混淆认知’的组合，确保题目既考查核心能力，又具备挑战性”，完成对难度适配性的自主验证，确保题目符合“难题”定位。  
   整体虽参考用户对“选择题题型”的形式要求，但未依赖任何具体题干内容或答案，核心设计环节均为自主生成，完全体现“独立命题的自然思路”。

2. **判断是否泄露“已知提示（真题/答案）”痕迹**  
   泄露提示的关键特征是“先接触真题/答案，再反向推导设计逻辑”，此思考过程无任何此类痕迹：  
   - 未提及“已存在的真题题干”“预设的标准答案”，所有关键元素（总人数9000、1/5的比例、涉枪重罪的设问方向）均来源于“阅读材料关键信息”，而非“已知真题内容”；  
   - 虽提到“用户提供的示例”“follow the tips”，但仅参考“选择题题型”这一形式框架和“设计难题”的通用建议，未涉及示例的具体题干细节、选项数值或答案，不存在“以真题为模板反推”的情况；  
   - 正确答案（1800）的得出逻辑是“阅读材料数据→比例计算规则→结果”，是正向的解题与设计结合过程，而非“先确定答案1800，再反向设定总人数9000和比例1/5”，无反向推导痕迹。  
   综上，思考过程仅在形式层面适配用户要求，内容层面完全自主，无任何已知提示泄露。


```json
{
  "符合要求得分": 5
}
```



[输入]
Okay, let's tackle this problem. The user wants me to create a difficult question based on the provided reading material about non-negative real numbers and inequalities. The example given is a problem where we have non-negative real numbers x₁, x₂, ..., xₙ such that the sum of 1/(1 + x_i) equals 1, and we need to prove that the sum of x_i/(1 + x_i²) is less than or equal to 1. The solution involves trigonometric substitution and Lagrange multipliers.

First, I need to understand the core of the problem. The key here is the substitution x_i = tanθ_i, which transforms the given condition into a sum of cos²θ_i equaling 1. Then, the target expression becomes sinθ_i cosθ_i, which can be maximized under the constraint. The maximum value found was 1/√2, but the problem states the inequality is ≤1, so the maximum is indeed 1 when two variables are 1 and the rest are infinity.

The question is already provided in the example, but I need to ensure it's constructed following the guidelines. The user wants a question that tests the application of trigonometric substitution and Lagrange multipliers, with the answer being 1. The thinking process should show how to arrive at this conclusion step by step.

Starting with the problem statement: non-negative real numbers x₁ to xₙ with the given sum condition. The task is to prove the inequality. The solution uses substitution x_i = tanθ_i, leading to cos²θ_i summing to 1, and then maximizing sinθ_i cosθ_i under that constraint. The maximum value is achieved when two angles are π/4 and the rest are 0, giving the sum as 1/√2, but the problem states the inequality is ≤1, so the maximum is indeed 1 when two variables are 1 and the rest are infinity.

The question is a proof question, so the answer is a boxed 1. The thinking process involves recognizing the substitution, applying trigonometric identities, using Cauchy-Schwarz or Lagrange multipliers to find the maximum, and then concluding the inequality holds.

I need to make sure the question is presented clearly, with the conditions and the target expression. The answer is straightforward once you go through the steps, but the question itself is the core. The user's example question is already given, so I just need to format it correctly with the answer boxed.

The final question is structured as a proof question, so the answer is a single value. The thinking process would involve the steps mentioned: substitution, trigonometric identities, applying the constraint, and finding the maximum value. The key is to show that the maximum is 1, which is achieved when two variables are 1 and the rest are infinity, satisfying the given condition.

So, putting it all together, the question is ready. The answer is 1, boxed. The thinking process confirms the steps needed to reach that conclusion. I should ensure that the question is presented in the required format with the answer boxed.


[输出]
### 详细分析过程
1. **判断是否遵循“独立命题自然思路”**  
   独立命题的自然思路应遵循“明确命题目标→拆解考点→设计题干/选项/答案→验证难度”的逻辑。但此思考过程的起点并非自主确定命题方向，而是直接基于“the user provided reading material”中的“example given”（即已存在的具体题目：“非负实数x₁,x₂,…,xₙ满足∑1/(1+xᵢ)=1，证明∑xᵢ/(1+xᵢ²)≤1”），且明确提及“the question is already provided in the example”“the user's example question is already given”，说明思考过程并非“独立设计题目”，而是围绕“已有的示例题目”展开。

2. **判断是否泄露“已知提示（真题/答案）”痕迹**  
   思考过程中多次直接引用“示例题目”的完整题干（条件与待证结论）、预设解法（“The solution involves trigonometric substitution and Lagrange multipliers”）及最终答案（“the answer is a boxed 1”），且后续所有分析均以“如何格式化已有题目”为核心（“I just need to format it correctly with the answer boxed”“putting it all together, the question is ready”），而非自主推导考点、设计题干。整体逻辑是“先看到已有题目（类似真题）和答案，再反推如何呈现”，而非“独立命题”，存在严重的“已知提示泄露”痕迹。


```json
{
  "符合要求得分": 1
}
```



[输入]
Okay, let's tackle this problem. The user wants me to create a difficult question based on the provided reading material about position, velocity, and acceleration, using derivatives and infinitesimals. The question needs to be challenging but solvable with the concepts from the text. 

First, I need to understand the core concepts from the reading. The material covers position functions, velocity as the first derivative, acceleration as the second derivative, and the role of infinitesimals in defining derivatives. The examples given include calculating velocity from position functions and understanding the relationship between Δx, dx, and Δt.

The question should test the understanding of these concepts. Since it's a difficult question, maybe it should involve multiple steps or require applying the derivative concept in a scenario. The user's example in the final question is about explaining what dx/dt represents and the difference between dx and Δx. That seems like a good angle because it touches on both the physical meaning and the mathematical distinction.

Looking at the reading, the key points are:
- Position function x(t) gives position at time t.
- Velocity v(t) is dx/dt, the derivative of x(t).
- Acceleration a(t) is the second derivative, d²x/dt².
- Infinitesimals are used in the limit definition of derivatives, where Δx/Δt approaches dx/dt as Δt approaches zero.

The question asks about the physical meaning of dx/dt and the difference between dx and Δx. The answer should be that dx/dt is velocity, and dx is an infinitesimal change in position, while Δx is a finite change. 

To make it a question, perhaps present a scenario where someone might confuse these concepts. For example, maybe a question where they have to identify the correct interpretation of a given derivative expression in a physics context. But the user's example given is a short answer question, so maybe it's better to structure it as a question that requires explaining the concepts.

The user's final question example is: "Consider a physical system where the position of an object is described by the function x(t). When taking the derivative of position with respect to time, dx/dt, what does this derivative represent in terms of physical meaning? How does this relate to the concept of infinitesimals, and what is the difference between dx and Δx? Provide a clear and concise explanation, including any relevant mathematical equations or examples."

This question is exactly what the user wants. It asks for the physical meaning of dx/dt (which is velocity), the relation to infinitesimals, and the difference between dx and Δx. The answer is straightforward but tests understanding of the concepts.

Now, I need to ensure that the question is difficult enough. The difficulty comes from requiring the student to connect the mathematical concept (derivative) with its physical interpretation and the distinction between infinitesimals and finite changes. The example given in the reading material has similar questions, so this aligns well.

The answer is provided as "dx/dt = v, where v is the velocity of the object" for the first part, and the second part explains the difference between dx and Δx. Since the question is a short answer, the answer should be concise but clear.

I think this fits the requirements. The question is based on the material, tests the key concepts, and the answer is direct but requires understanding the relationship between the mathematical operation and its physical meaning. The user's example in the final question matches this exactly, so I can proceed with that.


Consider a physical system where the position of an object is described by the function x(t). When taking the derivative of position with respect to time, dx/dt, what does this derivative represent in terms of physical meaning? How does this relate to the concept of infinitesimals, and what is the difference between dx and Δx? Provide a clear and concise explanation, including any relevant mathematical equations or examples.

Answer: dx/dt = v, where v is the velocity of the object. In terms of infinitesimals, dx/dt is defined as the limit of Δx/Δt as Δt approaches 0 (i.e., dx/dt = lim(Δt→0) Δx/Δt), where Δx = x(t+Δt) - x(t) is the finite change in position over the finite time interval Δt. dx represents an infinitesimal (infinitely small) change in position, which is a mathematical abstraction used to formalize the derivative; it is not a measurable finite quantity like Δx. For example, if x(t) = t² (in meters, with t in seconds), then dx/dt = 2t (in m/s), which is the object’s velocity at time t. If Δt = 0.1 s at t=1 s, Δx = (1.1)² - 1² = 0.21 m, so Δx/Δt = 2.1 m/s (average velocity over 0.1 s), while dx/dt at t=1 s is 2 m/s (instantaneous velocity, the limit as Δt shrinks to 0).

[输出]
### 详细分析过程
1. **判断是否泄露“已知提示（真题/答案）”痕迹**  
   该思考过程存在明显的“依赖已知真题框架”痕迹：多次明确提及“the user's example in the final question”“The user's final question example is: 'Consider a physical system where the position of an object...'”“The user's example in the final question matches this exactly”，表明其核心并非自主生成题目，而是以“用户预先提供的最终问题示例”（本质为已知真题雏形）为基准，后续所有分析（如“判断题目是否符合难度”“确认答案是否合理”）均围绕该示例展开，而非从零开始构思题干、设问与答案。同时，思考过程中直接引用示例题干的完整表述，并明确对应示例的标准答案（“dx/dt = v，where v is the velocity of the object”等），进一步印证了“先接触已知真题与答案，再反推其合理性”的逻辑，存在严重的已知提示泄露问题。

2. **判断是否模拟“独立命题时的自然思路”**  
   独立命题的自然逻辑应遵循“明确命题目标→拆解核心考点→自主设计题干/设问→推导答案方向→验证难度适配性”，但该思考过程完全偏离此逻辑：  
   - **命题目标层面**：未主动界定“基于‘位置、速度、加速度与导数、无穷小量’知识，设计一道考查‘数学概念与物理意义关联’的难题”，而是被动以“用户已有示例是否符合需求”为目标；  
   - **考点拆解层面**：未独立梳理“dx/dt的物理意义、无穷小量在导数定义中的角色、dx与Δx的本质差异”等考点，而是从用户示例中反向提取考点，缺乏主动拆解的过程；  
   - **题干设计层面**：未自主构建新的物理场景（如“物体沿曲线运动的位置函数x(t)=sin t + t”）或设计递进式设问（如“1. 推导dx/dt并说明其物理含义；2. 结合具体Δt值对比Δx与dx的差异”），而是直接沿用用户示例的题干结构，仅做“是否合理”的验证；  
   - **难度验证层面**：未自主分析“通过‘关联抽象数学定义与具体物理场景’‘区分 infinitesimal 与 finite 量’是否达到难题标准”，而是仅确认“用户示例是否符合难度要求”，无主动判断过程。  
   整体逻辑是“先看到真题示例，再反推其出题合理性”，而非“独立构思题目”，完全未体现独立命题的自然思路。


```json
{
  "符合要求得分": 1
}
```



"""

HINT_LEAKAGE_DETECTION_TEMPLATE = """
现在需要你对下面的思考过程进行分析，判断是否符合要求并给出得分（1-5分）分数越高表示越符合要求


[输入]
{doc}

[输出]
"""


QUESTION_DESIGN_THOUGHT_JUDGE_FEWSHOTS = """
任务：大模型出题思考过程评价
输入：我会给你一段大模型出题的思考过程，需要你对模型的出题思考过程进行打分（1-5分）


# 大模型出题思考过程评分规范（1-5分）
## 一、核心评价维度（权重排序）
1. 步骤完整性：是否覆盖“需求定位→设计（场景/题干/难点/答案）→试做反馈→迭代优化→定稿”全流程，核心环节缺失则降档；
2. 细节详实度：各环节是否包含“设计依据/逻辑说明”，无说明则降档；
3. 反馈闭环性：是否形成“试做→错误分析→调整→验证”闭环，无闭环则降档。


## 二、各分数档具体规范
# 大模型出题思考过程评分规范（5分制）
## 一、核心评价维度（权重排序不变）
1. **步骤完整性**：是否覆盖“需求定位→完整设计（场景/题干/难点/答案解析）→试做反馈→迭代优化→定稿”全流程，核心环节缺失直接限定分数上限；
2. **细节详实度**：各环节是否包含“设计依据/逻辑说明”（如知识点选择理由、场景适配逻辑、易错点设计目的等），无说明则按缺失数量降分；
3. **反馈闭环性**：是否形成“试做数据采集→错误归因→针对性调整→二次验证”闭环，闭环断裂则无法进入最高分数档。


### 5分（优秀：全流程闭环+细节极致）
#### 核心界定
覆盖所有核心环节，各环节均有**明确且充分的依据说明**，试做反馈闭环完整（含二次验证），无任何信息缺失。
#### 具体规范要求
1. 需求定位：明确学科、学段、知识点、考察目标，**详细说明知识点选择理由**（如贴合学段重点、弥补常见知识漏洞等）；
2. 设计环节：
   - 场景设计：说明场景与知识点的适配逻辑（如生活场景对应应用类考点，学术场景对应理论类考点）；
   - 题干设计：说明表述逻辑依据（如先背景后问题的递进式表述，避免信息歧义）；
   - 难点/易错点设计：明确易错点数量（2个及以上），**逐一说明每个易错点的设计目的**（如干扰项对应易混淆概念，陷阱条件对应审题疏漏）；
   - 答案/解析设计：给出完整答案+分层解析（如“解题步骤→关键思路→易错点提醒”），说明解析的分层逻辑依据；
3. 试做反馈：说明试做人群（如“初二学生30人+教师5人”）、整体正确率、**各错误类型及占比**（如“概念混淆类错误占60%，计算失误类占30%”）；
4. 迭代优化：说明具体调整动作（如“删除1个歧义干扰项，补充题干背景说明”），**明确调整依据**（如“依据试做中40%学生因题干信息不全出错”）；
5. 定稿确认：说明调整后二次试做的具体结果（如“正确率从60%提升至85%，目标错误类型占比降至10%以下”），给出清晰的定稿判断标准（如“二次试做正确率≥80%且核心错误率≤15%”）。


### 4分（良好：全流程覆盖+反馈闭环待补）
#### 核心界定
覆盖所有核心环节（需求定位→设计→试做→迭代→定稿），各环节**大部分有依据说明**（仅1个环节缺失），试做反馈有迭代但无二次验证，或二次验证无明确判断标准。
#### 具体规范要求
1. 需求定位：明确学科、学段、知识点、考察目标，**简要说明知识点选择理由**（如“为巩固近期所学”）；
2. 设计环节：
   - 场景/题干/难点/答案解析均完整设计，仅1项缺失设计依据（如“有场景设计，但未说明场景适配逻辑”）；
   - 难点/易错点数量明确（1-2个），至少说明1个易错点设计目的；
   - 解析为分层结构，但未说明分层逻辑；
3. 试做反馈：说明试做人群、整体正确率、各错误类型及占比，数据完整；
4. 迭代优化：说明具体调整动作及调整依据，逻辑清晰；
5. 定稿确认：以“调整后题目符合预期”定稿，**未进行二次试做**，或有二次试做但未给出定稿判断标准（如仅说“二次试做效果变好”，无具体数据）。


### 3分（合格：基础流程完整+试做环节缺失）
#### 核心界定
仅覆盖“需求定位→完整设计（场景/题干/难点/答案解析）”，无试做反馈及迭代优化环节，设计细节有部分缺失。
#### 具体规范要求
1. 需求定位：明确学科、学段、知识点、考察目标，**未说明知识点选择理由**；
2. 设计环节：
   - 场景设计存在（题干有具体语境，非抽象表述），但未说明场景选择依据；
   - 题干表述无歧义，但未说明表述逻辑依据；
   - 难点/易错点有设计（至少1个），但未说明设计目的（如仅说“设置1个易错点”，无后续理由）；
   - 答案完整，解析为非分层结构（如仅“答案：XXX，解析：XXX”，无步骤拆分）；
3. 无试做反馈、迭代优化环节，直接以“设计完成”定稿。


### 2分（不足：基础设计残缺+细节空白）
#### 核心界定
仅覆盖“需求定位→基础设计（题干+答案）”，场景或难点设计仅存1项（非完整），无任何细节说明，无试做迭代。
#### 具体规范要求
1. 需求定位：明确学科、学段、知识点，**未说明考察目标**；
2. 设计环节：
   - 仅存在场景或难点设计中的1项（如“有简单场景，但无难点设计”，或“有1个难点，但题干无具体场景，为抽象表述”）；
   - 题干表述简洁，但可能存在轻微歧义（如“未明确限定条件”）；
   - 仅给出完整答案，**无解析**（或解析仅“答案正确”，无实质内容）；
   - 所有设计环节均无依据说明（如“有场景但未说为何选此场景，有难点但未说为何设此难点”）；
3. 无试做反馈、迭代优化环节，直接定稿。


### 1分（极差：基础信息缺失+核心设计空白）
#### 核心界定
仅完成“知识点模糊确定→题干+答案”，需求定位核心要素缺失，无场景、无难点设计，无任何思考逻辑，无试做迭代。
#### 具体规范要求
1. 需求定位：仅提及“某知识点”（如“考三角形”），**未明确学科（如数学）、学段（如初中）、考察目标**；
2. 设计环节：
   - 无场景设计（题干为纯抽象表述，如“计算三角形面积”，无具体边长、图形背景等）；
   - 无难点/易错点设计（题目仅单一考察点，无干扰项、无陷阱条件，如“已知三角形底3cm、高2cm，求面积”）；
   - 仅给出题干及简单答案（如“答案：3cm²”），**无任何设计依据、逻辑说明**；
3. 无试做反馈、迭代优化环节，仅呈现“题目+答案”。


## 三、评分判断优先级（修订版）
1. **先定步骤完整性上限**：
   - 缺“试做反馈+迭代优化”：最高得3分（无论设计细节多完善）；
   - 缺“场景设计且缺难点设计”：最高得1分；
   - 仅缺“场景设计或仅缺难点设计”：最高得2分；
2. **再扣细节详实度分数**：
   - 某环节缺失1个设计依据：在步骤完整性上限基础上降0.5分（如4分档缺1个依据→3.5分）；
   - 某环节缺失2个及以上设计依据：在步骤完整性上限基础上降1分（如3分档缺2个依据→2分）；
3. **最后补反馈闭环性差异**：
   - 有试做+有调整，但无二次试做：4分档内最高得4分（无法升5分）；
   - 有试做+有调整+有二次试做，但无定稿判断标准：4分档内得4.5分（无法升5分）；
   - 有试做+有调整+有二次试做+有定稿标准：可进入5分档。


# 输出要求
- 先按照方法进行详细、准确、仔细地分析
- 再最后的部分按下面的JSON格式输出最后的结论（符合要求得分）
```json
{
  "得分": ***,
}
```


下面是一些例子

[输入]
Okay, let's tackle this problem. I need to create a difficult question based on the provided reading material about logarithmic and quadratic functions. The question should be designed with a specific background, educational level, and require the application of the concepts from the text. Let me start by understanding the core concepts from the reading material.

The reading material covers logarithmic functions, their properties, solving inequalities and equations involving logarithms, quadratic equations, and function composition. The key points include the definition of logarithmic functions, their properties like the change of base formula, solving logarithmic inequalities, and determining the number of solutions using the discriminant analysis.

The question should be challenging, so it needs to integrate these concepts in a way that requires multiple steps and possibly some creative thinking. Let's think about the structure of the question. It should have two parts, as shown in the final question provided. Part (1) asks for the range of a when f(1) < 2, and part (2) discusses the number of zeros of the function g(x) = f(x) - log₂[(a-4)x + 2a - 5].

First, let's focus on part (1). The function f(x) is given as log₂(1/x + a). To find the range of a when f(1) < 2, we need to substitute x = 1 into the function and solve the inequality. This will involve solving log₂(1 + a) < 2, which simplifies to 0 < 1 + a < 4, leading to a ∈ (-1, 3). This part is straightforward but requires understanding of logarithmic inequalities and their properties.

Now, let's move to part (2). The function g(x) is defined as f(x) - log₂[(a-4)x + 2a - 5]. To find the number of zeros of g(x), we need to set g(x) = 0 and solve the resulting equation. This will involve equating log₂(1/x + a) to log₂[(a-4)x + 2a - 5], which simplifies to 1/x + a = (a-4)x + 2a - 5. Rearranging this equation, we get a quadratic equation in terms of x: (a-4)x² + (a-5)x - 1 = 0.

The number of zeros of g(x) depends on the number of solutions to this quadratic equation, which can be determined using the discriminant. The discriminant Δ = (a-5)² + 4(a-4) = a² - 10a + 25 + 4a - 16 = a² - 6a + 9 = (a-3)². The discriminant is always non-negative, so the quadratic equation has real solutions. However, we need to consider the domain of the logarithmic functions involved, which requires that the arguments of the logarithms be positive.

Let's analyze the quadratic equation (a-4)x² + (a-5)x - 1 = 0. The solutions are given by the quadratic formula: x = [-(a-5) ± sqrt((a-3)²)] / 2(a-4). This simplifies to x = [5-a ± (a-3)] / 2(a-4). Therefore, the solutions are x = 1 or x = -1/(a-4).

Now, we need to check the domain of the logarithmic functions. For log₂(1/x + a) to be defined, 1/x + a > 0, and for log₂[(a-4)x + 2a - 5] to be defined, (a-4)x + 2a - 5 > 0.

Let's consider the cases for a:

1. When a = 4, the quadratic equation becomes -x -1 = 0, so x = -1. We need to check if this solution is valid. For x = -1, 1/x + a = -1 + 4 = 3 > 0, and (a-4)x + 2a - 5 = 0 + 8 - 5 = 3 > 0. Both conditions are satisfied, so x = -1 is a valid solution. Therefore, when a = 4, there is one zero.

2. When a = 3, the quadratic equation becomes -x² -2x -1 = 0, which factors to -(x+1)² = 0, so x = -1. We need to check the domain. For x = -1, 1/x + a = -1 + 3 = 2 > 0, and (a-4)x + 2a - 5 = -1 + 6 - 5 = 0. Since the argument of the logarithm must be positive, x = -1 is not a valid solution. Therefore, when a = 3, there are no zeros.

3. When a ≠ 3 and a ≠ 4, the quadratic equation has two solutions: x = 1 and x = -1/(a-4). We need to check the domain for these solutions.

For x = 1, 1/x + a = 1 + a > 0, and (a-4)x + 2a - 5 = a - 4 + 2a - 5 = 3a - 9 > 0. This requires a > 3.

For x = -1/(a-4), we need to ensure that 1/x + a > 0 and (a-4)x + 2a - 5 > 0.

1. For 1/x + a > 0: 1/x + a > 0 → 1 + ax > 0 → ax > -1. Since x = -1/(a-4), this becomes -1/(a-4) * a > -1 → -a/(a-4) > -1 → a/(a-4) < 1. This inequality holds for a < 4 or a > 5.

2. For (a-4)x + 2a - 5 > 0: (a-4)(-1/(a-4)) + 2a - 5 > 0 → -1 + 2a - 5 > 0 → 2a - 6 > 0 → a > 3.

Combining these conditions, we get a > 3 and a ≠ 4. Therefore, when a > 3 and a ≠ 4, there are two zeros.

Now, let's summarize the results:

- When a ≤ 4/5, there are no zeros.
- When 4/5 < a ≤ 1, there is one zero.
- When a > 1 and a ≠ 3 and a ≠ 4, there are two zeros.
- When a = 4, there is one zero.
- When a = 3, there are no zeros.

This is a bit complex, but it covers all the cases. Now, let's think about how to present this as a question. The question should be designed to test the understanding of logarithmic functions, quadratic equations, and domain restrictions. The background scenario can be a mathematical problem or a real-world application, but since it's a mathematical question, perhaps a theoretical problem is appropriate.

The question should be structured with two parts, as in the final question provided. Part (1) is a straightforward inequality solving, while part (2) requires solving a quadratic equation and analyzing the domain restrictions. This ensures that the question tests both algebraic manipulation and conceptual understanding.

Now, let's ensure that the question is challenging. The key difficulties are:

1. Solving the inequality for part (1) requires understanding logarithmic inequalities and their properties.
2. Solving the quadratic equation for part (2) requires applying the quadratic formula and analyzing the domain restrictions.
3. The number of zeros depends on the discriminant and the domain of the logarithmic functions, which requires careful consideration of the conditions.

To make the question more difficult, we can add some red herrings or require the student to consider multiple cases. For example, the student must consider the domain restrictions for each solution of the quadratic equation, which can be a source of errors if not carefully analyzed.

The final question should be presented in the specified format, with the answer in a box. The thinking process should be detailed, showing the step-by-step reasoning from the core concepts to the final answer.

Let me write down the thinking process:

1. The core concepts are logarithmic functions, quadratic equations, and domain restrictions.
2. Part (1) requires solving log₂(1/a + 1) < 2, which simplifies to 0 < 1/a + 1 < 4, leading to a ∈ (-1, 3).
3. Part (2) involves solving the quadratic equation (a-4)x² + (a-5)x - 1 = 0 and analyzing the domain restrictions for each solution.
4. The number of zeros depends on the discriminant and the domain of the logarithmic functions.
5. The student must consider the cases a = 4, a = 3, and a ≠ 3,4, and analyze the domain for each solution.

The final answer for part (1) is a ∈ (-1, 3), and for part (2), the number of zeros is determined by the cases mentioned above.

Now, let's write the question in the required format.


[输出]
### 一、大模型出题思考过程详细分析
#### 1. 步骤完整性分析（核心维度1）
根据评分规范，完整流程需覆盖“需求定位→设计（场景/题干/难点/答案解析）→试做反馈→迭代优化→定稿”，核心环节缺失则降档。  
- **需求定位**：明确学科为数学，核心知识点为“对数函数性质、二次方程求解、定义域限制、判别式分析”，但**未明确学段**（如高中数学）、**未清晰说明考察目标**（如“考察分类讨论能力与定义域意识”），且无知识点选择理由（如未说明为何结合对数与二次函数，是否针对学生常见薄弱点）。  
- **设计环节**：  
  - 场景设计：题干为纯理论数学问题（无生活场景如“函数模型拟合”、学术场景如“数据分析”），属于**抽象表述，无场景设计**，不符合“场景存在（具体语境）”的基础要求。  
  - 题干设计：采用“Part1求范围+Part2分析零点”的两问结构，表述无歧义，但未说明“先易后难”的表述逻辑依据（如为何先考察不等式再考察分类讨论）。  
  - 难点/易错点设计：隐含“定义域忽略”“分类讨论遗漏（a=3、a=4特殊情况）”两个易错点，但**未明确易错点数量**，也未说明设计目的（如“设计a=3的特殊情况是因为学生常忽略对数真数为0的无效性”）。  
  - 答案/解析设计：给出详细解题步骤（如Part1解对数不等式、Part2分情况讨论二次方程解），但解析为“步骤堆砌”，**无分层逻辑说明**（如未解释“先求方程解再验定义域”的分层必要性）。  
- **试做反馈**：完全缺失，未提及试做人群（如“高二学生20人”）、正确率、错误类型及占比（如“60%学生漏验定义域”）。  
- **迭代优化**：因无试做数据，无任何调整动作（如“修改题干提示定义域”）及调整依据，环节缺失。  
- **定稿**：直接以“写出最终问题”定稿，无定稿判断标准（如“正确率达标”），也无二次试做验证。  

综上，步骤完整性仅覆盖“需求定位（部分要素缺失）→基础设计（题干+答案）”，缺失“试做反馈、迭代优化”核心环节，且设计环节中场景缺失，不符合3分档“基础流程完整（需求定位+完整设计）”要求，更贴近2分档“基础设计残缺”的界定。

#### 2. 细节详实度分析（核心维度2）
各环节需包含“设计依据/逻辑说明”，无说明则降分。  
- **需求定位**：未明确学段、考察目标，且无知识点选择理由，比2分档“明确学科、学段、知识点，未说明考察目标”的要求更残缺（缺少学段）。  
- **设计环节**：  
  - 场景：无场景，无任何场景适配逻辑说明，不符合“有场景但无依据”的情况，属于“场景空白”。  
  - 题干：无表述逻辑依据，符合2分档“未说明依据”的情况。  
  - 难点：有易错点但无设计目的，符合2分档“无依据说明”的情况。  
  - 解析：无分层逻辑依据，符合2分档“无解析逻辑说明”的情况。  
- **试做、迭代**：无任何数据或调整说明，细节空白。  

综上，细节详实度多环节缺失（需求定位要素不全、设计无依据），仅满足“题干+答案完整”的最低要求，符合2分档“细节空白”的界定。

#### 3. 反馈闭环性分析（核心维度3）
需形成“试做→错误分析→调整→验证”闭环，无闭环则降档。  
思考过程无试做数据，自然无错误分析、调整动作及二次验证，完全无闭环，符合2分档“无试做迭代”的要求，无法达到4分及以上“有反馈或闭环”的标准。


### 二、评分结论
该大模型出题思考过程符合2分档（不足：基础设计残缺+细节空白）的核心界定：仅覆盖“需求定位（部分要素缺失）→基础设计（题干+答案）”，场景缺失、难点设计不完整，无任何细节依据说明，且无试做反馈与迭代优化环节。

```json
{
  "得分": 2
}
```




[输入]
Okay, let's tackle this problem. I need to create a difficult question based on the provided reading material about topological spaces, fundamental groups, and homotopy. The question should be designed according to the given guidelines, with a detailed thinking process and a final question in the specified format.

First, I'll review the reading material to understand the key concepts. The material covers topological spaces, continuous functions, fundamental groups, and homotopy. The example of the punctured torus is mentioned, and the question is to show that it deformation retracts to a union of two circles intersecting at one point. The answer is S¹ ∨ S¹, which is the wedge sum of two circles.

Now, I need to design a question that tests the understanding of deformation retraction and fundamental groups. The question should be challenging, requiring the student to apply the concepts of homotopy, fundamental groups, and deformation retraction.

Let's start by thinking about the background scenario. The punctured torus is a common topological space, and its deformation retraction to a wedge sum of two circles is a well-known result. The question should present this in a way that requires the student to justify the deformation retraction using the fundamental group.

The thinking process should involve several steps:
1. Understanding the punctured torus and its topological properties.
2. Knowing that a deformation retraction preserves the fundamental group.
3. Computing the fundamental group of the punctured torus.
4. Showing that the fundamental group of the punctured torus is isomorphic to the fundamental group of a wedge sum of two circles.
5. Concluding that the punctured torus deformation retracts to a wedge sum of two circles.

Now, let's consider the difficulty aspects. The student needs to apply the concept of deformation retraction, which is a homotopy equivalence that deformation retracts one space onto another. They also need to compute the fundamental group of the punctured torus, which is a non-trivial task. The fundamental group of a punctured torus is the free group on two generators, which is isomorphic to the fundamental group of a wedge sum of two circles.

The question should be presented in a way that requires the student to derive the fundamental group of the punctured torus and then show that it is isomorphic to the fundamental group of a wedge sum of two circles. This involves understanding the Seifert-van Kampen theorem or other methods to compute the fundamental group.

Let's structure the question step by step:

1. **Question Stem**: Ask the student to show that the punctured torus deformation retracts to a union of two circles intersecting at one point.
2. **Background Knowledge**: The student needs to know about deformation retraction, fundamental groups, and the fundamental group of a punctured torus.
3. **Key Steps**:
   - Compute the fundamental group of the punctured torus.
   - Show that this fundamental group is isomorphic to the fundamental group of a wedge sum of two circles.
   - Conclude that the punctured torus deformation retracts to a wedge sum of two circles.

The answer is S¹ ∨ S¹, so the student should provide this as the final answer.

Now, let's think about the thinking process. The student should start by recalling the definition of a deformation retraction. They need to understand that a deformation retraction is a homotopy equivalence that deformation retracts one space onto another. They should also know that the fundamental group of a deformation retracted space is isomorphic to the fundamental group of the base space.

The student should compute the fundamental group of the punctured torus. This can be done using the Seifert-van Kampen theorem. The punctured torus can be represented as a quotient space of the plane with a disk removed and identified with the torus. Alternatively, it can be thought of as a torus with a point removed.

The fundamental group of a torus is Z², and removing a point from a torus results in a space with fundamental group Z². However, the punctured torus is homeomorphic to a wedge sum of two circles, so its fundamental group is the free group on two generators, which is isomorphic to Z².

Therefore, the student should show that the fundamental group of the punctured torus is Z², and then conclude that it deformation retracts to a wedge sum of two circles.

The question should be presented in a way that requires the student to perform these steps, and the answer should be the wedge sum of two circles.

Now, let's write the final question and answer in the specified format.

Question: Show that the punctured torus \( T \setminus \{p\} \) deformation retracts to a union of two circles intersecting at exactly one point.
Answer: S¹ ∨ S¹

The thinking process should be detailed, showing the student's understanding of deformation retraction, fundamental groups, and the computation of the fundamental group of the punctured torus.


[输出]
### 一、大模型出题思考过程详细分析
#### 1. 步骤完整性分析（对应评分规范核心维度1）
该思考过程仅覆盖“需求定位→设计（场景/题干/难点/答案）”基础流程，**缺失“试做反馈”“迭代优化”“定稿确认”三个核心环节**：
- 需求定位：明确了数学学科下的拓扑学相关知识点（变形收缩、基本群、同伦），考察目标为“测试对变形收缩和基本群的理解”，但未明确学段（如本科高年级、研究生），且未说明知识点选择理由（如“为何选择穿孔环面作为考察对象，而非其他拓扑空间”）；
- 设计环节：包含场景（穿孔环面）、题干（证明穿孔环面变形收缩至两圆交一点）、难点（计算穿孔环面基本群、应用Seifert-van Kampen定理）、答案（S¹∨S¹），设计要素完整，但无后续试做数据采集、错误分析及迭代调整；
- 无试做反馈（未提及试做人群、正确率、错误类型）、无迭代优化（无基于反馈的调整动作）、无定稿确认（未给出定稿判断标准，直接呈现题目+答案）。

#### 2. 细节详实度分析（对应评分规范核心维度2）
各设计环节均**缺失关键依据说明**，符合3分档“设计细节部分缺失”的特征：
- 场景设计：选择“穿孔环面”作为场景，但未说明场景与知识点的适配逻辑（如“穿孔环面的拓扑性质与变形收缩考点的关联的合理性”）；
- 题干设计：采用“证明类”表述，但未说明表述逻辑依据（如“为何按‘结论→要求’的结构组织题干，而非‘背景→问题’的递进式表述”）；
- 难点设计：提及2个难点，但未说明难点设计目的（如“为何将‘应用Seifert-van Kampen定理’设为难点，是否因该定理是学生常见知识漏洞”）；
- 答案/解析设计：仅给出最终答案（S¹∨S¹），解析为纯描述性内容（如“学生应回忆变形收缩定义、计算基本群”），未形成“解题步骤→关键思路→易错点提醒”的分层结构，且无解析逻辑说明。

#### 3. 反馈闭环性分析（对应评分规范核心维度3）
因完全缺失“试做反馈”环节，**未形成任何反馈闭环**：无试做数据采集（无人群、正确率、错误类型），自然无错误归因、针对性调整及二次验证，不符合4分档“有迭代但无二次验证”或5分档“完整闭环”的要求。


#### 4. 分数档匹配判断
结合上述分析，该思考过程符合**3分（合格）** 档的核心界定：
- 步骤完整性：仅覆盖“需求定位→完整设计”，缺试做、迭代、定稿环节，匹配3分档“基础流程完整+试做环节缺失”；
- 细节详实度：需求定位未说明知识点选择理由，设计环节各要素均无依据说明，解析为非分层结构，匹配3分档“设计细节部分缺失”；
- 反馈闭环性：无反馈闭环，符合3分档“无试做迭代”的特征，无降档至2分/1分的条件（2分需“基础设计残缺”，1分需“无场景/无难点”，本思考过程设计要素完整）。


```json
{
  "得分": 3
}
```


"""


QUESTION_DESIGN_THOUGHT_JUDGE_TEMPLATE = """
现在需要你对下面的思考过程进行打分（1-5分）


[输入]
{doc}

[输出]
"""


REPETITION_DETECTION_FEWSHOTS = """
# 内容重复现象检测
## 一、任务背景说明
给你一段大模型出题的思考过程要求判断思考过程是否包含严重的内容重复现象。按照下面的格式返回你的结果

#### 输出要求
[先简短分析]
```json
{
  "内容重复": "是/否"
}
```

下面是一个例子



[输入]
The scores of 100 students in a test out of 10 were:

| Score | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|---|---|---|---|---|---|---|----|
| Frequency | 5 | 11 | 19 | 24 | 21 | 12 | 6 | 2 |

State the range of the scores.

The answer is 7.

Now, let's think about the thinking process:

1. The student reads the scenario and understands that they need to calculate the range of the scores.
2. The student looks at the table and identifies the scores and their frequencies.
3. The student needs to find the highest and lowest scores from the table.
4. The student identifies the highest score as 10 and the lowest score as 3.
5. The student subtracts the lowest score from the highest score to get the range: 10 - 3 = 7.
6. The student checks if there are any distractor scores, such as the score of 11, and confirms that it is not used in the calculation.

To make it more difficult, I can add a step where the student needs to calculate the range from the table, but there might be a mistake in the table. For example, the table might have a mistake in the highest or lowest score, or the student needs to check the table for the highest and lowest scores.

Let's think about the final question again:

The scores of 100 students in a test out of 10 were:

| Score | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|---|---|---|---|---|---|---|----|
| Frequency | 5 | 11 | 19 | 24 | 21 | 12 | 6 | 2 |

State the range of the scores.

The answer is 7.

Now, let's think about the thinking process:

1. The student reads the scenario and understands that they need to calculate the range of the scores.
2. The student looks at the table and identifies the scores and their frequencies.
3. The student needs to find the highest and lowest scores from the table.
4. The student identifies the highest score as 10 and the lowest score as 3.
5. The student subtracts the lowest score from the highest score to get the range: 10 - 3 = 7.
6. The student checks if there are any distractor scores, such as the score of 11, and confirms that it is not used in the calculation.

To make it more difficult, I can add a step where the student needs to calculate the range from the table, but there might be a mistake in the table. For example, the table might have a mistake in the highest or lowest score, or the student needs to check the table for the highest and lowest scores.

Let's think about the final question again:

The scores of 100 students in a test out of 10 were:

| Score | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|---|---|---|---|---|---|---|----|
| Frequency | 5 | 11 | 19 | 24 | 21 | 12 | 6 | 2 |

State the range of the scores.

The answer is 7.

Now, let's think about the thinking process:

1. The student reads the scenario and understands that they need to calculate the range of the scores.
2. The student looks at the table and identifies the scores and their frequencies.
3. The student needs to find the highest and lowest scores from the table.
4. The student identifies the highest score as 10 and the lowest score as 3.
5. The student subtracts the lowest score from the highest score to get the range: 10 - 3 = 7.
6. The student checks if there are any distractor scores, such as the score of 11, and confirms that it is not used in the calculation.

To make it more difficult, I can add a step where the student needs to calculate the range from the table, but there might be a mistake in the table. For example, the table might have a mistake in the highest or lowest score, or the student needs to check the table for the highest and lowest scores.

Let's think about the final question again:

The scores of 100 students in a test out of 10 were:

| Score | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|---|---|---|---|---|---|---|----|
| Frequency | 5 | 11 | 19 | 24 | 21 | 12 | 6 | 2 |

State the range of the scores.

The answer is 7.

Now, let's think about the thinking process:

1. The student reads the scenario and understands that they need to calculate the range of the scores.
2. The student looks at the table and identifies the scores and their frequencies.
3. The student needs to find the highest and lowest scores from the table.
4. The student identifies the highest score as 10 and the lowest score as 3.
5. The student subtracts the lowest score from the highest score to get the range: 10 - 3 = 7.
6. The student checks if there are any distractor scores, such as the score of 11, and confirms that it is not used in the calculation.

To make it more difficult, I can add a step where the student needs to calculate the range from the table, but there might be a mistake in the table. For example, the table might have a mistake in the highest or lowest score, or the student needs to check the table for the highest and lowest scores.

Let's think about the final question again:

The scores of 100 students in a test out of 10 were:

| Score | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|---|---|---|---|---|---|---|----|
| Frequency | 5 | 11 | 19 | 24 | 21 | 12 | 6 | 2 |

State the range of the scores.

The answer is 7.


[输出]
[输入内容多次重复呈现相同的题目（含表格）、答案“7”以及完全一致的思考过程（6个步骤），且重复段落中“To make it more difficult”后的建议也无变化，未添加任何新信息，存在严重的内容重复现象。]
```json
{
  "内容重复": "是"
}
```



"""

REPETITION_DETECTION_TEMPLATE = """

现在需要你对下面的思考过程进行判断是否存在内容重复


[输入]
{doc}

[输出]
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


class RepetitionDetection(BatchCallOpenAPI):
    def __init__(self, prompt_field="prompt", answer_field="answer"):
        self.prompt_field = prompt_field
        self.answer_field = answer_field

    @classmethod
    def task_desc(cls):
        return "内容重复"

    def prompt_fn(self, example):
        prompt = REPETITION_DETECTION_FEWSHOTS + REPETITION_DETECTION_TEMPLATE.format(
            doc=example["doc"],

        )
        return prompt

    def postprocess(self, response: str):
        s = response
        try:
            conclusion = s.strip()
            judge = re.findall(
                r'\"内容重复\": \"(.*)\"', conclusion)
            if len(judge) > 0 and judge[0] in ("是", "否"):
                return judge[0] == "是"
        except Exception as err:
            raise PostprocessError(f'{err}')


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


class QuestionDesignJudge(BatchCallOpenAPI):
    def __init__(self, prompt_field="prompt", answer_field="answer"):
        self.prompt_field = prompt_field
        self.answer_field = answer_field

    @classmethod
    def task_desc(cls):
        return "思维打分"

    def prompt_fn(self, example):
        prompt = QUESTION_DESIGN_THOUGHT_JUDGE_FEWSHOTS + QUESTION_DESIGN_THOUGHT_JUDGE_TEMPLATE.format(
            doc=example["doc"],

        )
        return prompt

    def postprocess(self, response: str):
        s = response
        try:
            score = int(re.findall(
                r'\"得分\": (\d+)', s)[0].strip())

            if score not in (1, 2, 3, 4, 5):
                raise PostprocessError(f'invalid similarity score={score}')
            return score
        except Exception as err:
            raise PostprocessError(f'{err}')


class HintLeakageDetection(BatchCallOpenAPI):
    def __init__(self, prompt_field="prompt", answer_field="answer"):
        self.prompt_field = prompt_field
        self.answer_field = answer_field

    @classmethod
    def task_desc(cls):
        return "提示泄漏"

    def prompt_fn(self, example):
        prompt = HINT_LEAKAGE_DETECTION_FEWSHOTS + HINT_LEAKAGE_DETECTION_TEMPLATE.format(
            doc=example["doc"],

        )
        return prompt

    def postprocess(self, response: str):
        s = response
        try:
            score = int(re.findall(
                r'\"符合要求得分\": (\d+)', s)[0].strip())

            if score not in (1, 2, 3, 4, 5):
                raise PostprocessError(f'invalid similarity score={score}')
            return score
        except Exception as err:
            raise PostprocessError(f'{err}')


class DocContainsQuestion(BatchCallOpenAPI):
    def __init__(self, prompt_field="prompt", answer_field="answer"):
        self.prompt_field = prompt_field
        self.answer_field = answer_field

    @classmethod
    def task_desc(cls):
        return "是否题目泄漏"

    def prompt_fn(self, example):
        prompt = LECTURE_CONTAIN_QUESTION_FEWSHOTS + LECTURE_CONTAIN_QUESTION_TEMPLATE.format(
            doc=example["doc"],
            question=example["question"],
            solution=example["solution"],
        )
        return prompt

    def postprocess(self, response: str):
        s = response
        try:
            conclusion = s.strip()
            judge = re.findall(
                r'\"题目泄漏\": \"(.*)\"', conclusion)
            if len(judge) > 0 and judge[0] in ("是", "否"):
                return judge[0] == "是"

            conclusion = conclusion[conclusion.index(
                "```json")+len("```json"):].strip()
            conclusion = conclusion[:conclusion.index("```")].strip()
            try:
                conclusion = json.loads(conclusion)
                if conclusion["题目泄漏"] not in ("是", "否"):
                    raise PostprocessError(f'corrupt')
                return conclusion["题目泄漏"] == "是"
            except Exception as err:
                try:
                    conclusion = re.findall(
                        r'\"题目泄漏\": \"(.*)\"', conclusion)[0]
                    if not conclusion in ("是", "否"):
                        raise PostprocessError(f'corrupt')
                    return conclusion == "是"
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


def general_qa_parse_solution_fn(solution_str: str, extract_question_fn=parse_question_solution_fn):
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


class LengthReward(PenaltyOrReward):
    def __init__(self, parse_solution_fn, min_score, max_score, abbrev="Length", max_token=8192):
        super().__init__(
            parse_solution_fn=parse_solution_fn, min_score=min_score, max_score=max_score, abbrev=abbrev
        )
        self.max_token = max_token
        self.max_reward = 0.65

    def get_penalty_or_reward(self, solution_str, ground_truth):
        lang_code = ground_truth["lang_code"]
        solution_str = self.parse_solution_fn(solution_str)

        if solution_str is None:
            return 0.0

        doc = solution_str[1]

        base_score = 0.0
        tokens = tokenize(doc, lang_code)
        return min(len(tokens), self.max_token) / self.max_token * self.max_reward


Process = namedtuple("Process", "name,function,is_async")


class Doc2QuerySelfTaughtComputeScore(object):
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
        self.task_name = "DOC2QUERY_SELF_TAUGHT"
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
        self.init_question_judge_agent()

    def init_verify_agent(self):
        self.verify_agent = Agent(
            **self.args["verify_agent"]["model"])

    def init_question_judge_agent(self):
        self.question_judge_agent = Agent(
            **self.args["question_judge_agent"]["model"])

    def init_rm_agent(self):
        self.rm_agent = RewardModelAgent(
            urls=self.args["reward_model_args"]["urls"],
            postprocess_solution_fn=lambda x: self.parse_thought_and_conclusion_fn(x)[
                0],
            parse_result_failure_score=self.min_reward
        )

    def thought_similarity_reward(
        self,
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
    ):
        rewards = []
        for solution_str, gt in zip(batch_solution_str, batch_ground_truth):
            solution_str = self.parse_thought_and_conclusion_fn(solution_str)

            if solution_str is None:
                rewards.append(0.0)
            try:
                thought = solution_str[0]
                lang_code = gt["lang_code"]
                gt_tokens = [" ".join(tokenize(ref.lower(), lang_code))
                             for ref in gt["references"]]
                sl_tokens = " ".join(tokenize(thought.lower(), lang_code))
                bleu = sacrebleu.sentence_bleu(sl_tokens, gt_tokens).score
                similarity = bleu / 100
            except Exception as err:
                similarity = 0.0
            rewards.append(similarity)
        return rewards

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

    async def repetition_detect(
        self,
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
    ):
        task = RepetitionDetection()
        indices = []
        batch_inputs = []

        result_index2queue_index = {}
        for i, (gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            result = self.parse_solution_fn(sol)
            if result is not None and gt.get("question", None):
                batch_inputs.append({"doc": result[1]})
                indices.append(i)
                result_index2queue_index[len(indices)-1] = i
            else:
                continue

        evaluations = await task.do_job(
            agent=self.verify_agent,
            batch_inputs=batch_inputs,
            max_concurrent_requests=self.args["verify_agent"]["max_concurrent_requests"],
        )
        full_rewards = [0] * len(batch_solution_str)
        for j, eval_result in enumerate(evaluations):
            queue_index = result_index2queue_index[j]
            if eval_result:
                full_rewards[queue_index] = -2.0
        return full_rewards

    async def design_judge_reward(
        self,
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
    ):
        task = QuestionDesignJudge()
        indices = []
        questions = []

        for i, (gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            result = self.parse_thought_and_conclusion_fn(sol)
            lang_code = gt["lang_code"]

            if result is not None:
                tokens = tokenize(result[0], lang_code)
                if len(tokens) < 1000:
                    continue
                questions.append({"doc": result[0]})
                indices.append(i)
            else:
                continue

        quality = await task.do_job(
            agent=self.question_judge_agent,
            batch_inputs=questions,
            max_concurrent_requests=self.args["question_judge_agent"]["max_concurrent_requests"],
        )

        scores = [0.0] * len(batch_solution_str)
        for _quality, index in zip(quality, indices):
            if _quality is None:
                pass
            else:
                scores[index] = 1.0 * (_quality-1.0) / 4
        return scores

    async def hint_leakage_reward(
        self,
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
    ):
        task = HintLeakageDetection()
        indices = []
        questions = []

        for i, (gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            result = self.parse_thought_and_conclusion_fn(sol)
            lang_code = gt["lang_code"]

            if result is not None:
                tokens = tokenize(result[0], lang_code)
                if len(tokens) < 800:
                    continue
                questions.append({"doc": result[0]})
                indices.append(i)
            else:
                continue

        quality = await task.do_job(
            agent=self.verify_agent,
            batch_inputs=questions,
            max_concurrent_requests=self.args["verify_agent"]["max_concurrent_requests"],
        )

        scores = [0.0] * len(batch_solution_str)
        for _quality, index in zip(quality, indices):
            if _quality is None:
                pass
            else:
                scores[index] = 1.0 * (_quality-1.0) / 4
        return scores

    def get_processes(self):
        return [
            Process(name="rep",
                    function=self.repetition_detect, is_async=True),
            Process(name="ans_cons",
                    function=self.same_answer_reward, is_async=True),
            Process(name="q_cons",
                    function=self.question_similarity_reward, is_async=True),
            Process(name="rm", function=partial(self.rm_agent.compute_rm_score,
                    judge_prompt_key="rm_judge_prompt"), is_async=False),
            Process(name="hint_leakage",
                    function=self.hint_leakage_reward, is_async=True),
            Process(name="cot", function=self.design_judge_reward, is_async=True)
        ]

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

        rewards_results = {}
        for process in self.get_processes():
            if process.is_async:
                results = await process.function(
                    batch_data_sources,
                    batch_solution_str,
                    batch_ground_truth,
                )
            else:
                results = process.function(
                    batch_data_sources,
                    batch_solution_str,
                    batch_ground_truth,
                )
            rewards_results[process.name] = results

        final_results = []
        for i in range(len(batch_solution_str)):
            scores = copy.deepcopy(penalty[i])
            penalties = ["Parse"]+[_.abbrev for _ in self._penalties]
            penalty_log_str = "/".join([f'{p}={s:.3f}' for p,
                                        s in zip(penalties, scores)])

            reward_log_str = []
            for k, v in rewards_results.items():
                _reward = v[i]
                scores.append(_reward)
                reward_log_str.append(f'{k}={_reward:.3f}')
            reward_log_str = "; ".join(reward_log_str)

            cur_score = np.sum(scores)
            final_results.append(cur_score)

            log_flag = f"[{self.task_name} VALID]" if self.split == "valid" else f"[{self.task_name} TRAIN]"
            log = False
            if cur_score <= self.min_reward:
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
                f'[Final Reward]={cur_score:.3f}({reward_log_str})|{penalty_log_str}\n')

            if log:
                print(f'[Thought]\n{batch_solution_str[i]}')
                print()
        return final_results

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
            return repr(self.clip_string(solution))
        return norm[0]

    def clip_string(self, s: str):
        if len(s) > 1500:
            return f'{s[:700]}... [省略] ...{s[-800:]}'
        return s

    def log_ground_truth(self, ground_truth):
        return ground_truth["question"]


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Query2Doc
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def parse_doc_fn(solution_str: str):
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
        conclusion = re.findall(r'\[LECTURE\](.*)\[/LECTURE\]',
                                solution_str, re.DOTALL)[0]
    except Exception as err:
        return None

    if "[LECTURE]" in conclusion or "[/LECTURE]" in conclusion:
        return None
    return thought, conclusion


class Query2DocComputeScore(Doc2QuerySelfTaughtComputeScore):
    def __init__(self,
                 parse_solution_fn,
                 split="train",
                 args=None,
                 min_reward=-2.0,
                 parse_thought_and_conclusion_fn=parse_question_solution_fn,
                 thought_log_prob=0.05
                 ):

        super().__init__(
            parse_solution_fn=parse_solution_fn, split=split,
            args=args,
            min_reward=min_reward,
            parse_thought_and_conclusion_fn=parse_doc_fn,
            thought_log_prob=thought_log_prob,
        )
        self.task_name = "QUERY2DOC"

    def init_agent(self):
        self.agents = {}
        self.init_verify_agent()
        self.init_solver_agent()
        self.init_rm_agent()

    @classmethod
    def rule_based_penalties(cls):
        return [
            LengthReward,
        ]

    def init_solver_agent(self):
        self.solver_agent = Agent(
            **self.args["learnable_args"]["model"])
        self.agents["learnable_args"] = self.solver_agent

    def init_rm_agent(self):
        self.rm_agent = RewardModelAgent(
            urls=self.args["reward_model_args"]["urls"],
            postprocess_solution_fn=lambda x: self.parse_thought_and_conclusion_fn(x)[
                1],
            parse_result_failure_score=self.min_reward
        )

    async def get_learnable_reward(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            skip_run=None):

        correctness = await self.simulate_respondent(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            skip_run=skip_run,
        )

        full_rewards, pass_rates = [], []
        run_args = self.args["learnable_args"]

        for i in range(len(batch_solution_str)):
            if i in list(correctness.values())[0]:
                base_score = 0.0
                pass_rates.append({
                    k: f'{np.sum(v[i])}/{len(v[i])}' for k, v in correctness.items()
                })

                try:
                    base_score = np.mean([
                        np.mean(v[i]) for k, v in correctness.items()
                    ])
                    full_rewards.append(base_score * run_args["weight"])
                except Exception as err:
                    print(f'[ERROR] {err}')
                    full_rewards.append(base_score)
            else:
                pass_rates.append({})
                full_rewards.append(0.0)
        return full_rewards, pass_rates

    def response_postprocess(self, s):
        if "</think>" in s:
            s = s[s.index("</think>")+len("</think>"):]

        return s

    async def simulate_respondent(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            skip_run=None
    ):
        verify_task = AnswerConsistency()
        return await self._simulate_respondent(
            batch_data_sources=batch_data_sources,
            batch_solution_str=batch_solution_str,
            batch_ground_truth=batch_ground_truth,
            skip_run=skip_run,
            run_args={"learnable_args": self.args["learnable_args"]},
            batch_verify_fn=partial(
                self.batch_verify_results, verify_task=verify_task),
            resp_postprocess_fn=self.response_postprocess
        )

    @classmethod
    def respond_w_context(cls, result, gt):
        prompt = f'[LECTURE]\n{result[1]}\n[/LECTURE]\n\n{gt["question"]}'
        return prompt

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

                skip = False
                for module in self._penalties:
                    cur_score = module.get_penalty_or_reward(
                        solution_str, gt
                    )
                    if cur_score < 0.0:
                        skip = True
                        break
                if skip:
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
                desc=f'[{run_args[name]["desc"]} {run_args[name]["model"]["model"]} 解题]',
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
                        # FIXME: Be Carefull
                        response=batch_ground_truth[index]["question"],
                        extra=r,
                        ground_truth=batch_ground_truth[index]["answer"]))

        correctness = await batch_verify_fn(
            verify_queue=verify_queue,
            max_concurrent_requests=self.args["verify_agent"]["max_concurrent_requests"],
            group_names=task_names
        )
        return correctness

    async def question_leakage(
        self,
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
    ):
        task = DocContainsQuestion()
        indices = []
        batch_inputs = []

        result_index2queue_index = {}
        for i, (gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            result = self.parse_solution_fn(sol)
            if result is not None and gt.get("question", None):
                batch_inputs.append(
                    {"doc": result[1], "question": gt["question"], "solution": gt["answer"]})
                indices.append(i)
                result_index2queue_index[len(indices)-1] = i
            else:
                continue

        evaluations = await task.do_job(
            agent=self.verify_agent,
            batch_inputs=batch_inputs,
            max_concurrent_requests=self.args["verify_agent"]["max_concurrent_requests"],
        )

        full_rewards = [1.5] * len(batch_solution_str)
        for j, eval_result in enumerate(evaluations):
            queue_index = result_index2queue_index[j]
            if eval_result:
                full_rewards[queue_index] = 0.0
        return full_rewards

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
                    (solver_response, example.extra, example.ground_truth))

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
                        (1.0 if eval_result else 0.0, eval_input[0]))
                else:
                    correctness[queue_elem.tag][queue_elem.index].append(
                        1.0 if eval_result else 0.0)
        return correctness

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

        reward1, extra_info = await self.get_learnable_reward(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )
        reward2 = await self.question_leakage(
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
                log_flag = f"[{self.task_name} VALID CORRUPT RESPONSE]" if self.split == "valid" else f"[{self.task_name} TRAIN CORRUPT RESPONSE]"

            if random.random() < self.thought_log_prob:
                log = True

            source = batch_ground_truth[i]["source"]

            print(
                f"--------------------------------{log_flag}--------------------------------")
            print(
                f"【Golden{i}】({source})`{self.log_ground_truth(batch_ground_truth[i])}`")
            print(
                f'[Final Reward]={cur_score:.3f}(learnable={reward1[i]:.3f};leakage={reward2[i]:.3f};reward={reward3[i]:.3f})|{penalty_log_str}\n')

            if log:
                print(f'[Thought]\n{batch_solution_str[i]}')
                print()
        return final_results

    def log_solution(self, solution):
        norm = self.parse_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return norm[1]


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Query2Doc
# ------------------------------------------------------------------------------------------------------------------------------------------------------
DOC2QUERY_ST_DEFAULT_PARAMS = {
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
        "max_concurrent_requests": 512
    },
    "question_judge_agent": {
        "model": {
            "model": "Qwen2.5-32B-Instruct",
            "base_url": "https://sd262bskcm47j59r1292g.apigateway-cn-beijing.volceapi.com/v1",
            "api_keys": "caa6246b-afbe-4d9b-ab34-87bf9922032b",
            "request_kwargs": {
                "temperature": 0.6,
                "timeout": 360,
                "max_tokens": 4096,
            },
        },
        "max_concurrent_requests": 512
    },
    "similarity_run_args":  {
        "threshold": {
            1: 0.01,
            2: 0.02,
            3: 0.05,
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

QUERY2DOC_DEFAULT_PARAMS = {
    "learnable_args": {
        "model": {
            "model": "Qwen2.5-32B-Instruct",
            "base_url": "https://sd262bskcm47j59r1292g.apigateway-cn-beijing.volceapi.com/v1",
            "api_keys": "caa6246b-afbe-4d9b-ab34-87bf9922032b",
            "request_kwargs": {
                "temperature": 0.8,
                "timeout": 360,
                "max_tokens": 2048,
            },
        },
        "fn": "respond_w_context",
        "repeat": 8,
        "desc": 'RAG回答',
        "max_concurrent_requests": 1024,
        "weight": 1.5,
    },
    "verify_agent": {
        "model": {
            "model": "Qwen2.5-32B-Instruct",
            "base_url": "https://sd262bskcm47j59r1292g.apigateway-cn-beijing.volceapi.com/v1",
            "api_keys": "caa6246b-afbe-4d9b-ab34-87bf9922032b",
            "request_kwargs": {
                "temperature": 0.6,
                "timeout": 360,
                "max_tokens": 2048,
            },
        },
        "max_concurrent_requests": 1024
    },
    "similarity_run_args":  {
        "threshold": {
            1: 0.01,
            2: 0.02,
            3: 0.05,
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


_default_doc2query_self_taught_compute_score_train = Doc2QuerySelfTaughtComputeScore(
    general_qa_parse_solution_fn, split="train", args=DOC2QUERY_ST_DEFAULT_PARAMS)
_default_doc2query_self_taught_compute_score_valid = Doc2QuerySelfTaughtComputeScore(
    general_qa_parse_solution_fn, split="valid", args=DOC2QUERY_ST_DEFAULT_PARAMS)
doc2query_self_taught_compute_score_train = _default_doc2query_self_taught_compute_score_train.compute_score
doc2query_self_taught_compute_score_valid = _default_doc2query_self_taught_compute_score_valid.compute_score


_default_query2doc_compute_score_train = Query2DocComputeScore(
    parse_doc_fn, split="train", args=QUERY2DOC_DEFAULT_PARAMS)
_default_query2doc_compute_score_valid = Query2DocComputeScore(
    parse_doc_fn, split="valid", args=QUERY2DOC_DEFAULT_PARAMS)
query2doc_compute_score_train = _default_query2doc_compute_score_train.compute_score
query2doc_compute_score_valid = _default_query2doc_compute_score_valid.compute_score
