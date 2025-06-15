import re
import sys
import json
import uuid
import copy
import math
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
from typing import Any, Dict, Callable, List
from decimal import Decimal, ROUND_HALF_UP
from tqdm import tqdm as tqdm_nonasync
from collections import namedtuple, defaultdict
from sacremoses import MosesTokenizer, MosesDetokenizer


from openai import OpenAI, RateLimitError, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------
en_mt = MosesTokenizer(lang='en')


RM_URLS = [
    "http://10.130.2.51:25473",
    "http://10.130.2.51:25954",
    "http://10.130.2.51:32560",
    "http://10.130.2.51:33547",
    "http://10.130.2.51:28764",
    "http://10.130.2.51:34113",
    "http://10.130.2.51:33871",
    "http://10.130.2.51:29538",
]

VERIFIER_MODEL_NAME = "qwen25_7B_fabricate_qa_criteria_judge_ehance_0518"
VERIFIER_MODEL_PATH = "http://10.130.133.200:8000/v1"
DEFAULT_PARSE_FAILURE_REWARD = -2.
MAX_CONCURRENT = 192
# MAX_CONCURRENT = 160


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
    if "<|endoftext|>" in solution_str:
        return solution_str[:solution_str.index("<|endoftext|>")].strip()
    return solution_str


async def rm_request_with_retry(urls, data, max_retries=3, retry_delay=5, suffix="/reward"):
    retries = 0
    while retries < max_retries:
        try:
            url = random.choice(urls)
            async with aiohttp.ClientSession() as session:
                async with session.post(f'{url}{suffix}', json=data, timeout=aiohttp.ClientTimeout(total=3000)) as response:
                    response.raise_for_status()
                    return await response.json()
        except (aiohttp.ClientError, aiohttp.ClientResponseError) as e:
            print(f"请求(数据总量={len(data)})失败，错误信息: {e}，重试第 {retries + 1} 次...")
            retries += 1
            if retries < max_retries:
                await asyncio.sleep(retry_delay)
    print(f"达到最大重试次数，请求失败。")
    return None


async def compute_rm_score(
        urls: List[str],
        batch_solution_str,
        batch_ground_truth,
        postprocess_solution_fn,
        parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD,
        judge_prompt_key="ground_truth",
        desc=""
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
        for batch in tqdm_nonasync(batchify(input_datas, n=32), desc=f'[RM{desc}][{urls}] batchify inference (batch=32)'):
            output_datas = await rm_request_with_retry(urls, batch)
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


class PenaltyOrReward(object):
    @abstractmethod
    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Criteria构造
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def criteria_parse_solution_fn(solution_str: str):
    solution_str = postprocess_solution(solution_str)
    try:
        thought = re.findall(r'think>.*</think>',
                             solution_str, re.DOTALL)[0]
    except Exception as err:
        return None

    conclusion = solution_str.replace(thought, "")
    try:
        conclusion = conclusion[conclusion.index(
            "[CRITERIA]"):conclusion.index("[/CRITERIA]")+len("[/CRITERIA]")].strip()
        scores = re.findall(r'\[SCORE=(\d+)\]', conclusion)
        if len(scores) > 20:
            return None
        if not all(int(_) > 0 and int(_) <= 3 for _ in scores):
            return None
    except Exception as err:
        return None
    return conclusion


def get_total_score(criteria):
    scores = re.findall(r'SCORE=(\d+)', criteria)
    scores = [int(_) for _ in scores]
    return sum(scores)


async def criteria_get_score(questions, criteria, max_concurrent_requests=32):
    def postprocess(s, max_score):
        try:
            s = s.strip()
            conclusion = s.split("\n")[-1]
            conclusion = conclusion[conclusion.index(
                "最终得分：")+len("最终得分："):].strip()
            score = int(conclusion)
            if score > max_score:
                raise PostprocessError(
                    f'score exceeds maximum value. ({score}>{max_score})')
            return score / max_score
        except Exception as err:
            raise PostprocessError(f'{err}')

    TEMPLATE = '# 目标概述 \n给定一道题目，和一个要点评分表，你需要严格按照要点评分表的内容和分值，逐项对问题进行打分，并在最后给出最终分数。\n\n## 回答格式要求\n1. 你应当逐项按照列点的方式，对问题和评价项目进行比较，计算单个项目的分值，注意分值应当是整数，避免小数的出现，例如0.5\n2. 包含计算总分的计算步骤\n3. 在回答的最后部分，必须以“最终得分：***”这样的格式，给出总分\n\n\n下面提供你具体的范例，你需要参考范例进行回答\n\n\n[题目]\nFor a certain site, cement-mixed columns are planned to be used for ground reinforcement. It is known that the foundation depth is 2.0m, the diameter of the mixing column is 600mm, the column length is 14.0m, and the column strength is $f_{\\\\mathrm{cu}}=0.8\\\\, \\\\mathrm{MPa}$. The column strength reduction factor is taken as $\\\\eta=0.3$, the inter-column soil bearing capacity reduction factor is $\\\\beta=0.6$, and the column end soil bearing capacity reduction factor is $\\\\alpha=0.4$. The center-to-center spacing of the mixing columns is 1.0m, arranged in an equilateral triangular pattern. Which of the following options is appropriate for the characteristic value of the bearing capacity of the composite foundation with mixed columns?\n\n[打分表]\n[CRITERIA]\n- [SCORE=2] 题目是否属于选择题型\n- [SCORE=2] 题目是否涉及土木工程领域的地基处理\n- [SCORE=2] 题目是否要求计算复合地基的承载力\n- [SCORE=2] 题目是否提供所有必要参数（如桩长、桩径、强度、折减系数等）\n- [SCORE=2] 题目是否涉及多个折减系数的综合应用\n- [SCORE=2] 题目是否要求考虑桩间距和排列方式的影响\n- [SCORE=2] 题目是否需要综合运用公式进行数值计算\n- [SCORE=1] 题目是否涉及桩端土和桩间土的承载力计算\n- [SCORE=1] 题目是否要求选择一个合适的承载力值\n- [SCORE=1] 题目是否包含清晰的题干描述\n- [SCORE=1] 题目是否设置合理的选项\n- [SCORE=1] 题目是否考查记忆和理解能力\n- [SCORE=1] 题目是否考查应用和分析能力\n- [SCORE=1] 题目是否考查创造能力\n[/CRITERIA]\n\n\n[输出]\n### 评分表逐项打分（列点格式）  \n1. **题目是否属于选择题型**  \n   - 得分：2  \n   - 说明：题目以“Which of the following options is appropriate...”结尾，明确要求从选项中选择答案，属于选择题型。  \n\n2. **题目是否涉及土木工程领域的地基处理**  \n   - 得分：2  \n   - 说明：题目描述水泥搅拌桩（cement-mixed columns）用于地基加固（ground reinforcement），属于土木工程地基处理范畴。  \n\n3. **题目是否要求计算复合地基的承载力**  \n   - 得分：2  \n   - 说明：问题目标为求解“复合地基承载力特征值”（characteristic value of the bearing capacity of the composite foundation），需进行承载力计算。  \n\n4. **题目是否提供所有必要参数**  \n   - 得分：2  \n   - 说明：题目完整提供了桩长（14.0m）、桩径（600mm）、桩体强度（\\(f_{\\mathrm{cu}}=0.8\\,\\mathrm{MPa}\\)）、折减系数（\\(\\eta=0.3, \\beta=0.6, \\alpha=0.4\\)）、桩间距（1.0m）及排列方式（等边三角形），无参数缺失。  \n\n5. **题目是否涉及多个折减系数的综合应用**  \n   - 得分：2  \n   - 说明：涉及桩体强度折减系数 \\(\\eta\\)、桩间土承载力折减系数 \\(\\beta\\)、桩端土承载力折减系数 \\(\\alpha\\)，需在公式中综合考虑各系数的作用。  \n\n6. **题目是否要求考虑桩间距和排列方式的影响**  \n   - 得分：2  \n   - 说明：桩间距（1.0m）和等边三角形排列用于计算置换率 \\(m\\)（即桩的面积占总地基面积的比例），直接影响复合地基承载力公式中的权重分配。  \n\n7. **题目是否需要综合运用公式进行数值计算**  \n   - 得分：2  \n   - 说明：需应用复合地基承载力公式（如 \\(f_{\\mathrm{spk}} = m \\cdot \\eta \\cdot f_{\\mathrm{cu}} + \\beta(1-m) \\cdot f_{\\mathrm{sk}}\\)，其中 \\(f_{\\mathrm{sk}}\\) 可能涉及桩端土承载力），并结合几何参数计算置换率，属于数值计算类问题。  \n\n8. **题目是否涉及桩端土和桩间土的承载力计算**  \n   - 得分：1  \n   - 说明：桩间土承载力通过 \\(\\beta(1-m) \\cdot f_{\\mathrm{sk}}\\) 体现，桩端土承载力通过桩体承载力公式中的桩端阻力项（需考虑 \\(\\alpha\\) 折减）间接涉及，考查两者的协同作用。  \n\n9. **题目是否要求选择一个合适的承载力值**  \n   - 得分：1  \n   - 说明：题目要求从选项中选择“合适的”承载力特征值，属于结果选择类问题，需结合计算结果匹配选项。  \n\n10. **题目是否包含清晰的题干描述**  \n    - 得分：1  \n    - 说明：题干明确列出所有参数、工程背景（地基加固）及问题目标，无歧义或模糊表述，信息完整。  \n\n11. **题目是否设置合理的选项**  \n    - 得分：1  \n    - 说明：虽然题目未列出具体选项，但作为标准选择题，默认选项设置合理（如涵盖计算可能的误差范围或常见错误值）。  \n\n12. **题目是否考查记忆和理解能力**  \n    - 得分：1  \n    - 说明：需记忆复合地基承载力公式的结构及各参数定义（如折减系数的物理意义），考查对基本概念的理解。  \n\n13. **题目是否考查应用和分析能力**  \n    - 得分：1  \n    - 说明：需将公式应用于具体参数，分析桩间距和排列方式对置换率的影响，以及折减系数如何调整桩体和土体的承载力贡献。  \n\n14. **题目是否考查创造能力**  \n    - 得分：0  \n    - 说明：问题为公式直接应用，无需创新方法或创造性思维，仅需按步骤计算，故不涉及创造能力考查。  \n\n\n### 最终分数计算  \n利用乘法运算（几个相同的数相加用乘法表示更简便）将所有得分相加：\n$\n\\begin{align*}\n\\begin{align*}\n&2 + 2 + 2 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 0\\\\\n=&2\\times7 + 1\\times6+0\\\\\n=&14 + 6+0\\\\\n=&20+0\\\\\n=&20\n\\end{align*}\n\n最终得分：20\n\n\n\n[题目]\nUsing a magnetoelectric induction speed sensor to measure vibration at $f = 50 \\\\, \\\\text{Hz}$, the amplitude error should be less than $5\\\\%$, and the damping ratio is $\\\\xi = 0.6$. What should be the natural frequency of the sensor?\n\n[打分表]\n[CRITERIA]\n- [SCORE=2] 题目是否属于计算题类型，需应用特定公式进行计算\n- [SCORE=2] 题目是否涉及机械工程或物理学领域，特别是振动分析\n- [SCORE=2] 题目是否考察传感器的频率响应特性\n- [SCORE=2] 是否涉及阻尼比（ξ）的相关计算或概念理解\n- [SCORE=2] 是否需要在给定误差范围内计算参数\n- [SCORE=2] 题干描述是否清晰明确，提供了足够的已知条件\n- [SCORE=2] 问题是否明确，指向一个具体的计算目标\n- [SCORE=2] 是否考查学生对模态分析或机械振动的理解\n- [SCORE=2] 是否涉及传感器测量原理的基本理解\n- [SCORE=1] 题目是否需要应用特定公式或方程进行计算\n- [SCORE=1] 是否需要进行误差分析或计算误差范围\n- [SCORE=1] 是否涉及传感器的频率响应曲线或特性\n- [SCORE=1] 是否考查传感器的选择与应用原则\n- [SCORE=1] 是否需要考虑传感器的动态特性\n- [SCORE=1] 是否涉及振动测量中的基本概念或术语\n- [SCORE=1] 是否需要进行单位换算或量纲分析\n- [SCORE=1] 题目是否提供了明确的技术参数或性能指标\n- [SCORE=1] 是否涉及传感器测量误差的评估方法\n- [SCORE=1] 是否需要综合运用传感器与振动分析的知识点\n- [SCORE=1] 是否考查传感器在动态测量中的应用能力\n- [SCORE=1] 是否涉及传感器的选择与应用原则\n- [SCORE=1] 是否需要考虑传感器的安装与使用注意事项\n- [SCORE=1] 是否涉及传感器测量中的噪声与干扰问题\n[/CRITERIA]\n\n\n[输出]\n### 评分表逐项打分（列点格式）  \n1. **题目是否属于计算题类型，需应用特定公式进行计算**  \n   - 得分：2  \n   - 说明：题目要求根据给定的振动频率、阻尼比和振幅误差计算传感器的固有频率，需应用频率响应特性公式，属于计算题类型。  \n\n2. **题目是否涉及机械工程或物理学领域，特别是振动分析**  \n   - 得分：2  \n   - 说明：磁电感应速度传感器用于振动测量，涉及机械工程中的振动分析领域。  \n\n3. **题目是否考察传感器的频率响应特性**  \n   - 得分：2  \n   - 说明：题目中明确提到振动频率 \\(f = 50 \\, \\\\text{Hz}\\)、阻尼比 \\(\\xi = 0.6\\) 和振幅误差要求，直接关联传感器的频率响应特性分析。  \n\n4. **是否涉及阻尼比（ξ）的相关计算或概念理解**  \n   - 得分：2  \n   - 说明：阻尼比 \\(\\xi = 0.6\\) 是频率响应公式中的关键参数，需用于计算固有频率与振幅误差的关系。  \n\n5. **是否需要在给定误差范围内计算参数**  \n   - 得分：2  \n   - 说明：题目要求振幅误差小于 \\(5\\%\\)，需通过误差约束条件反推传感器的固有频率。  \n\n6. **题干描述是否清晰明确，提供了足够的已知条件**  \n   - 得分：2  \n   - 说明：题干明确给出振动频率、阻尼比、误差要求及目标参数（固有频率），信息完整无歧义。  \n\n7. **问题是否明确，指向一个具体的计算目标**  \n   - 得分：2  \n   - 说明：问题直接询问“传感器的固有频率应为多少”，目标明确，指向单一计算结果。  \n\n8. **是否考查学生对模态分析或机械振动的理解**  \n   - 得分：2  \n   - 说明：频率响应分析是模态分析的核心内容，需理解固有频率、阻尼比与振幅误差的动态关系。  \n\n9. **是否涉及传感器测量原理的基本理解**  \n   - 得分：2  \n   - 说明：磁电感应传感器的测量原理基于电磁感应，其频率响应特性是正确使用传感器的基础。  \n\n10. **题目是否需要应用特定公式或方程进行计算**  \n    - 得分：1  \n    - 说明：需应用幅频特性公式 \\(|H(\\omega)| = \\\\frac{1}{\\sqrt{(1 - (\\omega/\\omega_n)^2)^2 + (2\\xi\\omega/\\omega_n)^2}}\\)，通过误差条件建立方程求解 \\(\\omega_n\\)。  \n\n11. **是否需要进行误差分析或计算误差范围**  \n    - 得分：1  \n    - 说明：需将振幅误差 \\(|H(\\omega)| \\leq 1.05\\)（或 \\(0.95 \\leq |H(\\omega)| \\leq 1.05\\)）转化为数学约束条件，分析固有频率的取值范围。  \n\n12. **是否涉及传感器的频率响应曲线或特性**  \n    - 得分：1  \n    - 说明：振幅误差与频率的关系通过频率响应曲线的幅频特性体现，需理解曲线形状对误差的影响。  \n\n13. **是否考查传感器的选择与应用原则**  \n    - 得分：1  \n    - 说明：根据测量需求（频率、误差）选择合适固有频率的传感器，属于传感器应用原则的考查。  \n\n14. **是否需要考虑传感器的动态特性**  \n    - 得分：1  \n    - 说明：频率响应特性属于传感器的动态特性，需分析其在动态测量中的表现。  \n\n15. **是否涉及振动测量中的基本概念或术语**  \n    - 得分：1  \n    - 说明：题目涉及振动频率、固有频率、阻尼比、振幅误差等振动测量的核心概念。  \n\n16. **是否需要进行单位换算或量纲分析**  \n    - 得分：0  \n    - 说明：题目中频率单位为 \\(\\\\text{Hz}\\)，参数单位统一，无需进行单位换算或量纲分析。  \n\n17. **题目是否提供了明确的技术参数或性能指标**  \n    - 得分：1  \n    - 说明：提供了振动频率（\\(50 \\, \\text{Hz}\\)）、阻尼比（\\(0.6\\)）、振幅误差限制（\\(<5\\%\\)）等明确技术指标。  \n\n18. **是否涉及传感器测量误差的评估方法**  \n    - 得分：1  \n    - 说明：通过幅频特性公式评估不同固有频率下的振幅误差，属于误差评估方法的应用。  \n\n19. **是否需要综合运用传感器与振动分析的知识点**  \n    - 得分：1  \n    - 说明：需结合传感器的频率响应特性（传感器知识）与振动系统的幅频特性分析（振动分析知识）完成计算。  \n\n20. **是否考查传感器在动态测量中的应用能力**  \n    - 得分：1  \n    - 说明：振动测量属于动态测量范畴，题目考查如何通过传感器参数设计满足动态测量的精度要求。  \n\n21. **是否涉及传感器的选择与应用原则**（重复项，按条目保留评分）  \n    - 得分：1  \n    - 说明：同第13项，仍属于传感器选择与应用原则的考查。  \n\n22. **是否需要考虑传感器的安装与使用注意事项**  \n    - 得分：0  \n    - 说明：题干未提及传感器的安装方式、环境条件等使用注意事项。  \n\n23. **是否涉及传感器测量中的噪声与干扰问题**  \n    - 得分：0  \n    - 说明：题目未涉及噪声源、抗干扰措施等相关内容。  \n\n\n\n### 最终分数计算  \n利用乘法运算（几个相同的数相加用乘法表示更简便）将所有得分相加：  \n$\n\\\\begin{align*}\n&2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 0 + 1 + 1 + 1 + 1 + 1 + 0 + 0\\\\\n=&2 \\\\times 9 + 1 \\\\times 11 + 0 \\times 3\\\\\n=&18 + 11 + 0\\\\\n=&29 + 0\\\\\n=&29\n\\end{align*}\n$  \n\n最终得分：29'

    prompts, postprocesses = [], []
    for q, c in zip(questions, criteria):
        prompt = TEMPLATE + \
            f'\n\n现在需要你对下面的问题计算分数。\n\n[题目]\n{q}\n\n[打分表]\n{c}\n\n[输出]\n'
        prompts.append(prompt)
        postprocesses.append(
            partial(postprocess, max_score=get_total_score(c)))

    results = await agent.run(prompts, max_concurrent_requests, desc="[Judge Question w Criteria]", postprocess_fns=postprocesses)
    scores = [_[1] for _ in results]
    return scores


async def decode_to_question(criteria, max_concurrent_requests=32):
    def postprocess(s):
        try:
            thought = re.findall(r'think>.*</think>', s, re.DOTALL)[0]
            conclusion = s.replace(thought, "")
            conclusion = conclusion[conclusion.index(
                "[QUESTION]")+len("[QUESTION]"):conclusion.index("[/QUESTION]")].strip()
            return conclusion
        except Exception as err:
            raise PostprocessError(f'{err}')

    TEMPLATE = """
# 目标概述
你是一个出题大师，你的任务是基于一个题目打分表，你需要构建一道题目，完美符合打分表内的所有项目。即构造的题目，如果用打分表进行打分，将会获得满分。

## 回答格式要求
```
<think>
... ...
<think>

[QUESTION]
{构造好的问题}}
[/QUESTION]

```

下面是一个具体的例子：


[打分表]
[CRITERIA]
- [SCORE=2] 题目是否属于选择题型
- [SCORE=2] 题目是否涉及土木工程领域的地基处理
- [SCORE=2] 题目是否要求计算复合地基的承载力
- [SCORE=2] 题目是否提供所有必要参数（如桩长、桩径、强度、折减系数等）
- [SCORE=2] 题目是否涉及多个折减系数的综合应用
- [SCORE=2] 题目是否要求考虑桩间距和排列方式的影响
- [SCORE=2] 题目是否需要综合运用公式进行数值计算
- [SCORE=1] 题目是否涉及桩端土和桩间土的承载力计算
- [SCORE=1] 题目是否要求选择一个合适的承载力值
- [SCORE=1] 题目是否包含清晰的题干描述
- [SCORE=1] 题目是否设置合理的选项
- [SCORE=1] 题目是否考查记忆和理解能力
- [SCORE=1] 题目是否考查应用和分析能力
- [SCORE=1] 题目是否考查创造能力
[/CRITERIA]


[输出]
<think>
用户让我构造一道符合打分表所有项目的选择题，涉及土木工程的地基处理，特别是复合地基承载力计算。首先，我需要确保题目是选择题型，所以格式上要有选项。接下来，必须涵盖地基处理，复合地基承载力计算，所以题干需要包括桩长、桩径、强度、折减系数等参数。打分表提到多个折减系数的综合应用，比如桩间土和桩身强度的折减，还有桩间距和排列方式的影响，比如正方形布置，桩间距影响面积置换率。
然后，需要考虑桩端土和桩间土的承载力计算，可能需要分别计算桩体承载力和桩间土承载力，再用面积置换率综合。题目需要数值计算，所以要给出具体的数值，比如桩长 10m，桩径 0.5m，压缩模量，天然地基承载力，折减系数等。选项设置要合理，正确答案需要通过公式计算得出，其他选项可以是计算过程中的中间值或错误应用公式的结果。
还要考查记忆、理解、应用、分析和创造能力，虽然创造能力在选择题中可能较难，但通过综合运用多个公式和考虑不同因素，可以体现分析和应用能力。题干描述要清晰，参数齐全，比如明确桩的布置方式，计算复合地基承载力特征值，并选择合适的选项。需要检查每个打分点是否覆盖，比如选择题型、地基处理、复合地基计算、必要参数、多个折减系数、桩间距和排列、数值计算、桩端和桩间土计算、选择合适值、清晰题干、合理选项，以及考查多种能力。确保每个评分项都满足，特别是分数高的项目，如前几项每个 2 分的点，必须全部涵盖。最后，组织题干和选项，确保计算过程正确，选项中的正确答案符合公式应用，比如复合地基承载力公式是桩体承载力乘以置换率加上桩间土承载力乘以（1 - 置换率），置换率根据桩间距和桩径计算，正方形布置时置换率为桩径平方除以桩间距平方。
</think>

[QUESTION]
某多层建筑采用水泥粉煤灰碎石桩（CFG 桩）复合地基，正方形布桩，桩径 d=0.5m，桩间距 s=1.5m，桩长 L=10m。已知桩体试块抗压强度平均值 f_cu=30MPa，桩间土承载力特征值 f_sk=120kPa，桩间土承载力折减系数 β=0.9，桩体强度折减系数 η=0.8，桩端天然地基土承载力特征值 q_p=800kPa，桩侧阻力忽略不计。按《建筑地基处理技术规范》计算，该复合地基的承载力特征值（kPa）最接近下列哪个选项？
A. 210
B. 245
C. 280
D. 315
[/QUESTION]

"""

    prompts = []
    for c in criteria:
        prompt = TEMPLATE + \
            f'\n\n现在需要你基于下面打分表构造一道完美符合要求的问题。\n*注意*最终你只需要生成问题，不要尝试解答问题\n\n[打分表]\n{c}\n\n[输出]\n'
        prompts.append(prompt)

    results = await agent.run(prompts, max_concurrent_requests, desc="[Fabricate QA]", postprocess_fns=[postprocess]*len(prompts))
    return [_[1] for _ in results]


async def question_similarity(agent, authentic, fabricate, max_concurrent_requests=32):
    TEMPLATE = """### 问题相似程度评价标准（1-5分）

| **相似等级** | **判定标准**                                                                 |
|--------------|------------------------------------------------------------------------------|
| **1分**      | 完全不同：题目类型、核心条件、求解目标、解题思路毫无关联，无任何共同要素。       |
| **2分**      | 弱相关：仅单一维度相关（如同属数学题中的几何/代数大类），其余要素无重合。         |
| **3分**      | 部分相似：题目类型相同（如均为“三维立方体隐藏块计数”），但核心条件、目标或步骤存在关键差异（如可见面数不同、求解方向相反）。 |
| **4分**      | 高度相似：题目类型、求解目标、解题框架完全一致，仅数据或参数不同（如隐藏块数值、矩阵元素等），核心步骤和逻辑完全复用。 |
| **5分**      | 完全相同：题目内容（数据、表述、目标、步骤）完全一致，无任何差异。               |


### 使用说明
1. **类型优先**：若题目类型不同（如几何vs代数），直接≤2分；类型相同是≥3分的前提。
2. **核心要素判断**：
   - **1分**：无任何交集（如几何题vs代数题）。
   - **2分**：仅同属大类别（如同为数学题，但具体子类型不同，如几何vs数列）。
   - **3分**：子类型相同但存在关键差异（如可见面数不同、递推公式不同）。
   - **4分**：子类型、目标、解题逻辑一致，仅数据不同（如求同一类型问题的不同参数解）。
   - **5分**：题目完全复现，无任何改动。

### 输出格式
<think>
... ...
</think>
[CONCLUSION START]
SIMILARITY=*
[CONCLUSION END]
```
（*处填写1-5的整数，需严格符合上述判定标准）



下面是具体的例子

[原问题]
Find the minimum value of\n\\[x^2 + 2xy + 3y^2 - 6x - 2y,\\]over all real numbers $x$ and $y.$

[对比问题]
Find the minimum value of the quadratic function \\( f(x) = x^2 - 6x + 7 \\).

[输出]
<think>
用户让我比较两个数学问题的相似程度。首先，我需要仔细看一下两个问题的内容。原问题是求一个二元二次多项式的最小值，变量是 x 和 y，表达式是 x² + 2xy + 3y² - 6x - 2y。待评价的问题是求一元二次函数 f (x) = x² - 6x + 7 的最小值。
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
<think>
用户让我比较两个矩阵求逆的问题的相似程度。首先，根据类型优先原则，两个问题都是求矩阵的逆，属于同一题目类型，所以类型相同，至少 3 分以上。接下来看核心要素：原问题是求一个 2x2 矩阵的逆，对比问题也是 2x2 矩阵，求解目标都是求逆矩阵，解题框架都是使用矩阵求逆的方法，比如伴随矩阵法或者行变换。两者的不同在于矩阵中的数据不同，原问题的矩阵是 [[5,-4],[0,1]]，对比问题是 [[2,1],[3,4]]。根据判定标准，4 分的情况是题目类型、求解目标、解题框架一致，仅数据或参数不同，核心步骤和逻辑完全复用。这里显然符合 4 分的条件，因为只是矩阵元素不同，解题方法完全一样，没有关键差异。所以相似等级应该是 4 分。
</think>
[CONCLUSION START]
SIMILARITY=4
[CONCLUSION END]



"""

    def postprocess(s):
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

    prompts = []
    prompts = defaultdict(list)
    for index, (a, b) in enumerate(zip(authentic, fabricate)):
        prompt = TEMPLATE + \
            f'\n\n现在需要你比较下面两个问题的相似度。\n\n[原问题]\n{a}\n\n[对比问题]\n{b}\n\n[输出]\n'
        prompts[prompt].append(index)

    results = await agent.run(list(prompts.keys()), max_concurrent_requests, desc="[QA Similarity]", postprocess_fns=[postprocess]*len(list(prompts.keys())))

    results_mapper = {}
    for (k, v) in results:
        for _ in prompts[k]:
            results_mapper[_] = v

    outputs = []
    for i, _ in enumerate(authentic):
        if i in results_mapper and results_mapper[i] is not None:
            outputs.append(results_mapper[i])
        else:
            outputs.append(0)
    return outputs


class QwQLongCoTCreateCriteriaComputeScore(object):
    def __init__(self,
                 split="train",
                 parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD,
                 max_concurrent_requests=32):
        self.split = split
        self.parse_result_failure_score = parse_result_failure_score
        self.max_concurrent_requests = max_concurrent_requests

    async def calc_classify_acc_reward(
        self,
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
        max_concurrent_requests=32,
        diff_threshold=0.1,
        return_single_score=True
    ):
        """
            计算Criteria是否可以准确区分出真题/合成题
            Score分为两个部分
            Score1: 有效区分出真题/合成题的准确率 (真题分数-合成题分数 >= diff_threshold) 分值-1～+1
            Score2: 真题通过Criteria判定的得分，越接近1越好。分值正则化到-1～+1

        """
        questions, criteria = [], []

        indices = []
        for i, (solution_str, gt) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            criterion = criteria_parse_solution_fn(solution_str)
            if criterion is not None:
                questions.append(gt["positive"])
                criteria.append(criterion)
                indices.append((i, "positive"))

                for negative in gt["negatives"]:
                    questions.append(negative)
                    criteria.append(criterion)
                    indices.append((i, "negative"))

        results = await criteria_get_score(
            questions, criteria, max_concurrent_requests=max_concurrent_requests)

        pos = {}
        neg = defaultdict(list)
        for result, index in zip(results, indices):
            index, tag = index
            if tag == "positive":
                pos[index] = result
            else:
                neg[index].append(result)

        score1, score2 = [0.0] * \
            len(batch_solution_str), [0.0] * len(batch_solution_str)

        def normalize(value):
            return 2 * value - 1.0

        for i in range(len(batch_solution_str)):
            # 解析错误
            if i not in pos:
                continue
            else:
                if pos[i] == None:
                    continue
                else:
                    score2[i] = normalize(pos[i])

                    acc = []
                    for val in neg[i]:
                        if val is not None:
                            _acc = (pos[i] - val) >= diff_threshold
                            if _acc:
                                acc.append(normalize(1.0))
                            else:
                                acc.append(normalize(0.0))
                    if len(acc) > 0:
                        score1[i] = np.mean(acc)
                    else:
                        score1[i] = 0.0

        if return_single_score:
            total_score = []
            for x, y in zip(score1, score2):
                total_score.append(x+y)
            return total_score
        else:
            return score1, score2

    async def calc_compression_ratio_reward(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            max_concurrent_requests=32,
            similarity_threshold=4,
    ):
        """
            计算Criteria对应原问题的信息压缩率, score取值区间-1～+1
        """

        indices = []
        criteria = []
        for i, (_gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            criterion = criteria_parse_solution_fn(sol)
            if criterion is not None:
                criteria.append(criterion)
                indices.append(i)
            else:
                continue

        results = await decode_to_question(criteria)
        new_indices, fabricates, authentics = [], [], []

        for fabricate, index in zip(results, indices):
            if fabricate is None:
                continue
            else:
                fabricates.append(fabricate)
                new_indices.append(index)
                authentics.append(batch_ground_truth[index]["positive"])

        similarity = await question_similarity(
            authentic=authentics,
            fabricate=fabricates,
            max_concurrent_requests=max_concurrent_requests
        )
        scores = [0.0] * len(batch_solution_str)
        for sim, index in zip(similarity, new_indices):
            if sim is None:
                pass
            else:
                # 相似度过高，泄题 => 负
                if sim >= similarity_threshold:
                    scores[index] = -1.0
                else:
                    scores[index] = 1.0

        return scores

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
        scores1 = await self.calc_classify_acc_reward(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            self.max_concurrent_requests
        )
        scores2 = await self.calc_compression_ratio_reward(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            self.max_concurrent_requests
        )

        final_results = []
        for gt, solution, score1, score2 in zip(batch_ground_truth, batch_solution_str, scores1, scores2):
            criteria = criteria_parse_solution_fn(solution)
            if criteria is None:
                final_results.append(self.parse_result_failure_score)
            else:
                final_results.append(score1+score2)

            if self.split == "valid" or (self.split == "train" and random.random() < 0.1):
                log = True
                log_flag = "[VALID]" if self.split == "valid" else "[TRAIN]"
            else:
                log = False

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f"【Question】`{self.log_ground_truth(gt)}`")
                print(
                    f"【Criteria】`{self.log_solution(solution)}`")
                print(
                    f'[Final Reward]={final_results[-1]:.3f}\n')
        return final_results

    def log_solution(self, solution):
        criteria = criteria_parse_solution_fn(solution)
        if criteria is None:
            return repr(self.clip_string(solution))
        return repr(self.clip_string(criteria))

    def log_ground_truth(self, ground_truth):
        return repr(self.clip_string(ground_truth["positive"]))

    def clip_string(self, s: str):
        if len(s) > 1500:
            return f'{s[:700]}... [省略] ...{s[-800:]}'
        return s


_qwq_longcot_create_criteria_compute_score_train = QwQLongCoTCreateCriteriaComputeScore(
    split="train", max_concurrent_requests=MAX_CONCURRENT)
_qwq_longcot_create_criteria_compute_score_valid = QwQLongCoTCreateCriteriaComputeScore(
    split="valid", max_concurrent_requests=MAX_CONCURRENT)
qwq_longcot_create_criteria_compute_score_train = _qwq_longcot_create_criteria_compute_score_train.compute_score
qwq_longcot_create_criteria_compute_score_valid = _qwq_longcot_create_criteria_compute_score_valid.compute_score
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Criteria构造
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Doc2Query
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def doc2query_parse_solution_fn(solution_str: str, remove_option_letter=True):
    # FIXME: QwQ tokenizer config
    solution_str = f'<think>\n{solution_str}'

    if solution_str.count("</question>") > 1:
        return None

    if solution_str.count("</think>") > 1:
        return None

    solution_str = postprocess_solution(solution_str)

    if not solution_str.startswith("<think>"):
        return None

    if not solution_str.endswith("</question>"):
        return None

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

    try:
        question = conclusion[conclusion.index(
            "Question: ")+len("Question: "):conclusion.index("Options:")].strip()
        options = conclusion[conclusion.index(
            "Options:")+len("Options:"):conclusion.index("Answer:")].strip()
        if remove_option_letter:
            options = re.findall(r'[A-W]\)\s*(.*)', options)
        else:
            options = re.findall(r'([A-W]\)\s*.*)', options)
        options = [_.strip() for _ in options]

        answer = conclusion[conclusion.index("Answer:"):].strip()
        answer = re.findall(r'Answer:\s*([A-W])', answer)[0].strip()

        # 选项有重复
        if len(options) != len(set(options)):
            return None
        return question, options, answer
    except Exception as err:
        return None


class Doc2QueryFormatReward(PenaltyOrReward):
    def __init__(self, base_reward=0.1, penalty=-2.0, doc2query_parse_solution_fn=doc2query_parse_solution_fn):
        self.base_reward = base_reward
        self.penalty = penalty
        self.doc2query_parse_solution_fn = doc2query_parse_solution_fn

    def get_penalty_or_reward(self, solution_str, ground_truth):
        solution_str = self.doc2query_parse_solution_fn(solution_str)
        if solution_str is None:
            return self.penalty

        question, options, answer = solution_str

        if ground_truth.get("option_num", None) is None:
            return self.penalty
        if len(options) == ground_truth["option_num"]:
            return self.base_reward
        return self.penalty / 2.0


class QuestionSimilarity(PenaltyOrReward):
    def __init__(self, doc2query_parse_solution_fn=doc2query_parse_solution_fn, key="authentic_question"):
        self.doc2query_parse_solution_fn = doc2query_parse_solution_fn
        self.key = key

    def get_penalty_or_reward(self, solution_str, ground_truth):
        if ground_truth.get(self.key, None) is None:
            return 0.0
        try:
            solution_str = self.doc2query_parse_solution_fn(solution_str)

            if solution_str is None:
                return 0.0
            question, options, answer = solution_str

            if ground_truth.get(self.key, None):
                gt = ground_truth[self.key]
            else:
                return 0.0

            gt_tokens = " ".join(tokenize(gt.lower(), "en"))
            sl_tokens = " ".join(tokenize(question.lower(), "en"))
            bleu = sacrebleu.sentence_bleu(sl_tokens, [gt_tokens]).score
            return bleu / 100
        except Exception as err:
            return 0.0


class RuleBasedOptionMatch(PenaltyOrReward):
    def __init__(self, doc2query_parse_solution_fn=doc2query_parse_solution_fn):
        self.doc2query_parse_solution_fn = doc2query_parse_solution_fn

        self.keywords = [
            # 数学与物理符号
            '$', '\\frac', '^', '_', '\\sqrt', '\\vec', '\\approx', '\\pm', '\\times', '\\cdot', '/', '=',
            '(', ')', '[', ']', '→', '\\hat', '%', '\\Delta', '\\odot', '\\rm', '\\ddot', '\\mu', '\\epsilon',
            '\\mathsf', '\\mathbf', '\\ln', '\\cos', '\\exp', '\\sum', '\\int', '\\partial', '\\infty',
            '\\pi', '\\zeta', '\\omega', '\\lambda', '\\sigma', '\\rho', '\\theta', '×10^', '×10^-', 'E+', 'E-',

            # 单位与物理量符号
            'm', 'cm', 'mm', 'in', 'km', 'ft', 's', 'μs', 'ms', 'min', 'h', 'a', 'sec', 'N', 'N/m²', 'Pa', 'kg',
            'kg/m³', 'm/s', 'm/s²', 'rad/s', 'J', 'kJ', 'GJ', 'W', 'W/m²', 'V', 'nV', 'kV', 'A', 'Ω', 'Hz', 'dB',
            'C', 'F', 'mol', 'L', 'mL', 'g', 'g/kg', '(liquid)', '(gas)', 'U', 'rpm', 'ppm', 'ppb',


            # 编号与结构符号
            '[ ]', '{ }', '( )', '〈〉', 'Ⅰ', 'Ⅱ', 'III', 'IV', '(1)', '(2)', '①', '②', 'n=', 'N=', 'No.', '→',
            '+', '=', 'H₂O', 'Mg²⁺',

            # 通用修饰词与状态词
            'approximately', 'around', 'about', 'respectively', 'perfectly', 'small', 'big', 'non-', 'anti-',
            '-hinged', '-order', 'dr.', 'national', 'university', 'initial', 'final', 'mean', 'total', 'effective',
            'original', 'renewed',

            # 其他符号与特殊标记
            '$', '¥', '€', '%', '‰', '\\', '|', '*', '^T', 'file a request for', 'accounting for', 'originating from'
        ]

    def get_common_keywords(self, options):
        common_keywords = [_ for _ in self.keywords if all(
            _ in option for option in options)]
        return common_keywords

    def get_common_words(self, options):
        if len(options[0].split(" ")) > 1:
            words = options[0].split(" ")
            return [_ for _ in words if all(_ in option for option in options)]
        else:
            common_tokens = [options[0][:i] for i in range(len(options[0])) if all(
                options[0][:i] in option for option in options)]
            if len(common_tokens) > 0 and common_tokens[-1] != '':
                return [common_tokens[-1]]
            else:
                return []

    def get_penalty_or_reward(self, solution_str, ground_truth):
        if ground_truth.get("options", None) is None:
            return 0.0
        try:
            raw_solution_str = solution_str
            solution_str = self.doc2query_parse_solution_fn(solution_str)
            if solution_str is None:
                return 0.0

            question, options, answer = solution_str
            options_sol = [_.lower().strip() for _ in options]
            options_gt = [_.lower().strip() for _ in ground_truth["options"]]

            targets = set(self.get_common_keywords(options_gt) +
                          self.get_common_words(options_gt))

            score = 0.0
            # 共同词缀奖励
            if len(targets) > 0:
                gt_match, sol_match = 0, 0
                for _ in targets:
                    this_gt_match = len(
                        [opt for opt in options_gt if _ in opt])
                    gt_match += this_gt_match
                    # 确保 this_sol_match 不会超过 this_gt_match
                    this_sol_match = min(
                        len([opt for opt in options_sol if _ in opt]), this_gt_match)
                    sol_match += this_sol_match

                score += (sol_match/gt_match * 0.1)
            else:
                pass

            # 选项匹配
            option_gt_matched = {_: False for _ in options_gt}
            for _ in options_sol:
                if _ in options_gt:
                    option_gt_matched[_] = True

            score += 0.2 * \
                len([k for k, v in option_gt_matched.items() if v]) / \
                len(option_gt_matched)

            # 答案匹配
            try:
                sol_answer = options_sol[ord(answer) - ord('A')]
                gt_tokens = " ".join(
                    tokenize(ground_truth["answer"].lower(), "en"))
                sl_tokens = " ".join(tokenize(sol_answer.lower(), "en"))
                bleu = sacrebleu.sentence_bleu(sl_tokens, [gt_tokens]).score

                score += 0.1 * bleu / 100
            except Exception as err:
                pass
            return score
        except Exception as err:
            return 0.0


class QwQLongCoTDoc2QueryComputeScore(object):
    MULTICHOICE_LETTER = ('A', 'B', 'C', 'D', 'E', 'F', 'G',
                          'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T')

    def __init__(self,
                 split="train", add_difficulty_rewards=False, difficulty_bon=8, parse_solution_fn=doc2query_parse_solution_fn):
        self.split = split
        self.doc2query_parse_solution_fn = parse_solution_fn

        self.format = Doc2QueryFormatReward(
            doc2query_parse_solution_fn=self.doc2query_parse_solution_fn)
        self.question_similarity = QuestionSimilarity(
            doc2query_parse_solution_fn=self.doc2query_parse_solution_fn, key="question")
        self.rule_base = RuleBasedOptionMatch(
            doc2query_parse_solution_fn=self.doc2query_parse_solution_fn)
        self.add_difficulty_rewards = add_difficulty_rewards
        self.difficulty_bon = difficulty_bon

        self.agent = Agent(**{
            "model": "qwen25_32B_instruct",
            "base_url": "http://10.130.131.138:8000/v1",
            "api_keys": "EMPTY",
            "request_kwargs": {
                "temperature": 0.9,
                "timeout": 360,
                "max_tokens": 2048,
            },
        })
        self.verify_agent = self.agent

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "Format": self.format.get_penalty_or_reward,
            "QSim": self.question_similarity.get_penalty_or_reward,
            "RuleBased": self.rule_base.get_penalty_or_reward,
        }

    async def chat_completion_with_retry(self, url, data, max_retries=3, retry_delay=5, suffix="/generate"):
        retries = 0
        while retries < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f'{url}{suffix}', json=data, timeout=aiohttp.ClientTimeout(total=2400)) as response:
                        response.raise_for_status()
                        return await response.json()
            except (aiohttp.ClientError, aiohttp.ClientResponseError) as e:
                print(
                    f"{url}请求(数据总量={len(data)})失败，错误信息: {e}，重试第 {retries + 1} 次...")
                retries += 1
                if retries < max_retries:
                    await aio.sleep(retry_delay)
        print(f"{url}达到最大重试次数，请求失败。")
        return None

    async def run_tasks_in_queues(self, tasks):
        """将任务分成n个队列并行执行"""
        n = len(self.get_respondent_urls())

        # 创建n个队列
        queues = [[] for _ in range(n)]

        # 平均分配任务到各个队列
        for i, task in enumerate(tasks):
            queue_id = i % n
            queues[queue_id].append(task)

        parallel_tasks = []
        for i, queue in enumerate(queues):
            parallel_tasks.append(self.chat_completion_with_retry(
                url=self.get_respondent_urls()[i],
                data=queue
            ))
        flattened_results = []
        for f in tqdm.asyncio.tqdm.as_completed(parallel_tasks, dynamic_ncols=True, desc=f'[Generate {len(tasks)} Responses]'):
            results = await f
            for result in results:
                flattened_results.append(result)

        return flattened_results

    def get_respondent_urls(self):
        suffixes = [
        ]
        return [f'http://{_}' for _ in suffixes]

    def response_postprocess(self, s):
        ans = None
        try:
            s = s.strip()
            conclusion = s.split("\n")[-1]
            conclusion = conclusion[conclusion.index(
                "Answer:")+len("Answer:"):].strip()
            if conclusion not in self.MULTICHOICE_LETTER:
                ans = None
            else:
                ans = conclusion
        except Exception as err:
            ans = None

        if ans is None:
            matched = re.findall(r'Answer:\s*([A-W])', s)
            if len(matched) > 0:
                return matched[0]
            return None
        return ans

    async def generate_responses(self, prompts):
        prompts_w_ids = [{"prompt": _, "uuid": uuid.uuid4().hex}
                         for _ in prompts]
        ids = [_["uuid"] for _ in prompts_w_ids]

        random.shuffle(prompts_w_ids)
        # prompts_w_ids = sorted(prompts_w_ids, key=lambda x: x["prompt"])
        results = await self.run_tasks_in_queues(prompts_w_ids)

        post_results = {}
        for result in results:
            if result and "uuid" in result and "response" in result:
                post_results[result["uuid"]] = (
                    result["prompt"],
                    self.response_postprocess(result["response"])
                )

        outputs = []
        for prompt, _uuid in zip(prompts, ids):
            if _uuid in post_results:
                outputs.append(post_results[_uuid])
            else:
                outputs.append((prompt, None))
        return outputs

    async def get_difficulty_reward(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth, max_concurrent_requests=MAX_CONCURRENT, repeat=8):

        prompts = []
        wo_content_prompts, w_content_prompts = defaultdict(
            list), defaultdict(list)

        for i, (solution_str, gt) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            result = self.doc2query_parse_solution_fn(solution_str)
            if result is not None:
                question, options, answer = result

                lang_code = gt["lang_code"]
                if lang_code == "zh":
                    instruct = '回答以下单项选择题。只有一个正确答案。你回应的最后一行必须采用 “Answer: $LETTER” 的格式（不带引号），其中 LETTER 为选项字母之一。你必须首先通过非常详细的思考过程逐步分析。'
                else:
                    instruct = 'Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format "Answer: $LETTER" (without quotes), where LETTER is one of the option letters. You must first think step by step with very detail thinking process.'

                prompt = f'{instruct}\n\n' + self.prepare_question_for_test(
                    question, options, lang_code=lang_code)
                wo_content_prompts[prompt].append(i)

                prompts.extend([prompt]*repeat)
                prompt = f'[LECTURE]\n{gt["document"]}\n[/LECTURE]\n\n' + f'{instruct}\n\n' + self.prepare_question_for_test(
                    question, options, lang_code=lang_code)
                w_content_prompts[prompt].append(i)

                prompts.extend([prompt]*repeat)

        _results = await self.agent.run(list(set(prompts)), max_concurrent_requests, desc=f"[Generate Responses {self.agent.model}]", postprocess_fns=[self.response_postprocess] * len(list(set(prompts))))
        results_mapper = defaultdict(list)
        for (k, v) in _results:
            results_mapper[k].append(v)

        wo_contents, w_contents = defaultdict(list), defaultdict(list)
        for k, v in results_mapper.items():
            if k in wo_content_prompts:
                for index in wo_content_prompts[k]:
                    wo_contents[index].extend(v)
            elif k in w_content_prompts:
                for index in w_content_prompts[k]:
                    w_contents[index].extend(v)
            else:
                raise NotImplementedError

        full_rewards = []
        pass_rates = []

        for i in range(len(batch_solution_str)):
            if i in wo_contents:
                base_score = 0.0

                wo_content, w_content = wo_contents[i], w_contents[i]

                wo_content = [_ for _ in wo_content if _ is not None]
                w_content = [_ for _ in w_content if _ is not None]

                # 正确回答
                result = self.doc2query_parse_solution_fn(
                    batch_solution_str[i])
                if result is not None:
                    _, _options, answer = result
                else:
                    answer, _options = "", []
                ans = answer

                wo_content_correct = [_ for _ in wo_content if _ == ans]
                w_content_correct = [_ for _ in w_content if _ == ans]

                pass_rates.append({
                    "wo_content": f'{len(wo_content_correct)}/{len(wo_content)} {wo_content}, ans={ans}',
                    "w_content": f'{len(w_content_correct)}/{len(w_content)} {w_content}, ans={ans}',
                })

                try:
                    if wo_content.count(self.MULTICHOICE_LETTER[len(
                            _options)]) >= self.difficulty_bon/4:
                        base_score -= 3.0
                    if wo_content.count(self.MULTICHOICE_LETTER[len(
                            _options)+1]) >= self.difficulty_bon/4:
                        base_score -= 3.0

                    # 无参考 majority vote
                    wo_content_majority_votes = defaultdict(int)
                    for v in wo_content:
                        wo_content_majority_votes[v] += 1
                    wo_content_majority_votes = sorted(
                        wo_content_majority_votes.items(), key=lambda x: x[1], reverse=True)
                    if len(wo_content_majority_votes) > 0:
                        wo_majority_vote_ans = wo_content_majority_votes[0][0]
                        if ans == self.MULTICHOICE_LETTER[len(_options)] or ans == self.MULTICHOICE_LETTER[len(_options)+1]:
                            base_score -= 3.0
                except Exception as err:
                    pass

                # 不带参考 模型也有机会rollout对 否则问题可能过于长尾
                if wo_content.count(ans) < self.difficulty_bon/4:  # 至少对两次
                    full_rewards.append(base_score)
                    continue

                # 带参考 应该比 不带参考 显著好
                if w_content.count(ans) - wo_content.count(ans) < self.difficulty_bon/4:
                    full_rewards.append(base_score)
                    continue

                # 完全做不对
                if len(wo_content_correct) == 0 or len(w_content_correct) == 0:
                    pass
                # 全对
                elif len(wo_content_correct) == len(wo_content):
                    pass
                else:
                    # 无参考正确率在一定区间
                    if len(wo_content_correct) >= 1 and len(wo_content_correct)/len(wo_content) <= 0.75:
                        wo_acc = len(wo_content_correct)/len(wo_content)
                        # 难度越大越好(min_threshold=0.2)
                        base_score += 1-max(wo_acc, 0.2)

                        # 有/无参考正确率差异越大越好
                        diff = (len(w_content_correct) / len(w_content)
                                ) - ((len(wo_content_correct))/(len(wo_content)))
                        diff = max(diff, 0.0)
                        base_score += diff

                        # 有参考 majority vote是正确答案加分
                        w_content_majority_votes = defaultdict(int)
                        for v in w_content:
                            w_content_majority_votes[v] += 1

                        w_content_majority_votes = sorted(
                            w_content_majority_votes.items(), key=lambda x: x[1], reverse=True)
                        try:
                            if w_content_majority_votes[0][0] == ans:
                                base_score += 0.5
                        except Exception as err:
                            pass

                full_rewards.append(base_score)
            else:
                pass_rates.append({})
                full_rewards.append(0.0)
        return full_rewards, pass_rates

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
        penalty = defaultdict(dict)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            for key, fn in self.get_penalties().items():
                penalty[key][i] = fn(solution_str, ground_truth)

        final_results = []

        if self.add_difficulty_rewards:
            difficulty_rewards, pass_rates = await self.get_difficulty_reward(
                batch_data_sources,
                batch_solution_str,
                batch_ground_truth,
                max_concurrent_requests=MAX_CONCURRENT,
                repeat=self.difficulty_bon
            )

        for i in range(len(batch_solution_str)):
            if self.add_difficulty_rewards:
                score = difficulty_rewards[i]
            else:
                score = 0.0

            penalty_log_str = []
            for name, _penalty in penalty.items():
                penalty_log_str.append(
                    f'{name}={_penalty[i]:.2f}')
                score += _penalty[i]

            final_results.append(score)

            if (self.split == "valid" and random.random() < 0.5) or (self.split == "train" and random.random() < 0.1):
                log = True
                log_flag = "[VALID]" if self.split == "valid" else "[TRAIN]"
            else:
                log = False

            difficulty = batch_ground_truth[i]["difficulty"]
            domain = batch_ground_truth[i]["domain"]

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f"【Solution】({domain})`{self.log_solution(batch_solution_str[i])}`")
                try:
                    print(
                        f"【Ground Truth】({difficulty})`{self.log_ground_truth(batch_ground_truth[i])}`")
                except Exception as err:
                    pass
                if self.add_difficulty_rewards:
                    print(
                        f'[Pass@{self.difficulty_bon}]={pass_rates[i]}|[Final Reward]={score:.3f}|Difficulty={difficulty_rewards[i]:.3f}|{"|".join(penalty_log_str)}\n')
                else:
                    print(
                        f'[Pass@{self.difficulty_bon}]={pass_rates[i]}|[Final Reward]={score:.3f}|{"|".join(penalty_log_str)}\n')
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
        norm = self.doc2query_parse_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return repr(self.format_question(norm[0], norm[1], norm[2]))

    def format_question(self, question, options, answer):
        options_str = "\n".join([f'{x}) {y}' for x, y in zip(
            self.MULTICHOICE_LETTER, options)])
        if answer is not None:
            return f'Question: {question}\n\nOptions:\n{options_str}\n\nAnswer: {answer}'
        else:
            return f'Question: {question}\n\nOptions:\n{options_str}'

    def prepare_question_for_test(self, question, options, lang_code):
        if lang_code == "zh":
            na = '以上都不对'
        else:
            na = 'None of the above'

        new_options = copy.deepcopy(options)
        if na not in new_options:
            new_options.append(na)

        if lang_code == "zh":
            error = '题目存在错误（包括题目信息不完整 / 前提矛盾或问题设定有缺陷 / 表述不当等等各种错误 / 同时存在多个正确答案、无法单选）'
        else:
            error = 'The question contains errors (cases including incomplete conditions, contradictory statements, Cannot be determined/Unable to determine, insufficient data/contradictory premises or problem is flawed/ill-posed, multiple correct answers simultaneously or etc.)'

        new_options.append(error)

        options_str = "\n".join([f'{x}) {y}' for x, y in zip(
            self.MULTICHOICE_LETTER, new_options)])

        if lang_code == "zh":
            return f'问题：{question}\n\n选项：\n{options_str}'
        else:
            return f'Question: {question}\n\nOptions:\n{options_str}'

    def log_ground_truth(self, ground_truth):
        return repr(self.format_question(
            ground_truth["question"],
            ground_truth["options"],
            ground_truth["answer"])
        )

    def clip_string(self, s: str):
        if len(s) > 1500:
            return f'{s[:700]}... [省略] ...{s[-800:]}'
        return s


_qwq_longcot_doc2query_compute_score_train = QwQLongCoTDoc2QueryComputeScore(
    split="train", add_difficulty_rewards=True)
_qwq_longcot_doc2query_compute_score_valid = QwQLongCoTDoc2QueryComputeScore(
    split="valid", add_difficulty_rewards=True)
qwq_longcot_doc2query_compute_score_train = _qwq_longcot_doc2query_compute_score_train.compute_score
qwq_longcot_doc2query_compute_score_valid = _qwq_longcot_doc2query_compute_score_valid.compute_score
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Doc2Query
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Doc2Query V2
# ------------------------------------------------------------------------------------------------------------------------------------------------------

def doc2query_v2_parse_solution_fn(solution_str: str, remove_option_letter=True):
    if solution_str.count("</question>") > 1:
        return None

    if solution_str.count("</think>") > 1:
        return None

    solution_str = postprocess_solution(solution_str)
    if not solution_str.startswith("<think>"):
        return None

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

    try:
        question = conclusion[conclusion.index(
            "Question: ")+len("Question: "):conclusion.index("Answer:")].strip()

        answer = conclusion[conclusion.index(
            "Answer:")+len("Answer:"):conclusion.index("Answer Type:")].strip()

        answer_type = conclusion[conclusion.index(
            "Answer Type:")+len("Answer Type:"):].strip()
        return question, answer, answer_type
    except Exception as err:
        return None


class AnswerFormat(object):
    @abstractmethod
    def verify(self, answer: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def rectify(self, answer: str) -> str:
        raise NotImplementedError


class NumericalAnswer(object):
    def __init__(self):
        pass

    def initial_recognize(self, answer) -> bool:
        """
        检测数值字符串是否符合规范要求。
        返回 True（符合）或 False（不符合）。
        """
        # 去除首尾空格
        s = answer.strip()

        # 正则表达式：分数（支持符号和前导零）
        pattern_fraction = r'^[+-]?\d+/[1-9]\d*$'  # 允许符号，分子可为任意整数，分母无lead zero

        # 正则表达式：浮点数（支持多种格式）
        # 支持 123.45, .45, 123., -0.5 等格式
        pattern_float = r'^[+-]?(\d+\.\d*|\.\d+)$'

        # 正则表达式：整数（支持符号和前导零）
        pattern_int = r'^[+-]?0$|^[+-]?[1-9]\d*$'  # 允许符号，允许单独的0或无lead zero的整数

        if re.fullmatch(pattern_fraction, s):
            return True
        elif re.fullmatch(pattern_float, s):
            return True
        elif re.fullmatch(pattern_int, s):
            return True
        else:
            return False

    def rectify(self, answer):
        """处理数字，判断整数/小数并格式化（四舍五入保留三位小数）"""
        num = answer
        # 处理分数形式
        if isinstance(num, str) and '/' in num:
            try:
                numerator, denominator = map(int, num.split('/'))
                value = numerator / denominator
                return f'\\boxed' + "{" + format_decimal(value) + "}"
            except:
                return f'\\boxed' + "{" + num + "}"  # 转换失败返回原始值

        # 处理二进制字符串
        if isinstance(num, str) and num.startswith('0b'):
            try:
                return f'\\boxed' + "{" + str(int(num, 2)) + "}"
            except:
                return f'\\boxed' + "{" + num + "}"  # 转换失败返回原始值

        # 处理普通字符串表示的数字
        try:
            value = float(num)
            return f'\\boxed' + "{" + self.format_decimal(value) + "}"
        except:
            return f'\\boxed' + "{" + num + "}"  # 非数字类型直接返回

    def format_decimal(self, value):
        """核心格式化函数：使用Decimal进行精确四舍五入"""
        # 使用Decimal进行精确计算
        decimal_value = Decimal(str(value))

        # 四舍五入保留三位小数
        rounded = decimal_value.quantize(
            Decimal('0.001'), rounding=ROUND_HALF_UP)

        # 判断是否为整数
        if rounded == rounded.to_integral_value():
            return int(rounded)
        else:
            # 转换为字符串，确保保留三位小数
            return f"{rounded:.3f}"

    def exclude_common_answer_pattern(self, answer):
        if answer in (
            '\\boxed{-2}', '\\boxed{-1}', '\\boxed{0}', '\\boxed{1}', '\\boxed{2}', '\\boxed{3}',
            '\\boxed{4}', '\\boxed{5}', '\\boxed{6}', '\\boxed{7}', '\\boxed{1.000}', '\\boxed{0.000}',
                '\\boxed{2.000}', '\\boxed{3.000}', '\\boxed{-1.000}',):
            return False
        return True

    def verify(self, answer):
        """
        检测答案是否符合 \boxed{} 格式及数值规范（整数/浮点数）

        参数：
        answer_str (str)：待检测的答案字符串（如 "\boxed{5}", "boxed{0.210}", "\boxed{5/12}" 等）

        返回：
        (bool, str)：第一个元素为是否通过校验，第二个元素为错误提示（若失败）
        """
        answer_str = answer
        # 1. 校验 \boxed{} 格式
        boxed_pattern = r'^\\boxed\{(.*?)\}$'
        match = re.match(boxed_pattern, answer_str)
        if not match:
            # return False, "格式错误：答案需用 \\boxed{} 包裹，且大括号内无空格"
            return False

        # 提取数值内容
        content = match.group(1).strip()
        if not content:
            # return False, "格式错误：\\boxed{} 内内容为空"
            return False

        # 2. 校验数值规范（复用之前的数值校验逻辑）
        # 去除可能的残留空格（确保数值部分无空格）
        cleaned_content = content.replace(' ', '')
        result = self.verify_numeric_content(cleaned_content)  # 调用数值校验函数
        return result[0]

    def verify_numeric_content(self, content):
        """
        单独校验数值内容是否符合规范（整数/浮点数，禁止分数）
        """
        # 去除无关字符（仅保留数字和小数点）
        cleaned = re.sub(r'[^\d.]', '', content)

        # 检查是否为分数（先于数值校验，避免误判）
        if '/' in content:
            return False, "禁止使用分数形式，请转换为小数"

        # 校验整数或浮点数
        if re.match(r'^\d+$', cleaned):
            # 整数校验：无前导零
            if len(cleaned) > 1 and cleaned.startswith('0'):
                return False, "整数包含前导零"
            return True, "格式正确"
        elif re.match(r'^\d*\.\d+$', cleaned):
            # 拆分整数部分和小数部分
            parts = cleaned.split('.')
            if len(parts) != 2:
                return False, "浮点数格式错误（需包含一个小数点）"

            int_part, float_part = parts
            # 整数部分校验：0 或正整数（无前导零）
            if int_part != '0' and (len(int_part) > 1 and int_part.startswith('0')):
                return False, "整数部分包含前导零"
            # 小数部分校验：固定3位
            if len(float_part) != 3:
                return False, f"小数部分应为3位（当前{len(float_part)}位）"
            return True, "格式正确"
        else:
            return False, "无效数值格式（需为整数或3位小数）"


class WithUnitSymbol(object):
    def __init__(self):
        pass

    def initial_recognize(self, answer) -> bool:
        return self.is_valid_with_unit(answer)

    def verify(self, answer):
        return self.is_valid_with_unit(answer)

    def is_valid_with_unit(self, answer: str) -> bool:
        """
        验证答案是否符合数值与单位格式规范
        增强对科学计数法多种表示形式的支持
        """
        pattern = re.compile(r'''
            ^                   # 字符串起始
            ([+-]?)             # 可选的正负号
            (                   # 数值部分
                \d+\.?\d*       # 整数或小数（如123, 123.4）
                |               # 或
                \.\d+           # 纯小数（如.456）
            )
            (?:                 # 科学计数法部分（可选，非捕获组）
                \s*             # 允许乘号前有空格
                [×x*·\s]        # 允许乘号为×、x、*、·或空格
                \s*             # 允许乘号后有空格
                10\^[+-]?\d+    # 10的指数部分（如10^-15）
            )?                  # 科学计数法结束
            \s+                 # 至少一个空格分隔数值与单位
            (                   # 单位部分
                [A-Za-zμΩ°]+       # 基础单位（如m, Pa, mol）
                [²³⁰¹²³⁴⁵⁶⁷⁸⁹\-⁻]*  # 允许幂次符号和负号（如m², m³, m⁻¹）
                (?:             # 可选的SI前缀（如k, m, μ）
                    [yzafpnumcdhkMGTPEZY]
                )?
                (?:             # 分子中多个单位用·连接（如kJ·mol）
                    \u00B7[A-Za-zμΩ]+[²³⁰¹²³⁴⁵⁶⁷⁸⁹\-⁻]*
                )*
                (?:             # 分母部分（可选）
                    /           # 斜杠分隔符
                    (?:         # 分母两种格式：括号内或直接跟单位
                        # 括号内的单位（如(mol·K)）
                        \([A-Za-zμΩ°]+[²³⁰¹²³⁴⁵⁶⁷⁸⁹\-⁻]*(?:\u00B7[A-Za-zμΩ°]+[²³⁰¹²³⁴⁵⁶⁷⁸⁹\-⁻]*)*\)
                        |       # 或
                        # 直接跟单位（如mol·K）
                        [A-Za-zμΩ°]+[²³⁰¹²³⁴⁵⁶⁷⁸⁹\-⁻]*(?:\u00B7[A-Za-zμΩ°]+[²³⁰¹²³⁴⁵⁶⁷⁸⁹\-⁻]*)*
                    )
                )?
            )
            $                   # 字符串结束
        ''', re.VERBOSE | re.UNICODE)  # 启用详细模式和Unicode匹配

        return bool(pattern.match(answer.strip()))


class GenerateQAV2FormatReward(PenaltyOrReward):
    def __init__(self, doc2query_parse_solution_fn=doc2query_v2_parse_solution_fn):
        self.doc2query_parse_solution_fn = doc2query_parse_solution_fn

    def get_penalty_or_reward(self, solution_str, ground_truth):
        solution_str = self.doc2query_parse_solution_fn(solution_str)

        if solution_str is None:
            return 0.0

        question, answer, answer_type = solution_str

        if answer_type not in ("NumericalAnswer", "WithUnitSymbol"):
            return -1.5
        if answer_type == "NumericalAnswer":
            parser = NumericalAnswer()
        elif answer_type == "WithUnitSymbol":
            parser = WithUnitSymbol()
        else:
            raise NotImplementedError

        try:
            if parser.verify(answer):
                if answer_type == "NumericalAnswer":
                    # 特定校验（避免构造0、1、2等常见答案）
                    if not parser.exclude_common_answer_pattern(answer):
                        return -0.1
                return 0.0
            else:
                return -0.2
        except Exception as err:
            return -0.2


class AnswerFeatureMatch(PenaltyOrReward):
    def __init__(self, doc2query_parse_solution_fn=doc2query_parse_solution_fn):
        self.doc2query_parse_solution_fn = doc2query_parse_solution_fn

        self.keywords = [
            # 数学与物理符号
            '\\box', '$', '\\frac', '^', '_', '\\sqrt', '\\vec', '\\approx', '\\pm', '\\times', '\\cdot', '/', '=',
            '(', ')', '[', ']', '→', '\\hat', '%', '\\Delta', '\\odot', '\\rm', '\\ddot', '\\mu', '\\epsilon',
            '\\mathsf', '\\mathbf', '\\ln', '\\cos', '\\exp', '\\sum', '\\int', '\\partial', '\\infty',
            '\\pi', '\\zeta', '\\omega', '\\lambda', '\\sigma', '\\rho', '\\theta', '×10^', '×10^-', 'E+', 'E-',

            # 单位与物理量符号
            'm', 'cm', 'mm', 'in', 'km', 'ft', 's', 'μs', 'ms', 'min', 'h', 'a', 'sec', 'N', 'N/m²', 'Pa', 'kg',
            'kg/m³', 'm/s', 'm/s²', 'rad/s', 'J', 'kJ', 'GJ', 'W', 'W/m²', 'V', 'nV', 'kV', 'A', 'Ω', 'Hz', 'dB',
            'C', 'F', 'mol', 'L', 'mL', 'g', 'g/kg', '(liquid)', '(gas)', 'U', 'rpm', 'ppm', 'ppb',


            # 编号与结构符号
            '[ ]', '{ }', '( )', '〈〉', 'Ⅰ', 'Ⅱ', 'III', 'IV', '(1)', '(2)', '①', '②', 'n=', 'N=', 'No.', '→',
            '+', '=', 'H₂O', 'Mg²⁺',

            # 通用修饰词与状态词
            'approximately', 'around', 'about', 'respectively', 'perfectly', 'small', 'big', 'non-', 'anti-',
            '-hinged', '-order', 'dr.', 'national', 'university', 'initial', 'final', 'mean', 'total', 'effective',
            'original', 'renewed',

            # 其他符号与特殊标记
            '$', '¥', '€', '%', '‰', '\\', '|', '*', '^T', 'file a request for', 'accounting for', 'originating from'
        ]

    def get_common_keywords(self, answer):
        common_keywords = [_ for _ in self.keywords if _ in answer]
        return common_keywords

    def get_penalty_or_reward(self, solution_str, ground_truth):
        if ground_truth.get("answer", None) is None:
            return 0.0
        try:
            raw_solution_str = solution_str
            solution_str = self.doc2query_parse_solution_fn(solution_str)

            if solution_str is None:
                return 0.0

            gt_ans = ground_truth["answer"]

            question, answer, answer_type = solution_str
            targets = set(self.get_common_keywords(gt_ans))
            score = 0.0
            # 共同词缀奖励
            if len(targets) > 0:
                gt_match, sol_match = 0, 0
                for _ in targets:
                    if _ in gt_ans:
                        gt_match += 1
                    if _ in answer:
                        sol_match += 1
                score += max((sol_match/gt_match * 0.02), 0.02)
            else:
                pass
            return score
        except Exception as err:
            return 0.0


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


class QwQLongCoTDoc2QueryV2ComputeScore(QwQLongCoTDoc2QueryComputeScore):
    def __init__(self,
                 split="train", add_difficulty_rewards=False, difficulty_bon=8, parse_solution_fn=doc2query_v2_parse_solution_fn):
        super().__init__(
            split=split, add_difficulty_rewards=add_difficulty_rewards, difficulty_bon=difficulty_bon, parse_solution_fn=parse_solution_fn
        )
        self.format = GenerateQAV2FormatReward(
            doc2query_parse_solution_fn=self.doc2query_parse_solution_fn)
        self.answer_feature = AnswerFeatureMatch(
            doc2query_parse_solution_fn=self.doc2query_parse_solution_fn)

        self.wo_content_agent = self.agent
        self.w_content_agent = Agent(**{
            "model": "DeepSeek-V3-0324",
            "base_url": "https://sd138cdmeq1emkiunptm0.apigateway-cn-beijing.volceapi.com/v1",
            "api_keys": "EMPTY",
            "request_kwargs": {
                "temperature": 0.9,
                "timeout": 360,
                "max_tokens": 4096,
            }
        })

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "Format": self.format.get_penalty_or_reward,
            "QSim": self.question_similarity.get_penalty_or_reward,
            # "AnsFeature": self.answer_feature.get_penalty_or_reward,
        }

    def get_answer_format(self, answer_type, lang_code):
        WithUnitSymbol_zh = """带单位数值 (WithUnitSymbol) 规范要求
1. **数值表示**
   - 问题指令必须明确要求保留的小数点位数 科学计数法位数。
   - 大数用科学计数法，避免冗余空格，如 `$5.27×10^{5}\ \\text{Pa}$`。

2. **单位规范**
   - 问题指令必须明确要求返回答案的单位。
   - 单位符号用国际标准（SI），大小写严格区分：
     - 大写：N（牛）、Pa（帕）、J（焦）、W（瓦）、Hz（赫）等。
     - 小写：m（米）、kg（千克）、s（秒）、mol（摩）等。
   - 单位与数值间留空格：`2.91 m` ✅，`2.91m` ❌。
   - 复合单位用斜杠表示：`kJ/(mol·K)` ✅，禁止使用乘方形式（如 `kJ·mol⁻¹·K⁻¹` ❌）。
"""
        WithUnitSymbol_en = """Specifications for Numerical Answers with Unit Symbols (WithUnitSymbol)
1. Numerical Representation:
    - The question instructions must clearly require the number of decimal places or significant figures for scientific notation to be retained.
    - Use scientific notation for large numbers, use format correctly such as `5.27 × 10^5 Pa`. Do not add parentheses to the exponent part: `1.256 × 10^-67 J` ✅, `1.256 × 10^{-67} J` ❌, `1.92 × 10⁷ m⁻¹` ❌。

2. Unit Specifications:
    - The question instructions must clearly require the unit for the returned answer.
    - Use international standard (SI) unit symbols with strict case distinction:
      - Uppercase: N (newton), Pa (pascal), J (joule), W (watt), Hz (hertz), etc.
      - Lowercase: m (meter), kg (kilogram), s (second), mol (mole), etc.
    - Leave a space between the unit and the numerical value: `2.91 m` ✅, `2.91m` ❌.
    - Use a slash for composite units: `kJ/(mol·K)` ✅, and power forms are prohibited (such as `kJ·mol⁻¹·K⁻¹` ❌).
"""
        NumericalAnswer_zh = """数值答案 (NumericalAnswer) 规范要求
1. **类型允许**：
  - **整数**：正整数，无前导零（如 \(5, 275, 144\)）。
  - **浮点数**：由整数部分、小数点和小数部分组成，整数部分可为 \(0\) 或正整数（无前导零），**小数部分固定保留3位**（如 \(0.210, 40.200, 5.500\)）。
  - **禁止分数形式**，必须转换为小数形式（如 \(5/12\) 需表示为 \(0.417\)）。

2. **格式限制**：
  - 不允许包含空格、逗号、单位（如“元”）等无关字符。
  - 所有答案需用 \(\\boxed{}\) 包裹（如 \(\\boxed{5}\)、\(\\boxed{0.210}\)）。
"""
        NumericalAnswer_en = """
Specifications for Numerical Answers (NumericalAnswer)
1. **Permitted Types**:
   - **Integers**: Positive integers without leading zeros (e.g., \(5, 275, 144\)).
   - **Floating-point numbers**: Composed of an integer part, a decimal point, and a fractional part. The integer part can be \(0\) or a positive integer (no leading zeros), and the **fractional part must be fixed to 3 decimal places** (e.g., \(0.210, 40.200, 5.500\)).
   - **Fractional forms are prohibited** and must be converted to decimal form (e.g., \(5/12\) should be expressed as \(0.417\)).

2. **Format Restrictions**:
   - No irrelevant characters such as spaces, commas, or units (e.g., "yuan") are allowed.
   - All answers must be enclosed in \(\\boxed{}\) (e.g., \(\\boxed{5}\), \(\\boxed{0.210}\)).
"""
        return {
            "WithUnitSymbol": WithUnitSymbol_zh,
            "NumericalAnswer": NumericalAnswer_zh
        }[answer_type] if lang_code == "zh" else {
            "WithUnitSymbol": WithUnitSymbol_en,
            "NumericalAnswer": NumericalAnswer_en
        }[answer_type]

    def response_postprocess(self, s):
        try:
            s = s.strip()
            conclusion = s
            if "最终答案是" in conclusion:
                conclusion = conclusion[conclusion.index(
                    "最终答案是")+len("最终答案是"):].strip()
                return conclusion
            else:
                conclusion = conclusion[conclusion.index(
                    "the final answer is")+len("the final answer is"):].strip()
                return conclusion
        except Exception as err:
            try:
                s = s.strip()
                conclusion = s.split("\n")[-1].strip()

                if len(conclusion) < 5:
                    conclusion = "\n".join(s.split("\n")[-3:]).strip()
                return conclusion
            except Exception as err:
                raise PostprocessError(f'parse conclusion failure')

    def verify(self, conclusion, answer, answer_type):
        if answer_type == "WithUnitSymbol":
            score = 1.0 if answer in conclusion else 0.0
            if score > 0:
                return score
            return 1.0 if all(part in conclusion for part in answer.split(" ")) else 0.0
        elif answer_type == "NumericalAnswer":
            gt = extract_answer(answer)
            if gt is None:
                return 0.0
            if extract_answer(conclusion) == gt:
                return 1.0
            else:
                return 0.0

    async def verify_results(self, verify_queue, batch_solution_str, max_concurrent_requests, split_names):
        def validate_result(response):
            s = response
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

        verify_prompt = """### **基于标准答案判断回答是否正确**
任务描述：请根据提供的**题目**、**用户回答（答案部分）**和**标准答案**，判断用户回答是否正确，并按照指定格式输出结果。需严格比对答案，若用户回答与标准答案**内容一致**，则判定为正确，否则为错误。

**必须与标准答案完全一致**（含数值、单位、符号等），如果数值是科学记数法或者是小数，只允许非常小的计算误差（有效计算部分最后一位），否则判错。 |

#### 输出要求
```json
{
"判断结果": "正确/错误",
}
```

现在对下面的回答判断正确性
"""

        verify_template = """
#### **输入：**
##### 题目
```
{question}
```

##### 用户回答（答案部分）
{conclusion}

##### 标准答案
{answer}

#### **输出：**
"""
        correctness = {name: defaultdict(list) for name in split_names}

        verify_mapper = defaultdict(list)

        for example in verify_queue:
            index = example[0]
            sol_str = batch_solution_str[index]
            question, answer, answer_type = self.doc2query_parse_solution_fn(
                sol_str)
            # 基于规则解析答案
            if example[2] is None:
                correctness[example[1]][example[0]].append(0.0)
            else:
                correct = self.verify(example[2], answer, answer_type)

                if correct > 0.0:
                    correctness[example[1]][example[0]].append(correct)
                else:
                    # verify_mapper
                    instruct = f'仔细一步步思考，并回答下面的问题。你回应的最后一行必须采用 “... 最终答案是 $ANSWER 的格式（不带引号），其中 $ANSWER 的格式要求需要满足下面的说明。\n\n{self.get_answer_format(answer_type, "zh")}'
                    prompt = f'{instruct}\n\n' + question
                    eval_prompt = verify_prompt + "\n\n" + verify_template.format(
                        question=prompt,
                        answer=answer,
                        conclusion=example[2]
                    )
                    verify_mapper[eval_prompt].append((example[0], example[1]))

        _results = await self.verify_agent.run(list(verify_mapper.keys()), max_concurrent_requests, desc=f"[Eval Responses {self.verify_agent.model}]", postprocess_fns=[validate_result] * len(list(verify_mapper.keys()),))

        results_mapper = defaultdict(list)
        for (k, v) in _results:
            for meta in verify_mapper[k]:
                index, _type = meta
                if v is not None:
                    correctness[_type][index].append(1.0 if v else 0.0)

        return correctness

    async def get_difficulty_reward(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth, max_concurrent_requests=MAX_CONCURRENT, wo_content_bon=24, w_content_bon=6):

        wo_content_prompts, w_content_prompts = defaultdict(
            list), defaultdict(list)

        for i, (solution_str, gt) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            result = self.doc2query_parse_solution_fn(solution_str)
            if result is not None:
                question, answer, answer_type = result
                ans_format_strict = self.format.get_penalty_or_reward(
                    solution_str, gt
                )
                # 答案格式不符合规范
                if ans_format_strict < 0.0:
                    continue

                lang_code = gt["lang_code"]
                if lang_code == "zh":
                    instruct = f'仔细一步步思考，并回答下面的问题。你回应的最后一行必须采用 “... 最终答案是 $ANSWER 的格式（不带引号），其中 $ANSWER 的格式要求需要满足下面的说明。\n\n{self.get_answer_format(answer_type, lang_code)}'
                else:
                    instruct = f'Think step by step in detail and answer the following questions. The last line of your response must be in the format "... the final answer is $ANSWER" (without quotes), where the format requirements for $ANSWER need to meet the instructions below.\n\n{self.get_answer_format(answer_type, lang_code)}'

                prompt = f'{instruct}\n\n' + question
                wo_content_prompts[prompt].append(i)

                prompt = f'[LECTURE]\n{gt["document"]}\n[/LECTURE]\n\n' + \
                    f'{instruct}\n\n' + question
                w_content_prompts[prompt].append(i)

        _w_content_prompts = list(w_content_prompts.keys()) * w_content_bon
        _wo_content_prompts = list(wo_content_prompts.keys()) * wo_content_bon

        tasks = [
            self.wo_content_agent.run(_wo_content_prompts, max_concurrent_requests, desc=f"[Generate w/o Content Responses {self.wo_content_agent.model}]", postprocess_fns=[
                self.response_postprocess] * len(_wo_content_prompts)),
            self.w_content_agent.run(_w_content_prompts, max_concurrent_requests, desc=f"[Generate w Content Responses {self.w_content_agent.model}]", postprocess_fns=[
                self.response_postprocess] * len(_w_content_prompts))
        ]
        wo_results, w_results = await aio.gather(*tasks)

        results_mapper = defaultdict(list)
        for (k, v) in wo_results:
            results_mapper[k].append(v)
        for (k, v) in w_results:
            results_mapper[k].append(v)

        # 答案验证
        verify_queue = []
        for k, v in results_mapper.items():
            if k in wo_content_prompts:
                for index in wo_content_prompts[k]:
                    verify_queue.extend(
                        [(index, "w/o_content", _v) for _v in v])
            elif k in w_content_prompts:
                for index in w_content_prompts[k]:
                    verify_queue.extend([(index, "w_content", _v) for _v in v])

        correctness = await self.verify_results(
            verify_queue=verify_queue, batch_solution_str=batch_solution_str, max_concurrent_requests=MAX_CONCURRENT,
            split_names=["w/o_content", "w_content"]
        )

        wo_contents, w_contents = correctness["w/o_content"], correctness["w_content"]

        full_rewards = []
        pass_rates = []

        for i in range(len(batch_solution_str)):
            if i in wo_contents:
                base_score = 0.0

                wo_content_scores = wo_contents[i]
                w_content_scores = w_contents[i]

                pass_rates.append({
                    "wo_content": f'{np.sum(wo_content_scores)}/{len(wo_content_scores)}',
                    "w_content": f'{np.sum(w_content_scores)}/{len(w_content_scores)}',
                })

                try:
                    if len(wo_content_scores) == 0 or len(w_content_scores) == 0:
                        full_rewards.append(base_score)
                        continue

                    # 题目过于简单或困难
                    if np.mean(wo_content_scores) == 1.0 or np.mean(wo_content_scores) < (1.0/16) or np.mean(wo_content_scores) == 0.:
                        full_rewards.append(base_score)
                        continue

                    # 带参考 应该比 不带参考 显著好
                    if not (np.mean(w_content_scores) >= min(1/w_content_bon + np.mean(wo_content_scores), 1.0)):
                        full_rewards.append(base_score)
                        continue

                    # # 有参考置信度
                    # if np.mean(w_content_scores) < 0.3:
                    #     full_rewards.append(base_score)
                    #     continue

                    # 总分计算
                    difficulty1 = (1.0-math.log2(1+np.sum(wo_content_scores))/math.log2(
                        1+wo_content_bon))
                    difficulty2 = (1.0-math.log2(1+np.sum(w_content_scores)) /
                                   math.log2(1+w_content_bon))

                    confidence = (1.0 if np.mean(w_content_scores)
                                  > 0.5 else np.mean(w_content_scores))
                    base_score = [difficulty1, difficulty2, confidence]
                except Exception as err:
                    pass

                full_rewards.append(base_score)
            else:
                pass_rates.append({})
                full_rewards.append(0.0)
        return full_rewards, pass_rates

    def log_solution(self, solution):
        norm = self.doc2query_parse_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return repr(self.format_question(norm[0], norm[1]))

    def format_question(self, question, answer):
        return f'Question: {question}\nAnswer: {answer}'

    def log_ground_truth(self, ground_truth):
        return repr(self.format_question(
            ground_truth["question"],
            ground_truth["answer"])
        )

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):

        penalty = defaultdict(list)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            parsed = self.doc2query_parse_solution_fn(solution_str)
            if parsed is None:
                penalty[i].append(-2.0)
            else:
                penalty[i].append(0.0)

            for key in ("Format", "QSim"):
                penalty[i].append(self.get_penalties()[key]
                                  (solution_str, ground_truth))

        # difficulty_rewards, pass_rates = await self.get_difficulty_reward(
        #     batch_data_sources,
        #     batch_solution_str,
        #     batch_ground_truth,
        #     max_concurrent_requests=MAX_CONCURRENT,
        # )

        # FIXME
        difficulty_rewards, pass_rates = [
            0.0]*len(batch_solution_str), [{}] * len(batch_solution_str)

        final_results = []
        for i in range(len(batch_solution_str)):
            scores = copy.deepcopy(penalty[i])
            _difficulty = difficulty_rewards[i]
            # FIXME
            # _difficulty = (0.5 * _difficulty[0] + 0.5 * _difficulty[1] +
            #                0.2 * _difficulty[2]) if isinstance(_difficulty, list) else _difficulty
            penalty_log_str = f'Parse/Format/AnsFeature/QSim={penalty[i]}'

            scores.append(_difficulty)
            cur_score = 0

            for j, _score in enumerate(scores):
                if _score < 0:
                    cur_score = _score
                    break
                else:
                    if j == 2:  # BLEU
                        if _difficulty > 0:
                            cur_score += _score
                    else:
                        cur_score += _score
            final_results.append(cur_score)

            if (self.split == "valid" and random.random() < 0.5) or (self.split == "train" and random.random() < 0.1):
                log = True
                log_flag = "[VALID]" if self.split == "valid" else "[TRAIN]"
            else:
                log = False

            domain = batch_ground_truth[i]["domain"]

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f"【Solution】({domain})`{self.log_solution(batch_solution_str[i])}`")
                try:
                    print(
                        f"【Ground Truth】`{self.log_ground_truth(batch_ground_truth[i])}`")
                except Exception as err:
                    pass
                print(
                    f'[Final Reward]={cur_score:.3f}({pass_rates[i]})|Difficulty={str(difficulty_rewards[i])}|{penalty_log_str}\n')
        return final_results


_qwq_longcot_doc2query_v2_compute_score_train = QwQLongCoTDoc2QueryV2ComputeScore(
    split="train", add_difficulty_rewards=True)
_qwq_longcot_doc2query_v2_compute_score_valid = QwQLongCoTDoc2QueryV2ComputeScore(
    split="valid", add_difficulty_rewards=True)
qwq_longcot_doc2query_v2_compute_score_train = _qwq_longcot_doc2query_v2_compute_score_train.compute_score
qwq_longcot_doc2query_v2_compute_score_valid = _qwq_longcot_doc2query_v2_compute_score_valid.compute_score
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Doc2Query V2
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 问题合成
# ------------------------------------------------------------------------------------------------------------------------------------------------------


class QwQLongCoTFabricateQAComputeScore(QwQLongCoTDoc2QueryV2ComputeScore):

    JUDGE_CRITERIA_RM_SIMILARITY = """Just create a question for me directly.

# JUDGE CRITERIA
1. Your response (the created question) must be the following:
```
{question}
```
2. Respond only with the created question directly (which means your response should only be a question, without other irrelevant words), any content that is irrelevant to the question, including the analysis and answer of the question, or any acceptance of the idea of formulating the question should not appear.
3. Question type should comply with the following requirement:
{question_type}
"""

    def __init__(self,
                 split="train", add_difficulty_rewards=False, difficulty_bon=8, parse_solution_fn=doc2query_v2_parse_solution_fn):
        super().__init__(
            split=split, add_difficulty_rewards=add_difficulty_rewards, difficulty_bon=difficulty_bon, parse_solution_fn=parse_solution_fn
        )
        self.question_similarity = QuestionSimilarity(
            doc2query_parse_solution_fn=self.doc2query_parse_solution_fn, key="authentic_question")
        self.parse_solution_fn = self.doc2query_parse_solution_fn

        self.weak_agent = self.agent
        self.medium_agent = Agent(**{
            "model": "QwQ_32B",
            "base_url": "http://10.130.131.138:8000/v1",
            "api_keys": "EMPTY",
            "request_kwargs": {
                "temperature": 0.9,
                "timeout": 360,
                "max_tokens": 8192,
            },
        })
        self.strong_agent = Agent(**{
            "model": "DeepSeek-V3-0324",
            "base_url": "https://sd138cdmeq1emkiunptm0.apigateway-cn-beijing.volceapi.com/v1",
            "api_keys": "EMPTY",
            "request_kwargs": {
                "temperature": 0.9,
                "timeout": 360,
                "max_tokens": 4096,
            }
        })
        self.verify_agent = self.agent

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "Format": self.format.get_penalty_or_reward,
            "QSim": self.question_similarity.get_penalty_or_reward,
        }

    async def get_difficulty_reward(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth, max_concurrent_requests=MAX_CONCURRENT, weak_bon=16, strong_bon=6):

        weak_model_prompts, strong_model_prompts = defaultdict(
            list), defaultdict(list)

        for i, (solution_str, gt) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            result = self.parse_solution_fn(solution_str)
            if result is not None:
                question, answer, answer_type = result
                ans_format_strict = self.format.get_penalty_or_reward(
                    solution_str, gt
                )
                # 答案格式不符合规范
                if ans_format_strict < 0.0:
                    continue

                lang_code = gt["lang_code"]
                if lang_code == "zh":
                    instruct = f'仔细一步步思考，并回答下面的问题。你回应的最后一行必须采用 “... 最终答案是 $ANSWER 的格式（不带引号），其中 $ANSWER 的格式要求需要满足下面的说明。\n\n{self.get_answer_format(answer_type, lang_code)}'
                else:
                    instruct = f'Think step by step in detail and answer the following questions. The last line of your response must be in the format "... the final answer is $ANSWER" (without quotes), where the format requirements for $ANSWER need to meet the instructions below.\n\n{self.get_answer_format(answer_type, lang_code)}'

                prompt = f'{instruct}\n\n' + question
                weak_model_prompts[prompt].append(i)
                strong_model_prompts[prompt].append(i)

        # 调用弱模型
        _weak_prompts = list(weak_model_prompts.keys()) * weak_bon

        # 调用强模型
        _strong_prompts = list(strong_model_prompts.keys()) * strong_bon

        tasks = [
            self.weak_agent.run(_weak_prompts, max_concurrent_requests, desc=f"[Generate Weak Responses {self.weak_agent.model}]", postprocess_fns=[
                                self.response_postprocess] * len(_weak_prompts)),
            self.strong_agent.run(_strong_prompts, max_concurrent_requests, desc=f"[Generate Strong Responses {self.strong_agent.model}]", postprocess_fns=[
                                  self.response_postprocess] * len(_strong_prompts))
        ]
        results = await aio.gather(*tasks)
        _weak_results, _strong_results = results

        weak_results_mapper = defaultdict(list)
        for (k, v) in _weak_results:
            weak_results_mapper[k].append(v)

        strong_results_mapper = defaultdict(list)
        for (k, v) in _strong_results:
            strong_results_mapper[k].append(v)

        # 答案验证
        verify_queue = []
        for k, v in weak_results_mapper.items():
            for index in weak_model_prompts[k]:
                verify_queue.extend([(index, "weak", _v) for _v in v])

        for k, v in strong_results_mapper.items():
            for index in strong_model_prompts[k]:
                verify_queue.extend([(index, "strong", _v) for _v in v])

        correctness = await self.verify_results(
            verify_queue=verify_queue, batch_solution_str=batch_solution_str, max_concurrent_requests=MAX_CONCURRENT,
            split_names=["weak", "strong"]
        )

        weak, strong = correctness["weak"], correctness["strong"]

        full_rewards = []
        pass_rates = []

        for i in range(len(batch_solution_str)):
            if i in weak:
                base_score = 0.0

                weak_scores = weak[i]
                strong_scores = strong[i]

                pass_rates.append({
                    "weak": f'{np.sum(weak_scores)}/{len(weak_scores)}',
                    "strong": f'{np.sum(strong_scores)}/{len(strong_scores)}',
                })

                try:
                    if len(weak_scores) == 0 or len(strong_scores) == 0:
                        full_rewards.append(base_score)
                        continue

                    # 题目过于简单或困难
                    if np.mean(weak_scores) == 1. or np.mean(weak_scores) < (1.0/weak_bon) or np.mean(weak_scores) == 0.:
                        full_rewards.append(base_score)
                        continue

                    # 总分计算
                    # difficulty = 0.5 * (1.0 - np.mean(weak_scores))
                    # if np.mean(strong_scores) > 0.:
                    #     difficulty += 0.5 * (1.0 - np.mean(strong_scores))

                    difficulty = 0.0
                    difficulty1 = (1.0-math.log2(1+np.sum(weak_scores))/math.log2(
                        1+weak_bon))

                    difficulty += difficulty1
                    if np.mean(strong_scores) > 0.:
                        difficulty2 = (1.0-math.log2(1+np.sum(strong_scores)) /
                                       math.log2(1+strong_bon))
                        difficulty += difficulty2

                    base_score = difficulty
                except Exception as err:
                    pass

                full_rewards.append(base_score)
            else:
                pass_rates.append({})
                full_rewards.append(0.0)
        return full_rewards, pass_rates

    async def llm_as_judge_similarity(
        self,
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
        max_concurrent_requests=128,
    ):
        indices = []
        fabricates, authentics = [], []
        for i, (gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            fabricate = self.parse_solution_fn(sol)
            if fabricate is not None:
                fabricates.append(fabricate)
                authentics.append(gt["authentic_question"])
                indices.append(i)
            else:
                continue

        similarity = await question_similarity(
            agent=self.verify_agent,
            authentic=authentics,
            fabricate=fabricates,
            max_concurrent_requests=max_concurrent_requests
        )

        scores = [0.0] * len(batch_solution_str)
        for sim, index in zip(similarity, indices):
            if sim is None:
                pass
            else:
                if sim < 3:
                    pass
                elif sim >= 4:
                    scores[index] = 1.0
                elif sim == 3:
                    scores[index] = 0.5
        return scores

    def log_solution(self, solution):
        norm = self.parse_solution_fn(solution)
        if norm is None:
            return repr(self.clip_string(solution))
        return repr(self.format_question(norm[0], norm[1]))

    def log_ground_truth(self, ground_truth):
        return repr(self.clip_string(ground_truth["authentic_question"]))

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
        penalty = defaultdict(list)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            parsed = self.doc2query_parse_solution_fn(solution_str)
            if parsed is None:
                penalty[i].append(-2.0)
            else:
                penalty[i].append(0.0)
            for key in ("Format", "QSim"):
                penalty[i].append(self.get_penalties()[key]
                                  (solution_str, ground_truth))

        difficulty_rewards, pass_rates = await self.get_difficulty_reward(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            max_concurrent_requests=MAX_CONCURRENT,
        )

        # # FIXME
        # difficulty_rewards, pass_rates = [
        #     0.0]*len(batch_solution_str), [{}] * len(batch_solution_str)

        similarity_rewards = await self.llm_as_judge_similarity(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            max_concurrent_requests=MAX_CONCURRENT,
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            scores = copy.deepcopy(penalty[i])
            scores.append(difficulty_rewards[i])
            cur_score = 0

            for j, _score in enumerate(scores):
                if _score < 0:
                    cur_score = _score
                    break
                else:
                    if j == 2:  # BLEU
                        if difficulty_rewards[i] > 0:
                            cur_score += _score
                    else:
                        cur_score += _score

            if difficulty_rewards[i] > 0:
                cur_score += similarity_rewards[i]

            penalty_log_str = f'Parse/Format/AnsFeature/QSim={penalty[i]}'

            final_results.append(cur_score)

            if (self.split == "valid" and random.random() < 0.5) or (self.split == "train" and random.random() < 0.1):
                log = True
                log_flag = "[VALID]" if self.split == "valid" else "[TRAIN]"
            else:
                log = False

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f"【Solution】`{self.log_solution(batch_solution_str[i])}`")
                try:
                    print(
                        f"【Ground Truth】`{self.log_ground_truth(batch_ground_truth[i])}`")
                except Exception as err:
                    print(f'[ERROR] {err}')
                print(
                    f'[Final Reward]={cur_score:.3f}|[Pass@{self.difficulty_bon}]={pass_rates[i]}|Sim={similarity_rewards[i]:.3f}|Difficulty={difficulty_rewards[i]}|{penalty_log_str}\n')
        return final_results

    def clip_string(self, s: str):
        if len(s) > 1500:
            return f'{s[:700]}... [省略] ...{s[-800:]}'
        return s


_qwq_longcot_fabricate_qa_compute_score_train = QwQLongCoTFabricateQAComputeScore(
    split="train")
_qwq_longcot_fabricate_qa_compute_score_valid = QwQLongCoTFabricateQAComputeScore(
    split="valid")
qwq_longcot_fabricate_qa_compute_score_train = _qwq_longcot_fabricate_qa_compute_score_train.compute_score
qwq_longcot_fabricate_qa_compute_score_valid = _qwq_longcot_fabricate_qa_compute_score_valid.compute_score
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 问题合成
# ------------------------------------------------------------------------------------------------------------------------------------------------------
