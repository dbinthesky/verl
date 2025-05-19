import re
import math
import json
import jieba
import random
import aiohttp
import asyncio
import requests
import numpy as np
import asyncio as aio
import tqdm.asyncio
from functools import partial
from asyncio import Semaphore
from abc import abstractmethod
from tqdm import tqdm as tqdm_nonasync
from typing import Dict, Callable, List, Any
from collections import defaultdict

from openai import OpenAI, RateLimitError, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------


RM_URLS = [
    "http://10.130.1.220:27626",
    "http://10.130.1.220:30652",
    "http://10.130.1.220:34800",
    "http://10.130.1.220:27635",
    "http://10.130.1.220:30842",
    "http://10.130.1.220:34782",
    "http://10.130.1.220:34911",
]

FAISS_SEARCH_URLS = [
    "http://10.130.0.128:21783",
    "http://10.130.0.128:24819",
    "http://10.130.0.79:26169",
    "http://10.130.0.79:28140",
    "http://10.130.0.79:23346",
    "http://10.130.0.128:27029"
]


VERIFIER_MODEL_NAME = "qwen25_7B_fabricate_qa_criteria_judge_ehance_0518"
VERIFIER_MODEL_PATH = "http://10.130.133.200:8000/v1"
DEFAULT_PARSE_FAILURE_REWARD = -2.


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
        return solution_str[:solution_str.index("<|im_end|>")]
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
    print("达到最大重试次数，请求失败。")
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
        for batch in tqdm_nonasync(batchify(input_datas, n=64), desc=f'[RM{desc}][{urls}] batchify inference (batch=64)'):
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


async def faiss_search_scores(
        urls: List[str],
        prompts,
        responses,
        desc=""
):
    input_datas = []

    for i, (p, r) in enumerate(zip(prompts, responses)):
        input_data = {
            "prompt": p, "response": r, "id": i
        }
        input_datas.append(input_data)

    rewards, nearest_docs, distances = [], [], []
    if len(input_datas) > 0:
        for batch in tqdm_nonasync(batchify(input_datas, n=64), desc=f'[RM{desc}][{urls}] batchify inference (batch=64)'):
            output_datas = await rm_request_with_retry(urls, batch, suffix="/faiss_reward")
            rewards.extend(output_datas["reward"])
            nearest_docs.extend(output_datas["nearest_docs"])
            distances.extend(output_datas["distances"])

    return {"reward": rewards, "nearest_docs": nearest_docs, "distances": distances}
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 预训练数据挖掘
# ------------------------------------------------------------------------------------------------------------------------------------------------------


async def question_validation(questions, max_concurrent_requests=32):
    _TEMPLATE = """
### 信息完备性检查（优先级排序）  
**1. 是否完全脱离上下文依赖（核心优先级）**  
   - 问题中是否包含"上文数据""前款规则""右图图表"等隐性指代？  
   - 关键校验点：  
     ▶ 剥离所有外部上下文后，问题能否独立完整表述？  
     ▶ 示例：  
       ❌ 错误："按之前方案，如何优化第二部分？"（依赖未说明的"之前方案"和"第二部分"）  
       ✅ 正确："基于2025年4月市场调研报告中'Z产品二线城市渗透率下降8%'的数据，如何制定区域渠道优化方案？"  

**2. 前提条件是否明确（基础约束层）**  
   - 是否隐含未说明的核心假设？（如：资源上限/时间阈值/主体范围/技术边界等）  
   - 关键校验点：  
     ▶ 问题涉及的"主体对象"是否明确？（如：针对C端用户/企业客户/内部团队？）  
     ▶ 约束条件是否清晰？（如：预算≤50万元、实施周期≤2周、需兼容现有系统？）  
     ▶ 示例：  
       ❌ 模糊："如何提升营销效果？"（缺少"产品类型""目标用户""现有转化率"等前提）  
       ✅ 明确："在预算30万元/月、目标用户为25-35岁女性的前提下，如何将D化妆品线上广告ROI从1.2提升至1.8？"  

**3. 核心概念是否定义清晰（语义共识层）**  
   - 关键术语是否存在多义性？是否需锚定具体场景/指标/范围？  
   - 关键校验点：  
     ▶ "用户体验""效率提升""技术优化"等抽象概念是否具化？  
     ▶ 示例：  
       ❌ 歧义："优化系统性能"（未明确"响应速度""并发量""故障率"等具体指标）  
       ✅ 精准："在日均10万次访问量下，如何将系统接口平均响应时间从800ms压缩至500ms以内？"  

**4. 目标指向是否具体可解（执行导向层）**  
   - 问题是否明确"解决路径的终点"？能否转化为可量化/可操作的任务？  
   - 关键校验点：  
     ▶ 避免"如何做好XX"等空泛表述，需包含"在XX场景下，达成XX结果"的结构  
     ▶ 示例：  
       ❌ 笼统："如何优化供应链？"  
       ✅ 可解："在旺季订单量增长40%的预期下，如何通过仓储布局调整将物流时效提升20%？"  


现在需要你对下面的问题进行判断，问题是否满足完备性要求。你需要先仔细思考，再得出结论，回复格式如下
<think>
... ...
</think>

[CONCLUSION START]
COMPLETE=True/False
[CONCLUSION END]
"""

    def postprocess(s):
        conclusion = s[s.index("[CONCLUSION START]"):s.index("[CONCLUSION END]")]
        conclusion = conclusion[conclusion.index("COMPLETE="):]
        if "True" in conclusion:
            return True
        elif "False" in conclusion:
            return False
        return None

    prompts, postprocesses = [], []
    for q in questions:
        prompt = _TEMPLATE + \
            f'\n\n[问题]\n{q}\n\n[输出]\n'
        prompts.append(prompt)
        postprocesses.append(postprocess)

    results = await agent.run(prompts, max_concurrent_requests, desc="[Question Validation]", postprocess_fns=postprocesses)
    scores = [_[1] for _ in results]
    return scores


def parse_solution_fn(solution_str: str):
    try:
        solution_str = postprocess_solution(solution_str)
        xml = re.findall(r'```xml(.*)```', solution_str, re.DOTALL)[0].strip()
    except Exception as err:
        return []
    qa = re.findall(
        r'<question>(.*?)</question>\n*<cot-in-document>(.*?)</cot-in-document>', xml, re.DOTALL)
    return [(_[0].strip(), _[1].strip()) for _ in qa]


class CoTRecall(PenaltyOrReward):
    def __init__(self,
                 postprocess_solution_fn,
                 parse_result_failure_score=0.,
                 ):
        self.postprocess_solution_fn = postprocess_solution_fn
        self.parse_result_failure_score = parse_result_failure_score

    def get_penalty_or_reward(self, solution_str, ground_truth, lang_code=None):
        try:
            qas = self.postprocess_solution_fn(solution_str)
            if qas is None or len(qas) == 0:
                return self.parse_result_failure_score
            raw_doc = ground_truth["ground_truth"]
            return len([_[1] for _ in qas if _[1] in raw_doc])/len(qas)
        except Exception as err:
            return self.parse_result_failure_score


class QwQLongCoTPretrainMiningComputeScore(object):
    JUDGE_CRITERIA_SINGLE_QUESTION_ZH = """### 高质量提问评价标准

    #### 一、核心评价维度与评分细则

    以下从五个维度评估提问质量，每个维度按0-4分打分（4分为最高）：

    1. **相关性**
       - **0分**：问题与文档核心内容无关，涉及背景知识外的话题（如询问作者学术背景、文档格式等）。
       - **1分**：问题指向边缘细节，但对理解核心内容有辅助作用（如追问术语定义，有助于理解后续内容）。
       - **2分**：问题涉及次要逻辑环节，如询问具体公式推导步骤，但未指出其中假设漏洞。
       - **3分**：问题精准定位关键知识点或逻辑断层，并明确指出矛盾点。
       - **4分**：问题直接指向文档中理解晦涩或过于简略的部分，且明确指出未解释清楚的内容，并能引导读者进一步思考。

    2. **逻辑深度**
       - **0分**：停留在事实复述或表面现象。
       - **1分**：基于单一因果关系提问，未涉及多因素关联。
       - **2分**：追问方法或过程，并能指出潜在的假设。
       - **3分**：涉及知识原理或潜在假设，并能指出假设的合理性。
       - **4分**：追问知识体系的底层逻辑或潜在风险，涉及多因素关联，并能引导读者进行系统性思考。

    3. **引导性**
       - **0分**：封闭性问题，答案为“是/否”或单一事实。
       - **1分**：问题仅需单一解释。
       - **2分**：问题隐含步骤提示，并能引导读者思考。
       - **3分**：问题明确引导逻辑链条。
       - **4分**：问题构建系统性思考框架，并能引导读者进行多角度思考，且能提供具体思考路径。

    4. **批判性视角**
       - **0分**：无质疑，仅请求解释或复述内容。
       - **1分**：表面质疑，未具体指出漏洞。
       - **2分**：指出方法局限性或现实矛盾，并能提出改进建议。
       - **3分**：质疑原文假设或逻辑漏洞，并能提出具体质疑点。
       - **4分**：探索替代方案或逆向思考，提供具体替代方法，并能引导读者进行深入探讨。

    5. **具体性**
       - **0分**：问题过于宽泛，难以具体回答。
       - **1分**：问题有一定的具体性，但仍有较大的回答空间。
       - **2分**：问题具体明确，指向明确的内容。
       - **3分**：问题不仅具体明确，还包含背景信息，便于回答。
       - **4分**：问题具体明确，包含详细的背景信息，并能引导读者进行深入思考。

    #### 二、综合评分计算方法

    1. **权重分配**：五个维度权重均等，各占20%。
    2. **得分计算**：总分 = （相关性得分 + 逻辑深度得分 + 引导性得分 + 批判性视角得分 + 具体性得分） × 0.2，满分4分。
       - **示例**：
         - 提问“若数据存在时空相关性，原文的平稳性假设失效，此时应如何修正模型？”
         - 相关性4分（指向假设漏洞），逻辑深度3分（涉及理论适用），引导性3分（隐含修正步骤），批判性视角4分（探索替代方案），具体性3分（具体明确且包含背景信息）。
         - 总分 = (4+3+3+4+3)×0.2 = 3.4分，属于“逻辑较深且具有批判意识的高质量提问”。

    ### 说明

    - **相关性**：增加对问题是否能引导读者深入理解文档核心内容的评估。
    - **逻辑深度**：增加对问题是否能引导读者进行多角度思考的评估。
    - **引导性**：增加对问题是否能引导读者进行系统性思考的评估。
    - **批判性视角**：增加对问题是否能引导读者进行逆向思考的评估。
    - **具体性**：增加对问题具体性和明确性的评估，确保问题能够明确指出未解释清楚的内容。
    """

    JUDGE_CRITERIA_SINGLE_QUESTION_EN = """### High-quality Question Evaluation Criteria

    #### I. Core Evaluation Dimensions and Scoring Rules

    The quality of questions is evaluated from the following five dimensions, with each dimension scored from 0 to 4 points (4 points being the highest).

    1. **Relevance**
       - **0 points**: The question is unrelated to the core content of the document and involves topics outside of the background knowledge (such as asking about the author's academic background, document format, etc.).
       - **1 point**: The question points to marginal details but has an auxiliary role in understanding the core content (such as asking for the definition of a term, which helps in understanding the subsequent content).
       - **2 points**: The question involves secondary logical links, such as asking for the derivation steps of a specific formula, but does not point out the loopholes in the assumptions.
       - **3 points**: The question precisely locates key knowledge points or logical discontinuities and clearly points out contradictions.
       - **4 points**: The question directly points to the parts of the document that are difficult to understand or too brief, clearly points out the content that is not explained clearly, and can guide the reader to think further.

    2. **Logical Depth**
       - **0 points**: Stays at the level of fact repetition or surface phenomena.
       - **1 point**: Asks questions based on a single causal relationship without involving the correlation of multiple factors.
       - **2 points**: Asks about the method or process and can point out potential assumptions.
       - **3 points**: Involves knowledge principles or potential assumptions and can point out the rationality of the assumptions.
       - **4 points**: Asks about the underlying logic or potential risks of the knowledge system, involves the correlation of multiple factors, and can guide the reader to think systematically.

    3. **Guidedness**
       - **0 points**: Closed-ended question with an answer of "yes/no" or a single fact.
       - **1 point**: The question only requires a single explanation.
       - **2 points**: The question implies step hints and can guide the reader to think.
       - **3 points**: The question clearly guides the logical chain.
       - **4 points**: The question constructs a systematic thinking framework, can guide the reader to think from multiple angles, and can provide a specific thinking path.

    4. **Critical Perspective**
       - **0 points**: No questioning, only requests for explanation or repetition of content.
       - **1 point**: Superficial questioning without specifically pointing out loopholes.
       - **2 points**: Points out the limitations of the method or real-world contradictions and can propose improvement suggestions.
       - **3 points**: Questions the assumptions or logical loopholes in the original text and can put forward specific points of doubt.
       - **4 points**: Explores alternative solutions or thinks in reverse, provides specific alternative methods, and can guide the reader to conduct in-depth discussions.

    5. **Specificity**
       - **0 points**: The question is too broad to be answered specifically.
       - **1 point**: The question has some specificity but still leaves a large space for answering.
       - **2 points**: The question is specific and clear, pointing to clear content.
       - **3 points**: The question is not only specific and clear but also includes background information, making it easy to answer.
       - **4 points**: The question is specific and clear, includes detailed background information, and can guide the reader to think deeply.

    #### II. Comprehensive Scoring Calculation Method

    1. **Weight Allocation**: The five dimensions have equal weight, each accounting for 20%.
    2. **Score Calculation**: Total score = (Relevance score + Logical depth score + Guidedness score + Critical perspective score + Specificity score) × 0.2, with a full score of 4 points.
       - **Example**:
         - Question: "If there is spatiotemporal correlation in the data and the stationarity assumption in the original text fails, how should the model be revised?"
         - Relevance: 4 points (points to the assumption loophole), Logical depth: 3 points (involves the application of theory), Guidedness: 3 points (implies the steps of revision), Critical perspective: 4 points (explores alternative solutions), Specificity: 3 points (specific and clear, includes background information).
         - Total score = (4 + 3 + 3 + 4 + 3)×0.2 = 3.4 points, which belongs to "a high-quality question with relatively deep logic and critical awareness".

    ### Explanation

    - **Relevance**: Adds an assessment of whether the question can guide the reader to deeply understand the core content of the document.
    - **Logical Depth**: Adds an assessment of whether the question can guide the reader to think from multiple angles.
    - **Guidedness**: Adds an assessment of whether the question can guide the reader to think systematically.
    - **Critical Perspective**: Adds an assessment of whether the question can guide the reader to think in reverse.
    - **Specificity**: Adds an assessment of the specificity and clarity of the question to ensure that the question can clearly point out the content that is not explained clearly.
    """

    JUDGE_CRITERIA_QUESTION_DIVERSITY_ZH = """### **问题多样性评价标准**：

    ### 一、问题类型多样性
    1. **高阶思维类问题**：
       - **跨域整合**：结合多学科知识提出问题，例如“结合法律和经济学视角，分析知识产权保护的有效性”。
       - **批判性思考**：质疑既有结论，提出研究设计反思的问题，例如“分析文献中样本偏差对研究结果的影响”。
       - **创新性拓展**：提出假设推演与知识延伸的问题，例如“探讨重力常数变化对航天模型的影响”。

    2. **应用分析类问题**：
       - **场景适配**：提出特定场景适用性分析的问题，例如“分析高并发场景下的技术选型”。
       - **风险评估**：提出潜在问题预判与局限性分析的问题，例如“分析自动驾驶技术的瓶颈”。
       - **优劣比较**：提出方案对比与适用情境区分的问题，例如“比较不同研究方法的优劣”。

    3. **基础认知类问题**：
       - **原理推导**：提出逻辑链条构建的问题，例如“从麦克斯韦方程组推导电磁波方程”。
       - **方法解析**：提出技术路线分解的问题，例如“解析层次分析法权重计算”。
       - **概念阐释**：提出基础定义解读的问题，例如“解释机器学习中的过拟合”。
       - **具体应用**：针对文档中的简略或晦涩部分，提出具体问题，例如“‘本改进工艺试验为提高甘油产品质量提供了一个较好的方法’，具体改进了哪些工艺参数？”。

    ### 二、问题视角多样性
    1. **利益相关者视角**：
       - **决策者视角**：提出政策影响与资源配置的问题，例如“碳关税对企业创新的激励”。
       - **研究者视角**：提出研究方法优化的问题，例如“统计模型选择的依据”。
       - **使用者视角**：提出用户体验与产品适配的问题，例如“APP界面设计原理”。

    2. **场景维度多元性**：
       - **实践应用**：提出技术开发、商业决策等现实场景的问题，例如“芯片产业的机遇与挑战”。
       - **学术研究**：提出科研全流程的问题，例如“文献差异原因分析”。
       - **教育教学**：提出教学专属的问题，例如“试题设计思路分析”。

    3. **学科覆盖广度**：
       - **交叉学科**：提出跨学科融合的问题，例如“认知心理学与多媒体教学”。
       - **人文社科**：提出法学、经济学等领域的问题，例如“历史赋税制度的经济逻辑”。
       - **自然科学**：提出基础学科与应用技术的问题，例如“柯西不等式向量形式证明”。

    ### 三、分析深度多样性
    1. **定性与定量分析融合度**：
       - **定量建模**：提出数据统计与数学建模的问题，例如“使用Logistic回归进行风险评估”。
       - **定性描述**：提出现象归纳与特征阐释的问题，例如“分析小说中的主题思想”。

    2. **正向推导与逆向思考**：
       - **正向推导**：从已知条件出发，推导出结论，例如“从益气活血法的疗效出发，推导其作用机制”。
       - **逆向思考**：从结果出发，反推原因，例如“分析自动驾驶技术的瓶颈，反推其技术难题”。

    3. **单维分析与系统分析**：
       - **单维分析**：从单一角度分析问题，例如“从单个药物的作用机制出发，分析其在益气活血法中的贡献”。
       - **系统分析**：从多个角度综合分析问题，例如“分析益气活血法在不同地区和不同人群中的适用性”。

    ### 四、思维路径多样性
    1. **归纳与演绎**：
       - **归纳**：从具体案例中归纳出一般规律，例如“从历史案例中归纳出取消协议的常见原因”。
       - **演绎**：从一般规律推导出具体结论，例如“从物理原理出发，推导引射喷管的性能”。

    2. **发散与收敛**：
       - **发散**：从不同角度探讨问题，例如“从不同角度探讨取消协议的可能后果”。
       - **收敛**：将不同角度的分析综合成一个结论，例如“综合各方意见，提出最优解决方案”。

    3. **线性与非线性思维**：
       - **线性思维**：按步骤顺序分析问题，例如“分析自动驾驶技术的渐进过程”。
       - **非线性思维**：从非线性角度分析问题，例如“分析自动驾驶技术的突变
    """
    JUDGE_CRITERIA_QUESTION_DIVERSITY_EN = """### Problem Diversity Evaluation Criteria:

    ### I. Problem Type Diversity
    1. **High-order Thinking Problems**:
       - **Cross-domain Integration**: Propose questions that combine knowledge from multiple disciplines. For example, "Analyze the effectiveness of intellectual property protection from the perspectives of law and economics."
       - **Critical Thinking**: Raise questions that challenge existing conclusions and involve reflections on research designs. For example, "Analyze the impact of sample bias in the literature on research results."
       - **Innovative Expansion**: Put forward questions that involve hypothesis deduction and knowledge extension. For example, "Explore the impact of changes in the gravitational constant on aerospace models."

    2. **Application and Analysis Problems**:
       - **Scenario Adaptability**: Pose questions about the analysis of applicability in specific scenarios. For example, "Analyze the technology selection in high-concurrency scenarios."
       - **Risk Assessment**: Raise questions about predicting potential problems and analyzing limitations. For example, "Analyze the bottlenecks of autonomous driving technology."
       - **Advantages and Disadvantages Comparison**: Propose questions about comparing different solutions and distinguishing applicable situations. For example, "Compare the advantages and disadvantages of different research methods."

    3. **Basic Cognitive Problems**:
       - **Principle Deduction**: Put forward questions about constructing logical chains. For example, "Deduce the electromagnetic wave equation from Maxwell's equations."
       - **Method Analysis**: Raise questions about decomposing technical routes. For example, "Analyze the weight calculation in the Analytic Hierarchy Process."
       - **Concept Interpretation**: Pose questions about interpreting basic definitions. For example, "Explain overfitting in machine learning."
       - **Specific Application**: Raise specific questions regarding the concise or obscure parts in documents. For example, "In the statement 'This improved process experiment provides a good method for improving the quality of glycerol products', which process parameters are specifically improved?"

    ### II. Problem Perspective Diversity
    1. **Stakeholder Perspectives**:
       - **Decision-maker Perspective**: Raise questions about the impact of policies and resource allocation. For example, "The incentive of carbon tariffs for corporate innovation."
       - **Researcher Perspective**: Pose questions about optimizing research methods. For example, "The basis for choosing a statistical model."
       - **User Perspective**: Raise questions about user experience and product adaptation. For example, "The design principles of an APP interface."

    2. **Diversity of Scenario Dimensions**:
       - **Practical Application**: Propose questions related to real-world scenarios such as technology development and business decision-making. For example, "The opportunities and challenges in the chip industry."
       - **Academic Research**: Raise questions covering the entire process of scientific research. For example, "Analysis of the reasons for differences in literature."
       - **Education and Teaching**: Pose questions specific to teaching. For example, "Analysis of the design ideas of test questions."

    3. **Breadth of Subject Coverage**:
       - **Interdisciplinary**: Raise questions that integrate multiple disciplines. For example, "Cognitive psychology and multimedia teaching."
       - **Humanities and Social Sciences**: Pose questions in fields such as law and economics. For example, "The economic logic of historical tax systems."
       - **Natural Sciences**: Raise questions in basic disciplines and applied technologies. For example, "Proof of the vector form of the Cauchy inequality."

    ### III. Diversity of Analysis Depth
    1. **Integration of Qualitative and Quantitative Analysis**:
       - **Quantitative Modeling**: Raise questions about data statistics and mathematical modeling. For example, "Use Logistic regression for risk assessment."
       - **Qualitative Description**: Put forward questions about summarizing phenomena and explaining characteristics. For example, "Analyze the theme in a novel."

    2. **Forward Deduction and Reverse Thinking**:
       - **Forward Deduction**: Starting from known conditions, derive conclusions. For example, "Starting from the curative effect of the method of replenishing qi and promoting blood circulation, deduce its mechanism of action."
       - **Reverse Thinking**: Starting from the results, infer the causes. For example, "Analyze the bottlenecks of autonomous driving technology and infer its technical difficulties."

    3. **Single-dimensional Analysis and Systematic Analysis**:
       - **Single-dimensional Analysis**: Analyze problems from a single perspective. For example, "Starting from the mechanism of action of a single drug, analyze its contribution to the method of replenishing qi and promoting blood circulation."
       - **Systematic Analysis**: Analyze problems comprehensively from multiple angles. For example, "Analyze the applicability of the method of replenishing qi and promoting blood circulation in different regions and among different groups of people."

    ### IV. Diversity of Thinking Paths
    1. **Induction and Deduction**:
       - **Induction**: Summarize general laws from specific cases. For example, "Summarize the common reasons for canceling agreements from historical cases."
       - **Deduction**: Derive specific conclusions from general laws. For example, "Starting from physical principles, deduce the performance of an ejector nozzle."

    2. **Divergent and Convergent Thinking**:
       - **Divergent**: Explore problems from different angles. For example, "Explore the possible consequences of canceling agreements from different angles."
       - **Convergent**: Synthesize analyses from different angles into a conclusion. For example, "Propose the optimal solution by integrating various opinions."

    3. **Linear and Non-linear Thinking**:
       - **Linear Thinking**: Analyze problems step by step. For example, "Analyze the progressive process of autonomous driving technology."
       - **Non-linear Thinking**: Analyze problems from a non-linear perspective. For example, "Analyze the sudden changes in autonomous driving technology."

    """

    def __init__(self,
                 split="train",
                 parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD,
                 weights_file="/cpfs01/shared/llm_ddd/tongjian/es_seeds/seed_all_0510.jsonl",
                 max_qa_pairs=50,
                 ):
        self.split = split
        self.parse_result_failure_score = parse_result_failure_score

        self.recall = CoTRecall(
            postprocess_solution_fn=parse_solution_fn)

        self.seed_weights = {}
        self.hits = defaultdict(float)

        self.max_qa_pairs = max_qa_pairs
        self.single_qa_pair_max_reward = 2.0 / max_qa_pairs

        with open(weights_file, "rt") as f:
            for i, line in enumerate(f):
                example = json.loads(line)
                dataset = example["self_improvement"]["dataset"]
                if dataset == "cmmlu":
                    self.seed_weights[i] = 0.5
                elif dataset == "gpqa":
                    self.seed_weights[i] = 2.0
                elif dataset == "mmlu":
                    self.seed_weights[i] = 0.5
                elif dataset == "mmlu_pro":
                    self.seed_weights[i] = 1.0
                elif dataset == "super_gpqa":
                    self.seed_weights[i] = 2.0

    def get_penalties(self) -> Dict[str, Callable]:
        return {
            "CoTRecall": self.recall.get_penalty_or_reward
        }

    def get_penalty_coef(self):
        return {
            "CoTRecall": 1.00
        }

    async def process_queue(self, queue, semaphore):
        """处理单个队列，确保队列内任务串行执行"""
        async with semaphore:  # 限制并发队列数量
            results = []
            for task in queue:
                result = await task
                results.append(result)
            return results

    async def run_tasks_in_queues(self, tasks, n):
        """将任务分成n个队列并行执行"""
        # 创建n个队列
        queues = [[] for _ in range(n)]

        # 平均分配任务到各个队列
        for i, task in enumerate(tasks):
            queues[i % n].append(task)

        # 创建信号量限制并发队列数量
        semaphore = Semaphore(n)

        # 并行处理所有队列
        queue_results = await asyncio.gather(
            *[self.process_queue(queue, semaphore) for queue in queues]
        )

        # 展平结果列表（保持原始顺序）
        flattened_results = []
        for i in range(len(tasks)):
            queue_idx = i % n
            task_idx = i // n
            flattened_results.append(queue_results[queue_idx][task_idx])

        return flattened_results

    async def get_single_question_judge_rm_rewards(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
            urls=RM_URLS,
            default_penalty=-1.0,
            reward_rectifier_value=0.
    ):
        """
            评价单条提问质量
        """
        indices, sizes = [], []
        addition_judges = []
        new_batch_solution_strs = []

        for i, (_gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            lang_code = contain_chinese(_gt["ground_truth"])
            if lang_code == "zh":
                judge_template = self.JUDGE_CRITERIA_SINGLE_QUESTION_ZH
            else:
                judge_template = self.JUDGE_CRITERIA_SINGLE_QUESTION_EN

            qas = parse_solution_fn(sol)

            if lang_code == "zh":
                questions = [
                    f'- 原文中需要进行提问的部分： \n"{_[1]}"\n- 提问：\n"{_[0]}"' for _ in qas]
                judge_prompt = f'任务：针对文档中理解晦涩、过于简略的部分进行提问。\n\n\n# 评价标准\n{judge_template}'
            else:
                questions = [
                    f'- Identify the parts in the original text that need to be questioned.\n"{_[1]}"\n- The question raised.\n"{_[0]}"' for _ in qas]
                judge_prompt = f'Task: Ask questions about the obscure and overly brief parts in the document.\n\n\n# Judge Criteria\n{judge_template}'

            for question in questions:
                addition_judges.append({"ground_truth": judge_prompt})
                new_batch_solution_strs.append(question)
            if len(questions) > 0:
                sizes.append(len(questions))
                indices.append(i)

        tasks = []
        n = len(urls)

        for i, batch in enumerate(batchify(zip(addition_judges, new_batch_solution_strs), n=64)):
            addition_judge = [_[0] for _ in batch]
            new_batch_solution_str = [_[1] for _ in batch]
            tasks.append(
                compute_rm_score(
                    batch_solution_str=new_batch_solution_str,
                    batch_ground_truth=addition_judge,
                    postprocess_solution_fn=lambda x: x,
                    parse_result_failure_score=self.parse_result_failure_score,
                    desc="-single_question_judge",
                    urls=[urls[i % n]]
                )
            )

        results = await self.run_tasks_in_queues(tasks, n=n)

        rewards = []
        for _ in results:
            rewards.extend(_)
        rewards_group = []
        for size in sizes:
            rewards_group.append(rewards[:size])
            rewards = rewards[size:]

        full_rewards = []
        for i in range(len(batch_solution_str)):
            if i in indices:
                full_rewards.append(np.mean(rewards_group[indices.index(i)]))
            else:
                full_rewards.append(default_penalty)
        return full_rewards

    async def question_validation(
        self,
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
        max_concurrent_requests=32,
    ):
        questions = []

        indices, sizes = [], []
        for i, (solution_str, gt) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            qas = parse_solution_fn(solution_str)
            questions.extend([_[0] for _ in qas])
            if len(qas) > 0:
                indices.append(i)
                sizes.append(len(qas))

        results = await question_validation(
            questions, max_concurrent_requests=max_concurrent_requests)
        rewards = results

        rewards_group = []
        for size in sizes:
            rewards_group.append(rewards[:size])
            rewards = rewards[size:]

        full_rewards = []
        for i in range(len(batch_solution_str)):
            if i in indices:
                group = rewards_group[indices.index(i)]
                group = [_ for _ in group if _ is not None]
                if len(group) > 0:
                    return group.count(True) / len(group)
                else:
                    full_rewards.append(0.)
            else:
                full_rewards.append(-1.0)
        return full_rewards

    async def get_question_diversity_rm_rewards(
        self,
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
        urls=RM_URLS,
        default_penalty=-0.1,
    ):
        """
            整体评价提问的多样性
        """
        addition_judges = []
        new_batch_solution_strs = []
        indices = []

        for i, (_gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            lang_code = contain_chinese(_gt["ground_truth"])
            if lang_code == "zh":
                judge_template = self.JUDGE_CRITERIA_QUESTION_DIVERSITY_ZH
            else:
                judge_template = self.JUDGE_CRITERIA_QUESTION_DIVERSITY_EN

            qas = parse_solution_fn(sol)

            if lang_code == "zh":
                questions = [
                    f'- 原文中需要进行提问的部分： \n"{_[1]}"\n- 提问：\n"{_[0]}"' for _ in qas]
                judge_prompt = f'任务：针对文档中理解晦涩、过于简略的部分进行提问。\n\n\n# 评价标准\n{judge_template}'
            else:
                questions = [
                    f'- Identify the parts in the original text that need to be questioned.\n"{_[1]}"\n- The question raised.\n"{_[0]}"' for _ in qas]
                judge_prompt = f'Task: Ask questions about the obscure and overly brief parts in the document.\n\n\n# Judge Criteria\n{judge_template}'

            addition_judges.append({
                "ground_truth": judge_prompt
            })
            indices.append(i)
            new_batch_solution_strs.append("\n\n".join(questions))

        tasks = []
        n = len(urls)

        for i, batch in enumerate(batchify(zip(addition_judges, new_batch_solution_strs), n=64)):
            addition_judge = [_[0] for _ in batch]
            new_batch_solution_str = [_[1] for _ in batch]
            tasks.append(
                compute_rm_score(
                    batch_solution_str=new_batch_solution_str,
                    batch_ground_truth=addition_judge,
                    postprocess_solution_fn=lambda x: x,
                    parse_result_failure_score=self.parse_result_failure_score,
                    desc="-question_diversity_judge",
                    urls=[urls[i % n]]
                )
            )

        results = await self.run_tasks_in_queues(tasks, n=n)
        rewards = []
        for _ in results:
            rewards.extend(_)

        full_rewards = []
        for i in range(len(batch_solution_str)):
            if i in indices:
                full_rewards.append(rewards[indices.index(i)])
            else:
                full_rewards.append(default_penalty)
        return full_rewards

    def get_weight(self, doc_id, distance, r0=1.0, r_min=0.1, k=0.05):
        # 计算衰减  \max\left( R_{\text{min}}, \ R_0 - k \cdot \ln(n + 1) \right)
        hit = self.hits[doc_id]
        r_coef = max(r0 - k * math.log(hit + 1), r_min)
        self.hits[doc_id] += distance
        weight = self.seed_weights[doc_id]
        return weight * r_coef

    def union_reward_for_doc(self, reward):
        scores = []
        for r, doc_id, dist in zip(reward["reward"], reward["nearest_docs"], reward["distances"]):
            weight = self.get_weight(doc_id, dist)
            score = weight * (r + dist) / 2 * self.single_qa_pair_max_reward
            scores.append(score)

        # 边际系数公式
        scores = sorted(scores, reverse=True)
        scores = scores[:self.max_qa_pairs]
        final_score = 0.0
        for i, score in enumerate(scores):
            final_score += score * (1/(1+0.1*(i-1)))
        return final_score

    async def bank_covery_rewards(
        self,
        batch_data_sources,
        batch_solution_str,
        batch_ground_truth,
        urls=FAISS_SEARCH_URLS,
        default_penalty=-0.1
    ):
        prompts, responses = [], []
        indices, sizes = [], []

        for i, (_gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            qas = parse_solution_fn(sol)
            for qa in qas:
                prompts.append(qa[0])
                responses.append(qa[1])

            if len(qas) > 0:
                sizes.append(len(qas))
                indices.append(i)

        tasks = []
        n = len(urls)

        for i, batch in enumerate(batchify(zip(prompts, responses), n=64)):
            _prompts = [_[0] for _ in batch]
            _responses = [_[1] for _ in batch]
            tasks.append(
                faiss_search_scores(
                    prompts=_prompts,
                    responses=_responses,
                    desc="-bank_covery",
                    urls=[urls[i % n]]
                )
            )

        results = await self.run_tasks_in_queues(tasks, n=n)
        rewards, nearest_docs, distances = [], [], []

        for _ in results:
            rewards.extend(_["reward"])
            nearest_docs.extend(_["nearest_docs"])
            distances.extend(_["distances"])

        rewards_group = []
        for size in sizes:
            rewards_group.append({
                "reward": rewards[:size],
                "nearest_docs": nearest_docs[:size],
                "distances": distances[:size],
            })

            rewards = rewards[size:]
            nearest_docs = nearest_docs[size:]
            distances = distances[size:]

        self.union_reward_for_doc(rewards_group[0])

        full_rewards = []
        for i in range(len(batch_solution_str)):
            if i in indices:
                full_rewards.append(
                    self.union_reward_for_doc(rewards_group[indices.index(i)]))
            else:
                full_rewards.append(default_penalty)
        return full_rewards

    def compute_score(self,
                      batch_data_sources,
                      batch_solution_str,
                      batch_ground_truth,
                      ):
        async def main():
            return await self._compute_score(batch_data_sources, batch_solution_str, batch_ground_truth)
        return asyncio.run(main())

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
        penalty = defaultdict(dict)
        for i, (data_source, solution_str, ground_truth) in enumerate(zip(batch_data_sources, batch_solution_str, batch_ground_truth)):
            for key, fn in self.get_penalties().items():
                penalty[key][i] = fn(solution_str, ground_truth)
        question_diversity = await self.get_question_diversity_rm_rewards(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )
        question_quality = await self.get_single_question_judge_rm_rewards(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )
        bank_covery = await self.bank_covery_rewards(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )
        validation = await self.question_validation(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            penalty_log_str = []
            _reward = question_diversity[i] + \
                question_quality[i] + bank_covery[i] + validation[i]

            for name, _penalty in penalty.items():
                if i in _penalty:
                    _reward += _penalty[i] * self.get_penalty_coef()[name]
                    try:
                        penalty_log_str.append(
                            f'{name}={_penalty[i]:.3f}*{self.get_penalty_coef()[name]}')
                    except Exception as _:
                        pass
            final_results.append(_reward)

            qas = self.get_qa(batch_solution_str[i])

            if self.split == "valid" or (self.split == "train" and random.random() < 0.01):
                log = True
                log_flag = "[VALID]" if self.split == "valid" else "[TRAIN]"
            else:
                log = False

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f'【Raw】`{self.log_ground_truth(batch_ground_truth[i])}`')
                print(
                    f'[Final Reward]={_reward:.3f}|VALIDATION={validation[i]:.3f}|DIVERSITY={question_diversity[i]:.3f}|QUALITY={question_quality[i]:.3f}|COVERY={bank_covery[i]:.3f}|{"|".join(penalty_log_str)}[{self.get_penalty_coef()}]\n')
                for j, (q, a) in enumerate(qas):
                    print(
                        f'\t【新增提问{j}】Q: {repr(q)} A: {repr(a)}')
        return final_results

    def get_qa(self, solution):
        qas = parse_solution_fn(solution)
        return qas

    def log_ground_truth(self, ground_truth):
        return repr(self.clip_string(ground_truth["ground_truth"]))

    def clip_string(self, s: str):
        if len(s) > 1500:
            return f'{s[:700]}... [省略] ...{s[-800:]}'
        return s


_qwq_longcot_pretrain_mining_compute_score_train = QwQLongCoTPretrainMiningComputeScore(
    split="train")
_qwq_longcot_pretrain_mining_compute_score_valid = QwQLongCoTPretrainMiningComputeScore(
    split="valid")
qwq_longcot_pretrain_mining_compute_score_train = _qwq_longcot_pretrain_mining_compute_score_train.compute_score
qwq_longcot_pretrain_mining_compute_score_valid = _qwq_longcot_pretrain_mining_compute_score_valid.compute_score
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 预训练数据挖掘
# ------------------------------------------------------------------------------------------------------------------------------------------------------
