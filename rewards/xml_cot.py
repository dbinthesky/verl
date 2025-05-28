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


EVAL_TEMPLATE = """### **基于标准答案判断回答是否正确**  
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
"""


class XMLCoTComputeScore(object):
    def __init__(self,
                 split="train", parse_result_failure_score=-1.0):
        self.split = split
        self.parse_result_failure_score = parse_result_failure_score

    async def get_accuracy(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth, max_concurrent_requests=256, majority_vote=3):
        #     def postprocess(s):
        #         try:
        #             s = s.strip()
        #             conclusion = s.split("\n")[-1]
        #             conclusion = conclusion[conclusion.index(
        #                 "Answer:")+len("Answer:"):].strip()
        #             if conclusion not in self.MULTICHOICE_LETTER:
        #                 raise PostprocessError(
        #                     f'{conclusion} is not valid')
        #             return conclusion
        #         except Exception as err:
        #             raise PostprocessError(f'{err}')

        #     prompts = []
        #     wo_content_prompts, w_content_prompts = {}, {}

        for i, (solution_str, gt) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            root = xml_cot_parse_solution_fn(solution_str)

            if root is not None:
                conclusion = [
                    child for child in root if child.tag == "conclusion"][0]

                conclusion = conclusion.text.strip()
                print(conclusion)
                print("="*80)
                #             question, options, answer = result

                #             instruct = 'Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format "Answer: $LETTER" (without quotes), where LETTER is one of the option letters. You must first think step by step with very detail thinking process.'
                #             prompt = f'{instruct}\n\n' + self.format_question(
                #                 question, options, answer=None)
                #             wo_content_prompts[prompt] = i

                #             prompts.extend([prompt]*repeat)
                #             prompt = f'[LECTURE]\n{gt["document"]}\n[/LECTURE]\n\n' + f'{instruct}\n\n' + self.format_question(
                #                 question, options, answer=None)
                #             w_content_prompts[prompt] = i

                #             prompts.extend([prompt]*repeat)

                #     results = await respondent_agent.run(prompts, max_concurrent_requests, desc="[Call Respondent]", postprocess_fns=[postprocess]*len(prompts))

                #     wo_contents, w_contents = defaultdict(list), defaultdict(list)

                #     for prompt, conclusion in results:
                #         if prompt in wo_content_prompts:
                #             index = wo_content_prompts[prompt]
                #             wo_contents[index].append(conclusion)
                #         elif prompt in w_content_prompts:
                #             index = w_content_prompts[prompt]
                #             w_contents[index].append(conclusion)
                #         else:
                #             raise NotImplementedError

                #     full_rewards = []
                #     for i in range(len(batch_solution_str)):
                #         if i in wo_contents:
                #             base_score = 0.0

                #             wo_content, w_content = wo_contents[i], w_contents[i]

                #             wo_content = [_ for _ in wo_content if _ is not None]
                #             w_content = [_ for _ in w_content if _ is not None]

                #             ans = batch_ground_truth[i]["options"].tolist().index(
                #                 batch_ground_truth[i]["answer"])
                #             ans = self.MULTICHOICE_LETTER[ans]

                #             wo_content_correct = [_ for _ in wo_content if _ == ans]
                #             w_content_correct = [_ for _ in w_content if _ == ans]

                #             # 完全做不对
                #             if len(wo_content_correct) == 0 or len(w_content_correct) == 0:
                #                 pass
                #             # 全对
                #             elif len(wo_content_correct) == len(wo_content):
                #                 pass
                #             else:
                #                 # [无参考] 正确率1/5-4/5区间
                #                 if len(wo_content_correct) >= 1 and len(wo_content_correct) <= 4:
                #                     if len(wo_content_correct) == 4:
                #                         base_score += 0.25
                #                     else:
                #                         base_score += 0.5

                #                     diff = (len(w_content_correct) / len(w_content)
                #                             ) - ((len(wo_content_correct))/(len(wo_content)))
                #                     diff = max(diff, 0.0)
                #                     base_score += diff

                #             full_rewards.append(base_score)
                #         else:
                #             full_rewards.append(0.0)
                #     return full_rewards
                # ------------------------------------------------------------------------------------------------------------------------------------------------------
                # XML CoT
                # ------------------------------------------------------------------------------------------------------------------------------------------------------
