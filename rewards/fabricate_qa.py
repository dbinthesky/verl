import re
import jieba
import random
import requests
import tqdm.asyncio
import asyncio as aio
from functools import partial
from functools import partial
from abc import abstractmethod
from typing import Any, Dict, Callable
from tqdm import tqdm as tqdm_nonasync
from collections import namedtuple, defaultdict


from openai import OpenAI, RateLimitError, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------

RM_URLS = [
    "http://10.130.0.174:5020"
]
VERIFIER_MODEL_NAME = "qwen25_72B_instruct"
VERIFIER_MODEL_PATH = "http://10.130.247.138:8000/v1"
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
        "timeout": 180,
        "max_tokens": 4096,
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


def rm_request_with_retry(RM_URLS, data, max_retries=3, retry_delay=1, suffix="/reward"):
    retries = 0
    while retries < max_retries:
        try:
            url = random.choice(RM_URLS)
            response = requests.post(f'{url}{suffix}', json=data, timeout=600)
            response.raise_for_status()  # 如果状态码不是 200，抛出异常
            return response.json()
        except requests.RequestException as e:
            print(f"请求(数据总量={len(data)})失败，错误信息: {e}，重试第 {retries + 1} 次...")
            retries += 1
            if retries < max_retries:
                time.sleep(retry_delay)
    print("达到最大重试次数，请求失败。")
    return None


def compute_rm_score(
        batch_solution_str,
        batch_ground_truth,
        postprocess_solution_fn,
        parse_result_failure_score=DEFAULT_PARSE_FAILURE_REWARD,
        judge_prompt_key="ground_truth"
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
        for batch in tqdm_nonasync(batchify(input_datas, n=128), desc=f'[RM][{RM_URLS}] batchify inference (batch=128)'):
            output_datas = rm_request_with_retry(RM_URLS, batch)
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

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# BASE
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Criteria构造
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def postprocess_solution(solution_str):
    if "<|im_end|>" in solution_str:
        return solution_str[:solution_str.index("<|im_end|>")]
    return solution_str


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
        if not all(int(_) > 0 and int(_) <= 5 for _ in scores):
            return None
    except Exception as err:
        return None
    return conclusion


def get_total_score(criteria):
    scores = re.findall(r'SCORE=(\d+)', criteria)
    scores = [int(_) for _ in scores]
    return sum(scores)


async def criteria_get_score(questions, criteria, max_concurrent_requests=32, parse_result_failure_score=-1.0):
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

    TEMPLATE = """
# 目标概述 
给定一道题目，和一个要点评分表，你需要严格按照要点评分表的内容和分值，逐项对问题进行打分，并在最后给出最终分数。

## 回答格式要求
1. 你应当逐项按照列点的方式，对问题和评价项目进行比较，计算单个项目的分值，注意分值应当是整数，避免小数的出现，例如0.5
2. 包含计算总分的计算步骤
3. 在回答的最后部分，必须以“最终得分：***”这样的格式，给出总分


下面提供你一个具体的范例，你需要参考范例进行回答


[题目]
For a certain site, cement-mixed columns are planned to be used for ground reinforcement. It is known that the foundation depth is 2.0m, the diameter of the mixing column is 600mm, the column length is 14.0m, and the column strength is $f_{\\mathrm{cu}}=0.8\\, \\mathrm{MPa}$. The column strength reduction factor is taken as $\\eta=0.3$, the inter-column soil bearing capacity reduction factor is $\\beta=0.6$, and the column end soil bearing capacity reduction factor is $\\alpha=0.4$. The center-to-center spacing of the mixing columns is 1.0m, arranged in an equilateral triangular pattern. Which of the following options is appropriate for the characteristic value of the bearing capacity of the composite foundation with mixed columns?

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
### 评分表逐项打分（列点格式）  
1. **题目是否属于选择题型**  
   - 得分：2  
   - 说明：题目以“Which of the following options is appropriate...”结尾，明确要求从选项中选择答案，属于选择题型。  

2. **题目是否涉及土木工程领域的地基处理**  
   - 得分：2  
   - 说明：题目描述水泥搅拌桩（cement-mixed columns）用于地基加固（ground reinforcement），属于土木工程地基处理范畴。  

3. **题目是否要求计算复合地基的承载力**  
   - 得分：2  
   - 说明：问题目标为求解“复合地基承载力特征值”（characteristic value of the bearing capacity of the composite foundation），需进行承载力计算。  

4. **题目是否提供所有必要参数**  
   - 得分：2  
   - 说明：题目完整提供了桩长（14.0m）、桩径（600mm）、桩体强度（\(f_{\mathrm{cu}}=0.8\,\mathrm{MPa}\)）、折减系数（\(\eta=0.3, \beta=0.6, \alpha=0.4\)）、桩间距（1.0m）及排列方式（等边三角形），无参数缺失。  

5. **题目是否涉及多个折减系数的综合应用**  
   - 得分：2  
   - 说明：涉及桩体强度折减系数 \(\eta\)、桩间土承载力折减系数 \(\beta\)、桩端土承载力折减系数 \(\alpha\)，需在公式中综合考虑各系数的作用。  

6. **题目是否要求考虑桩间距和排列方式的影响**  
   - 得分：2  
   - 说明：桩间距（1.0m）和等边三角形排列用于计算置换率 \(m\)（即桩的面积占总地基面积的比例），直接影响复合地基承载力公式中的权重分配。  

7. **题目是否需要综合运用公式进行数值计算**  
   - 得分：2  
   - 说明：需应用复合地基承载力公式（如 \(f_{\mathrm{spk}} = m \cdot \eta \cdot f_{\mathrm{cu}} + \beta(1-m) \cdot f_{\mathrm{sk}}\)，其中 \(f_{\mathrm{sk}}\) 可能涉及桩端土承载力），并结合几何参数计算置换率，属于数值计算类问题。  

8. **题目是否涉及桩端土和桩间土的承载力计算**  
   - 得分：1  
   - 说明：桩间土承载力通过 \(\beta(1-m) \cdot f_{\mathrm{sk}}\) 体现，桩端土承载力通过桩体承载力公式中的桩端阻力项（需考虑 \(\alpha\) 折减）间接涉及，考查两者的协同作用。  

9. **题目是否要求选择一个合适的承载力值**  
   - 得分：1  
   - 说明：题目要求从选项中选择“合适的”承载力特征值，属于结果选择类问题，需结合计算结果匹配选项。  

10. **题目是否包含清晰的题干描述**  
    - 得分：1  
    - 说明：题干明确列出所有参数、工程背景（地基加固）及问题目标，无歧义或模糊表述，信息完整。  

11. **题目是否设置合理的选项**  
    - 得分：1  
    - 说明：虽然题目未列出具体选项，但作为标准选择题，默认选项设置合理（如涵盖计算可能的误差范围或常见错误值）。  

12. **题目是否考查记忆和理解能力**  
    - 得分：1  
    - 说明：需记忆复合地基承载力公式的结构及各参数定义（如折减系数的物理意义），考查对基本概念的理解。  

13. **题目是否考查应用和分析能力**  
    - 得分：1  
    - 说明：需将公式应用于具体参数，分析桩间距和排列方式对置换率的影响，以及折减系数如何调整桩体和土体的承载力贡献。  

14. **题目是否考查创造能力**  
    - 得分：0  
    - 说明：问题为公式直接应用，无需创新方法或创造性思维，仅需按步骤计算，故不涉及创造能力考查。  


### 最终分数计算  
利用乘法运算（几个相同的数相加用乘法表示更简便）将所有得分相加：
$
\begin{align*}
\begin{align*}
&2 + 2 + 2 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 0\\
=&2\times7 + 1\times6+0\\
=&14 + 6+0\\
=&20+0\\
=&20
\end{align*}

最终得分：20
"""

    prompts, postprocesses = [], []
    for q, c in zip(questions, criteria):
        prompt = TEMPLATE + \
            f'\n\n现在需要你对下面的问题计算分数。\n\n[题目]\n{q}\n\n[打分表]\n{c}\n\n[输出]\n'
        prompts.append(prompt)
        postprocesses.append(
            partial(postprocess, max_score=get_total_score(c)))

    results = await agent.run(prompts, max_concurrent_requests, desc="[Judge Question w Criteria]", postprocess_fns=postprocesses)
    scores = [_[1] if _[1]
              is not None else parse_result_failure_score for _ in results]
    return scores


async def decode_to_question(criteria, max_concurrent_requests=32):
    def postprocess(s):
        try:
            thought = re.findall(r'think>.*</think>', s, re.DOTALL)[0]
            conclusion = s.replace(thought, "")
            conclusion = conclusion[conclusion.index(
                "[QUESTION]")+len("[QUESTION]"):conclusion.index("[/QUESTION]")].strip()
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
            f'\n\n现在需要你基于下面打分表构造一道完美符合要求的问题。\n\n[打分表]\n{c}\n\n[输出]\n'
        prompts.append(prompt)

    results = await agent.run(prompts, max_concurrent_requests, desc="[Fabricate QA]", postprocess_fns=[postprocess]*len(prompts))
    return results

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Criteria构造
# ------------------------------------------------------------------------------------------------------------------------------------------------------
