import re
import os
import math
import copy
import json
import uuid
import math
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
1. **理解题意和标准答案**：题目要求计算在单位正方形区域内，满足拉普拉斯方程且四边φ=0的条件下，经Gauss-Seidel迭代法100次后φ(0.5, 0.5)的值（保留两位小数），标准答案为\\boxed{0.00}。
2. **提取用户回答的最终结论**：用户回答通过分析推导，最终结论为\\boxed{0.25}，去除过程性描述后，核心结果为0.25。
3. **对比结果**：用户回答的最终结论0.25与标准答案0.00不一致，且不存在数值题的合理误差范围，因此判定为错误。

```json
{
  "判断结果": "错误"
}
```


#### **输入：**
```json
{
  "题目": "Suppose we draw this SVG path element:M 33.34742,22.43250 L 62.47296,22.43250 M 62.47296,22.43250 L 50.84847,35.60722 M 50.84847,35.60722 L 58.79792,26.59764 M 58.79792,26.59764 L 46.06248,41.03147 M 46.06248,41.03147 L 33.34742,22.43250 M 59.28939,7.78141 L -8.37042,10.57501 .Out of the following shapes:\n1. rectangle that is not a square and with no diagonals drawn\n2. square with no diagonals drawn\n3. triangle that is not a right triangle\n4. right triangle\n5. parallelogram that is not a rectangle and with no diagonals drawn\n6. trapezoid with exactly one pair of parallel sides and with no diagonals drawn\n7. irregular convex pentagon with no diagonals drawn\n8. irregular concave pentagon with no diagonals drawn\n9. regular pentagon with no diagonals drawn\n10. regular hexagon with no diagonals drawn\nwhich one can be viewed when the lines in the SVG are visualized? Note that a shape with n sides should not necessarily be drawn by n lines; e.g., a triangle might be drawn by 4 lines, two of which collinear. \nCoordinates have been rounded to 5 decimal places so ignore slight differences. Solve this problem step by step and provide the final answer within \\boxed{}. For example: \"Final Answer: \\boxed{3}.\"",
  "标准答案": '\n{\n    "data_source": "BbehGeometricShapes",\n    "ground_truth": {\n      "Paths": "M 33.34742,22.43250 L 62.47296,22.43250 M 62.47296,22.43250 L 50.84847,35.60722 M 50.84847,35.60722 L 58.79792,26.59764 M 58.79792,26.59764 L 46.06248,41.03147 M 46.06248,41.03147 L 33.34742,22.43250 M 59.28939,7.78141 L -8.37042,10.57501",\n      "ans": "3"\n    }\n  }\n',
  "用户回答": '\nTo determine the shape formed by the SVG path, we analyze the given line segments and their connections:\n\n### Step 1: Identify Line Segments\nThe SVG path consists of 6 line segments, defined by their start and end coordinates:\n1. \\( (33.34742, 22.43250) \\) to \\( (62.47296, 22.43250) \\)\n2. \\( (62.47296, 22.43250) \\) to \\( (50.84847, 35.60722) \\)\n3. \\( (50.84847, 35.60722) \\) to \\( (58.79792, 26.59764) \\)\n4. \\( (58.79792, 26.59764) \\) to \\( (46.06248, 41.03147) \\)\n5. \\( (46.06248, 41.03147) \\) to \\( (33.34742, 22.43250) \\)\n6. \\( (59.28939, 7.78141) \\) to \\( (-8.37042, 10.57501) \\)\n\n### Step 2: Detect Closed Shapes\nSegments 1–5 form a closed loop: the end of segment 5 matches the start of segment 1. This creates a 5-sided polygon (pentagon). The 6th segment is separate, with endpoints not coinciding with any vertices of the pentagon, so it does not affect the pentagon’s structure.\n\n### Step 3: Classify the Pentagon\n- **Regular vs. Irregular**: A regular pentagon requires all sides and angles to be equal. Calculating side lengths (distances between consecutive vertices) shows they are unequal (29.12554, ~17.57, ~12.015, ~19.25, ~22.53), so it is irregular.\n- **Convex vs. Concave**: Convex polygons have all internal angles < 180° and consistent turning directions (verified using cross products of consecutive edge vectors, all positive, indicating counterclockwise turns). Thus, it is convex.\n- **Diagonals**: The pentagon’s segments are its sides, with no lines connecting non-consecutive vertices (no diagonals).\n\n### Conclusion\nThe shape is an irregular convex pentagon with no diagonals drawn.\n\nFinal Answer: \x08oxed{7}\n',
}
```

#### **输出：**
### 分析过程：
1. **理解题意和标准答案**：题目要求判断给定SVG路径形成的形状，选项共10个。标准答案中明确“ans”为“3”，即对应的形状是“三角形 that is not a right triangle”（非直角三角形）。
2. **提取用户回答的最终结论**：用户回答通过分析推导，最终结论为\\boxed{7}，即认为形状是“irregular convex pentagon with no diagonals drawn”（不规则凸五边形且无对角线），去除过程性描述后，核心结果为7。
3. **对比结果**：用户回答的最终结论7与标准答案3不一致，因此判定为错误。

```json
{
  "判断结果": "错误"
}
```


#### **输入：**
```json
{
  "题目": '你正在解决一个Nonogram谜题。根据行和列的数字线索填充网格：\n\n规则说明：\n1. 数字表示连续填充的单元格块，块间至少间隔一个空格\n2. 行线索从左到右排列，列线索从上到下排列\n3. 用'X'表示填充，用空格或'.'表示空白\n\n行线索：\n第1行: 无\n第2行: [1, 1]\n第3行: 无\n第4行: [1]\n第5行: [1, 1]\n\n列线索：\n第1列: [1, 2]\n第2列: 无\n第3列: 无\n第4列: [1, 1]\n第5列: 无\n\n请将最终答案放在[answer]标签内，每行用'X'和空格表示填充状态：\n示例：\n[answer]\nXX X\n XXX\nX  X\n[/answer]'
  "标准答案": '{\n  \"data_source\": \"Nonograms\",\n  \"ground_truth\": {\n    \"rows\": [\n      [],\n      [\n        1,\n        1\n      ],\n      [],\n      [\n        1\n      ],\n      [\n        1,\n        1\n      ]\n    ],\n    \"columns\": [\n      [\n        1,\n        2\n      ],\n      [],\n      [],\n      [\n        1,\n        1\n      ],\n      []\n    ]\n  }\n}'
  "用户回答": '\n[answer]\n     \nX  X\n     \nX    \nX  X\n[/answer]\n'
}
```

#### **输出：**
### 分析过程：
1. **理解题意和核心判断依据**：题目为Nonogram谜题，判断标准是用户回答的网格是否符合行线索和列线索。行线索和列线索是核心依据，标准答案中“ground_truth”的行和列线索即为此谜题的规则约束。
2. **提取用户回答的内容**：用户回答的网格为：
   - 第1行：全空格（符合行线索“无”）
   - 第2行：X  X（两个单独的1格块，符合行线索[1,1]）
   - 第3行：全空格（符合行线索“无”）
   - 第4行：X    （一个1格块，符合行线索[1]）
   - 第5行：X  X（两个单独的1格块，符合行线索[1,1]）
3. **验证列线索符合性**：
   - 第1列：第2行X、第4行X、第5行X，形成1格块（第2行）和2格块（第4-5行），符合列线索[1,2]
   - 第2列：全空格，符合列线索“无”
   - 第3列：全空格，符合列线索“无”
   - 第4列：第2行X、第5行X，形成两个1格块，符合列线索[1,1]
   - 第5列：全空格，符合列线索“无”
4. **对比结果**：用户回答的网格内容完全符合所有行和列线索，内容正确。

```json
{
  "判断结果": "正确"
}
```


#### **输入：**
```json
{
  "题目": 'Solve this Binairo puzzle (size 4x4):\n\nRules:\n1. Equal 0s/1s in each row/column\n2. No three consecutive identical digits\n3. All rows/columns must be unique\n4. Exactly one valid solution\n\nPuzzle:\nRow 1: _ 1 _ _\nRow 2: 1 _ _ _\nRow 3: _ 1 _ 0\nRow 4: _ 0 _ _\n\nPlace your final answer between [answer] and [/answer] tags as:\n\n[answer]\n1 0 1 0\n0 1 0 1\n...[/answer]'
  "标准答案": '\n{\n    "data_source": "Binairo",\n    "ground_truth": {\n      "puzzle": [\n        [\n          null,\n          1,\n          null,\n          null\n        ],\n        [\n          1,\n          null,\n          null,\n          null\n        ],\n        [\n          null,\n          1,\n          null,\n          0\n        ],\n        [\n          null,\n          0,\n          null,\n          null\n        ]\n      ],\n      "solution": [\n        [\n          0,\n          1,\n          0,\n          1\n        ],\n        [\n          1,\n          0,\n          1,\n          0\n        ],\n        [\n          0,\n          1,\n          1,\n          0\n        ],\n        [\n          1,\n          0,\n          0,\n          1\n        ]\n      ]\n    }\n  }\n\n'
  "用户回答": '\n[answer]\n0 1 1 0\n1 0 0 1\n1 1 0 0\n0 0 1 1\n[/answer]\n'
}
```

#### **输出：**
### 分析过程：
1. **理解题意和标准答案**：题目为4x4的Binairo谜题，需依据给定规则求解。标准答案中的“solution”明确四行正确答案依次为“0 1 0 1”“1 0 1 0”“0 1 1 0”“1 0 0 1”。
2. **提取用户回答的最终结论**：用户回答通过[answer]标签给出的最终结果为“0 1 1 0”“1 0 0 1”“1 1 0 0”“0 0 1 1”，去除标签后，核心结果为这四行数字。
3. **对比结果**：用户回答的四行结果与标准答案的四行结果均不一致，因此判定为错误。

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


JUDGE_WITH_CRITERIA_FEWSHOTS = """
任务：基于打分表对下面回答的大模型响应作出打分

#### 输出要求
先在<think>...</think>里给出详细的思考过程，然后在最终结论部分给出评分。（其中评分分为两个部分，一个是对于回答的评分，另一个说明该评分表下满分为多少）

```json
{{
     "最终评分": ***, # int
     "满分总分": *** # int
}}
```

下面是一些例子

#### 用户提问
```
You will be given a definition of a task first, then some input of the task.\nIn this task, you are given the name of an Indian food dish. You need to classify the dish as a \"main course\", \"dessert\" or \"snack\".\n\nDum aloo\nOutput:
```

#### 模型响应
```
The dish Dum aloo is a main course dish from Indian cuisine, commonly served with rice or roti. It is a spicy appetizer or side dish made of baby potatoes cooked in a spicy and tangy tamarind gravy. The main ingredients include the baby potatoes, tamarind, oil, onion, ginger, garlic, tomatoes, and various spices such as turmeric, cumin, and garam masala.\n\nJustification: Dum aloo is typically served as a main course in Indian restaurants and households, and is often accompanied by a side of rice or roti and a vegetable dish. Therefore, it falls under the category of main course dishes. As a main course dish, it is not typically served as a dessert or snack, and its flavor profile is not typically sweet.
```

#### 评分表
```
# 评价标准\n\n  ## 评估方法简述\n  本评价标准从「有用性」、「真实性」、「安全性」和「核心任务相关性」四个维度对回答进行评估。每个维度包含具体的评分细则，确保评估结果的客观性和可量化性。\n\n  ## 有用性（20分）\n  1. 回答是否明确输出分类结果（10分）：\n     - 明确输出分类结果：10分\n     - 仅包含菜品名称：5分\n     - 未输出分类结果：0分\n  2. 是否包含菜品名称（10分）：\n     - 包含完整菜品名称：10分\n     - 部分包含菜品名称：5分\n     - 未包含菜品名称：0分\n\n  ## 真实性（40分）\n  1. 分类是否准确（30分）：\n     - 分类完全正确：30分\n     - 分类部分正确：15分\n     - 分类完全错误：0分\n  2. 是否包含正确分类依据（10分）：\n     - 提供合理分类依据：10分\n     - 仅提供分类结果：5分\n     - 无分类依据：0分\n\n  ## 安全性（20分）\n  1. 回答是否包含误导信息（10分）：\n     - 无误导信息：10分\n     - 轻微误导信息：5分\n     - 严重误导信息：0分\n  2. 是否包含有害信息（10分）：\n     - 无有害信息：10分\n     - 包含有害信息：0分\n\n  ## 核心任务相关性（20分）\n  1. 分类依据是否合理（10分）：\n     - 依据充分且合理：10分\n     - 依据部分合理：5分\n     - 依据完全不合理：0分\n  2. 是否考虑地域差异（10分）：\n     - 考虑地域差异：10分\n     - 未考虑地域差异：5分\n     - 忽视地域差异：0分
```


#### 输出
<think>
我现在需要根据用户提供的评分表，对给定的模型响应进行打分。首先，我需要仔细理解评分表的各个维度和具体细则，然后逐一对照模型响应来评分。

首先看评分表的四个维度：有用性（20 分）、真实性（40 分）、安全性（20 分）、核心任务相关性（20 分），总分是 20+40+20+20=100 分，所以满分总分是 100 分。

接下来逐个分析：

有用性（20 分）：

回答是否明确输出分类结果（10 分）：模型响应中明确提到 “Dum aloo 是主菜（main course）”，明确输出了分类结果，所以这一项得 10 分。
是否包含菜品名称（10 分）：模型响应中包含完整的菜品名称 “Dum aloo”，所以这一项得 10 分。
有用性总分：10+10=20 分。

真实性（40 分）：

分类是否准确（30 分）：Dum aloo 确实是印度的主菜，分类正确，所以这一项得 30 分。
是否包含正确分类依据（10 分）：模型响应中提到 “通常在印度餐厅和家庭中作为主菜供应，搭配米饭或面包等”，提供了合理的分类依据，所以这一项得 10 分。
真实性总分：30+10=40 分。

安全性（20 分）：

回答是否包含误导信息（10 分）：模型响应中没有误导信息，所以得 10 分。
是否包含有害信息（10 分）：模型响应中没有有害信息，所以得 10 分。
安全性总分：10+10=20 分。

核心任务相关性（20 分）：

分类依据是否合理（10 分）：模型提到 “通常作为主菜供应，搭配米饭或面包”，依据充分合理，得 10 分。
是否考虑地域差异（10 分）：模型中没有提到地域差异相关内容，根据细则，未考虑地域差异得 5 分。
核心任务相关性总分：10+5=15 分。

现在把各维度分数相加：20（有用性）+40（真实性）+20（安全性）+15（核心任务相关性）=95 分。

检查是否有遗漏：有用性的两项都满足，真实性分类正确且有依据，安全性没问题，核心任务相关性中依据合理但未考虑地域差异。所以总分是 95 分，满分 100 分。
</think>
```json
{{
     "最终评分": 95,
     "满分总分": 100
}}
```
"""

JUDGE_WITH_CRITERIA_TEMPLATE = """
现在对下面的大模型响应基于评分表进行打分

#### 用户提问
```
{instruction}
```

#### 模型响应
```
{response}
```

#### 评分表
```
{criteria}
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


class RFTRLVRVerify(RLVRVerify):
    def __init__(self):
        pass

    @classmethod
    def task_desc(cls):
        return "RLVR验证"

    def prompt_fn(self, example):
        solver_response, gt = example
        content = {
            "题目": gt["instruction"],
            "用户回答": solver_response,
            "标准答案": gt["criteria"],
        }
        prompt = RLVR_VERIFY_FEWSHOTS + "\n\n\n" + RLVR_VERIFY_TEMPLATE.format(
            content=json.dumps(content, ensure_ascii=False, indent="  ")
        )
        return prompt


class JudgeWithCriteria(BatchCallOpenAPI):
    def __init__(self):
        pass

    @classmethod
    def task_desc(cls):
        return "评分表打分"

    def prompt_fn(self, example):
        solver_response, gt = example
        prompt = JUDGE_WITH_CRITERIA_FEWSHOTS + "\n\n\n" + JUDGE_WITH_CRITERIA_TEMPLATE.format(
            instruction=gt["instruction"], response=solver_response, criteria=gt["criteria"]
        )
        return prompt

    def postprocess(self, response: str):
        s = response
        try:
            if "</think>" in s:
                s = s[s.index("</think>"):].strip()

            conclusion = s.strip()
            score = re.findall(
                r'\"最终评分\": ([\d\.]+)', conclusion)[0]
            total_score = re.findall(
                r'\"满分总分\": (.*)', conclusion)[0]
            final_score = min(float(score) / float(total_score), 1.0)
            if math.isnan(final_score) or np.isnan(final_score):
                return 0.0
            return final_score
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
                outputs.append(np.mean(scores))
            else:
                outputs.append(0.0)
        return outputs


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
        self.task_name = "CRITERIA_RM_RECALL"

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

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# CRITERIA_RM_RFT
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def criteria_rft_parse_solution_fn(solution_str: str):
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

            # Verify <conclusion> node should not have children.
            has_child = False
            for child in conclusion:
                has_child = True
            if has_child:
                return None

            conclusion = conclusion.text.strip()

        except Exception as err:
            return None
    else:
        return None
    return conclusion


class CriteriaRFTComputeScore(CriteriaRMRecallComputeScore):
    def __init__(self,
                 parse_solution_fn,
                 split="train",
                 args=None,
                 min_reward=-2.0
                 ):
        super().__init__(
            split=split, parse_solution_fn=parse_solution_fn, args=args, min_reward=min_reward
        )
        self.task_name = "CRITERIA_RM_RFT"
        self.task = JudgeWithCriteria()

    def init_agent(self):
        self.agents = {}
        self.init_verify_agent()

    def init_verify_agent(self):
        self.verify_agent = Agent(
            **self.args["verify_agent"]["model"])

    async def get_accuracy(
            self,
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth):

        task = self.task

        indices = []
        eval_inputs = []
        result_index2queue_index = {}
        for i, (gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            result = self.parse_solution_fn(sol)
            if result is not None:
                eval_inputs.append((result, gt))
                indices.append(i)
                result_index2queue_index[len(
                    eval_inputs)-1] = i
            else:
                continue

        evaluations = await task.do_job(
            agent=self.verify_agent,
            batch_inputs=eval_inputs,
            max_concurrent_requests=self.args["verify_agent"]["max_concurrent_requests"],
        )
        full_rewards = [0.0] * len(batch_solution_str)
        for j, (eval_result, eval_input) in enumerate(zip(evaluations, eval_inputs)):
            queue_index = result_index2queue_index[j]
            if eval_result:
                full_rewards[queue_index] = eval_result
        return full_rewards

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
        self.init_save_rollouts()
        accuracy = await self.get_accuracy(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            _reward = accuracy[i]
            final_results.append(_reward)

            if (_reward > 0 and random.random() < 0.1) or (self.split == "valid" and random.random() < 0.05) or (self.split == "train" and random.random() < 0.05):
                log = True
                log_flag = f"[{self.task_name} VALID]" if self.split == "valid" else f"[{self.task_name} TRAIN]"
            else:
                log = False
                log_flag = ""

            # 保存Rollout信息
            if self.split == "train":
                self.update_rollout_info(
                    solution_str=batch_solution_str[i],
                    ground_truth=batch_ground_truth[i],
                    score=_reward,
                )

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f'【Question】`{repr(batch_ground_truth[i]["instruction"])}`')
                print(
                    f'【Criteria】`{repr(batch_ground_truth[i]["criteria"])}`')
                print(
                    f'[Final Reward]={_reward:.3f}\n')
                print(
                    f'【Solution】')
                print(batch_solution_str[i])

        self.save_rollout_info()
        return final_results

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# CRITERIA_RM_RFT
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# XML_COT
# ------------------------------------------------------------------------------------------------------------------------------------------------------


class RFTComputeScore(CriteriaRFTComputeScore):
    def __init__(self,
                 parse_solution_fn,
                 split="train",
                 args=None,
                 min_reward=-2.0
                 ):
        super().__init__(
            split=split, parse_solution_fn=parse_solution_fn, args=args, min_reward=min_reward
        )
        self.task_name = "RFT"
        self.task = RFTRLVRVerify()

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
        self.init_save_rollouts()

        struct_scores = defaultdict(list)
        accuracy = await self.get_accuracy(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            _uuid = batch_ground_truth[i]["extra_info"]["uuid"]
            _reward = accuracy[i]

        final_results = []
        for i in range(len(batch_solution_str)):
            _uuid = batch_ground_truth[i]["extra_info"]["uuid"]
            _reward = accuracy[i]

            struct_info_str = f'[TODO]'

            final_results.append(_reward)
            if (_reward > 0 and random.random() < 0.05) or (self.split == "valid" and random.random() < 0.01) or (self.split == "train" and random.random() < 0.01):
                log = True
                log_flag = f"[{self.task_name} VALID]" if self.split == "valid" else f"[{self.task_name} TRAIN]"
            else:
                log = False
                log_flag = ""

            # 保存Rollout信息
            if self.split == "train":
                self.update_rollout_info(
                    solution_str=batch_solution_str[i],
                    ground_truth=batch_ground_truth[i],
                    score=_reward,
                )

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f'【Question】`{repr(batch_ground_truth[i]["instruction"])}`')
                print(
                    f'【Criteria】`{repr(batch_ground_truth[i]["criteria"])}`')
                print(
                    f'[Final Reward]={_reward:.3f}|{struct_info_str}\n')
                print(
                    f'【Solution】')
                print(batch_solution_str[i])
        self.save_rollout_info()
        return final_results


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# XML_COT
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# XML_COT_TRANSLATION
# ------------------------------------------------------------------------------------------------------------------------------------------------------

class XMLCoTTranslation(RFTComputeScore):
    def __init__(self,
                 parse_solution_fn,
                 split="train",
                 args=None,
                 min_reward=-2.0
                 ):
        super().__init__(
            split=split, parse_solution_fn=parse_solution_fn, args=args, min_reward=min_reward
        )
        self.task_name = "XML COT TRANSLATION"

    def compute_meta_cognition_recall(self,
                                      solution_str,
                                      ground_truth,
                                      ):
        meta = ground_truth["meta_cognition"]
        return len([_ for _ in meta if _ in solution_str]) / len(meta)

    def detect_meta_cognition_hack(self, solution_str):
        # 匹配</mutter>后接任意空白字符，再接<mutter>的模式
        pattern = r'</mutter>\s*<mutter>'
        # 查找所有匹配项
        matches = re.findall(pattern, solution_str, re.IGNORECASE)
        return len(matches) > 0

    def compute_tree_max_depth(self, solution_str):
        solution_str = postprocess_solution(solution_str)
        if not solution_str.startswith("<think>"):
            solution_str = f'<think>\n{solution_str}'

        try:
            root = xml_cot_parse_solution_fn(solution_str)
        except Exception as err:
            return 0.0

        return self.calculate_max_depth(root)

    def calculate_max_depth(self, element):
        """
        计算XML元素的最大深度，处理element为None的情况

        参数:
            element: ET.Element or None - 要计算深度的XML元素，可能为None

        返回:
            int - 最大深度值（根元素深度为1，element为None时返回0）
        """
        # 如果元素为None，返回0
        if element is None:
            return 0

        # 如果没有子元素，深度为1
        children = list(element)
        if not children:
            return 1

        # 递归计算所有子元素的深度，当前深度加1
        max_child_depth = 0
        for child in children:
            child_depth = self.calculate_max_depth(child)
            if child_depth > max_child_depth:
                max_child_depth = child_depth

        return max_child_depth + 1

    def traverse_recursive(self, node, all_nodes=None, depth=0, correct=[]):
        """
        递归遍历XML节点及其所有后代节点（DFS方式）

        参数:
            node: 当前要处理的节点（Element对象）
            all_nodes: 用于收集节点信息的列表（默认自动初始化）
            depth: 当前节点的深度（根节点为0）

        返回:
            list: 包含所有节点信息的列表
        """
        # 处理node为None的情况
        if node is None:
            return []

        # 初始化节点列表（仅在第一次调用时）
        if all_nodes is None:
            all_nodes = []

        # 收集当前节点信息
        node_info = {
            'tag': node.tag,
            'text': node.text.strip() if node.text else None,
            'attributes': dict(node.attrib),
            'depth': depth
        }
        all_nodes.append(node_info)

        # 递归处理每个子节点，深度+1
        for i, child in enumerate(node):
            if child is not None and child.tag == "mutter":
                if i == 0:
                    correct.append(0)
                else:
                    correct.append(1)
            self.traverse_recursive(child, all_nodes, depth + 1, correct)

        return all_nodes

    def compute_meta_cognition_precision(self, solution_str):
        solution_str = postprocess_solution(solution_str)
        if not solution_str.startswith("<think>"):
            solution_str = f'<think>\n{solution_str}'

        try:
            root = xml_cot_parse_solution_fn(solution_str)
        except Exception as err:
            return 0.0

        if root is None:
            return 0.0

        corrects = []
        self.traverse_recursive(root, None, 0, corrects)
        if len(corrects) == 0:
            return 0
        return np.mean(corrects)

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
        accuracy = await self.get_accuracy(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            _uuid = batch_ground_truth[i]["extra_info"]["uuid"]
            _reward = accuracy[i]

        final_results = []
        for i in range(len(batch_solution_str)):
            _uuid = batch_ground_truth[i]["extra_info"]["uuid"]
            _reward = accuracy[i]
            recall = self.compute_meta_cognition_recall(
                batch_solution_str[i], batch_ground_truth[i]
            )
            _reward += recall * 0.5

            precision = self.compute_meta_cognition_precision(
                batch_solution_str[i]
            )
            _reward += precision * 0.5

            # 树结构化奖励
            max_depth = self.compute_tree_max_depth(
                batch_solution_str[i])
            if max_depth >= 5:
                _reward += 0.25
            elif max_depth >= 6:
                _reward += 0.5
            elif max_depth >= 7:
                _reward += 0.6

            hack_detection = self.detect_meta_cognition_hack(
                batch_solution_str[i]
            )
            if hack_detection:
                _reward = 0.0

            final_results.append(_reward)
            if (accuracy[i] > 0 and random.random() < 0.05) or (self.split == "valid" and random.random() < 0.01) or (self.split == "train" and random.random() < 0.01):
                log = True
                log_flag = f"[{self.task_name} VALID]" if self.split == "valid" else f"[{self.task_name} TRAIN]"
            else:
                log = False
                log_flag = ""

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f'【Question】`{repr(batch_ground_truth[i]["instruction"])}`')
                print(
                    f'【Criteria】`{repr(batch_ground_truth[i]["criteria"])}`')
                print(
                    f'[Final Reward]={_reward:.3f}|Recall={recall:.3f}|Precision={precision:.3f}\n')
                print(
                    f'【Solution】')
                print(batch_solution_str[i])
        return final_results

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# XML_COT_TRANSLATION
# ------------------------------------------------------------------------------------------------------------------------------------------------------


AUTOPE_DEFAULT_PARAMS = {
    "verify_agent": {
        "model": {
            "model": "Qwen2.5-32B-Instruct",
            "base_url": "https://sd262bskcm47j59r1292g.apigateway-cn-beijing.volceapi.com/v1",
            "api_keys": "caa6246b-afbe-4d9b-ab34-87bf9922032b",
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
            "model": "Qwen2.5-32B-Instruct",
            "base_url": "https://sd262bskcm47j59r1292g.apigateway-cn-beijing.volceapi.com/v1",
            "api_keys": "caa6246b-afbe-4d9b-ab34-87bf9922032b",
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
            "model": "Qwen2.5-32B-Instruct",
            "base_url": "https://sd262bskcm47j59r1292g.apigateway-cn-beijing.volceapi.com/v1",
            "api_keys": "caa6246b-afbe-4d9b-ab34-87bf9922032b",
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
            "model": "Qwen2.5-32B-Instruct",
            "base_url": "https://sd262bskcm47j59r1292g.apigateway-cn-beijing.volceapi.com/v1",
            "api_keys": "caa6246b-afbe-4d9b-ab34-87bf9922032b",
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
        "default_local_dir": "/cpfs01/shared/llm_ddd/tongjian/ckpts/datareview_rl_test/verl/grpo/criteria_rm_recall"
    }
}

CRITERIA_RM_RFT_DEFAULT_PARAMS = {
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
        "max_concurrent_requests": 32
    },
    "save_rollouts": {
        "default_local_dir": "/cpfs01/shared/llm_ddd/tongjian/ckpts/datareview_rl_test/verl/grpo/criteria_rft"
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


criteria_rft_score_train = CriteriaRFTComputeScore(
    criteria_rft_parse_solution_fn, split="train", args=CRITERIA_RM_RFT_DEFAULT_PARAMS).compute_score
criteria_rft_score_valid = CriteriaRFTComputeScore(
    criteria_rft_parse_solution_fn, split="valid", args=CRITERIA_RM_RFT_DEFAULT_PARAMS).compute_score


rft_score_train = RFTComputeScore(
    criteria_rft_parse_solution_fn, split="train", args=CRITERIA_RM_RFT_DEFAULT_PARAMS).compute_score
rft_score_valid = RFTComputeScore(
    criteria_rft_parse_solution_fn, split="valid", args=CRITERIA_RM_RFT_DEFAULT_PARAMS).compute_score


xml_cot_translation_score_train = XMLCoTTranslation(
    criteria_rft_parse_solution_fn, split="train", args=CRITERIA_RM_RFT_DEFAULT_PARAMS).compute_score
xml_cot_translation_score_valid = XMLCoTTranslation(
    criteria_rft_parse_solution_fn, split="valid", args=CRITERIA_RM_RFT_DEFAULT_PARAMS).compute_score
