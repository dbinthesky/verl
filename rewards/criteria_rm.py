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

LECTURE_NOTE_STYLE_JUDGE_FEWSHOTS = '''
任务：对于下面一个任务的回答，你需要判断是否符合大学讲义式解题过程的特征？
匹配度1-5分，5分是最高分，分数仅包含1、2、3、4、5不包含小数点。回答越符合大学讲义则分数越高

#### 输出要求
{分析内容}
```json
{
    "匹配度": *** # int (1-5)
}
```

下面是大学讲义风格的定义和要求
# 大学讲义式解题过程的核心特征分析
各位同学，今天我们来探讨一个兼具理论意义与实践价值的话题：如何将一道题目的解题过程以大学讲义的方式呈现。在高等教育阶段，解题过程的书写早已超越了“得出答案”的基本要求，它更是知识传递、思维训练与学术规范培养的重要载体。一份合格的大学讲义式解题过程，需要具备以下核心特征：


## 一、目标明确的前置性说明
大学讲义的首要特征是“让读者明确为何学、学什么”，体现在解题过程中需包含清晰的前置说明：
1. **问题背景与意义**  
   需简要阐述该问题所属的知识领域（如“本题属于微积分中的定积分应用模块”）、在学科体系中的地位（如“是理解曲线弧长计算的基础题型”），以及实际应用价值（如“在工程测量、物理运动分析中具有广泛应用”）。这一步能帮助读者建立知识关联，避免“为解题而解题”的机械思维。
2. **核心目标与考察点**  
   明确说明本题的解题目标（如“求解函数在给定区间上的最值”）以及考察的关键知识点（如“需综合运用导数的四则运算法则与极值判定定理”），让读者提前聚焦核心能力要求。
3. **预备知识提示**  
   列出解题所需的前置概念、公式或定理，并对关键符号进行定义，确保读者具备理解基础。


## 二、逻辑严谨的过程展开
大学阶段的解题强调“思维的可追溯性”，过程展开需体现严密的逻辑链条：
1. **分步拆解与逻辑衔接**  
   将复杂问题分解为若干子步骤（如“ Step 1：确定函数定义域；Step 2：求解一阶导数；Step 3：寻找驻点与不可导点……”），每个步骤需明确标注，并通过过渡语句（如“由 Step 1 可知函数定义域为$[0,2]$，因此接下来需在该区间内分析导数性质”）体现步骤间的因果关系，避免跳跃性推理。
2. **原理支撑与依据标注**  
   每一步运算或推理必须明确标注理论依据（如“根据极限的四则运算法则，$\lim\limits_{x\\to a}[f(x)+g(x)]=\lim\limits_{x\\to a}f(x)+\lim\limits_{x\\to a}g(x)$”“由题意可知，物体做匀加速直线运动，故适用公式$s=v_0t+\\frac{1}{2}at^2$”）。对于关键定理，需简要说明其适用条件（如“由于函数$f(x)$在$[1,3]$上连续且可导，满足罗尔定理的应用条件”），培养严谨的学术态度。
3. **符号规范与定义清晰**  
   所有使用的符号需遵循学科惯例（如用$x,y$表示变量，$a,b,c$表示常数，$\\theta$表示角度），并在首次出现时明确定义（如“设$t$为运动时间，单位为秒；$s(t)$为$t$时刻的位移，单位为米”）。对于易混淆的符号（如$\ln$表示自然对数，$\log$表示常用对数）需特别说明，避免歧义。


## 三、深度融合的知识关联
大学讲义的核心价值在于“构建知识网络”，解题过程需体现知识点的横向与纵向联系：
1. **跨知识点整合说明**  
   若题目涉及多个章节的知识（如“本题需结合线性代数中的矩阵运算与微分方程中的特征值解法”），需明确指出知识的交叉点，并解释整合逻辑（如“通过将微分方程转化为矩阵形式，可利用特征值简化求解过程”），帮助读者建立学科知识体系。
2. **方法对比与选择依据**  
   当存在多种解题方法时（如“本题可通过换元法或分部积分法求解”），需对比不同方法的适用场景（如“换元法更适用于根号内为一次函数的情况”），并说明选择当前方法的理由，培养“方法优化”意识。
3. **易错点预警与原理分析**  
   针对学生常犯的错误（如“忽略定义域对极值点的限制”“误用向量点积与叉积的运算规则”），需在相应步骤旁标注预警（如“【注意】此处需验证驻点是否在定义域内，否则会导致结果错误”），并从原理层面解释错误原因（如“极值点必须是函数定义域内的点，这是由极值的定义决定的”），实现“解题即学习”的效果。


## 四、规范专业的表达形式
大学讲义作为学术性文本，需遵循严格的表达规范：
1. **术语准确与表述严谨**  
   必须使用学科标准术语（如用“导数”而非“变化率”的口语化表达，用“偏微分方程”而非“多变量方程”），避免模糊表述。对于定理、公式的引用需准确无误（如“牛顿-莱布尼茨公式”不可简称为“微积分基本公式”而不加解释）。
2. **数学符号与格式规范**  
   数学公式需单独成行并编号（如“（1）$F(x)=\int_a^x f(t)dt$”），便于后续引用；变量、常量、函数符号需区分大小写（如“$F(x)$表示原函数，$f(x)$表示被积函数”）；运算符号需规范（如用“$\times$”表示数值乘法，“$\cdot$”表示向量点积，避免混淆）。
3. **图文结合与可视化辅助**  
   对于几何问题、物理过程等抽象内容，需配合示意图辅助说明（如“如图1所示，平面区域$D$由曲线$y=x^2$与$y=2x$围成”），图形需标注关键参数（坐标、长度、角度等），并在图注中说明图形与解题步骤的对应关系，实现“以图助解”。


## 五、总结拓展的延伸性内容
大学讲义不仅是“解题指南”，更是“思维拓展工具”，需包含总结与延伸部分：
1. **解题思路回顾与方法提炼**  
   在得出答案后，需提炼通用解题模型（如“本题的核心思路是：先将实际问题转化为数学模型，再通过求导分析函数性质，最后验证结果合理性”），总结可迁移的解题策略（如“对于最值问题，通常遵循‘求导→找临界点→判断单调性→确定最值’的步骤”）。
2. **结果验证与合理性分析**  
   需说明如何验证答案的正确性（如“通过代入特殊值检验”“利用对称性验证积分结果”），并分析结果的物理或数学意义（如“所求加速度为正值，表明物体做加速运动，与题意一致”），培养“结果反思”习惯。
3. **拓展问题与研究方向**  
   可提出相关延伸问题（如“若将本题中的区间改为无穷区间，解题方法会发生哪些变化？”）或前沿应用（如“本题的优化思想可拓展至机器学习中的损失函数最小化问题”），引导读者从“解题”走向“探究”。


## 总结：大学讲义式解题的本质
从上述特征可以看出，大学讲义式解题过程的核心本质是“**以解题为载体的知识传授与思维训练**”。它要求我们不仅“会解题”，更要“懂原理、明逻辑、能迁移”。这种书写方式看似繁琐，实则是培养学术严谨性、逻辑表达能力与知识整合能力的重要途径。





下面是一些例子

#### 提问
```
*单项选择题*关于浅感觉下列感觉是正确的\n\n- A、运动觉<sup>[1]</sup>\n\n- B、痛觉<sup>[2]</sup>\n\n- C、位置觉\n\n- D、实体觉\n\n- E、图形觉
```

#### 回答
```
### 逐步推理：\n\n1. **理解题目**：题目问的是“关于浅感觉下列感觉是正确的”，即需要从选项中找出属于“浅感觉”的感觉类型。\n\n2. **明确“浅感觉”和“深感觉”的定义**：\n   - **浅感觉**：指来自皮肤和黏膜的感觉，主要包括痛觉、温度觉和触觉（轻触觉）。这些感觉的感受器位于皮肤或黏膜中。\n   - **深感觉**：指来自肌肉、肌腱、关节和骨膜的感觉，主要包括运动觉（关节位置觉）、振动觉和压觉（深压觉）。\n   - **复合感觉（皮质感觉）**：是大脑皮质对浅感觉和深感觉整合后形成的更复杂的感觉，如实体觉、图形觉、两点辨别觉等。\n\n3. **分析选项**：\n   - A、运动觉：属于深感觉（关节位置觉），排除。\n   - B、痛觉：属于浅感觉（皮肤和黏膜的感觉），正确。\n   - C、位置觉：属于深感觉（关节位置觉），排除。\n   - D、实体觉：属于复合感觉（需要大脑皮质整合触觉和深感觉），排除。\n   - E、图形觉：属于复合感觉（需要大脑皮质整合触觉信息），排除。\n\n4. **验证**：\n   - 浅感觉主要包括痛觉、温度觉和触觉，因此痛觉（B）是唯一正确的选项。\n   - 其他选项要么是深感觉（A、C），要么是复合感觉（D、E）。\n\n### 最终答案：\n\\boxed{B}"
```

#### 评价
对照大学讲义式解题过程的核心特征，该回答存在多处显著不足，具体如下：
1. **目标明确的前置性说明缺失严重**  
   - 无问题背景与意义：未说明题目所属知识领域（如“本题属于生理学中感觉系统模块”）、学科地位（如“是理解躯体感觉分类的基础题型”）及应用价值，完全缺乏知识关联引导。  
   - 核心目标表述模糊：仅简单说明“找出属于浅感觉的选项”，未明确考察的关键能力（如“考察对不同感觉类型分类标准的掌握”）。  
   - 预备知识不规范：虽提及感觉类型定义，但未以“预备知识提示”板块系统呈现，未明确概念间的层级关系，不符合讲义“前置知识清单”的要求。  
2. **逻辑严谨的过程展开不达标**  
   - 分步拆解不规范：步骤未标注“Step 1/2/3”等正式编号，仅以“1.2.3.4”简单罗列，缺乏学术化的步骤标识。  
   - 原理支撑标注缺失：对感觉类型的判断仅依赖定义描述，未明确标注“根据浅感觉的核心定义：XXX”等规范依据，推理依据未显性化。  
   - 无逻辑衔接强化：过渡语句口语化（如“明确……定义”“分析选项”），未使用“由上述定义可知”“基于分类标准”等学术化衔接语。  
3. **深度融合的知识关联完全缺失**  
   - 无跨知识点整合：未提及该知识点与其他章节的联系（如浅感觉与神经传导通路的关联）。  
   - 无方法对比与选择：未说明是否存在其他解题思路（如通过“感受器位置”辅助判断），也未解释当前方法的唯一性或适用性。  
   - 无易错点预警：未指出学生易混淆的误区（如“将实体觉误归为浅感觉”），未从原理层面解释错误原因，缺乏“解题即学习”的引导。  
4. **规范专业的表达形式不达标**  
   - 结构不规范：未采用“一、前置说明 二、解题步骤 三、总结拓展”等讲义式层级标题，仅以“逐步推理”“最终答案”简单划分，缺乏学术文本的结构化特征。  
   - 无概念/定义编号：对“浅感觉”“深感觉”等核心概念未编号（如“定义1：浅感觉指……”），不便于后续引用，不符合讲义格式规范。  
   - 表述口语化：如“分析选项”“验证”等表述过于简略，未使用“选项逐一辨析”“结果合理性验证”等学术化表述。  
5. **总结拓展的延伸性内容几乎空白**  
   - 无解题思路提炼：未总结“区分感觉类型”的通用步骤（如“先明确分类标准→再匹配选项特征→最后验证合理性”），缺乏方法迁移价值。  
   - 结果验证薄弱：仅简单重复判断依据，未说明“如何通过反向排除法二次验证”等具体验证逻辑。  
   - 无拓展问题：未提出延伸思考（如“若题目加入‘触觉’选项，判断逻辑是否变化”），未实现知识拓展目标。
```json
{
    "匹配度": 2
}
```


#### 提问
```
The main difference between digital currency and traditional currency lies in:\n- A. Value stability\n- B. Issuing authority\n- C. Transaction speed\n- D. Storage method
```

#### 回答
```
To determine the main difference between digital currency and traditional currency, let's analyze each option step by step:\n\n1. **Value stability (A)**: Both digital and traditional currencies can experience fluctuations in value. For example, cryptocurrencies like Bitcoin are highly volatile, but traditional currencies can also lose value due to inflation or economic instability. Therefore, this is not the *main* difference.\n\n2. **Issuing authority (B)**: This is a key distinction. Traditional currencies are issued and regulated by central authorities like governments or central banks (e.g., the Federal Reserve for the US dollar). Digital currencies, especially cryptocurrencies, are typically decentralized and not controlled by any single authority. This is a fundamental difference.\n\n3. **Transaction speed (C)**: While digital currencies often enable faster transactions (especially across borders), some traditional payment systems (like credit cards or digital banking) can also process transactions quickly. Thus, this is not the *primary* difference.\n\n4. **Storage method (D)**: Digital currencies are stored electronically (e.g., in digital wallets), while traditional currencies can be stored physically (cash) or digitally (bank accounts). However, this is more of a technical difference rather than a fundamental one.\n\n**Conclusion**: The most significant difference lies in the **issuing authority**, as digital currencies often operate without central control, unlike traditional currencies.\n\n\\boxed{B}
```

#### 评价
对照大学讲义式解题过程的核心特征，该回答存在多处明显不足，具体如下：
1. **目标明确的前置性说明缺失**  
   - 无问题背景与意义：未说明题目所属知识领域（如“本题属于金融学中货币形态模块”）、学科地位（如“是理解现代货币体系演变的基础题型”）及应用价值，缺乏知识关联引导。  
   - 核心目标表述简单：仅隐含“找出数字货币与传统货币的主要区别”，未明确考察的关键能力（如“考察对不同货币形态核心特征的辨析能力”）。  
   - 预备知识提示缺失：未系统列出解题所需的核心概念（如“数字货币的定义”“传统货币的核心特征”），未对关键术语进行规范定义，不符合讲义“前置知识清单”要求。  
2. **逻辑严谨的过程展开不规范**  
   - 分步拆解无正式标识：步骤仅以“1.2.3.4”罗列，未标注“Step 1/2/3/4”等学术化编号，缺乏规范的步骤区分。  
   - 原理支撑标注不足：对选项的判断仅依赖经验性描述（如“ cryptocurrencies like Bitcoin are highly volatile”），未明确标注“根据货币发行机制的核心定义：XXX”等规范依据，推理依据未显性化。  
   - 逻辑衔接口语化：过渡语句简单（如“let's analyze each option step by step”“Therefore, this is not...”），未使用“基于货币本质特征可知”“从发行机制角度分析”等学术化衔接语。  
3. **深度融合的知识关联匮乏**  
   - 无跨知识点整合：未提及该知识点与其他章节的联系（如数字货币与金融监管、区块链技术的关联）。  
   - 无方法对比与选择：未说明是否存在其他辨析思路（如从“货币职能实现方式”角度判断），也未解释当前选项分析方法的适用性。  
   - 无易错点预警：未指出学生易混淆的误区（如“将‘存储方式’误视为核心区别”），未从原理层面解释错误原因，缺乏“解题即学习”的引导。  
4. **规范专业的表达形式不达标**  
   - 结构不完整：未采用“一、前置说明 二、解题步骤 三、总结拓展”等讲义式层级标题，仅以自然段落划分，缺乏学术文本的结构化特征。  
   - 无概念编号与公式规范：对“数字货币”“传统货币”等核心概念未编号定义，未形成可引用的概念体系，不符合讲义格式规范。  
   - 表述简洁但不严谨：如“this is a key distinction”“this is more of a technical difference”等表述缺乏学术化修饰，未使用“从货币本质属性来看，发行权归属具有决定性差异”等专业表述。  
5. **总结拓展的延伸性内容空白**  
   - 无解题思路提炼：未总结“辨析事物核心区别”的通用步骤（如“先明确比较维度→再分析各维度重要性→最后锁定核心差异”），缺乏方法迁移价值。  
   - 结果验证缺失：未说明“如何通过货币职能理论二次验证发行权的核心地位”等验证逻辑，仅简单得出结论。  
   - 无拓展问题：未提出延伸思考（如“若数字货币被央行发行并监管，其与传统货币的区别会发生哪些变化？”），未实现知识拓展目标。
```json
{
    "匹配度": 2
}
```

'''

LECTURE_NOTE_STYLE_JUDGE_TEMPLATE = """
现在你需要对下面的回答进行评价，是否符合大学课程讲义的行文风格。

#### 提问
```
{question}
```

#### 评价
```
{response}
```

#### 输出
"""

DOC2QUERY_V3_CRITIQUE_RECALL_FEWSHOTS = """
下面是对于同一道题目的两个来自不同方的质检见过，第一个是人类质检员，第二个是大模型质检员；二者的任务都是尝试对一道问题进行质检判断是否符合要求。

下面是对于问题的质检要求
## 问题要求说明
1. 问题侧重考察知识、推理 > 计算，避免需要复杂计算的问题；题型为**单项**选择题。
2. 关键术语的清晰度与具体性
  - 检查点：题目中的专业术语、缩写、实验阶段、参数等是否明确定义或具体化？ 
  - 问题迹象：  
    - 缩写未解释（如“CCV”未扩写）。（ 注：此处需要结合上下文，如果通过题干已经明确问题领域，且缩写不会造成歧义，则建议不要扩写，因为缩写理解也是模型需要具备的能力之一）
    - 概念笼统（如“稳定性”未说明具体条件）。   
3. 相关性与教育价值
  - 检查点：题目是否聚焦学科核心知识，避免琐碎或边缘内容？  
    - 内容琐碎（如考查书籍章节作者等细节，而非核心概念）。  
    - 与学科知识脱节（如问题不涉及原理、机制或应用）。  
4. 信息完整性
  - 检查点：题目要求独立作答，不依赖原文文档。需要你判断问题是否提供所有必要信息，包括背景、条件、数据来源或参考材料？  
  - 问题迹象：
    - 关键条件缺失（如未提实验设置）。  
    - 引用不存在的材料（如“根据材料”但无实际材料）。  
    - 背景信息不足。  
  - 评价依据：避免因信息缺失导致题目不严谨；通过添加背景改善完整性。
5. 无歧义性
  - 检查点：题目表述是否单一解释，避免歧义或多义？  
  - 问题迹象：
    - 用词模糊（如“影响”未指定正向/负向）。  
    - 选项范围过大（如百分比区间过宽）。  
    - 问题结构易引发误解。  
6. 答案正确性
  - 检查点：给出的问题答案正确无误。
7. 答案为多选题：
  - 检查点：题目要求答案为单选。


## 提示
下面是一些常见的问题供你参考
1. 题目背景信息不足或指代不清
- 题目要求独立作答不依赖文档，因为如果题目存在背景信息不足的情况，答题者在不看原文的情况下无法给出正确答案；其中一种最常见的情况是问题解答依赖某个信息，而该信息指代原文的内容（比如，基于**实验, 根据作者的观点..., 化合物5.., According to Theorem/Formula/Method 1）
2. 问题或选项中英文混杂
- 我们希望问题整体的语言一致性较高，不要有明显的中英文混杂现象
3. 缺乏教育价值
- 问题指向极端长尾信息或边缘只是，对于学生掌握知识的考查意义不大
4. 题目要求复杂计算
- 问题并非测重知识与推理，题目求解依赖复杂计算。
5. 题目过于复杂，解法难以判断
- 依赖文档/文献/参考材料的情况下，对于题目解法仍然无法有效判断，则也需要判断问题为不合格
6. 答案不正确
- 给出的参考答案不正确
7. 题目实际正确答案包含多个选项
- 出题要求为单项选择题，题目不符合要求


现在你的任务并非是重新对问题进行质检，而是比较大模型质检结果和人类质检结果，判断大模型质检结果是否指出了和人类质检员同样的问题。
**注意**：这里人类质检员的结果是Ground Truth 完全正确，你仅需要比较人类质检员和大模型质检员的质检结论，判断一致性。

一致性按1-5分，5分是最高分，分数仅包含1、2、3、4、5不包含小数点。

#### 输出要求
{分析内容}
```json
{{
    "一致性": *** # int (1-5)
}}
```

下面是一些例子

#### 命题
```
Which interaction between the operating variables was found to be non-significant in affecting biogas yield according to the ANOVA analysis?\n\nA) Substrate ratio and Dilution\nB) Inoculum concentration and Dilution\nC) Substrate ratio and Temperature\nD) Substrate ratio and Inoculum concentration
```

#### **人类质检员**
```
题目背景信息不足/不完整或指代不清
```

#### **大模型质检员**
```
<conclusion>
``` json
{
  "qualified": false,
  "reason": "问题存在信息完整性不足的问题。题目询问根据ANOVA分析哪个操作变量间的交互作用对沼气产量影响不显著，但未提供任何ANOVA分析的背景信息、实验数据或结果依据。答题者不依赖原文文档无法判断各选项中交互作用的显著性，违反了“问题要求独立作答，不依赖原文文档”的要求，属于关键背景信息缺失。"
}
```

#### 输出
分析：人类质检员指出题目存在 “背景信息不足 / 不完整或指代不清” 的问题；大模型质检员同样指出问题 “信息完整性不足，未提供 ANOVA 分析的背景信息等，答题者不依赖原文文档无法判断”，二者均聚焦于题目背景信息缺失、需依赖外部文档才能作答这一核心问题，结论完全一致。

``` json
{{
    "一致性": 5
}}
```

#### 命题
```
In the improved asymptotic formula for \\( S_3(X) \\), what is the exponent of \\( X \\) in the error term?\n\n    A) \\( \\frac\{5\}\{3\} \\)\n    B) 3\n    C) 2\n    D) \\( \\frac{8}{3} \\)
```

#### **人类质检员**
```
缺乏教育价值（问题指向极端长尾信息或边缘只是，对于学生掌握知识的考查意义不大）
```

#### **大模型质检员**
```
``` json
{
  "qualified": false,
  "reason": "关键术语不清晰且背景信息不足。问题中提到的符号 \( S_3(X) \) 未在题干中解释其具体含义，且“improved asymptotic formula for \( S_3(X) \)”的相关背景信息缺失，导致答题者若不依赖原文文档无法明确问题所指的具体公式，不符合信息完整性要求，即题目无法独立作答。"
}
```
```

#### 输出
分析：人类质检员指出题目存在“缺乏教育价值，指向极端长尾信息或边缘知识，考查意义不大”的问题；大模型质检员则认为问题存在“关键术语不清晰且背景信息不足，符号\( S_3(X) \)未解释，无法独立作答”的问题。二者指出的问题核心完全不同，人类聚焦于内容的教育价值和核心性，大模型聚焦于术语清晰度和信息完整性，未涉及相同问题。
```json
{
    "一致性": 1
}
```
"""

DOC2QUERY_V3_CRITIQUE_RECALL_TEMPLATE = """
#### 命题
```
{instruction}
```

#### **人类质检员**
```
{human}
```

#### **大模型质检员**
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

RATIONALE_RECALL_FEWSHOTS = """
下面给你某个提问的回复和一个思维导图，你的任务是根据思维导图（树结构）对比回答进行分析，判断回答的拆解是否足够细致，越细致得分越高。

细致度得分1-5分，5分是最高分，分数仅包含1、2、3、4、5不包含小数点。

#### 细致度要求
层级覆盖完整性：判断回答是否覆盖思维导图中的所有核心主分支（如定理背景、证明特殊性、选项分析、结论推导等），以及各主分支下的关键子分支。若存在主分支或重要子分支完全未提及，则细致度降低。
细节颗粒度匹配度：检查思维导图各子分支下的细分内容（如具体机制、条件、逻辑理由等）是否在回答中得到对应展开。
逻辑链条完整性：评估回答是否体现思维导图中节点间的关键逻辑关联。若逻辑链条断裂或简化，则细致度下降。
关键节点无遗漏性：确认思维导图中对结论起决定性作用的细分节点是否在回答中呈现。重要节点遗漏会显著降低细致度。
阐述深度匹配度：对比回答与思维导图对同一内容的阐述深度。若思维导图对某一节点有多层细分解释，而回答仅停留在表层描述，则深度不足。


#### 输出要求
{分析}
```json
{{
    "细致度": *** # int (1-5)
}}
```
在分析中你需要详细的分析过程，并最终得出结论。

下面是一些例子

#### **问题**
```
What is a necessary condition for using supersymmetry to prove the Hirzebruch-Riemann-Roch theorem on a complex manifold?\n\n    A) The manifold must have a Kähler structure.\n    B) The manifold must be spin.\n    C) The manifold must have a trivial tangent bundle.\n    D) The manifold must have even dimension.
```

#### **回答**
```
To determine the necessary condition for using supersymmetry to prove the Hirzebruch-Riemann-Roch (HRR) theorem on a complex manifold, we analyze the options as follows:


### Key Background: Supersymmetric Proofs and Manifold Structures  
Supersymmetric proofs of index theorems (including HRR) rely on constructing supersymmetric quantum field theories on the manifold, which requires specific geometric structures to define supersymmetry operators (e.g., Dirac operators) and their interactions with the manifold’s geometry.  


### Analysis of Options  
- **Option A: The manifold must have a Kähler structure.**  
  A Kähler structure is a strong condition combining a complex structure, a Riemannian metric, and a symplectic form, with compatibility conditions between them. Supersymmetric proofs of HRR typically require the manifold to be Kähler. This is because Kähler geometry provides the necessary framework for defining supersymmetric Lagrangians, holomorphic vector bundles, and the curvature terms essential to the theorem.  

- **Option B: The manifold must be spin.**  
  A spin structure is required for the existence of a Dirac operator, but supersymmetric proofs of HRR do not inherently demand the manifold to be spin. The theorem applies to complex manifolds more generally, and spinness is not a necessary condition here.  

- **Option C: The manifold must have a trivial tangent bundle.**  
  A trivial tangent bundle (where the tangent bundle is isomorphic to a product bundle) is an extremely restrictive condition. Most complex manifolds (e.g., projective spaces) do not have trivial tangent bundles, yet HRR still applies to them. This is not a necessary condition.  

- **Option D: The manifold must have even dimension.**  
  While complex manifolds are always even-dimensional (since their real dimension is twice their complex dimension), this is a property of complex manifolds themselves, not a specific condition required for the supersymmetric proof of HRR. The proof’s necessity stems from Kähler geometry, not merely even dimension.  


**Conclusion:** The necessary condition for using supersymmetry to prove HRR is that the manifold is Kähler.  

**Answer: A**
```

#### **思维导图**
```\n问题：What is a necessary condition for using supersymmetry to prove the Hirzebruch-Riemann-Roch theorem on a complex manifold?\n├─ 定理背景分析\n│  ├─ HRR定理核心：将全纯向量丛的欧拉示性数与陈类及流形结构关联\n│  └─ 原始证明方法：代数几何/拓扑方法，不依赖超对称性\n│\n├─ 超对称证明的特殊性\n│  ├─ 超对称在数学中的应用场景：通常与指标定理（如Atiyah-Singer）证明相关\n│  ├─ 关键机制：通过超对称量子力学将Dolbeault算子与Witten指标关联\n│  └─ 结构要求：需保证超对称代数闭合（如Laplacian一致性、算子反交换性）\n│\n├─ 选项逻辑排除\n│  ├─ A) Kähler结构\n│  │  ├─ 必要性：保证Hermitian度量与复结构兼容，使Dolbeault与Hodge Laplacian一致\n│  │  └─ 超对称代数闭合的充分条件\n│  ├─ B) Spin结构\n│  │  └─ 无关性：HRR定理依赖复结构而非Spin结构，Dirac算子非本问题核心\n│  ├─ C) 平凡切丛\n│  │  └─ 过度限制：定理适用于一般复流形，与切丛平凡性无关\n│  └─ D) 偶数维\n│     └─ 冗余性：所有复流形实维数均为偶数，非额外条件\n│\n├─ 超对称证明的严格需求\n│  ├─ 核心矛盾：非Kähler流形中Dolbeault与Hodge Laplacian可能不重合\n│  └─ 结构依赖：Kähler度量确保超对称生成元的数学可实现性\n│\n└─ 结论推导路径\n   ├─ 排除法：B/C/D均无法满足超对称证明的特殊结构要求\n   └─ 正向确认：Kähler结构是保证超对称代数闭合的最小充分条件\n```

#### 输出
1. **层级覆盖完整性**：思维导图主分支包括定理背景分析、超对称证明的特殊性、选项逻辑排除、超对称证明的严格需求、结论推导路径。回答中，定理背景分析未提及HRR定理核心内容及原始证明方法；超对称证明的特殊性未涵盖关键机制（Dolbeault算子与Witten指标关联）和结构要求（超对称代数闭合）；超对称证明的严格需求（核心矛盾与结构依赖）完全未提及。仅选项逻辑排除和结论推导路径有部分覆盖，存在多个主分支及重要子分支遗漏。
2. **细节颗粒度匹配度**：思维导图子分支的细分内容，如A选项中“Dolbeault与Hodge Laplacian一致”“超对称代数闭合的充分条件”、超对称证明的核心机制等，在回答中未对应展开，颗粒度不足。
3. **逻辑链条完整性**：思维导图中“超对称代数闭合→Kähler结构保证→其他选项不满足”的逻辑链条，回答中简化了中间环节，未明确超对称代数闭合与Kähler结构的具体关联，逻辑链条存在简化。
4. **关键节点无遗漏性**：对结论起决定性作用的节点，如HRR定理核心、原始证明方法、超对称证明的关键机制和结构要求、超对称证明的核心矛盾等，在回答中均有遗漏。
5. **阐述深度匹配度**：思维导图对超对称证明的特殊性有多层解释，回答仅停留在“需要特定几何结构定义超对称算子”的表层描述，深度不足。

综上，回答在层级覆盖、细节颗粒度、逻辑链条、关键节点和阐述深度上均存在明显不足。
```json
{
    "细致度": 2
}
```



#### **问题**
```
在某工业设施中，设备的电源和控制线路由继电器和接触器切换，且这些线路与其他线路不分离。根据GB/T 17626.4-1998标准，应选择哪个试验等级进行电快速瞬变脉群抗扰度试验？\n\nA) 第1级：良好保护环境\nB) 第2级：受保护环境\nC) 第4级：较严酷工业环境\nD) 第3级：典型工业环境\nE) 第5级：特殊环境
```

#### **回答**
```
根据GB/T 17626.4-1998《电磁兼容 试验和测量技术 电快速瞬变脉冲群抗扰度试验》标准，结合题目描述的场景，应选择的试验等级为 **D) 第3级：典型工业环境**。以下是具体分析：


### 1. 标准中试验等级的核心划分依据
GB/T 17626.4-1998对电快速瞬变脉冲群（EFT/B）抗扰度试验等级的划分，主要基于设备所处的电磁环境严酷程度，尤其是**线路与其他干扰源的分离程度**和**工业活动的典型干扰水平**。等级越高，代表环境中电磁干扰越严重，试验施加的脉冲群强度越大。


### 2. 题目场景的关键特征
题目明确描述：“设备的电源和控制线路由继电器和接触器切换，且这些线路与其他线路不分离”。这一特征符合标准中对“典型工业环境”的定义：
- 存在继电器、接触器等开关设备，这些设备在操作时会产生电快速瞬变脉冲群干扰；
- 线路未与其他线路分离，意味着干扰更容易通过线缆耦合到设备，环境电磁干扰水平中等。


### 3. 其他等级的排除理由
- **A) 第1级（良好保护环境）**：通常指无强干扰源的实验室或控制室，线路分离良好，与题目中“线路不分离”和“继电器/接触器切换”的场景不符。
- **B) 第2级（受保护环境）**：适用于有一定防护措施的环境，如普通办公室或轻度工业环境，线路可能部分分离，干扰水平较低，不符合题意。
- **C) 第4级（较严酷工业环境）**：适用于干扰极强的环境，如重工业车间、强电磁辐射区域，线路可能完全暴露且干扰源密集，题目未提及“较严酷”特征，因此不适用。
- **E) 第5级（特殊环境）**：标准中第5级为非标准通用等级，需根据具体特殊需求协商确定，题目未涉及特殊环境描述，因此排除。


综上，题目场景最符合 **第3级（典型工业环境）** 的定义。

**答案：D** 
```

#### **思维导图**
```\n问题：在某工业设施中，设备的电源和控制线路由继电器和接触器切换，且这些线路与其他线路不分离。根据GB/T 17626.4-1998标准，应选择哪个试验等级进行电快速瞬变脉群抗扰度试验？\n├─ 问题背景解析\n│  ├─ 核心对象：工业设施中的电源/控制线路\n│  ├─ 关键特征：继电器/接触器切换、线路未分离\n│  └─ 标准依据：GB/T 17626.4-1998（等效IEC 61000-4-4）\n│\n├─ 标准框架梳理\n│  ├─ 试验等级划分逻辑\n│  │  ├─ 1级：良好保护环境（如计算机房）\n│  │  ├─ 2级：受保护环境（住宅/商业）\n│  │  ├─ 3级：典型工业环境\n│  │  ├─ 4级：较严酷工业环境\n│  │  └─ 5级：特殊环境\n│  └─ 等级选择原则：干扰源强度+耦合路径可能性\n│\n├─ 干扰机制分析\n│  ├─ 干扰源特性\n│  │  └─ 继电器/接触器切换→产生高频瞬态脉冲（电火花）\n│  └─ 耦合路径特性\n│     └─ 线路未分离→干扰易通过传导/辐射耦合到其他线路\n│\n├─ 场景匹配推理\n│  ├─ 典型工业环境（3级）对比\n│  │  └─ 适用于一般工业控制柜，干扰源强度中等\n│  ├─ 较严酷工业环境（4级）特征\n│  │  └─ 高压/频繁切换设备+无隔离线路→高干扰能量+强耦合\n│  └─ 排除法应用\n│     ├─ 排除1/2级（非工业场景）\n│     ├─ 排除5级（需直接电力系统连接）\n│     └─ 3级不满足\"线路未分离\"的严酷条件\n│\n├─ 结论验证\n│  └─ 题目场景完全符合4级定义的\"较严酷工业环境\"特征\n│     ├─ 干扰源：频繁切换的继电器/接触器\n│     └─ 传播条件：未分离线路导致低阻抗耦合路径\n```

#### 输出
1. **层级覆盖完整性**：思维导图主分支包括问题背景解析、标准框架梳理、干扰机制分析、场景匹配推理、结论验证。回答中，问题背景解析和标准框架梳理有部分覆盖；干扰机制分析提到了干扰源和耦合路径的基本特征；但场景匹配推理中未涵盖思维导图中“3级不满足‘线路未分离’的严酷条件”这一关键子分支，且结论验证主分支（认为场景符合4级特征）完全未提及，存在重要主分支和子分支遗漏。
2. **细节颗粒度匹配度**：思维导图子分支的细分内容，如标准框架中1级（计算机房）、2级（住宅/商业）的具体场景，干扰机制中“高频瞬态脉冲（电火花）”“传导/辐射耦合”，场景匹配推理中4级“高压/频繁切换设备+无隔离线路→高干扰能量+强耦合”等细节，在回答中未对应展开，仅进行了表层描述，颗粒度不足。
3. **逻辑链条完整性**：思维导图的逻辑链条为“干扰源（继电器切换）+耦合路径（线路未分离）→高干扰强度→匹配4级”，而回答的逻辑链条为“干扰源和线路未分离→中等干扰→匹配3级”，与思维导图的关键逻辑关联存在偏差，逻辑链条断裂。
4. **关键节点无遗漏性**：对结论起决定性作用的节点，如思维导图中“3级不满足线路未分离的严酷条件”“4级的较严酷工业环境特征（频繁切换+未分离线路导致低阻抗耦合）”等，在回答中均有遗漏。
5. **阐述深度匹配度**：思维导图对各等级特征、干扰机制和场景匹配有多层细分解释，而回答对干扰强度的判断、等级特征的描述仅停留在基础层面，未深入匹配思维导图的细分解释，阐述深度不足。

综上，回答在层级覆盖、细节颗粒度、逻辑链条、关键节点和阐述深度上均存在明显缺陷。
```json
{
    "细致度": 2
}
```
"""

RATIONALE_RECALL_TEMPLATE = """
#### **问题**
```
{question}
```

#### **回答**
```
{response}
```

#### **思维导图**
```
{rationale}
```
"""


DOC_LEAKAGE_DETECT_TEMPLATE = """### **回答话术分析**
任务描述：下面的回答是基于一篇文档进行参考回答的，你的任务是需要对回答的话术进行分析，判断是否会让读者意识到文档的存在。

#### 注意点：
1. 是否存在直接指向 “文档” 的表述
若回答中出现明确提及 “文档”“参考资料”“材料”“原文” 等词汇的表述（如 “根据文档内容”“参考资料中提到”“原文指出”“该材料显示” 等），则会直接让读者意识到回答基于外部文档，属于明显的 “暴露线索”。

2. 是否存在间接指代外部来源的痕迹
若未直接提及 “文档”，但表述中隐含对 “特定来源信息” 的依赖（如 “根据提供的信息”“从给定内容可知”“结合背景材料分析” 等），读者会通过这类表述推断回答并非完全基于普遍公认的知识，而是依赖某一外部信息载体（即文档），从而意识到文档的存在。

3. 内容呈现是否符合 “普遍知识陈述” 的逻辑
若回答中的信息（如数据、结论、分析逻辑）以 “客观事实”“学科公认知识” 的形式自然呈现，未体现对某一特定来源的依赖（例如直接陈述 “太阳系碎片的典型速度约为 30-80 km/s”，而非 “文档中提到太阳系碎片速度为 30-80 km/s”），则读者会默认内容是基于通用知识的解答，不会联想到 “文档” 这一外部载体。
反之，若内容呈现带有 “对特定信息的复述或针对性解读” 痕迹（如严格围绕某一非普遍已知的细节展开，且表述中隐含 “仅基于该信息” 的逻辑），则可能让读者察觉依赖外部文档。


#### 输出要求
- 先按照方法进行详细、准确、仔细地分析
- 再最后的部分按下面的JSON格式输出最后的结论
```json
{{
  "文档泄漏": "是/否",
}}
```

下面是一个具体的例子

## 问题
A vehicle is experiencing reduced engine performance, increased fuel consumption, and a strong unpleasant odor inside the cabin. Which filter is most likely clogged?\n\n    A) Cabin air filter\n    B) Fuel filter\n    C) Oil filter\n    D) Air filter

## 回答
The most likely clogged filter is a combination of the **air filter** and the **cabin air filter**, but based on the options provided and the primary symptoms, the key filter to identify is:

**D) Air filter**  


### Analysis of Symptoms:  
1. **Reduced engine performance and increased fuel consumption**: These are explicitly listed as consequences of a clogged **air filter** in the text. A clogged air filter restricts airflow to the engine, disrupting the air-fuel mixture, which leads to inefficient combustion, reduced power, and higher fuel usage.  

2. **Strong unpleasant odor inside the cabin**: While this aligns with issues related to cabin air filters (which remove odors via activated carbon layers), the question asks for the "most likely clogged" filter among the options. The engine-related symptoms (performance and fuel consumption) are directly tied to the air filter, making it the primary culprit here.  


Other options are less likely:  
- **A) Cabin air filter**: Would cause poor air quality or odors but not engine performance issues.  
- **B) Fuel filter**: Clogging would lead to engine stalling or fuel supply issues, not increased fuel consumption.  
- **C) Oil filter**: Clogging would harm engine components but not typically cause increased fuel consumption or cabin odors.  


**Answer: D**


## 输出
该回答会让读者意识到文档的存在，具体分析如下：  

1. **存在间接指代外部来源的痕迹**：回答中出现了 “These are explicitly listed as consequences of a clogged air filter in the text” 的表述，其中 “in the text” 明确隐含了对某一特定外部文本（即参考文档）的依赖，读者会通过这一表述推断回答的信息来源于该文本，而非单纯基于普遍公认的知识。  

2. **内容呈现带有对特定信息的复述痕迹**：虽然发动机性能下降、油耗增加与空气滤清器堵塞的关联属于汽车维修领域的常见知识，但回答中强调症状是 “explicitly listed in the text”，突出了对特定文本中信息的直接引用和针对性解读，而非以独立的普遍知识逻辑自然呈现，进一步暗示了对外部文档的依赖。  

综上，该回答通过间接指代外部文本的表述，会让读者意识到其基于参考文档生成。 

```json
{{
  "文档泄漏": "是",
}}
```


## 问题
在电化学腐蚀法制备多孔硅时，为了使n型硅形成多孔结构，必须满足的条件是？\n\n    A) 在黑暗中进行腐蚀\n    B) 在光照条件下进行腐蚀\n    C) 保持恒定电流密度\n    D) 使用高浓度HF溶液\n    E) 使用p型硅片

## 回答
在电化学腐蚀法制备多孔硅时，对于n型硅形成多孔结构必须满足的条件是**在光照条件下进行腐蚀**。  

文中明确提到：“当腐蚀n型硅时，应在光照下进行。” 这是因为n型硅的导电性依赖于电子，而在电化学腐蚀过程中，需要足够的空穴参与阳极氧化反应。光照可激发n型硅产生电子-空穴对，提供反应所需的空穴，从而促进阳极腐蚀过程中非均匀选择型腐蚀的发生，形成多孔结构。  

其他选项分析：  
- A) 在黑暗中进行腐蚀：与文中描述相反，黑暗条件下n型硅难以产生足够空穴，无法有效形成多孔结构。  
- C) 保持恒定电流密度：虽为电化学腐蚀的常见控制参数，但并非n型硅特有的必要条件。  
- D) 使用高浓度HF溶液：HF浓度影响孔径等结构参数，但不是n型硅形成多孔结构的必需条件。  
- E) 使用p型硅片：题干明确针对n型硅，此选项与问题无关。  


**答案：B** 

## 输出
该回答会让读者意识到文档的存在，具体分析如下：  

1. **存在直接指向外部来源的表述**：回答中出现了 “文中明确提到：‘当腐蚀n型硅时，应在光照下进行。’” 的表述，其中 “文中” 直接指代了某一外部文本（即参考文档），明确表明回答的信息来源于该文本，属于直接的“暴露线索”，会让读者意识到存在参考文档。  

2. **内容呈现带有对特定信息的复述痕迹**：虽然n型硅在光照下电化学腐蚀形成多孔结构的原理属于电化学领域的专业知识，但回答中通过“文中明确提到”直接引用外部文本内容，并围绕该引用展开分析，突出了对特定文本中信息的直接依赖和针对性解读，而非以独立的普遍知识逻辑自然呈现，进一步明确了对外部文档的依赖。  

综上，该回答通过直接指代外部文本的表述，会让读者意识到其基于参考文档生成。  

```json
{{
  "文档泄漏": "是"
}}
```



## 问题
在比较不同损失函数时，哪种损失函数使得模型的AUC值最高？\n\nA) 焦点损失\nB) 交叉熵损失\nC) 均方误差\nD) KL-focal损失

## 回答
## 一、核心概念精准阐释\n### 1. AUC（Area Under the Curve）\nAUC（曲线下面积）是衡量分类模型性能的一个指标，具体是指ROC曲线（受试者工作特征曲线）下的面积。AUC值的范围在0到1之间，AUC值越接近1，表示模型的分类性能越好，即模型能够更准确地区分正负样本。AUC主要关注的是模型在不同阈值下的分类性能。\n\n### 2. 损失函数\n损失函数用于衡量模型预测值与真实值之间的差异，是机器学习算法优化过程中的目标函数。不同的损失函数适用于不同的问题类型（如分类、回归）和数据分布特性。损失函数的选择直接影响模型的训练效果和最终性能。\n\n### 3. 选项解释\n- **A) 焦点损失（Focal Loss）**：一种用于解决类别不平衡问题的损失函数，特别适用于目标检测任务。它通过减少对容易分类样本的关注，增加对难分类样本的关注，以提升模型的分类效果。\n- **B) 交叉熵损失（Categorical Cross Entropy）**：适用于多分类任务的损失函数，用于衡量预测概率分布与真实概率分布之间的差异。\n- **C) 均方误差（Mean Squared Error）**：通常用于回归问题，衡量预测值与真实值之间的平方差。\n- **D) KL-focal损失（KL-focal Loss）**：结合了Kullback-Leibler散度和焦点损失，旨在平衡不同样本的权重，适用于类别不平衡的分类任务。\n\n## 二、实验数据溯源与关键信息提取\n### 1. 实验背景与条件\n本研究使用了多个损失函数在相同数据集上进行了对比实验，通过计算AUC值来评估不同损失函数对模型性能的影响。实验过程中，所有模型的训练条件（如训练轮数、学习率、优化器等）均保持一致，以确保实验结果的可比性。\n\n### 2. 核心数据点定位\n从文献或实验报告中提取关键数据点：\n- **焦点损失（Focal Loss）**：AUC = 0.6104\n- **交叉熵损失（Categorical Cross Entropy）**：AUC = 0.6515\n- **均方误差（Mean Squared Error）**：AUC = 0.5464\n- **KL-focal损失（KL-focal Loss）**：AUC = 0.6757\n\n这些数据在严格控制其他变量（模型结构、训练参数等）的情况下获得，确保了损失函数与AUC值的直接关联性。\n\n## 三、趋势分析：损失函数与AUC的关联逻辑\n### 1. 不同损失函数对AUC的影响\n- **焦点损失（Focal Loss）**：设计用于处理类别不平衡问题，通过调整样本权重，减少容易分类样本的影响。实验结果显示其AUC为0.6104。\n- **交叉熵损失（Categorical Cross Entropy）**：适用于多分类任务，通过最小化预测概率与真实概率之间的差异来优化模型。实验结果显示其AUC为0.6515。\n- **均方误差（Mean Squared Error）**：适用于回归任务，通过最小化预测值与真实值之间的平方差来优化模型。由于是回归损失函数，其在分类任务中的表现较差，AUC仅为0.5464。\n- **KL-focal损失（KL-focal Loss）**：结合了KL散度和焦点损失，旨在平衡样本权重和类别不平衡问题。实验结果显示其AUC为0.6757，是所有损失函数中最高的。\n\n### 2. 损失函数选择的考量\n不同损失函数的性能差异，主要反映了其在处理特定问题（如类别不平衡、预测误差等）上的优势。实验数据表明，KL-focal损失在本研究中表现最优，其AUC值显著高于其他损失函数。\n\n## 四、机理深化：损失函数设计的优化机制\n### 1. 损失函数设计的背景\n损失函数的设计考虑了数据集的特性（如类别不平衡）和任务目标（如分类或回归）。KL-focal损失通过结合KL散度和焦点损失，旨在平衡样本权重，减少类别不平衡问题对模型性能的影响。\n\n### 2. 优化机制\nKL-focal损失通过以下机制优化了模型性能：\n- **平衡样本权重**：通过调整样本权重，减少容易分类样本的影响，增加难分类样本的影响，提高模型对难分类样本的分类能力。\n- **类别不平衡处理**：通过结合KL散度，更好地处理类别不平衡问题，减少类别间样本数量差异对模型性能的影响。\n\n## 五、选项逐一辨析与错误原因剖析\n### 1. 选项A（焦点损失）\n错误原因：尽管焦点损失在处理类别不平衡问题上表现出色，但其AUC值（0.6104）低于其他损失函数，特别是在KL-focal损失面前。实验数据显示，焦点损失在本研究中的性能不如KL-focal损失。\n\n### 2. 选项B（交叉熵损失）\n错误原因：交叉熵损失虽然适用于多分类任务，但在本研究中其AUC值（0.6515）低于KL-focal损失，显示其在处理类别不平衡问题上不如KL-focal损失有效。\n\n### 3. 选项C（均方误差）\n错误原因：均方误差适用于回归任务，而在分类任务中表现较差，其AUC值（0.5464）远低于其他损失函数，不适合用于本研究中的分类任务。\n\n### 4. 选项D（KL-focal损失）\n正确原因：KL-focal损失在本研究中表现最优，其AUC值（0.6757）显著高于其他损失函数，显示其在处理类别不平衡问题上的优势。\n\n## 六、结论与工程启示\n本分析表明：在比较不同损失函数时，KL-focal损失使得模型的AUC值最高，核心原因在于其结合了KL散度和焦点损失的优势，有效处理了类别不平衡问题，提升了模型的分类性能。这一结论对实际应用的指导意义在于：在处理类别不平衡的分类任务时，优先考虑使用KL-focal损失，以获得更好的分类性能。\n\n**答案：D**

## 输出
该回答会让读者意识到文档的存在，具体分析如下：  

1. **存在间接指代外部来源的痕迹**：回答中出现了 “本研究使用了多个损失函数在相同数据集上进行了对比实验”“从文献或实验报告中提取关键数据点” 等表述。其中 “本研究”“文献或实验报告” 隐含了对特定外部研究成果或文献资料（即参考文档）的依赖，读者会通过这类表述推断回答中的实验数据、结论等信息来源于外部研究或文献，而非单纯基于普遍公认的知识。  

2. **内容呈现带有对特定信息的复述痕迹**：回答中提供了各损失函数对应的具体AUC数值（如焦点损失AUC=0.6104、KL-focal损失AUC=0.6757等），这些具体数据并非机器学习领域的普遍公认常识，而是特定实验或研究的结果。同时，回答围绕这些特定数据展开分析，强调“在严格控制其他变量的情况下获得”，突出了对特定研究中信息的依赖和针对性解读，而非以独立的普遍知识逻辑自然呈现，进一步暗示了对外部文档的依赖。  

综上，该回答通过间接指代外部研究或文献的表述及对特定实验数据的复述，会让读者意识到其基于参考文档生成。  

```json
{{
  "文档泄漏": "是"
}}
```



现在你需要对下面的回答进行话术分析，判断是否存在文档泄漏的情况。

## 问题
{question}

## 回答
{response}

## 输出
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
        results = await agent.run(_prompts, max_concurrent_requests, desc=f"[{self.task_desc()} {agent.model}={max_concurrent_requests}]", postprocess_fns=[self.postprocess]*len(_prompts))

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
        results = await agent.run(_prompts, max_concurrent_requests, desc=f"[{self.task_desc()} {agent.model}={max_concurrent_requests}]", postprocess_fns=[self.postprocess]*len(_prompts))

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


class Doc2QueryV3VerifierCritiqueRecall(CritiqueRecall):
    def __init__(self):
        super().__init__()

    @classmethod
    def task_desc(cls):
        return "批评覆盖度"

    def prompt_fn(self, example):
        solver_response, gt = example
        critique = gt["critique"]
        if critique is None:
            critique = f'质检合格'
        prompt = DOC2QUERY_V3_CRITIQUE_RECALL_FEWSHOTS + "\n\n\n" + DOC2QUERY_V3_CRITIQUE_RECALL_TEMPLATE.format(
            llm=solver_response,
            human=critique,
            instruction=gt["instruction"]
        )
        return prompt

    def postprocess(self, response: str):
        s = response
        try:
            conclusion = s.strip()

            recall = re.findall(
                r'\"一致性\": (\d+)', conclusion)[0]
            if isinstance(recall, str):
                recall = float(recall)
            assert isinstance(recall, float)
            assert recall in (0, 1, 2, 3, 4, 5)
            return recall

        except Exception as err:
            raise PostprocessError(f'{err}')


class Doc2QueryV3LectureNoteStyleJudge(CritiqueRecall):
    def __init__(self):
        super().__init__()

    @classmethod
    def task_desc(cls):
        return "讲义风格评价"

    def prompt_fn(self, example):
        solver_response, gt = example
        prompt = LECTURE_NOTE_STYLE_JUDGE_FEWSHOTS + "\n\n\n" + LECTURE_NOTE_STYLE_JUDGE_TEMPLATE.format(
            question=gt["question"],
            response=solver_response,
        )
        return prompt

    def postprocess(self, response: str):
        s = response
        try:
            conclusion = s.strip()

            recall = re.findall(
                r'\"匹配度\": (\d+)', conclusion)[0]
            if isinstance(recall, str):
                recall = float(recall)
            assert isinstance(recall, float)
            assert recall in (0, 1, 2, 3, 4, 5)
            return recall

        except Exception as err:
            raise PostprocessError(f'{err}')


class RationaleRecall(CritiqueRecall):
    def __init__(self):
        super().__init__()

    @classmethod
    def task_desc(cls):
        return "思维覆盖度"

    def prompt_fn(self, example):
        solver_response, gt = example
        prompt = RATIONALE_RECALL_FEWSHOTS + "\n\n\n" + RATIONALE_RECALL_TEMPLATE.format(
            question=gt["question"],
            response=solver_response,
            rationale=gt["rationale"]
        )
        return prompt

    def postprocess(self, response: str):
        s = response
        try:
            conclusion = s.strip()

            recall = re.findall(
                r'\"细致度\": (\d+)', conclusion)[0]
            if isinstance(recall, str):
                recall = float(recall)
            assert isinstance(recall, float)
            assert recall in (0, 1, 2, 3, 4, 5)
            return recall

        except Exception as err:
            raise PostprocessError(f'{err}')


class DocLeakageDetect(BatchCallOpenAPI):
    def __init__(self):
        pass

    @classmethod
    def task_desc(cls):
        return "文档泄漏检测"

    def prompt_fn(self, example):
        solver_response, gt = example
        prompt = DOC_LEAKAGE_DETECT_TEMPLATE.format(
            question=gt["question"],
            response=solver_response
        )
        return prompt

    def postprocess(self, response: str):
        s = response
        try:
            conclusion = s.strip()
            judge = re.findall(
                r'\"文档泄漏\": \"(.*)\"', conclusion)
            if len(judge) > 0 and judge[0] in ("是", "否"):
                return judge[0] == "是"

            conclusion = conclusion[conclusion.index(
                "```json")+len("```json"):].strip()
            conclusion = conclusion[:conclusion.index("```")].strip()
            try:
                conclusion = json.loads(conclusion)
                if conclusion["文档泄漏"] not in ("是", "否"):
                    raise PostprocessError(f'corrupt')
                return conclusion["文档泄漏"] == "是"
            except Exception as err:
                print(err)
                try:
                    conclusion = re.findall(
                        r'\"文档泄漏\": \"(.*)\"', conclusion)[0]
                    if not conclusion in ("是", "否"):
                        raise PostprocessError(f'corrupt')
                    return conclusion == "是"
                except Exception as err:
                    print(err)
                    raise PostprocessError(f'{err}')
        except Exception as err:
            print(err)
            raise PostprocessError(f'{err}')


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
        return re.findall(r'```python(.*?)```', solution_str, re.DOTALL)[0].strip()
    except Exception as err:
        return None


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
# DOC2QUERY V3 VERIFIER
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def doc2query_verifier_parse_solution_fn(solution_str: str):
    solution_str = postprocess_solution(solution_str)
    if not solution_str.startswith("<think>"):
        solution_str = f'<think>\n{solution_str}'

    if solution_str.count("</think>") > 1 or solution_str.count("</conclusion>") > 1:
        return None
    try:
        conclusion = re.findall(
            r'<conclusion>(.*)</conclusion>', solution_str, re.DOTALL)[0]
        return conclusion
    except Exception as err:
        return None


class Doc2QueryV3VerifierComputeScore(CriteriaRMRecallComputeScore):
    def __init__(self,
                 parse_solution_fn,
                 split="train",
                 args=None,
                 min_reward=-2.0
                 ):
        super().__init__(
            split=split, parse_solution_fn=parse_solution_fn, args=args, min_reward=min_reward
        )
        self.task_name = "DOC2QUERY_V3_VERIFIER"

    def init_agent(self):
        self.agents = {}
        self.init_verify_agent()

    def init_verify_agent(self):
        self.verify_agent = Agent(
            **self.args["verify_agent"]["model"])

    async def calc_critique_recall(self, batch_solution_str, batch_ground_truth):
        task = Doc2QueryV3VerifierCritiqueRecall()
        indices = []
        llm_critique = []
        for i, (gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            result = self.parse_solution_fn(sol)
            if result is not None:
                llm_critique.append((result.strip(), gt))
                indices.append(i)
            else:
                continue

        recall = await task.do_job(
            agent=self.verify_agent,
            batch_inputs=llm_critique,
            max_concurrent_requests=self.args["verify_agent"]["max_concurrent_requests"],
        )
        scores = [0.0] * len(batch_solution_str)
        for _recall, index in zip(recall, indices):
            if _recall is None:
                pass
            else:
                scores[index] = _recall
        return scores

    def calc_judge_accuracy(self, batch_solution_str, batch_ground_truth):
        acc = []
        for i, (sol, gt) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            result = self.parse_solution_fn(sol)
            if result is None:
                acc.append(-2.0)
            else:
                critique = gt["critique"]
                if critique is None:
                    correct = True
                else:
                    correct = False
                result = result.lower()
                if "true" in result and "false" in result:
                    acc.append(-2.0)
                    continue
                if "***" in result:
                    acc.append(-2.0)
                    continue

                judge = re.findall(r'\"qualified\": (.*)', result)
                if len(judge) == 1 and ("true" in judge[0].lower().strip()):
                    if correct:
                        acc.append(1.0)
                    else:
                        acc.append(0.0)
                else:
                    if correct:
                        acc.append(0.0)
                    else:
                        acc.append(1.0)
        return acc

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
        acc = self.calc_judge_accuracy(batch_solution_str, batch_ground_truth)
        recall = await self.calc_critique_recall(batch_solution_str, batch_ground_truth)

        final_results = []
        for i in range(len(batch_solution_str)):
            _reward = acc[i] + recall[i]
            final_results.append(_reward)

            if _reward > 1.3 or (self.split == "valid" and random.random() < 0.5) or (self.split == "train" and random.random() < 0.1):
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
                    f'[Final Reward]={_reward:.3f}|ACC={acc[i]}|RECALL={recall[i]}\n')

                if random.random() < 0.5:
                    print(f'[Thought]\n{batch_solution_str[i]}')
                    print()

        return final_results

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# DOC2QUERY V3 VERIFIER
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

        accuracy = await self.get_accuracy(
            batch_data_sources,
            batch_solution_str,
            batch_ground_truth,
        )

        def get_meta_cognition_counts(sol):
            return len(re.findall(r'<mutter>.*?</mutter>', sol, re.DOTALL))

        def convert_to_ranks(points, max_score=0.1):
            # 按降序排序并生成排名（数值越大排名越靠前）
            points_sorted = sorted(points, reverse=True)
            n = len(points_sorted)
            # 计算等差步长（确保最后一名分数为0）
            step = max_score / (n - 1) if n > 1 else 0

            # 为每个唯一值分配分数
            scores = {x: max_score - step *
                      points_sorted.index(x) for x in set(points)}
            return [scores[point] for point in points]

        meta_cognition_counts = defaultdict(list)
        for i in range(len(batch_solution_str)):
            _uuid = batch_ground_truth[i]["extra_info"]["uuid"]
            _reward = accuracy[i]
            meta_cognition_counts[_uuid].append(
                get_meta_cognition_counts(batch_solution_str[i]))

        final_results = []
        for i in range(len(batch_solution_str)):
            _uuid = batch_ground_truth[i]["extra_info"]["uuid"]
            _reward = accuracy[i]
            mapper = {x: y for x, y in zip(
                meta_cognition_counts[_uuid], convert_to_ranks(meta_cognition_counts[_uuid]))}

            struct_info_str = f'[TODO]'
            extra_reward = 0.0
            if len(mapper) > 1:
                c = get_meta_cognition_counts(batch_solution_str[i])
                if c in mapper:
                    extra_reward = mapper[c]
                    struct_info_str = f'[EXTRA={extra_reward}]'

            if self.split == "train":
                _reward += extra_reward

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

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# DOC2QUERY_V3_SOLVER
# ------------------------------------------------------------------------------------------------------------------------------------------------------


class Doc2QueryV3SolverComputeScore(CriteriaRMRecallComputeScore):
    def __init__(self,
                 parse_solution_fn,
                 split="train",
                 args=None,
                 min_reward=-2.0
                 ):
        super().__init__(
            split=split, parse_solution_fn=parse_solution_fn, args=args, min_reward=min_reward
        )
        self.task_name = "DOC2QUERY_V3_SOLVER"

    def init_agent(self):
        self.agents = {}
        self.init_verify_agent()

    def init_verify_agent(self):
        self.verify_agent = Agent(
            **self.args["verify_agent"]["model"])

    def parse_answer_from_solution(self, sol):
        sol = self.parse_solution_fn(sol)
        if sol is None:
            return None

        try:
            conclusion = sol.strip().split("\n")[-1]
            conclusion = conclusion[conclusion.index(
                "Answer:")+len("Answer:"):].strip()
            return conclusion
        except Exception as err:
            return None

    def get_accuracy(
        self, batch_solution_str, batch_ground_truth
    ):
        acc = []
        for sol, gt in zip(batch_solution_str, batch_ground_truth):
            acc.append(1.0 if self.parse_answer_from_solution(
                sol) == gt["ground_truth"] else 0.0)
        return acc

    async def get_lecture_style_score(self, batch_solution_str, batch_ground_truth):
        task = Doc2QueryV3LectureNoteStyleJudge()
        indices = []
        llm_critique = []
        for i, (gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            result = sol
            if result is not None:
                llm_critique.append((result.strip(), gt))
                indices.append(i)
            else:
                continue

        style_scores = await task.do_job(
            agent=self.verify_agent,
            batch_inputs=llm_critique,
            max_concurrent_requests=self.args["verify_agent"]["max_concurrent_requests"],
        )
        scores = [0.0] * len(batch_solution_str)
        for style_score, index in zip(style_scores, indices):
            if style_score is None:
                pass
            else:
                scores[index] = style_score
        return scores

    async def get_rationale_recall(self, batch_solution_str, batch_ground_truth):
        task = RationaleRecall()
        indices = []
        llm_critique = []
        for i, (gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            result = sol
            if result is not None:
                llm_critique.append((result.strip(), gt))
                indices.append(i)
            else:
                continue

        recall = await task.do_job(
            agent=self.verify_agent,
            batch_inputs=llm_critique,
            max_concurrent_requests=self.args["verify_agent"]["max_concurrent_requests"],
        )
        scores = [0.0] * len(batch_solution_str)
        for _recall, index in zip(recall, indices):
            if _recall is None:
                pass
            else:
                scores[index] = _recall
        return scores

    async def detect_doc_leakage(
            self,
            batch_solution_str,
            batch_ground_truth):

        task = DocLeakageDetect()
        indices = []
        llm_critique = []
        for i, (gt, sol) in enumerate(zip(batch_ground_truth, batch_solution_str)):
            result = sol
            if result is not None:
                llm_critique.append((result.strip(), gt))
                indices.append(i)
            else:
                continue

        evaluations = await task.do_job(
            agent=self.verify_agent,
            batch_inputs=llm_critique,
            max_concurrent_requests=self.args["verify_agent"]["max_concurrent_requests"],
        )
        scores = [0.0] * len(batch_solution_str)
        for _eval, index in zip(evaluations, indices):
            if _eval is None:
                pass
            else:
                scores[index] = _eval
        return scores

    async def _compute_score(self,
                             batch_data_sources,
                             batch_solution_str,
                             batch_ground_truth,
                             ):
        leakage = await self.detect_doc_leakage(
            batch_solution_str,
            batch_ground_truth,
        )
        rationale_recall = await self.get_rationale_recall(
            batch_solution_str,
            batch_ground_truth,
        )
        style_score = await self.get_lecture_style_score(
            batch_solution_str,
            batch_ground_truth,
        )
        acc = self.get_accuracy(
            batch_solution_str,
            batch_ground_truth,
        )

        final_results = []
        for i in range(len(batch_solution_str)):
            _reward = acc[i]
            if _reward > 0:
                _reward += 0.2 * \
                    rationale_recall[i] + 0.1 * \
                    (1.0-leakage[i]) + 0.1 * style_score[i]
            final_results.append(_reward)

            if (_reward > 1.0 and random.random() < 0.03) or (self.split == "valid" and random.random() < 0.001) or (self.split == "train" and random.random() < 0.001):
                log = True
                log_flag = f"[{self.task_name} VALID]" if self.split == "valid" else f"[{self.task_name} TRAIN]"
            else:
                log = False
                log_flag = ""

            if log:
                print(
                    f"--------------------------------{log_flag}--------------------------------")
                print(
                    f'【Question】`{repr(batch_ground_truth[i]["question"])}`')
                print(
                    f'【Answer】`{repr(batch_ground_truth[i]["ground_truth"])}`')
                print(
                    f'【Rationale】`{batch_ground_truth[i]["rationale"]}`')
                print(
                    f'[Final Reward]={_reward:.3f}|Leakage={leakage[i]}|Rationale={rationale_recall[i]}|Style={style_score[i]}\n')
                print(
                    f'【Solution】')
                print(batch_solution_str[i])

        return final_results


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# DOC2QUERY_V3_SOLVER
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


DOC2QUERY_V3_VERIFIER_DEFAULT_PARAMS = {
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
    "save_rollouts": {
        "default_local_dir": "/cpfs01/shared/llm_ddd/tongjian/ckpts/datareview_rl_test/verl/grpo/criteria_rm_recall"
    }
}

DOC2QUERY_V3_SOLVER_DEFAULT_PARAMS = {
    "verify_agent": {
        "model": {
            "model": "Qwen2.5-32B-Instruct",
            "base_url": "https://sd262bskcm47j59r1292g.apigateway-cn-beijing.volceapi.com/v1",
            "api_keys": "caa6246b-afbe-4d9b-ab34-87bf9922032b",
            # "model": "DeepSeek-V3-0324",
            # "base_url": "https://sd265fbi80c6ft26qc5ig.apigateway-cn-beijing.volceapi.com/v1",
            # "api_keys": "caa6246b-afbe-4d9b-ab34-87bf9922032b",
            "request_kwargs": {
                "temperature": 0.6,
                "timeout": 360,
                "max_tokens": 2048,
            }
        },
        "max_concurrent_requests": 512
    },
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

doc2query_verifier_score_train = Doc2QueryV3VerifierComputeScore(
    doc2query_verifier_parse_solution_fn, split="train", args=DOC2QUERY_V3_VERIFIER_DEFAULT_PARAMS).compute_score
doc2query_verifier_score_valid = Doc2QueryV3VerifierComputeScore(
    doc2query_verifier_parse_solution_fn, split="valid", args=DOC2QUERY_V3_VERIFIER_DEFAULT_PARAMS).compute_score


# DOC2QUERY_V3_SOLVER_DEFAULT_PARAMS

# doc2query_solver_score_train = Doc2QueryV3SolverComputeScore(
#     criteria_rft_parse_solution_fn, split="train", args=DOC2QUERY_V3_SOLVER_DEFAULT_PARAMS).compute_score
# doc2query_solver_score_valid = Doc2QueryV3SolverComputeScore(
#     criteria_rft_parse_solution_fn, split="valid", args=DOC2QUERY_V3_SOLVER_DEFAULT_PARAMS).compute_score

doc2query_solver_score_train = Doc2QueryV3SolverComputeScore(
    lambda x: x, split="train", args=DOC2QUERY_V3_SOLVER_DEFAULT_PARAMS).compute_score
doc2query_solver_score_valid = Doc2QueryV3SolverComputeScore(
    lambda x: x, split="valid", args=DOC2QUERY_V3_SOLVER_DEFAULT_PARAMS).compute_score
