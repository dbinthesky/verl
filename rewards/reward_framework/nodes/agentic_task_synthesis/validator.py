"""
Agentic Task Synthesis - Validator Nodes

Validates rubric quality using LLM:
- CategoryOrthogonalityCheckNode: Checks if rubric categories are orthogonal
"""

from __future__ import annotations

from typing import Any, Dict, List

from ...core import NodeConfig
from ..base import MapNode
from .data import AgenticTaskSample, RubricCategory


__all__ = ['CategoryOrthogonalityCheckNode']


class CategoryOrthogonalityCheckNode(MapNode[AgenticTaskSample]):
    """Check orthogonality between rubric categories using LLM.

    This node judges whether categories are:
    1. Redundant (overlapping in meaning)
    2. Non-orthogonal (not independent)

    Processing:
        - Input: AgenticTaskSample (must already have expanded categories)
        - Uses LLM to judge category orthogonality
        - Stores rubric orthogonality judgment in sample metadata
        - Skips sample if check fails (controlled by config.skip_on_failure)

    Attributes:
        agent: LLM agent for judging
    """

    # 显式声明处理的数据类型
    data_type = AgenticTaskSample

    def __init__(self, config: NodeConfig, agent: 'Agent'):
        super().__init__(config)
        self.agent = agent

    def _build_prompt(
        self,
        task_description: str,
        categories: List[RubricCategory]
    ) -> str:
        """Build prompt for LLM to judge category orthogonality.

        Args:
            task_description: Task description from sample
            categories: List of RubricCategory to check

        Returns:
            Prompt string for LLM
        """
        # Build category list (只使用 category_name)
        category_list = "\n".join([
            f"{i+1}. {cat.category_name}"
            for i, cat in enumerate(categories)
        ])

        prompt = f"""【任务描述】
{task_description}


【Rubric 类别列表】
{category_list}


# Role: 任务评估基准审计官

## 1. 任务目标
你现在的角色是一个冷酷、严苛的"逻辑查重与正交性审计官"。
你的任务是接收一份新生成的【任务评价标准】的大的类目，并对其进行深度的"反冗余测试"。
你必须判断该 JSON 中的 Rubric 分类大类是否严格满足下面的原则（相互独立正交）。一旦发现任何维度的重叠、包含或同义反复，必须立刻予以否决。


## 2. 核心审计红线
在审查输入的 Rubrics 时，请严格使用以下三种违规模型进行匹配验证。只要触碰任何一条，即判定为"非正交（False）"。

### 违规类型 1：大类概念重叠
- **定义**：Rubric 类目在逻辑概念上没有清晰的边界，存在相互交织或涵盖的情况。
- **致命案例**：大类 A 叫"数据统计与分析"，大类 B 叫"关联强度推演"。（这两个阶段在实际执行时必然高度重合，执行者无法清晰区分一个动作到底属于 A 还是 B）。
- **合格标准**：大类之间必须像流水线一样是严格的时间先后关系或完全独立的模块（例如：A. 异常排查 -> B. 核心成因拆解 -> C. 优化方案

### 违规类型 2：语义同义重述
- **定义**：两个或多个 Rubric 类别，使用了不同的专业词汇或句式，但本质上测试的是**同一个目标或同一个信息检查点**。只要 Agent 达成了条目 A，就必然自动达成了条目 B。

### 违规类型 3：包含与子集关系
- **定义**：生成了宏观的"概括性"条目，同时又生成了隶属于该概括的"细节"条目。
- **致命案例**：
  - 条目 A（父集）："是否给出了两项评估标准存在冲突的完整成因分析"
  - 条目 B（子集）："是否指出了权重分布差异是导致冲突的成因"
  （如果你判定了父集，子集就多余了；如果你要验证子集，父集就成了无效的废话。违背了"判定原子性"）。

## 3. 审查逻辑链
请深呼吸，按照以下步骤对输入的 JSON 进行扫描：
**类目交叉比对**：进行两两比对，判定 Rubric 类目设计是否独立正交。

## 4. 输出格式
**通过时：**
```json
{{
  "reason": "所有大类相互独立，符合正交性要求",
  "pass": true
}}
```

**未通过时：**
```json
{{
  "reason": "具体违规原因（必须指出哪些大类违规，属于哪种违规类型）",
  "pass": false
}}
```

**字段说明：**
- `pass`: 布尔值，true 表示正交，false 表示违规
- `reason`: 字符串，简明扼要说明判定理由（失败时必须指出违规的大类和违规类型编号）

请先进行详尽的分析，然后在响应的最后部分严格按照以下 JSON 格式输出，只需包含两个字段；你不需要直接输出json结果字段
"""
        return prompt

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract orthogonality judgment.

        Args:
            response_text: Raw LLM response (expects binary format: {"pass": bool, "reason": str})

        Returns:
            Parsed judgment dict with keys:
                - pass (bool): True if orthogonal, False if violates
                - reason (str): Explanation
                - orthogonality_score (float): Derived from pass (1.0 or 0.0)
                - is_orthogonal (bool): Same as pass
                - reasoning (str): Same as reason

        Raises:
            ValueError: If parsing fails
        """
        import json
        import re

        try:
            # Try to extract JSON from response
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)

            if match:
                json_str = match.group(1)
            else:
                # Try to find bare JSON object with "pass" field
                json_pattern2 = r'\{[^}]*"pass"[^}]*\}'
                match2 = re.search(json_pattern2, response_text, re.DOTALL)
                if match2:
                    json_str = match2.group(0)
                else:
                    raise ValueError("No JSON found in response")

            data = json.loads(json_str)

            # Validate required fields (new binary format)
            if "pass" not in data:
                raise ValueError("Missing 'pass' field in response")
            if "reason" not in data:
                raise ValueError("Missing 'reason' field in response")

            # Extract binary judgment
            pass_flag = bool(data["pass"])
            reason = str(data["reason"])

            # Return with backward-compatible fields
            return {
                "pass": pass_flag,
                "reason": reason,
                "orthogonality_score": 1.0 if pass_flag else 0.0,
                "is_orthogonal": pass_flag,
                "reasoning": reason
            }

        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    async def map_one(self, data: AgenticTaskSample, context: Dict[str, Any]) -> None:
        """Judge category orthogonality for single sample (in-place).

        Modifies:
            - data.set_meta('rubric_orthogonality_score', score)
            - data.set_meta('rubric_orthogonality_judgment', judgment_dict)
            - Optionally marks as skipped based on config:
                - skip_on_none: Skip if LLM returns None
                - skip_on_negative: Skip if orthogonality check fails (pass=false)
                - skip_on_failure: Skip if parsing/execution error occurs
        """
        # Check if categories exist (使用类型过滤)
        categories = data.get_children(RubricCategory)

        if len(categories) == 0:
            # No categories to check
            data.set_meta('rubric_orthogonality_score', 1.0)
            data.set_meta('rubric_orthogonality_note', 'no_categories')
            return

        if len(categories) == 1:
            # Single category is always orthogonal
            data.set_meta('rubric_orthogonality_score', 1.0)
            data.set_meta('rubric_orthogonality_note', 'single_category')
            return

        try:
            # Build prompt
            prompt = self._build_prompt(data.task_description, categories)

            if not prompt:
                data.set_meta('rubric_orthogonality_error', 'empty_prompt')
                if self.config.skip_on_failure:
                    data.mark_skipped("empty_prompt", self.name)
                return

            # Call LLM
            response = await self.agent.generate(prompt)

            if response is None:
                # LLM调用失败返回None
                data.set_meta('rubric_orthogonality_error', 'llm_call_returned_none')
                if self.config.skip_on_none:
                    data.mark_skipped("llm_call_failed", self.name)
                return

            # Store raw response and prompt in metadata (for debugging/testing)
            data.set_meta('rubric_orthogonality_prompt', prompt)
            data.set_meta('rubric_orthogonality_raw_response', response.content)
            data.set_meta('rubric_orthogonality_finish_reason', response.finish_reason)
            if response.usage:
                data.set_meta('rubric_orthogonality_token_usage', response.usage)

            # Parse response
            judgment = self._parse_llm_response(response.content)

            # Store results in metadata
            data.set_meta('rubric_orthogonality_score', judgment['orthogonality_score'])
            data.set_meta('rubric_orthogonality_judgment', judgment)
            data.set_meta('rubric_orthogonality_reasoning', judgment['reasoning'])

            # Check if orthogonality failed (negative result)
            if not judgment['pass']:
                # 正交性检查未通过
                if self.config.skip_on_negative:
                    data.mark_skipped(f"orthogonality_failed: {judgment['reason']}", self.name)

        except Exception as e:
            # 解析或执行异常
            error_msg = f"Failed to parse LLM response: {str(e)}"
            data.set_meta('rubric_orthogonality_error', error_msg)

            if self.config.skip_on_failure:
                data.mark_skipped(f"rubric_orthogonality_check_error: {e}", self.name)
