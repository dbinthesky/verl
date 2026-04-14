"""
Agentic Task Synthesis - Validator Nodes

Validates rubric quality using LLM:
- CategoryOrthogonalityCheckNode: Checks if rubric categories are orthogonal
- CategoryClassificationCheckNode: Checks if rubrics are correctly classified under their category
- RubricQualityCheckNode: Base class for rubric item quality validation
"""

from __future__ import annotations

import json
import re
from abc import abstractmethod
from typing import Any, Dict, List

from ...core import NodeConfig
from ..base import MapNode
from .data import AgenticTaskSample, RubricCategory, RubricItem


__all__ = [
    'CategoryOrthogonalityCheckNode',
    'CategoryClassificationCheckNode',
    'RubricQualityCheckNode',
    'RubricRigidityCheckNode',
]


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
            task_description: Task description from parent sample
            categories: List of all rubric categories to check

        Returns:
            Prompt string for LLM
        """
        # Build categories list with names
        categories_text = "\n".join([f"{i+1}. {cat.category_name}" for i, cat in enumerate(categories)])

        prompt = f"""【任务描述】
{task_description}


【Rubric 大类列表】
{categories_text}


# Role: Rubric 大类正交性审计官

你的任务是判断以上 Rubric 大类之间是否存在概念重叠、冗余或非正交的情况。

## 审计标准

判断这些大类是否满足以下条件：

1. **概念独立性**：每个大类是否有清晰的概念边界，不存在交织或涵盖的情况
2. **语义唯一性**：不同大类是否用不同表述测试同一个维度
3. **层级独立性**：不存在包含与子集关系（父类与子类同时出现）

## 不合格标准（任一即不合格）

### 违规类型 1：概念重叠
- **定义**：大类在逻辑概念上没有清晰的边界，存在相互交织或涵盖的情况
- **致命案例**：大类 A "数据统计与分析"，大类 B "关联强度推演"（实际执行时无法清晰区分）

### 违规类型 2：语义同义重述
- **定义**：两个或多个大类，使用了不同的专业词汇或句式，但本质上测试的是同一个维度
- **致命案例**：大类 A "核心价值明确"，大类 B "主要价值说明"（两者本质相同）

### 违规类型 3：包含与子集关系
- **定义**：生成了宏观的"概括性"大类，同时又生成了隶属于该概括的"细节"大类
- **致命案例**：
  - 大类 A（父集）："内容完整性检查"
  - 大类 B（子集）："技术细节完整性"
  （如果 A 包含 B，则 B 是冗余的）


## 输出格式

```json
{{
  "reason": "详细说明判定理由。如果不合格，必须指出：哪些大类之间存在重叠/冗余（属于哪种违规类型）",
  "pass": true/false
}}
```

**字段说明**：
- `reason`: 字符串，详细说明判定理由
- `pass`: 布尔值，true 表示合格（大类之间正交），false 表示不合格（存在重叠或冗余）

请先进行详细分析，然后输出 JSON 格式的判定结果。
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

        Raises:
            ValueError: If parsing fails
        """
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

            # Validate required fields
            if "pass" not in data:
                raise ValueError("Missing 'pass' field in response")
            if "reason" not in data:
                raise ValueError("Missing 'reason' field in response")

            # Extract binary judgment
            pass_flag = bool(data["pass"])
            reason = str(data["reason"])

            return {
                "pass": pass_flag,
                "reason": reason
            }

        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    async def map_one(self, data: AgenticTaskSample, context: Dict[str, Any]) -> None:
        """Judge category orthogonality for single sample (in-place).

        Modifies:
            - data.set_meta('rubric_orthogonality_pass', bool)
            - data.set_meta('rubric_orthogonality_judgment', judgment_dict)
            - data.set_meta('rubric_orthogonality_reason', str)
            - Optionally marks as skipped based on config:
                - skip_on_none: Skip if LLM returns None
                - skip_on_negative: Skip if orthogonality check fails (pass=false)
                - skip_on_failure: Skip if parsing/execution error occurs
        """
        # Check if categories exist (使用类型过滤)
        categories = data.get_children(RubricCategory)

        if len(categories) == 0:
            # No categories to check
            data.set_meta('rubric_orthogonality_pass', True)
            data.set_meta('rubric_orthogonality_note', 'no_categories')
            return

        if len(categories) == 1:
            # Single category is always orthogonal
            data.set_meta('rubric_orthogonality_pass', True)
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
            data.set_meta('rubric_orthogonality_pass', judgment['pass'])
            data.set_meta('rubric_orthogonality_judgment', judgment)
            data.set_meta('rubric_orthogonality_reason', judgment['reason'])

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


class CategoryClassificationCheckNode(MapNode[RubricCategory]):
    """Check if rubrics are correctly classified under their category.

    This node validates whether the rubric items under each category
    truly belong to that category, or should be moved to other categories.

    Processing:
        - Input: Single RubricCategory (with expanded RubricItems)
        - Uses LLM to judge if items match the category
        - Marks category as skipped if classification is invalid
    """

    data_type = RubricCategory

    def __init__(self, config: NodeConfig, agent: 'Agent'):
        super().__init__(config)
        self.agent = agent

    def _build_prompt(
        self,
        current_category: RubricCategory,
        all_category_names: List[str],
        task_description: str
    ) -> str:
        """Build prompt for LLM to judge category classification.

        Args:
            current_category: The category being checked
            all_category_names: Names of all categories (for context)
            task_description: Task description from parent sample

        Returns:
            Prompt string for LLM
        """
        # Get rubric items under this category
        items = current_category.get_children(RubricItem)

        # Build item list
        item_details = []
        for i, item in enumerate(items, 1):
            justification_text = "\n      ".join(item.justification)
            item_details.append(f"""  {i}. {item.rubric_name}
     - 二元判断：{item.binary_statement}
     - 推理步骤：
       {justification_text}
     - 可追溯性：{item.traceability}""")

        items_text = "\n\n".join(item_details)

        # Build all categories list
        categories_text = "\n".join([f"{i+1}. {name}" for i, name in enumerate(all_category_names)])

        prompt = f"""【任务描述】
{task_description}


【所有 Rubric 大类】
{categories_text}


【当前审查的大类】
大类名称：{current_category.category_name}

该大类下的 Rubric 条目：
{items_text}


# Role: Rubric 分类与去重审计官

你的任务是对该大类下的 Rubric 条目进行双重审计：

## 审计任务一：分类准确性

判断以上列出的 Rubric 条目是否真的属于「{current_category.category_name}」这个大类。

**审计标准**：
1. **主题一致性**：所有条目是否围绕该大类的核心主题展开？
2. **分类准确性**：是否有条目更适合归入其他大类（参考上面的所有大类列表）？

**不合格标准**：
- 存在明显分类错误的条目（应该属于其他大类）
- 条目与大类名称主题不符


## 审计任务二：条目去重检查

判断该大类下的 Rubric 条目之间是否存在重复现象。

请严格使用以下三种违规模型进行检查，只要触碰任何一条，即判定为"不合格（False）"。

### 违规类型 1：概念重叠
- **定义**：Rubric 条目在逻辑概念上没有清晰的边界，存在相互交织或涵盖的情况。
- **致命案例**：条目 A 检查"数据统计与分析"，条目 B 检查"关联强度推演"。（这两个阶段在实际执行时必然高度重合，执行者无法清晰区分一个动作到底属于 A 还是 B）。
- **合格标准**：条目之间必须像流水线一样是严格的时间先后关系或完全独立的模块。

### 违规类型 2：语义同义重述
- **定义**：两个或多个 Rubric 条目，使用了不同的专业词汇或句式，但本质上测试的是**同一个目标或同一个信息检查点**。只要 Agent 达成了条目 A，就必然自动达成了条目 B。
- **致命案例**：条目 A："是否明确了核心价值"，条目 B："是否说明了主要价值点"。（两者本质相同）

### 违规类型 3：包含与子集关系
- **定义**：生成了宏观的"概括性"条目，同时又生成了隶属于该概括的"细节"条目。
- **致命案例**：
  - 条目 A（父集）："是否给出了两项评估标准存在冲突的完整成因分析"
  - 条目 B（子集）："是否指出了权重分布差异是导致冲突的成因"
  （如果你判定了父集，子集就多余了；如果你要验证子集，父集就成了无效的废话。违背了"判定原子性"）


## 综合判定

**合格标准（必须同时满足）**：
1. 所有条目的分类准确，主题一致
2. 条目之间相互独立，不存在上述三种重复违规

**不合格标准（任一即不合格）**：
1. 存在分类错误的条目
2. 存在重复的条目（概念重叠、语义同义、包含关系）


## 输出格式

```json
{{
  "reason": "详细说明判定理由。如果不合格，必须指出：(1) 哪些条目分类错误（应该归入哪个大类）；(2) 哪些条目存在重复（属于哪种违规类型）",
  "pass": true/false
}}
```

**字段说明**：
- `reason`: 字符串，详细说明判定理由
- `pass`: 布尔值，true 表示合格（分类准确且无重复），false 表示不合格

请先进行详细分析，然后输出 JSON 格式的判定结果。
"""
        return prompt

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract classification judgment.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed judgment dict with keys:
                - pass (bool): True if classification is valid
                - reason (str): Explanation
        """
        try:
            # Extract JSON from response
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)

            if match:
                json_str = match.group(1)
            else:
                # Try to find bare JSON object
                json_pattern2 = r'\{[^}]*"pass"[^}]*\}'
                match2 = re.search(json_pattern2, response_text, re.DOTALL)
                if match2:
                    json_str = match2.group(0)
                else:
                    raise ValueError("No JSON found in response")

            data = json.loads(json_str)

            # Validate required fields
            if "pass" not in data:
                raise ValueError("Missing 'pass' field in response")
            if "reason" not in data:
                raise ValueError("Missing 'reason' field in response")

            return {
                "pass": bool(data["pass"]),
                "reason": str(data["reason"]),
            }

        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    async def map_one(self, data: RubricCategory, context: Dict[str, Any]) -> None:
        """Check category classification for single category (in-place).

        Requires context keys:
            - 'parent_sample': AgenticTaskSample (to get other categories)
            - 'task_description': str (task description)

        Modifies:
            - data.set_meta('category_classification_pass', bool)
            - data.set_meta('category_classification_judgment', dict)
            - Optionally marks as skipped if classification fails
        """
        # Get parent sample from context
        parent_sample = context.get('parent_sample')
        if not parent_sample:
            data.set_meta('category_classification_error', 'missing_parent_sample_in_context')
            if self.config.skip_on_failure:
                data.mark_skipped("missing_parent_sample", self.name)
            return

        # Get task description from context
        task_description = context.get('task_description', '')
        if not task_description:
            data.set_meta('category_classification_error', 'missing_task_description_in_context')
            if self.config.skip_on_failure:
                data.mark_skipped("missing_task_description", self.name)
            return

        # Get all category names from parent
        all_categories = parent_sample.get_children(RubricCategory)
        all_category_names = [cat.category_name for cat in all_categories]

        # Check if items exist
        items = data.get_children(RubricItem)
        if not items:
            # No items to check
            data.set_meta('category_classification_pass', True)
            data.set_meta('category_classification_note', 'no_items')
            return

        try:
            # Build prompt
            prompt = self._build_prompt(data, all_category_names, task_description)

            if not prompt:
                data.set_meta('category_classification_error', 'empty_prompt')
                if self.config.skip_on_failure:
                    data.mark_skipped("empty_prompt", self.name)
                return

            # Call LLM
            response = await self.agent.generate(prompt)

            if response is None:
                data.set_meta('category_classification_error', 'llm_call_returned_none')
                if self.config.skip_on_none:
                    data.mark_skipped("llm_call_failed", self.name)
                return

            # Store raw response and prompt
            data.set_meta('category_classification_prompt', prompt)
            data.set_meta('category_classification_raw_response', response.content)
            data.set_meta('category_classification_finish_reason', response.finish_reason)
            if response.usage:
                data.set_meta('category_classification_token_usage', response.usage)

            # Parse response
            judgment = self._parse_llm_response(response.content)

            # Store results
            data.set_meta('category_classification_pass', judgment['pass'])
            data.set_meta('category_classification_judgment', judgment)
            data.set_meta('category_classification_reason', judgment['reason'])

            # Check if classification failed
            if not judgment['pass']:
                if self.config.skip_on_negative:
                    data.mark_skipped(f"classification_failed: {judgment['reason']}", self.name)

        except Exception as e:
            error_msg = f"Failed to check category classification: {str(e)}"
            data.set_meta('category_classification_error', error_msg)

            if self.config.skip_on_failure:
                data.mark_skipped(f"category_classification_check_error: {e}", self.name)


class RubricQualityCheckNode(MapNode[RubricItem]):
    """Base class for rubric item quality validation.

    This abstract base class provides a framework for validating individual
    rubric items against specific quality criteria. Subclasses should:
    1. Set judge_dimension_name to identify the validation dimension
    2. Implement _build_prompt() to generate LLM prompts
    3. Optionally override map_one() for custom validation logic

    Processing:
        - Input: Single RubricItem
        - Uses LLM to judge rubric quality on a specific dimension
        - Stores judgment results in item metadata

    Attributes:
        agent: LLM agent for judging
        judge_dimension_name: Name of the quality dimension being judged
                              (e.g., "traceability", "atomic", "measurable")
                              Used for metadata key generation

    Metadata keys (stored in RubricItem):
        - rubric_{dimension}_pass: bool
        - rubric_{dimension}_judgment: dict
        - rubric_{dimension}_reason: str
        - rubric_{dimension}_prompt: str (optional, for debugging)
        - rubric_{dimension}_raw_response: str (optional, for debugging)
        - rubric_{dimension}_error: str (if error occurs)
    """

    data_type = RubricItem

    def __init__(self, config: NodeConfig, agent: 'Agent', judge_dimension_name: str):
        """Initialize rubric quality check node.

        Args:
            config: Node configuration
            agent: LLM agent for validation
            judge_dimension_name: Quality dimension identifier
                                 (e.g., "traceability", "atomic", "measurable")
        """
        super().__init__(config)
        self.agent = agent
        self.judge_dimension_name = judge_dimension_name

    @abstractmethod
    def _build_prompt(
        self,
        rubric_item: RubricItem,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM to judge rubric quality.

        Subclasses must implement this method to generate prompts
        specific to their validation dimension.

        Args:
            rubric_item: The rubric item being validated
            context: Execution context (may include task_description, parent_category, etc.)

        Returns:
            Prompt string for LLM

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement _build_prompt()")

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract quality judgment.

        Default implementation expects JSON format: {"pass": bool, "reason": str}

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed judgment dict with keys:
                - pass (bool): True if rubric passes quality check
                - reason (str): Explanation of the judgment

        Raises:
            ValueError: If parsing fails
        """
        try:
            # Extract JSON from response
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)

            if match:
                json_str = match.group(1)
            else:
                # Try to find bare JSON object
                json_pattern2 = r'\{[^}]*"pass"[^}]*\}'
                match2 = re.search(json_pattern2, response_text, re.DOTALL)
                if match2:
                    json_str = match2.group(0)
                else:
                    raise ValueError("No JSON found in response")

            data = json.loads(json_str)

            # Validate required fields
            if "pass" not in data:
                raise ValueError("Missing 'pass' field in response")
            if "reason" not in data:
                raise ValueError("Missing 'reason' field in response")

            return {
                "pass": bool(data["pass"]),
                "reason": str(data["reason"]),
            }

        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    async def map_one(self, data: RubricItem, context: Dict[str, Any]) -> None:
        """Validate single rubric item quality (in-place).

        Default implementation:
        1. Builds prompt using _build_prompt()
        2. Calls LLM via agent
        3. Parses response using _parse_llm_response()
        4. Stores results in metadata

        Modifies:
            - data.set_meta(f'rubric_{dimension}_pass', bool)
            - data.set_meta(f'rubric_{dimension}_judgment', dict)
            - data.set_meta(f'rubric_{dimension}_reason', str)
            - Optionally marks as skipped based on config

        Args:
            data: RubricItem to validate
            context: Execution context
        """
        dimension = self.judge_dimension_name

        try:
            # Build prompt
            prompt = self._build_prompt(data, context)

            if not prompt:
                data.set_meta(f'rubric_{dimension}_error', 'empty_prompt')
                if self.config.skip_on_failure:
                    data.mark_skipped("empty_prompt", self.name)
                return

            # Call LLM
            response = await self.agent.generate(prompt)

            if response is None:
                data.set_meta(f'rubric_{dimension}_error', 'llm_call_returned_none')
                if self.config.skip_on_none:
                    data.mark_skipped("llm_call_failed", self.name)
                return

            # Store raw response and prompt (for debugging)
            data.set_meta(f'rubric_{dimension}_prompt', prompt)
            data.set_meta(f'rubric_{dimension}_raw_response', response.content)
            data.set_meta(f'rubric_{dimension}_finish_reason', response.finish_reason)
            if response.usage:
                data.set_meta(f'rubric_{dimension}_token_usage', response.usage)

            # Parse response
            judgment = self._parse_llm_response(response.content)

            # Store results
            data.set_meta(f'rubric_{dimension}_pass', judgment['pass'])
            data.set_meta(f'rubric_{dimension}_judgment', judgment)
            data.set_meta(f'rubric_{dimension}_reason', judgment['reason'])

            # Check if validation failed
            if not judgment['pass']:
                if self.config.skip_on_negative:
                    data.mark_skipped(f"{dimension}_check_failed: {judgment['reason']}", self.name)

        except Exception as e:
            error_msg = f"Failed to check rubric {dimension}: {str(e)}"
            data.set_meta(f'rubric_{dimension}_error', error_msg)

            if self.config.skip_on_failure:
                data.mark_skipped(f"rubric_{dimension}_check_error: {e}", self.name)


class RubricRigidityCheckNode(RubricQualityCheckNode):
    """Check rubric statement syntax and rigidity (Pass 1: Statement Syntax & Rigidity).

    This node validates the binary_statement of a rubric item against three criteria:
    1. Atomicity (判定原子性): No compound logic (AND/OR operators)
    2. Self-containment (信息自闭环): No external references
    3. Binary rigidity (二元刚性): No subjective adjectives or degree adverbs

    Processing:
        - Input: Single RubricItem
        - Checks only binary_statement field (no need for context/background)
        - Uses lightweight LLM (e.g., Qwen-Max or GPT-4o-mini)
        - Stores detailed validation results in metadata

    Example:
        config = NodeConfig(name="rigidity_check", node_type=NodeType.LLM_JUDGE)
        node = RubricRigidityCheckNode(config, agent)
        await node.process_one(rubric_item, context)

        # Check results
        if rubric_item.get_meta('rubric_rigidity_pass'):
            print("✅ Rigidity check passed")
        else:
            print(f"❌ Failed: {rubric_item.get_meta('rubric_rigidity_reason')}")
    """

    def __init__(self, config: NodeConfig, agent: 'Agent'):
        """Initialize rigidity check node.

        Args:
            config: Node configuration
            agent: LLM agent (lightweight model recommended: Qwen-Max, GPT-4o-mini)
        """
        super().__init__(config, agent, judge_dimension_name="rigidity")

    def _build_prompt(
        self,
        rubric_item: RubricItem,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM to check statement rigidity.

        Args:
            rubric_item: The rubric item being validated
            context: Execution context (may include task_description for background)

        Returns:
            Prompt string for LLM
        """
        binary_statement = rubric_item.binary_statement
        task_description = context.get('task_description', '')

        # Build task description section if available
        task_section = ""
        if task_description:
            task_section = f"""# 任务背景（仅供参考，不影响判定）
{task_description}


"""

        prompt = f"""{task_section}# 任务目标
你是 Rubric 语法合规审查官，负责执行零容忍的刚性检查。

**什么是 Rubric？**
Rubric 是用于评价任务执行结果的二元判定标准（通过/不通过）。一条合格的 Rubric 必须让判卷人在不依赖任何外部信息的情况下，仅凭陈述本身就能做出明确的是/否判断。

**审查职责：**
对下方输入的 Rubric 判定标准进行语法合规性扫描。你只对陈述的语言结构负责，不需要理解任务背景。任何触犯规则的 Rubric 将被直接否决。

# 扫描规则（三条死刑红线）
你必须找出该 Rubric 是否触犯了以下任何一条"死刑"红线：
1. **违背原子性**：句子中是否只包含**一个**单一的、不可分割的逻辑检验点。绝不允许将多个判定条件组合在一起（复杂的检验必须被拆分，这里只能有一件事）。
2. **违背自闭环**：句子中是否要求判卷人去寻找外部线索？（致命词汇例如："是否和原文一致"、"是否符合专家预期"、"是否和图表相符"。必须是写死在句子里的绝对数值或实体名称）。即 Rubric本身包含了量化判定的全部信息。
3. **违背二元刚性**：句子中是否是明确可量化判定的判断性陈述，是否出现了主观形容词或程度副词？（致命词汇例如："是否合理"、"是否详细"、"是否充分"、"是否正确地"）。

# 待检查的 Rubric 标准
【Rubric 陈述】：{binary_statement}

# 输出格式 (JSON)
请直接输出 JSON，不要附加任何废话：
```json
{{
  "is_atomic": true/false,
  "is_self_contained": true/false,
  "is_rigid": true/false,
  "rejection_reason": "如果任意一项为 false，请一句话指出触发了哪个违规词汇或结构。如果全为 true，填 null。"
}}
```
"""
        return prompt

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract rigidity judgment.

        Expected JSON format from LLM:
        {
          "is_atomic": bool,
          "is_self_contained": bool,
          "is_rigid": bool,
          "rejection_reason": str or null
        }

        Converts to standard format:
        {
          "pass": bool,
          "reason": str
        }

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed judgment dict with keys:
                - pass (bool): True if all three checks pass
                - reason (str): Explanation (from rejection_reason or success message)
                - is_atomic (bool): Atomicity check result
                - is_self_contained (bool): Self-containment check result
                - is_rigid (bool): Binary rigidity check result

        Raises:
            ValueError: If parsing fails
        """
        try:
            # Extract JSON from response
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)

            if match:
                json_str = match.group(1)
            else:
                # Try to find bare JSON object
                json_pattern2 = r'\{[^}]*"is_atomic"[^}]*\}'
                match2 = re.search(json_pattern2, response_text, re.DOTALL)
                if match2:
                    json_str = match2.group(0)
                else:
                    raise ValueError("No JSON found in response")

            data = json.loads(json_str)

            # Validate required fields
            if "is_atomic" not in data:
                raise ValueError("Missing 'is_atomic' field in response")
            if "is_self_contained" not in data:
                raise ValueError("Missing 'is_self_contained' field in response")
            if "is_rigid" not in data:
                raise ValueError("Missing 'is_rigid' field in response")

            # Extract values
            is_atomic = bool(data["is_atomic"])
            is_self_contained = bool(data["is_self_contained"])
            is_rigid = bool(data["is_rigid"])
            rejection_reason = data.get("rejection_reason", None)

            # Overall pass: all three checks must pass
            pass_flag = is_atomic and is_self_contained and is_rigid

            # Generate reason
            if pass_flag:
                reason = "陈述通过刚性检查：满足原子性、自闭环、二元刚性三项要求"
            else:
                reason = rejection_reason if rejection_reason else "未通过刚性检查"

            return {
                "pass": pass_flag,
                "reason": reason,
                "is_atomic": is_atomic,
                "is_self_contained": is_self_contained,
                "is_rigid": is_rigid,
            }

        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")
