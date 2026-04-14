"""
Agentic Task Synthesis - Expander Nodes

Expands hierarchical data structures:
- AgenticTaskSample → RubricCategory
- RubricCategory → RubricItem
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import ExpandNode
from .data import AgenticTaskSample, RubricCategory, RubricItem


__all__ = [
    'RubricCategoryExpanderNode',
    'RubricItemExpanderNode',
]


class RubricCategoryExpanderNode(ExpandNode[AgenticTaskSample]):
    """Expands AgenticTaskSample into RubricCategory nodes."""

    # 显式声明处理的数据类型（输入）
    data_type = AgenticTaskSample

    def expand_one(self, data: AgenticTaskSample, context: Dict[str, Any]) -> List[RubricCategory]:
        """Expand sample into category nodes."""
        categories = []

        if "verify_rubrics" not in data.parsed_json:
            return categories

        verify_rubrics = data.parsed_json["verify_rubrics"]

        for i, (category_name, rubrics) in enumerate(verify_rubrics.items()):
            category = RubricCategory(
                sample_idx=data.sample_idx,
                data_id=f"{data.data_id}/category_{i}",
                parent_id=data.data_id,
                category_name=category_name,
                category_rubrics=rubrics
            )
            categories.append(category)

        return categories


class RubricItemExpanderNode(ExpandNode[RubricCategory]):
    """Expands RubricCategory into RubricItem nodes."""

    # 显式声明处理的数据类型（输入）
    data_type = RubricCategory

    def expand_one(self, data: RubricCategory, context: Dict[str, Any]) -> List[RubricItem]:
        """Expand category into rubric item nodes."""
        items = []

        for i, rubric_data in enumerate(data.category_rubrics):
            item = RubricItem(
                sample_idx=data.sample_idx,
                data_id=f"{data.data_id}/rubric_{i}",
                parent_id=data.data_id,
                rubric_name=rubric_data.get("rubric_name", ""),
                binary_statement=rubric_data.get("binary_statement", ""),
                justification=rubric_data.get("justification", []),
                traceability=rubric_data.get("traceability", "")
            )
            items.append(item)

        return items
