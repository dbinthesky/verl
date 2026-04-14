"""
Agentic Task Synthesis - Data Structures

Defines the hierarchical data structure for agentic task synthesis:
    Level 1: AgenticTaskSample (LLM raw response)
    Level 2: RubricCategory (e.g., "核心价值主张锚定")
    Level 3: RubricItem (specific rubric to judge)
"""

from __future__ import annotations

from typing import Any, Dict, List
from dataclasses import dataclass, field

from ...core import PipelineDataBase


__all__ = [
    'AgenticTaskSample',
    'RubricCategory',
    'RubricItem',
]


@dataclass
class AgenticTaskSample(PipelineDataBase):
    """Level 1: LLM generated raw response (root node).

    Attributes:
        raw_response: Original LLM output (includes <think>, ```json```, etc.)
        task_description: Parsed task description
        parsed_json: Parsed JSON data
    """
    raw_response: str = ""
    task_description: str = ""
    parsed_json: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.data_id:
            self.data_id = f"sample_{self.sample_idx}"


@dataclass
class RubricCategory(PipelineDataBase):
    """Level 2: Rubric category (middle node).

    Attributes:
        category_name: Category name (e.g., "核心价值主张锚定")
        category_rubrics: Raw rubric data for this category
        category_score: Aggregated score from child rubrics
    """
    category_name: str = ""
    category_rubrics: List[Dict[str, Any]] = field(default_factory=list)
    category_score: float = 0.0

    def __post_init__(self):
        super().__post_init__()


@dataclass
class RubricItem(PipelineDataBase):
    """Level 3: Specific rubric item (leaf node).

    Attributes:
        rubric_name: Rubric name
        binary_statement: Binary judgment statement
        justification: Reasoning steps
        traceability: Traceability information
        judge_score: LLM judge score (0-1)
        judge_reason: Judge reasoning
    """
    rubric_name: str = ""
    binary_statement: str = ""
    justification: List[str] = field(default_factory=list)
    traceability: str = ""

    # Judge results
    judge_score: float = 0.0
    judge_reason: str = ""

    def __post_init__(self):
        super().__post_init__()
