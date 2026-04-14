"""
Agentic Task Synthesis Module

A complete pipeline for agentic task synthesis and validation:
- Data structures: AgenticTaskSample, RubricCategory, RubricItem
- Parser: AgenticTaskParserNode
- Expanders: RubricCategoryExpanderNode, RubricItemExpanderNode
- Validator: CategoryOrthogonalityCheckNode
- Pipeline: AgenticTaskPipeline

Usage:
    from reward_framework.nodes.agentic_task_synthesis import (
        AgenticTaskSample,
        AgenticTaskPipeline,
    )

    # Create pipeline
    pipeline = AgenticTaskPipeline(...)

    # Process single sample (verl interface)
    sample = AgenticTaskSample(raw_response="...")
    result = await pipeline.run_one(sample, context)
"""

from .data import AgenticTaskSample, RubricCategory, RubricItem
from .parser import AgenticTaskParserNode
from .expander import RubricCategoryExpanderNode, RubricItemExpanderNode
from .validator import CategoryOrthogonalityCheckNode
from .pipeline import AgenticTaskPipeline


__all__ = [
    # Data structures
    'AgenticTaskSample',
    'RubricCategory',
    'RubricItem',

    # Nodes
    'AgenticTaskParserNode',
    'RubricCategoryExpanderNode',
    'RubricItemExpanderNode',
    'CategoryOrthogonalityCheckNode',

    # Pipeline
    'AgenticTaskPipeline',
]
