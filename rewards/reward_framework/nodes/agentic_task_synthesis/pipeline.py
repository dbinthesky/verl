"""
Agentic Task Synthesis - Pipeline

Complete pipeline for agentic task synthesis with explicit orchestration.
"""

from __future__ import annotations

from typing import Dict, Any
import asyncio

from ...core import NodeConfig
from ...pipeline import Pipeline
from .data import AgenticTaskSample, RubricCategory
from .parser import AgenticTaskParserNode
from .expander import RubricCategoryExpanderNode, RubricItemExpanderNode
from .validator import CategoryOrthogonalityCheckNode


__all__ = ['AgenticTaskPipeline']


class AgenticTaskPipeline(Pipeline):
    """Complete pipeline for agentic task synthesis

    Pipeline stages:
        1. Parse raw LLM response into structured data
        2. Expand into rubric categories
        3. Parallel:
           - Expand categories into rubric items
           - Check category orthogonality (LLM judge)

    Example:
        # Create pipeline
        pipeline = AgenticTaskPipeline(
            parser_config=NodeConfig(...),
            category_expander_config=NodeConfig(...),
            item_expander_config=NodeConfig(...),
            validator_config=NodeConfig(...),
            validator_agent=agent
        )

        # Process single sample (verl interface)
        sample = AgenticTaskSample(raw_response="...")
        result = await pipeline.run_one(sample, context)

        # Process batch (offline evaluation)
        samples = [AgenticTaskSample(...) for _ in range(10)]
        results = await pipeline.run_batch(samples, context)
    """

    def __init__(
        self,
        parser_config: NodeConfig,
        category_expander_config: NodeConfig,
        item_expander_config: NodeConfig,
        validator_config: NodeConfig,
        validator_agent: 'Agent'
    ):
        """Initialize pipeline with all nodes

        Args:
            parser_config: Config for parser node
            category_expander_config: Config for category expander
            item_expander_config: Config for item expander
            validator_config: Config for orthogonality validator
            validator_agent: LLM agent for orthogonality check
        """
        self.parser = AgenticTaskParserNode(parser_config)
        self.category_expander = RubricCategoryExpanderNode(category_expander_config)
        self.item_expander = RubricItemExpanderNode(item_expander_config)
        self.orthogonality_checker = CategoryOrthogonalityCheckNode(
            validator_config,
            validator_agent
        )

    async def run_one(
        self,
        data: AgenticTaskSample,
        context: Dict[str, Any]
    ) -> AgenticTaskSample:
        """Process a single agentic task sample

        Args:
            data: Input sample with raw_response
            context: Execution context (must include 'llm_limiter')

        Returns:
            Processed sample with:
                - Parsed task_description and parsed_json
                - Expanded categories and items
                - Orthogonality judgment in metadata
        """
        # Step 1: Parse raw response
        await self.parser.process_one(data, context)

        # Early return if parsing failed
        if data.is_skipped:
            return data

        # Step 2: Expand into categories
        await self.category_expander.process_one(data, context)

        # Early return if expansion failed
        if data.is_skipped:
            return data

        # Step 3: Parallel processing
        categories = data.get_children(RubricCategory)

        if categories:
            await asyncio.gather(
                # Expand categories into items (concurrent)
                *[self.item_expander.process_one(cat, context) for cat in categories],
                # Check category orthogonality
                self.orthogonality_checker.process_one(data, context),
            )

        return data
