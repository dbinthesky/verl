"""
Pipeline Base Class

Provides the Pipeline abstraction for manual orchestration of node execution.
Replaces the legacy TopologyGraph/Executor pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import asyncio

from ..core import PipelineData


__all__ = ['Pipeline']


class Pipeline(ABC):
    """Pipeline base class for manual orchestration

    Users inherit from this class and implement run_one() to explicitly
    control node execution order and parallelism.

    Example:
        class MyPipeline(Pipeline):
            async def run_one(self, data, context):
                # Step 1: Parse
                await self.parser.process_one(data, context)

                # Step 2: Expand
                await self.expander.process_one(data, context)

                # Step 3: Parallel processing
                children = data.get_children()
                await asyncio.gather(
                    self.node_a.process_batch(children, context),
                    self.node_b.process_one(data, context),
                )

                return data
    """

    @abstractmethod
    async def run_one(
        self,
        data: PipelineData,
        context: Dict[str, Any]
    ) -> PipelineData:
        """Execute the pipeline for a single data item

        Args:
            data: Input data item
            context: Execution context (must include 'llm_limiter' if using LLM nodes)

        Returns:
            Processed data item
        """
        raise NotImplementedError

    async def run_batch(
        self,
        batch: List[PipelineData],
        context: Dict[str, Any]
    ) -> List[PipelineData]:
        """Execute the pipeline for a batch of data items

        Default implementation: concurrent run_one() for all items.
        Override if you need batch-specific logic.

        Args:
            batch: List of input data items
            context: Execution context

        Returns:
            List of processed data items
        """
        return await asyncio.gather(*[self.run_one(data, context) for data in batch])
