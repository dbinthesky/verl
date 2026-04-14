"""
Node Base Classes - Single-Item Processing

This module provides the base classes for all pipeline nodes.
Core principle: Nodes process single items by default, with optional batch optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generic, TypeVar
import asyncio

from ..core import PipelineData, NodeConfig


__all__ = ['DataT', 'Node', 'MapNode', 'ExpandNode', 'AggregateNode']


DataT = TypeVar('DataT', bound=PipelineData)


class Node(ABC, Generic[DataT]):
    """Node base class (single-item processing)

    Core principles:
    1. process_one() handles single item (required)
    2. process_batch() handles batch (default: concurrent process_one)
    3. Nodes modify data in-place
    4. data_type declares the type of data this node processes
    """

    # Explicitly declare the data type this node processes
    data_type: Optional[type] = None

    def __init__(self, config: NodeConfig):
        self.config = config
        self.name = config.name
        self.node_type = config.node_type

    @abstractmethod
    async def process_one(
        self,
        data: DataT,
        context: Dict[str, Any]
    ) -> DataT:
        """Process a single data item (subclass must implement)

        Args:
            data: Single data item (will be modified in-place)
            context: Execution context

        Returns:
            The processed data item (same object, modified in-place)
        """
        raise NotImplementedError

    async def process_batch(
        self,
        batch: List[DataT],
        context: Dict[str, Any]
    ) -> List[DataT]:
        """Process a batch of data items (default: concurrent process_one)

        Subclasses can override this for batch-specific optimizations
        (e.g., LLM nodes can use batch API).

        Args:
            batch: List of data items
            context: Execution context

        Returns:
            List of processed data items
        """
        return await asyncio.gather(*[self.process_one(data, context) for data in batch])

    # ===== Framework utility methods =====

    def should_skip(self, data: PipelineData) -> bool:
        """Check if data should be skipped"""
        if data.is_skipped and self.config.respect_skip:
            return True
        return False


class MapNode(Node[DataT]):
    """Map node: 1-to-1 processing

    Subclass only needs to implement map_one() method.
    """

    @abstractmethod
    async def map_one(self, data: DataT, context: Dict[str, Any]) -> None:
        """Process a single data item (in-place modification)

        Args:
            data: Data item to process
            context: Execution context
        """
        raise NotImplementedError

    async def process_one(
        self,
        data: DataT,
        context: Dict[str, Any]
    ) -> DataT:
        """Process a single item with skip and error handling"""
        # Skip logic
        if self.should_skip(data):
            return data

        # Process
        try:
            await self.map_one(data, context)
        except Exception as e:
            if self.config.skip_on_failure:
                data.mark_skipped(f"map_error: {e}", self.name)
            else:
                raise

        return data


class ExpandNode(Node[DataT]):
    """Expand node: 1-to-many processing

    Subclass only needs to implement expand_one() method.
    """

    @abstractmethod
    def expand_one(self, data: DataT, context: Dict[str, Any]) -> List[DataT]:
        """Expand a single data item into multiple children

        Args:
            data: Parent data item
            context: Execution context

        Returns:
            List of child data items
        """
        raise NotImplementedError

    async def process_one(
        self,
        data: DataT,
        context: Dict[str, Any]
    ) -> DataT:
        """Process a single item: expand and add children"""
        # Skip logic
        if self.should_skip(data):
            return data

        # Expand
        try:
            children = self.expand_one(data, context)

            # Add children to parent
            for child in children:
                data.add_child(child)

        except Exception as e:
            if self.config.skip_on_failure:
                data.mark_skipped(f"expand_error: {e}", self.name)
            else:
                raise

        return data


class AggregateNode(Node[DataT]):
    """Aggregate node: aggregate children results to parent

    Subclass only needs to implement aggregate_children() method.
    """

    @abstractmethod
    def aggregate_children(
        self,
        parent: DataT,
        children: List[DataT]
    ) -> float:
        """Aggregate children results

        Args:
            parent: Parent data item
            children: List of child data items

        Returns:
            Aggregated score
        """
        raise NotImplementedError

    async def process_one(
        self,
        data: DataT,
        context: Dict[str, Any]
    ) -> DataT:
        """Process a single item: aggregate children"""
        # Skip logic
        if self.should_skip(data):
            return data

        # Get children
        children = data.get_children()

        # Filter valid children
        valid_children = [c for c in children if not c.is_skipped]

        if not valid_children:
            if self.config.skip_on_none:
                data.mark_skipped("no_valid_children", self.name)
            return data

        # Aggregate
        try:
            score = self.aggregate_children(data, valid_children)
            data.set_meta('aggregated_score', score)
        except Exception as e:
            if self.config.skip_on_failure:
                data.mark_skipped(f"aggregate_error: {e}", self.name)
            else:
                raise

        return data
