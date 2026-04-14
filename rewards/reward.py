"""
Typed Pipeline Framework for RL-based Question Generation & Evaluation

This module provides a strongly-typed, declarative framework for building
multi-stage LLM pipelines with explicit topology definition.

Core Concepts:
    - PipelineData: Protocol-based data contract (framework agnostic)
    - Node: Atomic processing unit (in-place modification)
    - Topology: DAG defining node dependencies
    - Executor: Orchestrates node execution

Design Philosophy:
    - Protocol-driven (不依赖具体类型)
    - In-place modification (数据直接修改，不返回新数据)
    - Composition over inheritance
    - Type-safe interfaces with generic support
    - Single-file constraint (for verl framework compatibility)

Version: 2.0.0 (Refactored with Protocol-based design)
"""

from __future__ import annotations

import os
import sys
import math
import logging
import asyncio as aio
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Callable, TypeVar, Generic,
    Protocol, TypedDict, Union, Tuple, Set, Awaitable, Iterator,
    runtime_checkable
)
from dataclasses import dataclass, field, replace
from collections import defaultdict, deque
import numpy as np


__version__ = "2.0.0"


# ==============================================================================
# Framework Core: Data Protocol (最小契约)
# ==============================================================================

@runtime_checkable
class PipelineData(Protocol):
    """Pipeline数据必须实现的最小契约

    框架层只依赖这些接口，不关心具体实现。
    任务可以选择继承PipelineDataBase，或者自己实现这个Protocol。
    """

    # ===== 身份标识 =====
    @property
    def data_id(self) -> str:
        """全局唯一标识符（如 "sample_0" 或 "sample_0/part_1"）"""
        ...

    @property
    def sample_idx(self) -> int:
        """归属的样本索引（根节点）"""
        ...

    # ===== Skip状态 =====
    @property
    def is_skipped(self) -> bool:
        """是否被跳过"""
        ...

    def mark_skipped(self, reason: str, node_name: str) -> None:
        """标记为跳过"""
        ...

    def get_skip_info(self) -> Tuple[str, str]:
        """获取跳过信息：(reason, node_name)"""
        ...

    # ===== 层次结构（支持展开/聚合）=====
    def get_children(self) -> List['PipelineData']:
        """获取子项（如果有展开）"""
        ...

    def add_child(self, child: 'PipelineData') -> None:
        """添加子项"""
        ...

    @property
    def parent_id(self) -> Optional[str]:
        """父节点ID"""
        ...

    # ===== 元数据 =====
    def set_meta(self, key: str, value: Any) -> None:
        """设置元数据"""
        ...

    def get_meta(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        ...

    def get_all_meta(self) -> Dict[str, Any]:
        """获取所有元数据"""
        ...


# ==============================================================================
# Framework Core: Base Implementation (可选基类)
# ==============================================================================

@dataclass
class PipelineDataBase:
    """框架提供的基础实现（任务可以选择继承或自己实现PipelineData）

    实现了PipelineData的所有接口，提供开箱即用的功能。
    """

    # 必需字段
    sample_idx: int
    data_id: str = ""

    # Skip状态
    _is_skipped: bool = False
    _skip_reason: str = ""
    _skip_at_node: str = ""

    # 层次结构
    parent_id: Optional[str] = None
    _children: List['PipelineDataBase'] = field(default_factory=list)

    # 元数据
    _metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.data_id:
            self.data_id = f"sample_{self.sample_idx}"

    # ===== 实现PipelineData接口 =====

    @property
    def is_skipped(self) -> bool:
        return self._is_skipped

    def mark_skipped(self, reason: str, node_name: str) -> None:
        self._is_skipped = True
        self._skip_reason = reason
        self._skip_at_node = node_name

    def get_skip_info(self) -> Tuple[str, str]:
        return (self._skip_reason, self._skip_at_node)

    def get_children(self) -> List['PipelineDataBase']:
        return self._children

    def add_child(self, child: 'PipelineDataBase') -> None:
        self._children.append(child)

    def set_meta(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_meta(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def get_all_meta(self) -> Dict[str, Any]:
        return self._metadata.copy()

    # ===== 便捷方法 =====

    def add_score(self, score_name: str, score_value: float) -> None:
        """添加分数到元数据"""
        if 'scores' not in self._metadata:
            self._metadata['scores'] = {}
        self._metadata['scores'][score_name] = score_value

    def get_all_scores(self) -> Dict[str, float]:
        """获取所有分数"""
        return self._metadata.get('scores', {})

    def iter_all_descendants(self) -> Iterator['PipelineDataBase']:
        """递归迭代所有后代节点（DFS）"""
        for child in self._children:
            yield child
            yield from child.iter_all_descendants()


# ==============================================================================
# Node Configuration & Metadata
# ==============================================================================

class NodeType(Enum):
    """Node type enumeration."""
    PARSER = "parser"
    RULE = "rule"
    LLM_GENERATOR = "llm_generator"
    LLM_JUDGE = "llm_judge"
    AGGREGATOR = "aggregator"
    FILTER = "filter"
    TRANSFORMER = "transformer"
    EXPANDER = "expander"


@dataclass(frozen=True)
class NodeConfig:
    """节点配置（框架层）"""
    name: str
    node_type: NodeType

    # Skip策略
    respect_skip: bool = True              # 是否尊重上游skip标记
    skip_on_negative: bool = False
    skip_on_none: bool = False

    # 并行控制
    enable_internal_parallel: bool = False
    max_internal_concurrent: int = 10

    # 其他
    filter_only: bool = False
    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        if not self.name:
            raise ValueError("Node name cannot be empty")
        if self.weight < 0:
            raise ValueError(f"Node weight must be non-negative, got {self.weight}")


@dataclass
class ExecutionMetadata:
    """节点执行元数据（框架标准）"""
    node_name: str = ""
    processed_count: int = 0
    skipped_count: int = 0
    newly_skipped_ids: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Node Base Class (框架层)
# ==============================================================================

DataT = TypeVar('DataT', bound=PipelineData)


class Node(ABC, Generic[DataT]):
    """节点基类（框架层）

    核心原则：
    1. 只依赖PipelineData接口，不依赖具体类型
    2. In-place修改数据，不创建新数据
    3. 返回执行元数据（用于监控和日志）
    """

    def __init__(self, config: NodeConfig):
        self.config = config
        self.name = config.name
        self.node_type = config.node_type

    @abstractmethod
    async def process(
        self,
        batch: List[DataT],
        context: Dict[str, Any]
    ) -> ExecutionMetadata:
        """处理批量数据（子类实现）

        Args:
            batch: 数据列表（会被直接修改）
            context: 执行上下文

        Returns:
            执行元数据
        """
        raise NotImplementedError

    # ===== 框架提供的工具方法 =====

    def should_skip(self, data: PipelineData) -> bool:
        """判断是否跳过（框架标准逻辑）"""
        if data.is_skipped and self.config.respect_skip:
            return True
        return False

    def filter_valid(self, batch: List[DataT]) -> List[DataT]:
        """过滤出有效数据（未跳过）"""
        return [data for data in batch if not self.should_skip(data)]

    async def execute(
        self,
        batch: List[DataT],
        context: Dict[str, Any]
    ) -> ExecutionMetadata:
        """框架层的执行入口（封装了通用逻辑）

        此方法不应被子类override，子类只需实现process()
        """
        import time
        start = time.time()

        # 调用子类实现
        metadata = await self.process(batch, context)

        # 补充执行时间
        metadata.execution_time = time.time() - start
        metadata.node_name = self.name

        return metadata


# ==============================================================================
# Generic Node Types (框架提供)
# ==============================================================================

class MapNode(Node[DataT]):
    """映射节点（框架通用类型）

    功能：对每个数据项进行1对1处理
    用法：子类只需实现 map_one() 方法
    """

    @abstractmethod
    async def map_one(self, data: DataT, context: Dict[str, Any]) -> None:
        """处理单个数据项（子类实现，In-place修改）"""
        raise NotImplementedError

    async def process(
        self,
        batch: List[DataT],
        context: Dict[str, Any]
    ) -> ExecutionMetadata:
        """框架实现的映射逻辑（支持并行）"""
        processed = 0
        skipped = 0
        newly_skipped = []

        # 过滤有效数据
        valid_data = [d for d in batch if not self.should_skip(d)]
        skipped = len(batch) - len(valid_data)

        # 并行处理（如果配置了并发）
        if self.config.enable_internal_parallel and len(valid_data) > 1:
            tasks = [self.map_one(data, context) for data in valid_data]
            results = await aio.gather(*tasks, return_exceptions=True)

            # 处理异常
            for data, result in zip(valid_data, results):
                if isinstance(result, Exception):
                    data.mark_skipped(f"error: {result}", self.name)
                    newly_skipped.append(data.data_id)
                elif data.is_skipped:
                    # map_one 内部标记了 skip
                    newly_skipped.append(data.data_id)
        else:
            # 顺序处理
            for data in valid_data:
                try:
                    await self.map_one(data, context)
                except Exception as e:
                    data.mark_skipped(f"error: {e}", self.name)
                    newly_skipped.append(data.data_id)

                # 检查是否在 map_one 内部被标记为 skipped
                if data.is_skipped and data.data_id not in newly_skipped:
                    newly_skipped.append(data.data_id)

        processed = len(valid_data) - len(newly_skipped)

        return ExecutionMetadata(
            processed_count=processed,
            skipped_count=skipped,
            newly_skipped_ids=newly_skipped
        )


class ExpandNode(Node[DataT]):
    """展开节点（框架通用类型）

    功能：将每个数据项展开为多个子项
    用法：子类只需实现 expand_one() 方法
    """

    @abstractmethod
    def expand_one(self, data: DataT) -> List[DataT]:
        """展开单个数据项（子类实现）

        Returns:
            子数据项列表
        """
        raise NotImplementedError

    async def process(
        self,
        batch: List[DataT],
        context: Dict[str, Any]
    ) -> ExecutionMetadata:
        """框架实现的展开逻辑"""
        processed = 0
        skipped = 0

        for data in batch:
            if self.should_skip(data):
                skipped += 1
                continue

            try:
                # 调用子类实现的展开逻辑
                children = self.expand_one(data)

                # 添加子项
                for child in children:
                    data.add_child(child)

                processed += 1
            except Exception as e:
                data.mark_skipped(f"expand_error: {e}", self.name)
                skipped += 1

        return ExecutionMetadata(
            processed_count=processed,
            skipped_count=skipped
        )


class AggregateNode(Node[DataT]):
    """聚合节点（框架通用类型）

    功能：将子项结果聚合到父项
    用法：子类只需实现 aggregate_children() 方法
    """

    @abstractmethod
    def aggregate_children(
        self,
        parent: DataT,
        children: List[DataT]
    ) -> float:
        """聚合子项（子类实现）

        Args:
            parent: 父数据项
            children: 子数据项列表

        Returns:
            聚合后的分数
        """
        raise NotImplementedError

    async def process(
        self,
        batch: List[DataT],
        context: Dict[str, Any]
    ) -> ExecutionMetadata:
        """框架实现的聚合逻辑"""
        processed = 0
        skipped = 0

        for data in batch:
            if self.should_skip(data):
                skipped += 1
                continue

            children = data.get_children()

            # 过滤有效子项
            valid_children = [c for c in children if not c.is_skipped]

            if not valid_children:
                data.mark_skipped("no_valid_children", self.name)
                skipped += 1
                continue

            try:
                # 调用子类实现的聚合逻辑
                score = self.aggregate_children(data, valid_children)
                data.set_meta('aggregated_score', score)

                processed += 1
            except Exception as e:
                data.mark_skipped(f"aggregate_error: {e}", self.name)
                skipped += 1

        return ExecutionMetadata(
            processed_count=processed,
            skipped_count=skipped
        )


# ==============================================================================
# Topology Graph
# ==============================================================================

@dataclass
class Edge:
    """Edge in the topology graph."""
    from_node: str
    to_node: str
    condition: Optional[Callable[[Any], bool]] = None


class TopologyGraph:
    """Topology graph for defining node dependencies."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._adj_list: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, node: Node) -> 'TopologyGraph':
        """Add a node to the graph."""
        if node.name in self.nodes:
            raise ValueError(f"Node {node.name} already exists")
        self.nodes[node.name] = node
        return self

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Callable[[Any], bool]] = None
    ) -> 'TopologyGraph':
        """Add an edge between nodes."""
        if from_node not in self.nodes:
            raise ValueError(f"Node {from_node} not found")
        if to_node not in self.nodes:
            raise ValueError(f"Node {to_node} not found")

        # Check for cycles
        if self._would_create_cycle(from_node, to_node):
            raise ValueError(f"Adding edge {from_node} -> {to_node} would create a cycle")

        edge = Edge(from_node=from_node, to_node=to_node, condition=condition)
        self.edges.append(edge)
        self._adj_list[from_node].append(to_node)

        return self

    def _would_create_cycle(self, from_node: str, to_node: str) -> bool:
        """Check if adding an edge would create a cycle."""
        # BFS from to_node to see if we can reach from_node
        visited = set()
        queue = deque([to_node])

        while queue:
            node = queue.popleft()
            if node == from_node:
                return True

            if node in visited:
                continue
            visited.add(node)

            for neighbor in self._adj_list.get(node, []):
                queue.append(neighbor)

        return False

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order."""
        in_degree = {name: 0 for name in self.nodes}

        for edge in self.edges:
            in_degree[edge.to_node] += 1

        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor in self._adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return result

    def get_node(self, name: str) -> Node:
        """Get node by name."""
        if name not in self.nodes:
            raise KeyError(f"Node {name} not found")
        return self.nodes[name]

    def visualize(self) -> str:
        """Generate ASCII visualization of the topology."""
        lines = ["=" * 60, "Pipeline Topology", "=" * 60, ""]

        # Group by levels
        sorted_nodes = self.topological_sort()
        levels = self._compute_levels(sorted_nodes)

        for level, node_names in enumerate(levels):
            lines.append(f"Level {level}:")
            for node_name in node_names:
                node = self.nodes[node_name]
                type_str = f" [{node.config.node_type.value}]"

                config_str = []
                if node.config.skip_on_negative:
                    config_str.append("skip_on_neg")
                if node.config.filter_only:
                    config_str.append("filter_only")
                if node.config.weight != 1.0:
                    config_str.append(f"w={node.config.weight:.2f}")

                config_info = f" ({', '.join(config_str)})" if config_str else ""
                lines.append(f"  - {node_name}{type_str}{config_info}")
            lines.append("")

        lines.append("Edges:")
        for edge in self.edges:
            cond_str = " [conditional]" if edge.condition else ""
            lines.append(f"  {edge.from_node} -> {edge.to_node}{cond_str}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _compute_levels(self, sorted_nodes: List[str]) -> List[List[str]]:
        """Compute node levels for visualization."""
        levels: Dict[str, int] = {}

        for node_name in sorted_nodes:
            # Find max level of predecessors
            predecessors = [e.from_node for e in self.edges if e.to_node == node_name]

            if not predecessors:
                levels[node_name] = 0
            else:
                max_pred_level = max(levels[pred] for pred in predecessors)
                levels[node_name] = max_pred_level + 1

        # Group by level
        level_groups: Dict[int, List[str]] = defaultdict(list)
        for node_name, level in levels.items():
            level_groups[level].append(node_name)

        return [level_groups[i] for i in range(max(levels.values()) + 1)]

    def validate(self) -> None:
        """Validate topology structure."""
        if not self.nodes:
            return

        # Check for completely isolated nodes (no edges at all)
        all_connected = set()
        for edge in self.edges:
            all_connected.add(edge.from_node)
            all_connected.add(edge.to_node)

        isolated = set(self.nodes.keys()) - all_connected

        # If all nodes are isolated, allow single node (trivial pipeline)
        if len(isolated) == len(self.nodes):
            if len(isolated) > 1:
                raise ValueError(f"Multiple isolated nodes found: {isolated}")
        # Otherwise, any isolated node is an error
        elif isolated:
            raise ValueError(f"Isolated nodes found (not connected to pipeline): {isolated}")

        # Try topological sort (will raise if cycles exist)
        self.topological_sort()


# ==============================================================================
# Pipeline Executor
# ==============================================================================

class PipelineExecutor:
    """Pipeline执行器（简化版）"""

    def __init__(self, topology: TopologyGraph):
        self.topology = topology
        self.context: Dict[str, Any] = {}
        self.execution_log: List[ExecutionMetadata] = []

    async def execute(
        self,
        batch: List[PipelineData],
        context: Optional[Dict[str, Any]] = None
    ) -> List[PipelineData]:
        """执行pipeline

        Args:
            batch: 数据列表
            context: 执行上下文（可选）

        Returns:
            处理后的数据列表（同一个对象，已被修改）
        """
        self.context = context or {}
        self.execution_log = []

        print(f"[Executor] Starting pipeline with {len(batch)} samples")

        # 拓扑排序
        sorted_nodes = self.topology.topological_sort()

        print(f"[Executor] Execution plan: {len(sorted_nodes)} nodes\n")

        # 顺序执行节点
        for node_name in sorted_nodes:
            node = self.topology.get_node(node_name)

            if not node.config.enabled:
                print(f"[Executor] Skipping {node_name} (disabled)")
                continue

            print(f"[Executor] Executing {node_name}...")

            # 执行节点
            metadata = await node.execute(batch, self.context)
            self.execution_log.append(metadata)

            print(f"[Executor] {node_name} completed in {metadata.execution_time:.2f}s")
            print(f"[Executor]   Processed: {metadata.processed_count}, Skipped: {metadata.skipped_count}")

            if metadata.newly_skipped_ids:
                print(f"[Executor]   Newly skipped: {len(metadata.newly_skipped_ids)} samples")

        print(f"\n[Executor] Pipeline completed")
        return batch

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "total_nodes": len(self.execution_log),
            "total_time": sum(m.execution_time for m in self.execution_log),
            "node_summary": [
                {
                    "name": m.node_name,
                    "processed": m.processed_count,
                    "skipped": m.skipped_count,
                    "time": m.execution_time
                }
                for m in self.execution_log
            ]
        }


# ==============================================================================
# Utility Functions
# ==============================================================================

def create_simple_context(ground_truths: List[Any]) -> Dict[str, Any]:
    """创建简单的执行上下文（便捷函数）"""
    return {
        "ground_truths": ground_truths,
        "min_reward": 0.0
    }


# ==============================================================================
# PLACEHOLDER: Agent and Logging modules will be inserted here
# ==============================================================================

# Agent模块和Logging模块将保留在这里（从原文件拷贝）

# Structured Logging Module - Using structlog
# ==============================================================================

"""
Structured JSON logging with async support.

Installation:
    pip install structlog

Usage:
    from reward import setup_logging, get_logger

    # Setup once at startup
    setup_logging(log_dir="./logs", log_level="INFO")

    # Get logger
    logger = get_logger(__name__)

    # Log structured data
    logger.info("pipeline_started", batch_size=32, model="gpt-4")
    logger.error("api_error", error_code=500, retry_count=3)
"""

try:
    import structlog
    from structlog.types import FilteringBoundLogger
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    FilteringBoundLogger = Any  # Fallback type


def _extract_console_sample_rate(logger, method_name, event_dict):
    """Extract console_sample_rate and mark in event_dict.

    The console_sample_rate will be kept in event_dict with a special prefix
    so that SamplingFilter can extract it from the formatted message.
    After filtering, it will be removed before final formatting.
    """
    if 'console_sample_rate' in event_dict:
        # Store with special prefix for filter to find
        event_dict['__console_sample_rate__'] = event_dict.pop('console_sample_rate')

    return event_dict


def _remove_console_sample_rate(logger, method_name, event_dict):
    """Remove __console_sample_rate__ before final formatting.

    This runs after filtering but before rendering to JSON/console.
    """
    event_dict.pop('__console_sample_rate__', None)
    return event_dict


class SamplingFilter(logging.Filter):
    """Filter that samples log records based on per-log sample rate.

    Each log record can specify its own console_sample_rate:
        logger.info("event", key=value, console_sample_rate=0.1)

    If not specified, default_sample_rate is used (1.0 = no sampling).
    ERROR and CRITICAL are always logged regardless of rate.
    """
    def __init__(self, default_sample_rate: float = 1.0):
        super().__init__()
        if not 0.0 <= default_sample_rate <= 1.0:
            raise ValueError(f"default_sample_rate must be 0.0-1.0, got {default_sample_rate}")
        self.default_sample_rate = default_sample_rate

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if record should be logged."""
        # Always log ERROR and CRITICAL
        if record.levelno >= logging.ERROR:
            return True

        # Try to extract console_sample_rate from the formatted message
        # For structlog, the msg contains the event_dict before formatting
        sample_rate = self.default_sample_rate

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if record should be logged."""
        # Always log ERROR and CRITICAL
        if record.levelno >= logging.ERROR:
            return True

        # Extract console_sample_rate from record.msg (structlog event_dict)
        sample_rate = self.default_sample_rate

        if hasattr(record, 'msg') and isinstance(record.msg, dict):
            sample_rate = record.msg.get('__console_sample_rate__', self.default_sample_rate)

        # No sampling (keep all)
        if sample_rate >= 1.0:
            return True

        # Drop all (except ERROR/CRITICAL already handled)
        if sample_rate <= 0.0:
            return False

        # Sample based on probability
        import random
        return random.random() < sample_rate


def setup_logging(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    log_filename: Optional[str] = None,
    console_output: bool = True,
    console_sample_rate: float = 1.0,
    json_indent: Optional[int] = None
) -> Any:
    """Setup structured JSON logging with file + console output.

    Args:
        log_dir: Directory for log files
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_filename: Custom filename (default: app_YYYYMMDD_HHMMSS.jsonl)
        console_output: Enable console output (human-readable)
        console_sample_rate: Console sampling rate (0.0-1.0, 1.0=no sampling).
                            File output is always full. ERROR/CRITICAL always logged.
        json_indent: JSON indent (None=JSONL compact, 2=pretty)

    Returns:
        Root logger

    Raises:
        ImportError: If structlog is not installed

    Example:
        >>> # Full logging (default)
        >>> setup_logging(log_dir="./logs", log_level="INFO")
        >>>
        >>> # Sample 10% to console, full to file
        >>> setup_logging(log_dir="./logs", console_sample_rate=0.1)
        >>>
        >>> logger = get_logger(__name__)
        >>> logger.info("event", key="value", count=42)
    """
    if not STRUCTLOG_AVAILABLE:
        raise ImportError(
            "structlog is required for logging. Install: pip install structlog"
        )

    from pathlib import Path
    from datetime import datetime

    # Create log directory
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"app_{timestamp}.jsonl"

    log_file = log_dir_path / log_filename

    # Configure stdlib logging backend
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[]
    )

    # File handler (JSON/JSONL) - remove __console_sample_rate__ before rendering
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(indent=json_indent),
            foreign_pre_chain=[_remove_console_sample_rate]
        )
    )
    logging.root.addHandler(file_handler)

    # Console handler (human-readable with sampling)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(colors=True),
                foreign_pre_chain=[_remove_console_sample_rate]
            )
        )
        # Add sampling filter (file handler logs everything)
        # Filter is always added to allow per-log console_sample_rate
        console_handler.addFilter(SamplingFilter(default_sample_rate=console_sample_rate))
        logging.root.addHandler(console_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.CallsiteParameterAdder([
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]),
            _extract_console_sample_rate,  # Extract before formatting
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    sampling_info = f", Console Sampling: {console_sample_rate:.1%}" if console_sample_rate < 1.0 else ""
    print(f"[Logging] File: {log_file} (Full)")
    print(f"[Logging] Level: {log_level}, Console: {console_output}{sampling_info}")

    return get_logger("root")


def get_logger(name: str) -> Any:
    """Get structured logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("event", user_id=123, action="login")
    """
    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging
        return logging.getLogger(name)

    return structlog.get_logger(name)


# ==============================================================================
# Framework Version & Exports
# ==============================================================================

__all__ = [
    # Enums
    'NodeType',

    # Protocols & Data
    'PipelineData',
    'PipelineDataBase',

    # Configuration
    'NodeConfig',
    'ExecutionMetadata',

    # Core classes
    'Node',
    'MapNode',
    'ExpandNode',
    'AggregateNode',
    'Edge',
    'TopologyGraph',
    'PipelineExecutor',

    # LLM Agent
    'Agent',
    'AgentConfig',
    'LLMResponse',
    'LLMError',
    'RateLimitError',
    'APIError',
    'PostprocessError',

    # Logging
    'setup_logging',
    'get_logger',

    # Utilities
    'create_simple_context',
    'create_agent',

    # Agentic Task Synthesis
    'AgenticTaskSample',
    'RubricCategory',
    'RubricItem',
    'AgenticTaskParserNode',
    'RubricCategoryExpanderNode',
    'RubricItemExpanderNode',
]


# ==============================================================================
# LLM Agent Module - Async OpenAI Client with Strong Typing
# ==============================================================================

import os
from dataclasses import replace


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class RateLimitError(LLMError):
    """Rate limit exceeded."""
    pass


class PostprocessError(LLMError):
    """Postprocessing failed."""
    pass


class APIError(LLMError):
    """API call failed."""
    pass


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for LLM Agent.

    Attributes:
        model: Model name (e.g., 'gpt-4', 'claude-3-sonnet')
        base_url: API base URL (None for default OpenAI endpoint)
        api_key: Single API key (None to use env var OPENAI_API_KEY)
        system_message: System message for all requests
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature [0, 2]
        top_p: Nucleus sampling parameter [0, 1]
        seed: Random seed for reproducibility
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts for transient errors
        retry_min_wait: Minimum wait between retries (seconds)
        retry_max_wait: Maximum wait between retries (seconds)
    """
    model: str = "gpt-3.5-turbo"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    system_message: str = "You are a helpful and harmless assistant."
    max_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    seed: int = 100745534
    timeout: float = 60.0
    max_retries: int = 4
    retry_min_wait: float = 5.0
    retry_max_wait: float = 20.0

    def __post_init__(self):
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


@dataclass
class LLMResponse:
    """Response from LLM with metadata.

    Attributes:
        content: Generated text content
        prompt: Original prompt
        model: Model used
        finish_reason: Why generation stopped
        usage: Token usage stats
        latency: Response time in seconds
    """
    content: str
    prompt: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    latency: float = 0.0


class Agent:
    """Async LLM Agent with strong typing and robust error handling.

    Features:
    - Type-safe configuration
    - Automatic retries with exponential backoff
    - Batch processing with concurrency control
    - Prompt deduplication
    - Client connection pooling
    - Comprehensive error handling

    Example:
        config = AgentConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=2048
        )
        agent = Agent(config)

        prompts = ["Question 1", "Question 2"]
        responses = await agent.batch_generate(
            prompts=prompts,
            max_concurrent=10,
            postprocess_fn=lambda x: x.strip()
        )
    """

    def __init__(self, config: AgentConfig):
        """Initialize agent with configuration.

        Args:
            config: Agent configuration
        """
        self.config = config
        self._client: Optional[Any] = None
        self._logger = get_logger(f"agent.{config.model}")

        # Get API key
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self._logger.error("agent_init_failed",
                reason="missing_api_key",
                model=config.model
            )
            raise ValueError("API key not provided and OPENAI_API_KEY env var not set")

    async def _get_client(self):
        """Get or create AsyncOpenAI client (lazy initialization)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            except ImportError:
                self._logger.error("agent_init_failed",
                    reason="openai_not_installed",
                    model=self.config.model
                )
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    async def generate(self,
                      prompt: str,
                      postprocess_fn: Optional[Callable[[str], str]] = None,
                      **override_kwargs) -> Optional[LLMResponse]:
        """Generate single response.

        Args:
            prompt: Input prompt
            postprocess_fn: Optional function to clean up response
            **override_kwargs: Override config parameters for this call

        Returns:
            LLMResponse or None if generation failed
        """
        import time
        from tenacity import (
            retry, stop_after_attempt, wait_exponential,
            retry_if_exception_type
        )

        @retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait
            ),
            retry=retry_if_exception_type((RateLimitError, APIError))
        )
        async def _call_api():
            client = await self._get_client()

            # Build request kwargs
            request_kwargs = {
                'model': self.config.model,
                'messages': [
                    {'role': 'system', 'content': self.config.system_message},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': self.config.max_tokens,
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'seed': self.config.seed
            }

            # Apply overrides
            request_kwargs.update(override_kwargs)

            start = time.time()
            try:
                response = await client.chat.completions.create(**request_kwargs)
                latency = time.time() - start

                content = response.choices[0].message.content

                # Postprocess if needed
                if postprocess_fn:
                    try:
                        content = postprocess_fn(content)
                    except Exception as e:
                        self._logger.error("postprocess_failed",
                            error_message=str(e),
                            response_length=len(content),
                            prompt_length=len(prompt)
                        )
                        raise PostprocessError(f"Postprocess failed: {e}")

                return LLMResponse(
                    content=content,
                    prompt=prompt,
                    model=response.model,
                    finish_reason=response.choices[0].finish_reason,
                    usage={
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    } if response.usage else None,
                    latency=latency
                )

            except Exception as e:
                error_msg = str(e).lower()

                # Detect rate limit errors
                if 'rate' in error_msg or '429' in error_msg:
                    self._logger.error("api_call_failed",
                        error_type="rate_limit",
                        error_message=str(e),
                        model=self.config.model,
                        prompt_length=len(prompt)
                    )
                    raise RateLimitError(f"Rate limit exceeded: {e}")

                # Other API errors are retryable
                self._logger.error("api_call_failed",
                    error_type="api_error",
                    error_message=str(e),
                    model=self.config.model,
                    prompt_length=len(prompt)
                )
                raise APIError(f"API call failed: {e}")

        try:
            return await _call_api()
        except (RateLimitError, APIError) as e:
            self._logger.error("generation_failed_after_retries",
                error_type=type(e).__name__,
                error_message=str(e),
                max_retries=self.config.max_retries,
                model=self.config.model
            )
            return None
        except PostprocessError as e:
            self._logger.error("generation_failed_postprocess",
                error_message=str(e),
                model=self.config.model
            )
            return None

    async def batch_generate(self,
                            prompts: List[str],
                            max_concurrent: int = 10,
                            postprocess_fn: Optional[Callable[[str], str]] = None,
                            deduplicate: bool = True,
                            desc: Optional[str] = None,
                            show_progress: bool = True
                            ) -> List[Tuple[str, Optional[LLMResponse]]]:
        """Generate responses for batch of prompts with concurrency control.

        Args:
            prompts: List of input prompts
            max_concurrent: Maximum concurrent requests
            postprocess_fn: Optional postprocessing function
            deduplicate: If True, deduplicate prompts before calling API
            desc: Description for progress bar
            show_progress: Whether to show progress

        Returns:
            List of (prompt, response) tuples in original order
        """
        if not prompts:
            return []

        # Deduplicate prompts
        if deduplicate:
            unique_prompts = list(dict.fromkeys(prompts))  # Preserve order
            prompt_to_indices = defaultdict(list)
            for idx, prompt in enumerate(prompts):
                prompt_to_indices[prompt].append(idx)
        else:
            unique_prompts = prompts
            prompt_to_indices = {p: [i] for i, p in enumerate(prompts)}

        # Create semaphore for concurrency control
        semaphore = aio.Semaphore(max_concurrent)

        async def _generate_with_semaphore(prompt):
            async with semaphore:
                return prompt, await self.generate(prompt, postprocess_fn)

        # Execute all requests
        tasks = [_generate_with_semaphore(p) for p in unique_prompts]

        if show_progress and desc:
            print(f"[Agent] {desc} - {len(unique_prompts)} unique prompts "
                  f"(from {len(prompts)} total, concurrency={max_concurrent})")

        results = await aio.gather(*tasks)

        if show_progress and desc:
            success_count = sum(1 for _, resp in results if resp is not None)
            failed_count = len(unique_prompts) - success_count
            failure_rate = failed_count / len(unique_prompts) if unique_prompts else 0

            print(f"[Agent] {desc} - Completed: {success_count}/{len(unique_prompts)} successful")

            # Log high failure rate
            if failure_rate > 0.2:  # More than 20% failures
                self._logger.warning("batch_high_failure_rate",
                    desc=desc,
                    total=len(unique_prompts),
                    successful=success_count,
                    failed=failed_count,
                    failure_rate=failure_rate,
                    model=self.config.model
                )

        # Map back to original order if deduplicated
        if deduplicate:
            result_map = {prompt: resp for prompt, resp in results}
            outputs = []
            for prompt in prompts:
                outputs.append((prompt, result_map.get(prompt)))
            return outputs
        else:
            return results

    async def batch_generate_simple(self,
                                    prompts: List[str],
                                    max_concurrent: int = 10,
                                    postprocess_fn: Optional[Callable[[str], str]] = None,
                                    desc: Optional[str] = None
                                    ) -> List[Optional[str]]:
        """Simplified batch generation returning only content strings.

        Compatible with original Agent.run() interface.

        Args:
            prompts: List of input prompts
            max_concurrent: Maximum concurrent requests
            postprocess_fn: Optional postprocessing function
            desc: Description for logging

        Returns:
            List of generated strings (None for failed generations)
        """
        results = await self.batch_generate(
            prompts=prompts,
            max_concurrent=max_concurrent,
            postprocess_fn=postprocess_fn,
            desc=desc
        )

        return [resp.content if resp else None for _, resp in results]

    async def close(self):
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()


# Backward compatibility: create Agent from old-style kwargs
def create_agent(
    system: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_kwargs: Optional[Dict[str, Any]] = None
) -> Agent:
    """Create Agent from old-style parameters (backward compatibility).

    Args:
        system: System message
        model: Model name
        base_url: API base URL
        api_key: API key
        request_kwargs: Additional request parameters

    Returns:
        Agent instance
    """
    config = AgentConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        system_message=system or "You are a helpful and harmless assistant."
    )

    # Apply request_kwargs if provided
    if request_kwargs:
        config = replace(
            config,
            max_tokens=request_kwargs.get('max_tokens', config.max_tokens),
            temperature=request_kwargs.get('temperature', config.temperature),
            top_p=request_kwargs.get('top_p', config.top_p),
            seed=request_kwargs.get('seed', config.seed)
        )

    return Agent(config)


# ==============================================================================
# Agentic Task Synthesis - Data Structures & Nodes
# ==============================================================================

"""
Agentic Task Synthesis Implementation.

Evaluates generated agentic tasks with multi-level rubric evaluation:
    Level 1: AgenticTaskSample (LLM raw response)
    Level 2: RubricCategory (e.g., "核心价值主张锚定")
    Level 3: RubricItem (specific rubric to judge)
"""


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


class AgenticTaskParserNode(MapNode[AgenticTaskSample]):
    """Parser node: parses raw LLM response into AgenticTaskSample.

    Processing:
        1. Remove EOS markers
        2. Extract JSON from <think> and ```json``` blocks
        3. Validate schema
        4. Populate AgenticTaskSample fields
    """

    def _postprocess_solution(self, solution_str: str) -> str:
        """Remove common LLM end-of-sequence markers."""
        markers = [
            "<|im_end|>",
            "<｜end▁of▁sentence｜>",
            "<|endoftext|>"
        ]

        for marker in markers:
            if marker in solution_str:
                return solution_str[:solution_str.index(marker)].strip()

        return solution_str

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON content from LLM response.

        Steps:
            1. Remove EOS markers
            2. Cut from </think> onwards
            3. Extract between ```json and ```

        Returns:
            JSON string

        Raises:
            ValueError: If JSON cannot be extracted
        """
        import re

        # Step 1: Remove EOS markers
        cleaned = self._postprocess_solution(response)

        # Step 2: Cut from </think> onwards
        if "</think>" in cleaned:
            cleaned = cleaned[cleaned.index("</think>") + len("</think>"):].strip()

        # Step 3: Extract from ```json ... ```
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, cleaned, re.DOTALL)

        if not match:
            raise ValueError("No JSON block found (expected ```json...```)")

        return match.group(1).strip()

    def _validate_schema(self, data: Dict[str, Any]) -> bool:
        """Validate parsed JSON schema.

        Expected schema:
        {
            "task_description": str,
            "verify_rubrics": {
                "category": [
                    {
                        "rubric_name": str,
                        "binary_statement": str,
                        "justification": List[str],
                        "traceability": str
                    }
                ]
            }
        }
        """
        if not isinstance(data, dict):
            return False

        if "task_description" not in data or "verify_rubrics" not in data:
            return False

        if not isinstance(data["task_description"], str):
            return False

        if not isinstance(data["verify_rubrics"], dict):
            return False

        # Validate rubrics structure
        for category, rubrics in data["verify_rubrics"].items():
            if not isinstance(rubrics, list):
                return False

            for rubric in rubrics:
                if not isinstance(rubric, dict):
                    return False

                required_fields = ["rubric_name", "binary_statement", "justification", "traceability"]
                if not all(field in rubric for field in required_fields):
                    return False

                if not isinstance(rubric["rubric_name"], str):
                    return False
                if not isinstance(rubric["binary_statement"], str):
                    return False
                if not isinstance(rubric["justification"], list):
                    return False
                if not isinstance(rubric["traceability"], str):
                    return False

        return True

    async def map_one(self, data: AgenticTaskSample, context: Dict[str, Any]) -> None:
        """Parse single raw response (in-place modification).

        Modifies data.task_description and data.parsed_json.
        Marks as skipped if parsing fails.
        """
        try:
            # Extract JSON
            json_str = self._extract_json_from_response(data.raw_response)

            # Parse JSON
            import json
            parsed = json.loads(json_str)

            # Validate schema
            if not self._validate_schema(parsed):
                raise ValueError("Schema validation failed")

            # Populate fields (in-place)
            data.task_description = parsed["task_description"]
            data.parsed_json = parsed

        except Exception as e:
            # Mark as skipped on parse failure
            data.mark_skipped(f"parse_error: {e}", self.name)


class RubricCategoryExpanderNode(ExpandNode[AgenticTaskSample]):
    """Expands AgenticTaskSample into RubricCategory nodes."""

    def expand_one(self, data: AgenticTaskSample) -> List[RubricCategory]:
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

    def expand_one(self, data: RubricCategory) -> List[RubricItem]:
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


if __name__ == "__main__":
    # Framework self-test
    print(f"Typed Pipeline Framework v{__version__}")
    print(f"Available node types: {[t.value for t in NodeType]}")
    print("Framework loaded successfully!")
