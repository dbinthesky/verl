"""
Node Configuration and Metadata - Framework Core

This module defines node configuration and execution metadata structures.
"""

from __future__ import annotations

from typing import Any, Dict, List
from dataclasses import dataclass, field
from enum import Enum


__all__ = ['NodeType', 'NodeConfig', 'ExecutionMetadata']


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
    skip_on_failure: bool = True           # 处理失败时是否skip（默认True）

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
