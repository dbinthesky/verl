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

Version: 2.0.0 (Refactored with Protocol-based design)
"""

from __future__ import annotations

# Version
from .__version__ import __version__

# Core
from .core import (
    PipelineData,
    PipelineDataBase,
    NodeType,
    NodeConfig,
    ExecutionMetadata,
)

# Nodes
from .nodes import (
    Node,
    MapNode,
    ExpandNode,
    AggregateNode,
    AgenticTaskSample,
    RubricCategory,
    RubricItem,
    AgenticTaskParserNode,
    RubricCategoryExpanderNode,
    RubricItemExpanderNode,
    CategoryOrthogonalityCheckNode,
)

# Topology
from .topology import (
    Edge,
    TopologyGraph,
    PipelineExecutor,
)

# Agent
from .agent import (
    LLMError,
    RateLimitError,
    PostprocessError,
    APIError,
    LLMResponse,
    AgentConfig,
    Agent,
    create_agent,
)

# Logging
from .log import (
    setup_logging,
    get_logger,
)

# Utils
from .utils import (
    create_simple_context,
)


__all__ = [
    # Version
    '__version__',

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
    'CategoryOrthogonalityCheckNode',
]
