"""
Typed Pipeline Framework for RL-based Question Generation & Evaluation

This module provides a strongly-typed, declarative framework for building
multi-stage LLM pipelines with explicit topology definition.

Core Concepts:
    - Node: Atomic processing unit with typed inputs/outputs
    - Topology: DAG defining node dependencies
    - Executor: Orchestrates node execution with context management

Design Philosophy:
    - Composition over inheritance
    - Explicit topology as first-class citizen
    - Type-safe interfaces with generic support
    - Single-file constraint (for verl framework compatibility)
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
    Protocol, TypedDict, Union, Tuple, Set, Awaitable
)
from dataclasses import dataclass, field
from collections import defaultdict, deque, OrderedDict
import numpy as np


# ==============================================================================
# Type Variables & Generic Types
# ==============================================================================

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')
ParsedT = TypeVar('ParsedT')
ScoreT = Union[float, List[float]]


# ==============================================================================
# Core Type Definitions
# ==============================================================================

class NodeType(Enum):
    """Node type enumeration for categorization and visualization."""
    PARSER = "parser"
    RULE = "rule"
    LLM_GENERATOR = "llm_generator"
    LLM_JUDGE = "llm_judge"
    AGGREGATOR = "aggregator"
    FILTER = "filter"
    TRANSFORMER = "transformer"


class ExecutionContext(TypedDict, total=False):
    """Execution context passed to all nodes.

    Required fields:
        ground_truths: Ground truth data for each sample

    Optional fields:
        agents: Dictionary of LLM agents by name
        max_concurrent: Concurrency limits by agent name
        parse_fn: Solution parsing function
        parsed_questions: Parsed question data
        extra_context: Additional task-specific context
        min_reward: Minimum reward value for failed samples
        skip_indices: Set of sample indices to skip
    """
    ground_truths: List[Dict[str, Any]]
    agents: Dict[str, Any]
    max_concurrent: Dict[str, int]
    parse_fn: Callable[[str], Optional[Any]]
    parsed_questions: List[Any]
    extra_context: Optional[Any]
    min_reward: float
    skip_indices: Set[int]


@dataclass(frozen=True)
class NodeConfig:
    """Immutable configuration for a processing node.

    Attributes:
        name: Unique identifier for the node
        node_type: Category of the node
        skip_on_negative: If True, mark samples with negative scores for skipping
        filter_only: If True, node score is not included in final aggregation
        weight: Multiplicative weight for node's contribution to final score
        enabled: If False, node is skipped during execution
    """
    name: str
    node_type: NodeType
    skip_on_negative: bool = False
    filter_only: bool = False
    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        if not self.name:
            raise ValueError("Node name cannot be empty")
        if self.weight < 0:
            raise ValueError(f"Node weight must be non-negative, got {self.weight}")


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for a single solver in multi-solver evaluation.

    Attributes:
        name: Solver identifier (e.g., 'weak', 'adv')
        agent_key: Key to lookup agent in ExecutionContext.agents
        repeat: Number of times to repeat each question
        prompt_fn_key: Key to lookup prompt construction function in context
        max_concurrent: Maximum concurrent requests for this solver
    """
    name: str
    agent_key: str
    repeat: int
    prompt_fn_key: str
    max_concurrent: int

    def __post_init__(self):
        if self.repeat <= 0:
            raise ValueError(f"Repeat must be positive, got {self.repeat}")
        if self.max_concurrent <= 0:
            raise ValueError(f"Max concurrent must be positive, got {self.max_concurrent}")


@dataclass(frozen=True)
class DifficultyMetricConfig:
    """Configuration for difficulty score computation.

    Defines thresholds and weights for evaluating question difficulty
    based on weak and advanced solver performance.

    Attributes:
        weak_name: Name of weak solver
        adv_name: Name of advanced solver
        weak_weight: Weight for weak solver difficulty contribution
        adv_weight: Weight for advanced solver difficulty contribution
        weak_overcomplex_threshold: Below this, question is too hard for weak
        adv_overcomplex_threshold: Below this, question is too hard for advanced
        weak_oversimple_threshold: Above this, question is too easy for weak
        adv_oversimple_threshold: Above this, question is too easy for advanced
        advantage_gap_threshold: Minimum required gap (adv - weak) pass rate
        confidence_bonus_threshold: Advanced pass rate for bonus eligibility
        confidence_bonus_weight: Weight for confidence bonus
    """
    weak_name: str
    adv_name: str
    weak_weight: float = 0.4
    adv_weight: float = 0.6
    weak_overcomplex_threshold: float = 0.1
    adv_overcomplex_threshold: float = 0.3
    weak_oversimple_threshold: float = 0.9
    adv_oversimple_threshold: float = 0.95
    advantage_gap_threshold: float = 0.2
    confidence_bonus_threshold: float = 0.8
    confidence_bonus_weight: float = 0.1


@dataclass
class NodeResult(Generic[OutputT]):
    """Result of node execution with metadata.

    Attributes:
        outputs: List of outputs for each sample (None for failed samples)
        node_name: Name of the node that produced this result
        execution_time: Time taken to execute in seconds
        metadata: Additional node-specific metadata
    """
    outputs: List[Optional[OutputT]]
    node_name: str
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Protocol Definitions (for duck typing interfaces)
# ==============================================================================

class ParseFunction(Protocol):
    """Protocol for solution parsing functions."""

    def __call__(self, solution_str: str) -> Optional[ParsedT]:
        """Parse solution string into structured format.

        Args:
            solution_str: Raw solution string from model

        Returns:
            Parsed result or None if parsing failed
        """
        ...


class PostprocessFunction(Protocol):
    """Protocol for response postprocessing functions."""

    def __call__(self, response: str) -> Optional[str]:
        """Postprocess model response.

        Args:
            response: Raw model response

        Returns:
            Cleaned response or None if postprocessing failed
        """
        ...


class PromptFunction(Protocol):
    """Protocol for prompt construction functions."""

    def __call__(self, parsed: Any, ground_truth: Dict[str, Any],
                 extra_context: Optional[Any] = None) -> str:
        """Construct prompt for model.

        Args:
            parsed: Parsed question/solution data
            ground_truth: Ground truth data
            extra_context: Optional additional context

        Returns:
            Formatted prompt string
        """
        ...


class PenaltyOrRewardModule(Protocol):
    """Protocol for rule-based penalty/reward modules."""

    def get_penalty_or_reward(self, solution_str: str,
                             ground_truth: Dict[str, Any]) -> float:
        """Compute penalty or reward score.

        Args:
            solution_str: Solution string to evaluate
            ground_truth: Ground truth data

        Returns:
            Score (negative for penalty, positive for reward)
        """
        ...


# ==============================================================================
# Abstract Node Base Class
# ==============================================================================

class Node(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all processing nodes.

    Nodes are the atomic units of computation in the pipeline. Each node:
    - Has typed inputs and outputs
    - Executes asynchronously
    - Has access to shared execution context
    - Can mark samples for skipping

    Type Parameters:
        InputT: Type of input for each sample
        OutputT: Type of output for each sample
    """

    def __init__(self, config: NodeConfig):
        """Initialize node with configuration.

        Args:
            config: Immutable node configuration
        """
        self.config = config
        self.name = config.name
        self.node_type = config.node_type

    @abstractmethod
    async def execute(self,
                     batch_inputs: List[Optional[InputT]],
                     context: ExecutionContext) -> NodeResult[OutputT]:
        """Execute node logic on batch of inputs.

        Args:
            batch_inputs: List of inputs (None for skipped samples)
            context: Shared execution context

        Returns:
            NodeResult containing outputs and metadata
        """
        raise NotImplementedError

    def _filter_valid_inputs(self,
                            batch_inputs: List[Optional[InputT]]
                            ) -> Tuple[List[int], List[InputT]]:
        """Filter out None inputs and return valid indices and values.

        Args:
            batch_inputs: Batch with potential None values

        Returns:
            Tuple of (valid_indices, valid_inputs)
        """
        valid_indices = [i for i, inp in enumerate(batch_inputs) if inp is not None]
        valid_inputs = [batch_inputs[i] for i in valid_indices]
        return valid_indices, valid_inputs

    def _reconstruct_batch(self,
                          valid_indices: List[int],
                          valid_outputs: List[OutputT],
                          total_size: int) -> List[Optional[OutputT]]:
        """Reconstruct full batch with None for skipped samples.

        Args:
            valid_indices: Indices of valid samples
            valid_outputs: Outputs for valid samples
            total_size: Total batch size

        Returns:
            Full batch with None for skipped indices
        """
        outputs: List[Optional[OutputT]] = [None] * total_size
        for idx, output in zip(valid_indices, valid_outputs):
            outputs[idx] = output
        return outputs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.node_type.value})"


# ==============================================================================
# Topology Graph
# ==============================================================================

@dataclass(frozen=True)
class Edge:
    """Directed edge connecting two nodes.

    Attributes:
        from_node: Source node name
        to_node: Target node name
        condition: Optional condition function for conditional execution
    """
    from_node: str
    to_node: str
    condition: Optional[Callable[[ExecutionContext], bool]] = None

    def should_execute(self, context: ExecutionContext) -> bool:
        """Check if edge should be traversed given context.

        Args:
            context: Current execution context

        Returns:
            True if edge should be traversed
        """
        if self.condition is None:
            return True
        return self.condition(context)


class TopologyGraph:
    """Directed Acyclic Graph (DAG) defining pipeline topology.

    The topology graph:
    - Stores nodes and their dependencies
    - Validates DAG structure (no cycles)
    - Computes execution order via topological sort
    - Supports visualization for debugging

    Usage:
        topology = (TopologyGraph()
            .add_node(parser_node)
            .add_node(rule_node)
            .add_edge("parser", "rule"))
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, node: Node) -> TopologyGraph:
        """Add a node to the graph.

        Args:
            node: Node instance to add

        Returns:
            Self for method chaining

        Raises:
            ValueError: If node with same name already exists
        """
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' already exists in topology")

        self.nodes[node.name] = node
        return self

    def add_edge(self,
                 from_node: str,
                 to_node: str,
                 condition: Optional[Callable[[ExecutionContext], bool]] = None
                 ) -> TopologyGraph:
        """Add a directed edge between two nodes.

        Args:
            from_node: Name of source node
            to_node: Name of target node
            condition: Optional condition for conditional execution

        Returns:
            Self for method chaining

        Raises:
            ValueError: If either node doesn't exist or edge creates cycle
        """
        if from_node not in self.nodes:
            raise ValueError(f"Source node '{from_node}' not found in topology")
        if to_node not in self.nodes:
            raise ValueError(f"Target node '{to_node}' not found in topology")

        edge = Edge(from_node, to_node, condition)
        self.edges.append(edge)
        self.adjacency[from_node].append(to_node)
        self.reverse_adjacency[to_node].append(from_node)

        # Validate no cycles
        if self._has_cycle():
            # Rollback
            self.edges.pop()
            self.adjacency[from_node].pop()
            self.reverse_adjacency[to_node].pop()
            raise ValueError(f"Adding edge {from_node} -> {to_node} creates a cycle")

        return self

    def _has_cycle(self) -> bool:
        """Check if graph contains a cycle using DFS.

        Returns:
            True if cycle exists, False otherwise
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.adjacency.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return True

        return False

    def topological_sort(self) -> List[List[str]]:
        """Compute topological ordering as levels.

        Returns levels where:
        - Nodes in same level can execute in parallel
        - All dependencies of level N are in levels < N

        Returns:
            List of levels, where each level is a list of node names

        Raises:
            ValueError: If graph contains cycles (shouldn't happen due to add_edge validation)
        """
        in_degree = {name: 0 for name in self.nodes}
        for edge in self.edges:
            in_degree[edge.to_node] += 1

        levels: List[List[str]] = []
        queue = deque([name for name, deg in in_degree.items() if deg == 0])

        if not queue:
            raise ValueError("No nodes with zero in-degree found (cycle detected)")

        while queue:
            level: List[str] = []
            for _ in range(len(queue)):
                node_name = queue.popleft()
                level.append(node_name)

                for neighbor in self.adjacency.get(node_name, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            levels.append(level)

        # Verify all nodes were included
        total_nodes = sum(len(level) for level in levels)
        if total_nodes != len(self.nodes):
            raise ValueError("Topological sort failed: cycle detected")

        return levels

    def get_node(self, name: str) -> Node:
        """Get node by name.

        Args:
            name: Node name

        Returns:
            Node instance

        Raises:
            KeyError: If node not found
        """
        if name not in self.nodes:
            raise KeyError(f"Node '{name}' not found in topology")
        return self.nodes[name]

    def visualize(self, show_types: bool = True) -> str:
        """Generate ASCII visualization of topology.

        Args:
            show_types: If True, include node types in output

        Returns:
            Multi-line string representation of topology
        """
        lines = ["=" * 60, "Pipeline Topology", "=" * 60]

        try:
            levels = self.topological_sort()
        except ValueError as e:
            return f"Error: {e}"

        for level_idx, level_nodes in enumerate(levels):
            lines.append(f"\nLevel {level_idx}:")
            for node_name in level_nodes:
                node = self.nodes[node_name]
                if show_types:
                    type_str = f" [{node.node_type.value}]"
                else:
                    type_str = ""

                config_str = []
                if node.config.skip_on_negative:
                    config_str.append("skip_on_neg")
                if node.config.filter_only:
                    config_str.append("filter_only")
                if node.config.weight != 1.0:
                    config_str.append(f"w={node.config.weight:.2f}")

                config_info = f" ({', '.join(config_str)})" if config_str else ""

                lines.append(f"  - {node_name}{type_str}{config_info}")

        lines.append("\nEdges:")
        for edge in self.edges:
            cond_str = " [conditional]" if edge.condition else ""
            lines.append(f"  {edge.from_node} -> {edge.to_node}{cond_str}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def validate(self) -> None:
        """Validate topology structure.

        Raises:
            ValueError: If topology is invalid
        """
        # Check for isolated nodes
        all_connected = set()
        for edge in self.edges:
            all_connected.add(edge.from_node)
            all_connected.add(edge.to_node)

        isolated = set(self.nodes.keys()) - all_connected
        if len(isolated) > 1:  # Allow single root node
            raise ValueError(f"Isolated nodes found: {isolated}")

        # Try topological sort (will raise if cycles exist)
        self.topological_sort()


# ==============================================================================
# Pipeline Executor
# ==============================================================================

class PipelineExecutor:
    """Orchestrates execution of nodes according to topology.

    The executor:
    - Manages execution order via topological sort
    - Handles parallel execution within levels
    - Maintains execution context and results
    - Implements skip logic for failed samples
    - Aggregates node outputs into final scores

    Attributes:
        topology: The topology graph to execute
        context: Shared execution context
        results: Stores NodeResult for each executed node
        skip_indices: Set of sample indices marked for skipping
    """

    def __init__(self,
                 topology: TopologyGraph,
                 context: ExecutionContext):
        """Initialize executor.

        Args:
            topology: Validated topology graph
            context: Execution context with required fields
        """
        self.topology = topology
        self.context = context
        self.results: Dict[str, NodeResult] = {}
        self.skip_indices: Set[int] = set()

        # Add skip_indices to context
        self.context['skip_indices'] = self.skip_indices

        # Validate topology
        self.topology.validate()

    async def execute(self, batch_inputs: List[Any]) -> List[float]:
        """Execute complete pipeline on batch of inputs.

        Args:
            batch_inputs: Batch of raw inputs (e.g., solution strings)

        Returns:
            List of final scores, one per input
        """
        import time

        num_samples = len(batch_inputs)

        # Get execution order
        levels = self.topology.topological_sort()

        print(f"[Executor] Starting pipeline with {num_samples} samples")
        print(f"[Executor] Execution plan: {len(levels)} levels")

        # Execute level by level
        for level_idx, level_nodes in enumerate(levels):
            print(f"\n[Executor] === Level {level_idx} ===")
            print(f"[Executor] Nodes: {', '.join(level_nodes)}")

            # Get enabled nodes in this level
            enabled_nodes = [name for name in level_nodes
                           if self.topology.nodes[name].config.enabled]

            if not enabled_nodes:
                print(f"[Executor] All nodes in level {level_idx} disabled, skipping")
                continue

            # Prepare tasks for parallel execution
            tasks: List[Awaitable[Tuple[str, NodeResult]]] = []

            for node_name in enabled_nodes:
                node = self.topology.get_node(node_name)

                # Get inputs for this node
                node_inputs = self._get_node_inputs(node_name, batch_inputs)

                # Create task
                task = self._execute_node_wrapper(node, node_inputs)
                tasks.append(task)

            # Execute all nodes in level in parallel
            level_start = time.time()
            level_results = await aio.gather(*tasks)
            level_time = time.time() - level_start

            # Store results and update skip indices
            for node_name, result in level_results:
                self.results[node_name] = result
                self._update_skip_indices(node_name, result)

            print(f"[Executor] Level {level_idx} completed in {level_time:.2f}s")
            print(f"[Executor] Skipped samples: {len(self.skip_indices)}/{num_samples}")

        # Aggregate final scores
        print(f"\n[Executor] Aggregating final scores...")
        final_scores = self._aggregate_final_scores(num_samples)

        return final_scores

    def _get_node_inputs(self,
                        node_name: str,
                        original_inputs: List[Any]) -> List[Optional[Any]]:
        """Get inputs for a node from predecessor outputs or original inputs.

        Args:
            node_name: Name of node to get inputs for
            original_inputs: Original batch inputs

        Returns:
            List of inputs for the node (with None for skipped samples)
        """
        # Find predecessor nodes
        predecessors = self.topology.reverse_adjacency.get(node_name, [])

        if not predecessors:
            # Root node: use original inputs, apply skip mask
            return [inp if i not in self.skip_indices else None
                   for i, inp in enumerate(original_inputs)]

        # Strategy: Use output from first parser/transformer node in chain
        # For rule/judge nodes, use the parser's output
        # This ensures type consistency

        # Find the first parser/transformer predecessor
        for pred_name in predecessors:
            pred_node = self.topology.nodes[pred_name]
            if pred_node.node_type in (NodeType.PARSER, NodeType.TRANSFORMER, NodeType.LLM_GENERATOR):
                pred_result = self.results.get(pred_name)
                if pred_result is not None:
                    return pred_result.outputs

        # Fallback: use first predecessor's output
        pred_result = self.results.get(predecessors[0])
        if pred_result is None:
            # Predecessor hasn't executed yet (shouldn't happen with topo sort)
            raise RuntimeError(f"Predecessor {predecessors[0]} of {node_name} not executed")

        return pred_result.outputs

    async def _execute_node_wrapper(self,
                                    node: Node,
                                    inputs: List[Optional[Any]]
                                    ) -> Tuple[str, NodeResult]:
        """Wrapper for executing a node with timing.

        Args:
            node: Node to execute
            inputs: Inputs for the node

        Returns:
            Tuple of (node_name, result)
        """
        import time

        print(f"[Executor] Executing {node.name}...")

        start_time = time.time()
        try:
            result = await node.execute(inputs, self.context)
            result.execution_time = time.time() - start_time

            print(f"[Executor] {node.name} completed in {result.execution_time:.2f}s")
            return node.name, result

        except Exception as e:
            print(f"[Executor] ERROR in {node.name}: {e}")
            # Return empty result on error
            empty_result = NodeResult(
                outputs=[None] * len(inputs),
                node_name=node.name,
                execution_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
            return node.name, empty_result

    def _update_skip_indices(self, node_name: str, result: NodeResult) -> None:
        """Update skip indices based on node result.

        Args:
            node_name: Name of executed node
            result: Node execution result
        """
        node = self.topology.get_node(node_name)

        if not node.config.skip_on_negative:
            return

        # Mark samples with negative scores for skipping
        for i, output in enumerate(result.outputs):
            if output is not None and isinstance(output, (int, float)) and output < 0:
                self.skip_indices.add(i)

    def _aggregate_final_scores(self, num_samples: int) -> List[float]:
        """Aggregate node outputs into final scores.

        Args:
            num_samples: Number of samples in batch

        Returns:
            List of final scores
        """
        final_scores = [0.0] * num_samples

        # Collect all non-filter nodes with their weights
        score_nodes = [
            (name, node) for name, node in self.topology.nodes.items()
            if not node.config.filter_only and node.config.enabled
        ]

        for i in range(num_samples):
            if i in self.skip_indices:
                # Use minimum reward for skipped samples
                final_scores[i] = self.context.get('min_reward', -2.0)
                continue

            # Accumulate weighted scores from all nodes
            sample_score = 0.0

            for node_name, node in score_nodes:
                result = self.results.get(node_name)
                if result is None:
                    continue

                output = result.outputs[i]
                if output is None:
                    continue

                # Handle different output types
                if isinstance(output, (int, float)):
                    node_score = output
                elif isinstance(output, (list, tuple)):
                    # Check if it's a list of numbers
                    try:
                        if all(isinstance(x, (int, float)) for x in output):
                            node_score = sum(output)
                        else:
                            # Non-numeric list/tuple, skip
                            continue
                    except:
                        continue
                else:
                    # Unknown type, skip
                    continue

                sample_score += node.config.weight * node_score

            final_scores[i] = sample_score

        return final_scores

    def get_node_result(self, node_name: str) -> Optional[NodeResult]:
        """Get execution result for a specific node.

        Args:
            node_name: Name of node

        Returns:
            NodeResult if node was executed, None otherwise
        """
        return self.results.get(node_name)

    def print_summary(self) -> None:
        """Print execution summary."""
        print("\n" + "=" * 60)
        print("Execution Summary")
        print("=" * 60)

        for node_name, result in self.results.items():
            node = self.topology.get_node(node_name)

            non_none_count = sum(1 for o in result.outputs if o is not None)
            total_count = len(result.outputs)

            print(f"\n{node_name}:")
            print(f"  Type: {node.node_type.value}")
            print(f"  Time: {result.execution_time:.2f}s")
            print(f"  Valid outputs: {non_none_count}/{total_count}")

            if result.metadata:
                print(f"  Metadata: {result.metadata}")

        print("\n" + "=" * 60)


# ==============================================================================
# Utility Functions
# ==============================================================================

def create_context(
    ground_truths: List[Dict[str, Any]],
    agents: Optional[Dict[str, Any]] = None,
    max_concurrent: Optional[Dict[str, int]] = None,
    parse_fn: Optional[Callable] = None,
    min_reward: float = -2.0,
    **kwargs
) -> ExecutionContext:
    """Convenience function to create ExecutionContext.

    Args:
        ground_truths: Ground truth data for each sample
        agents: Dictionary of LLM agents
        max_concurrent: Concurrency limits by agent
        parse_fn: Solution parsing function
        min_reward: Minimum reward for failed samples
        **kwargs: Additional context fields

    Returns:
        ExecutionContext instance
    """
    context: ExecutionContext = {
        'ground_truths': ground_truths,
        'agents': agents or {},
        'max_concurrent': max_concurrent or {},
        'min_reward': min_reward,
        'skip_indices': set()
    }

    if parse_fn is not None:
        context['parse_fn'] = parse_fn

    # Add additional kwargs
    for key, value in kwargs.items():
        context[key] = value  # type: ignore

    return context


# ==============================================================================
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


def setup_logging(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    log_filename: Optional[str] = None,
    console_output: bool = True,
    json_indent: Optional[int] = None
) -> Any:
    """Setup structured JSON logging with file + console output.

    Args:
        log_dir: Directory for log files
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_filename: Custom filename (default: app_YYYYMMDD_HHMMSS.jsonl)
        console_output: Enable console output (human-readable)
        json_indent: JSON indent (None=JSONL compact, 2=pretty)

    Returns:
        Root logger

    Raises:
        ImportError: If structlog is not installed

    Example:
        >>> setup_logging(log_dir="./logs", log_level="INFO")
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

    # File handler (JSON/JSONL)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(indent=json_indent)
        )
    )
    logging.root.addHandler(file_handler)

    # Console handler (human-readable)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(colors=True)
            )
        )
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
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    print(f"[Logging] File: {log_file}")
    print(f"[Logging] Level: {log_level}, Console: {console_output}")

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

__version__ = "0.1.0"
__all__ = [
    # Enums
    'NodeType',

    # Type definitions
    'ExecutionContext',
    'NodeConfig',
    'SolverConfig',
    'DifficultyMetricConfig',
    'NodeResult',

    # Protocols
    'ParseFunction',
    'PostprocessFunction',
    'PromptFunction',
    'PenaltyOrRewardModule',

    # Core classes
    'Node',
    'Edge',
    'TopologyGraph',
    'PipelineExecutor',

    # LLM Agent
    'Agent',
    'AgentConfig',
    'LLMResponse',
    'LLMError',
    'RateLimitError',

    # Logging
    'setup_logging',
    'get_logger',

    # Utilities
    'create_context',
    'create_agent',
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


if __name__ == "__main__":
    # Framework self-test
    print(f"Typed Pipeline Framework v{__version__}")
    print(f"Available node types: {[t.value for t in NodeType]}")
    print("Framework loaded successfully!")
