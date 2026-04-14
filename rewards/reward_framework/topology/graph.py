"""
Topology Graph - Pipeline Structure

This module provides the topology graph structure for defining node dependencies.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

from ..nodes import Node


__all__ = ['Edge', 'TopologyGraph']


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
