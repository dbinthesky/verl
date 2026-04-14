"""
Pipeline Data Base Implementation - Framework Core

This module provides the base implementation of PipelineData protocol.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass, field


__all__ = ['PipelineDataBase']


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

    def get_children(self, child_type: Optional[type] = None) -> List['PipelineDataBase']:
        """获取子节点，可选按类型过滤

        Args:
            child_type: 子节点类型（None = 返回所有）

        Returns:
            子节点列表

        Examples:
            all_children = sample.get_children()
            categories = sample.get_children(RubricCategory)
        """
        if child_type is None:
            return self._children
        return [c for c in self._children if isinstance(c, child_type)]

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

    def iter_all_descendants(self, node_type: Optional[type] = None) -> Iterator['PipelineDataBase']:
        """递归迭代所有后代节点（DFS），可选按类型过滤

        Args:
            node_type: 节点类型（None = 返回所有）

        Examples:
            all_descendants = list(sample.iter_all_descendants())
            all_categories = list(sample.iter_all_descendants(RubricCategory))
        """
        for child in self._children:
            if node_type is None or isinstance(child, node_type):
                yield child
            yield from child.iter_all_descendants(node_type)

    def mark_skipped_recursive(self, reason: str, node_name: str) -> None:
        """递归标记自己和所有后代为 skipped

        Args:
            reason: Skip 原因
            node_name: 标记的节点名称
        """
        self.mark_skipped(reason, node_name)
        for child in self._children:
            child.mark_skipped_recursive(f"parent_skipped: {reason}", node_name)
