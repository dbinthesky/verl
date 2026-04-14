"""
Pipeline Data Protocol - Framework Core

This module defines the minimal contract that all pipeline data must implement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Protocol, runtime_checkable


__all__ = ['PipelineData']


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
