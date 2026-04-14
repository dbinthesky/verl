"""
Node Base Classes - Framework Core

This module provides the base classes for all pipeline nodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generic, TypeVar
import asyncio as aio

from ..core import PipelineData, NodeConfig, ExecutionMetadata


__all__ = ['DataT', 'Node', 'MapNode', 'ExpandNode', 'AggregateNode']


DataT = TypeVar('DataT', bound=PipelineData)


class Node(ABC, Generic[DataT]):
    """节点基类（框架层）

    核心原则：
    1. 只依赖PipelineData接口，不依赖具体类型
    2. In-place修改数据，不创建新数据
    3. 返回执行元数据（用于监控和日志）
    4. 通过 data_type 显式声明处理的数据类型（供 Executor 自动收集）
    """

    # 显式声明该节点处理的数据类型（子类应该覆盖）
    data_type: Optional[type] = None

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
