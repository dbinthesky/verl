"""
Pipeline Executor

This module provides the pipeline execution engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import PipelineData, ExecutionMetadata
from .graph import TopologyGraph


__all__ = ['PipelineExecutor']


class PipelineExecutor:
    """Pipeline执行器（简化版）"""

    def __init__(self, topology: TopologyGraph):
        self.topology = topology
        self.context: Dict[str, Any] = {}
        self.execution_log: List[ExecutionMetadata] = []

    def _collect_nodes_of_type(
        self,
        root_batch: List[PipelineData],
        target_type: type,
        skip_filtered: bool = True
    ) -> List[PipelineData]:
        """从 root batch 递归收集指定类型的所有节点

        Args:
            root_batch: 根节点列表
            target_type: 目标类型
            skip_filtered: 是否过滤掉 skip 的节点

        Returns:
            收集到的节点列表
        """
        collected = []

        for root in root_batch:
            # 检查 root 自己
            if isinstance(root, target_type):
                if not skip_filtered or not root.is_skipped:
                    collected.append(root)

            # 递归检查所有后代
            for descendant in root.iter_all_descendants(target_type):
                if not skip_filtered or not descendant.is_skipped:
                    collected.append(descendant)

        return collected

    async def execute(
        self,
        batch: List[PipelineData],
        context: Optional[Dict[str, Any]] = None
    ) -> List[PipelineData]:
        """执行pipeline

        Args:
            batch: 根数据列表（通常是 Sample 级别）
            context: 执行上下文（可选）

        Returns:
            处理后的数据列表（同一个对象，已被修改）
        """
        self.context = context or {}
        self.execution_log = []

        print(f"[Executor] Starting pipeline with {len(batch)} root samples")

        # 拓扑排序
        sorted_nodes = self.topology.topological_sort()

        print(f"[Executor] Execution plan: {len(sorted_nodes)} nodes\n")

        # 顺序执行节点
        for node_name in sorted_nodes:
            node = self.topology.get_node(node_name)

            if not node.config.enabled:
                print(f"[Executor] Skipping {node_name} (disabled)")
                continue

            # 🔴 核心改动：自动收集该 Node 需要的数据类型
            if node.data_type is not None:
                # 节点声明了 data_type，自动收集对应类型的节点
                node_batch = self._collect_nodes_of_type(
                    batch,
                    node.data_type,
                    skip_filtered=node.config.respect_skip
                )
                type_name = node.data_type.__name__
                print(f"[Executor] Executing {node_name} on {len(node_batch)} nodes of type {type_name}...")
            else:
                # 未声明 data_type，使用原始 root batch
                node_batch = batch
                print(f"[Executor] Executing {node_name} on {len(node_batch)} root nodes...")

            # 执行节点
            metadata = await node.execute(node_batch, self.context)
            self.execution_log.append(metadata)

            print(f"[Executor] {node_name} completed in {metadata.execution_time:.2f}s")
            print(f"[Executor]   Processed: {metadata.processed_count}, Skipped: {metadata.skipped_count}")

            if metadata.newly_skipped_ids:
                print(f"[Executor]   Newly skipped: {len(metadata.newly_skipped_ids)} nodes")

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
