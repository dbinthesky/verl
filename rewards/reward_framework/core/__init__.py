"""
Core Framework Module

This module contains the core framework components:
- PipelineData: Protocol defining data contract
- PipelineDataBase: Base implementation
- NodeConfig: Node configuration
- ExecutionMetadata: Execution metadata
"""

from .protocol import PipelineData
from .base import PipelineDataBase
from .config import NodeType, NodeConfig, ExecutionMetadata


__all__ = [
    'PipelineData',
    'PipelineDataBase',
    'NodeType',
    'NodeConfig',
    'ExecutionMetadata',
]
