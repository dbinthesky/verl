"""
Utility Functions

This module provides utility functions for the framework.
"""

from typing import Any, Dict, List


__all__ = ['create_simple_context']


def create_simple_context(ground_truths: List[Any]) -> Dict[str, Any]:
    """创建简单的执行上下文（便捷函数）"""
    return {
        "ground_truths": ground_truths,
        "min_reward": 0.0
    }
