"""
Typed Pipeline Framework for RL-based Question Generation & Evaluation

BACKWARD COMPATIBILITY SHIM
This file now redirects all imports to the new modular reward_framework package.

For new code, import directly from reward_framework:
    from reward_framework import PipelineData, Node, Agent, etc.

This shim is maintained for backward compatibility with existing code.
"""

# Import everything from the new modular package
from reward_framework import *  # noqa: F401, F403

# Maintain backward compatibility
__version__ = "2.0.0"

# Add deprecation notice for direct imports
import warnings
warnings.warn(
    "Direct import from reward.py is deprecated. "
    "Please use 'from reward_framework import ...' instead. "
    "This compatibility shim will be removed in version 3.0.0.",
    DeprecationWarning,
    stacklevel=2
)
