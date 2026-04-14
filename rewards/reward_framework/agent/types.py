"""
LLM Agent Types

This module defines exception types and response structures for the LLM agent.
"""

from __future__ import annotations

from typing import Dict, Optional
from dataclasses import dataclass


__all__ = [
    'LLMError',
    'RateLimitError',
    'PostprocessError',
    'APIError',
    'LLMResponse',
]


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
