"""
LLM Agent Configuration

This module defines configuration structures for the LLM agent.
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass


__all__ = ['AgentConfig']


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
        reasoning_effort: Reasoning effort level ('low', 'medium', 'high')
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts for transient errors
        retry_min_wait: Minimum wait between retries (seconds)
        retry_max_wait: Maximum wait between retries (seconds)
        show_progress: Whether to show progress bar for batch generation
    """
    model: str = "gpt-3.5-turbo"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    system_message: str = "You are a helpful and harmless assistant."
    max_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    seed: int = 100745534
    reasoning_effort: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 4
    retry_min_wait: float = 5.0
    retry_max_wait: float = 20.0
    show_progress: bool = True

    def __post_init__(self):
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.reasoning_effort is not None and self.reasoning_effort not in ['low', 'medium', 'high']:
            raise ValueError(f"reasoning_effort must be 'low', 'medium', or 'high', got {self.reasoning_effort}")
