"""
LLM Agent Module - Async OpenAI Client with Strong Typing

This module provides an async LLM agent with robust error handling,
automatic retries, and batch processing capabilities.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional, Callable, List, Tuple
from collections import defaultdict
import asyncio as aio

from .config import AgentConfig
from .types import LLMError, RateLimitError, PostprocessError, APIError, LLMResponse
from ..log import get_logger


__all__ = ['Agent', 'create_agent']


class Agent:
    """Async LLM Agent with strong typing and robust error handling.

    Features:
    - Type-safe configuration
    - Automatic retries with exponential backoff
    - Batch processing with concurrency control
    - Prompt deduplication
    - Client connection pooling
    - Comprehensive error handling

    Example:
        config = AgentConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=2048
        )
        agent = Agent(config)

        prompts = ["Question 1", "Question 2"]
        responses = await agent.batch_generate(
            prompts=prompts,
            max_concurrent=10,
            postprocess_fn=lambda x: x.strip()
        )
    """

    def __init__(self, config: AgentConfig):
        """Initialize agent with configuration.

        Args:
            config: Agent configuration
        """
        self.config = config
        self._client: Optional[Any] = None
        self._logger = get_logger(f"agent.{config.model}")

        # Get API key
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self._logger.error("agent_init_failed",
                reason="missing_api_key",
                model=config.model
            )
            raise ValueError("API key not provided and OPENAI_API_KEY env var not set")

    async def _get_client(self):
        """Get or create AsyncOpenAI client (lazy initialization)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            except ImportError:
                self._logger.error("agent_init_failed",
                    reason="openai_not_installed",
                    model=self.config.model
                )
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    async def generate(self,
                      prompt: str,
                      postprocess_fn: Optional[Callable[[str], str]] = None,
                      **override_kwargs) -> Optional[LLMResponse]:
        """Generate single response.

        Args:
            prompt: Input prompt
            postprocess_fn: Optional function to clean up response
            **override_kwargs: Override config parameters for this call

        Returns:
            LLMResponse or None if generation failed
        """
        from tenacity import (
            retry, stop_after_attempt, wait_exponential,
            retry_if_exception_type
        )

        @retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait
            ),
            retry=retry_if_exception_type((RateLimitError, APIError))
        )
        async def _call_api():
            client = await self._get_client()

            # Build request kwargs
            request_kwargs = {
                'model': self.config.model,
                'messages': [
                    {'role': 'system', 'content': self.config.system_message},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': self.config.max_tokens,
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'seed': self.config.seed
            }

            # Add reasoning_effort if configured
            if self.config.reasoning_effort is not None:
                request_kwargs['reasoning_effort'] = self.config.reasoning_effort

            # Apply overrides
            request_kwargs.update(override_kwargs)

            start = time.time()
            try:
                response = await client.chat.completions.create(**request_kwargs)
                latency = time.time() - start

                content = response.choices[0].message.content

                # Postprocess if needed
                if postprocess_fn:
                    try:
                        content = postprocess_fn(content)
                    except Exception as e:
                        self._logger.error("postprocess_failed",
                            error_message=str(e),
                            response_length=len(content),
                            prompt_length=len(prompt)
                        )
                        raise PostprocessError(f"Postprocess failed: {e}")

                return LLMResponse(
                    content=content,
                    prompt=prompt,
                    model=response.model,
                    finish_reason=response.choices[0].finish_reason,
                    usage={
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    } if response.usage else None,
                    latency=latency
                )

            except Exception as e:
                error_msg = str(e).lower()

                # Detect rate limit errors
                if 'rate' in error_msg or '429' in error_msg:
                    self._logger.error("api_call_failed",
                        error_type="rate_limit",
                        error_message=str(e),
                        model=self.config.model,
                        prompt_length=len(prompt)
                    )
                    raise RateLimitError(f"Rate limit exceeded: {e}")

                # Other API errors are retryable
                self._logger.error("api_call_failed",
                    error_type="api_error",
                    error_message=str(e),
                    model=self.config.model,
                    prompt_length=len(prompt)
                )
                raise APIError(f"API call failed: {e}")

        try:
            return await _call_api()
        except (RateLimitError, APIError) as e:
            self._logger.error("generation_failed_after_retries",
                error_type=type(e).__name__,
                error_message=str(e),
                max_retries=self.config.max_retries,
                model=self.config.model
            )
            return None
        except PostprocessError as e:
            self._logger.error("generation_failed_postprocess",
                error_message=str(e),
                model=self.config.model
            )
            return None

    async def batch_generate(self,
                            prompts: List[str],
                            max_concurrent: int = 10,
                            postprocess_fn: Optional[Callable[[str], str]] = None,
                            deduplicate: bool = True,
                            desc: Optional[str] = None,
                            show_progress: bool = True
                            ) -> List[Tuple[str, Optional[LLMResponse]]]:
        """Generate responses for batch of prompts with concurrency control.

        Args:
            prompts: List of input prompts
            max_concurrent: Maximum concurrent requests
            postprocess_fn: Optional postprocessing function
            deduplicate: If True, deduplicate prompts before calling API
            desc: Description for progress bar
            show_progress: Whether to show progress

        Returns:
            List of (prompt, response) tuples in original order
        """
        if not prompts:
            return []

        # Deduplicate prompts
        if deduplicate:
            unique_prompts = list(dict.fromkeys(prompts))  # Preserve order
            prompt_to_indices = defaultdict(list)
            for idx, prompt in enumerate(prompts):
                prompt_to_indices[prompt].append(idx)
        else:
            unique_prompts = prompts
            prompt_to_indices = {p: [i] for i, p in enumerate(prompts)}

        # Create semaphore for concurrency control
        semaphore = aio.Semaphore(max_concurrent)

        async def _generate_with_semaphore(prompt):
            async with semaphore:
                return prompt, await self.generate(prompt, postprocess_fn)

        # Execute all requests
        tasks = [_generate_with_semaphore(p) for p in unique_prompts]

        if show_progress and desc:
            print(f"[Agent] {desc} - {len(unique_prompts)} unique prompts "
                  f"(from {len(prompts)} total, concurrency={max_concurrent})")

        results = await aio.gather(*tasks)

        if show_progress and desc:
            success_count = sum(1 for _, resp in results if resp is not None)
            failed_count = len(unique_prompts) - success_count
            failure_rate = failed_count / len(unique_prompts) if unique_prompts else 0

            print(f"[Agent] {desc} - Completed: {success_count}/{len(unique_prompts)} successful")

            # Log high failure rate
            if failure_rate > 0.2:  # More than 20% failures
                self._logger.warning("batch_high_failure_rate",
                    desc=desc,
                    total=len(unique_prompts),
                    successful=success_count,
                    failed=failed_count,
                    failure_rate=failure_rate,
                    model=self.config.model
                )

        # Map back to original order if deduplicated
        if deduplicate:
            result_map = {prompt: resp for prompt, resp in results}
            outputs = []
            for prompt in prompts:
                outputs.append((prompt, result_map.get(prompt)))
            return outputs
        else:
            return results

    async def batch_generate_simple(self,
                                    prompts: List[str],
                                    max_concurrent: int = 10,
                                    postprocess_fn: Optional[Callable[[str], str]] = None,
                                    desc: Optional[str] = None
                                    ) -> List[Optional[str]]:
        """Simplified batch generation returning only content strings.

        Compatible with original Agent.run() interface.

        Args:
            prompts: List of input prompts
            max_concurrent: Maximum concurrent requests
            postprocess_fn: Optional postprocessing function
            desc: Description for logging

        Returns:
            List of generated strings (None for failed generations)
        """
        results = await self.batch_generate(
            prompts=prompts,
            max_concurrent=max_concurrent,
            postprocess_fn=postprocess_fn,
            desc=desc
        )

        return [resp.content if resp else None for _, resp in results]

    async def close(self):
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()


# Backward compatibility: create Agent from old-style kwargs
def create_agent(
    system: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_kwargs: Optional[dict] = None
) -> Agent:
    """Create Agent from old-style parameters (backward compatibility).

    Args:
        system: System message
        model: Model name
        base_url: API base URL
        api_key: API key
        request_kwargs: Additional request parameters

    Returns:
        Agent instance
    """
    from dataclasses import replace

    config = AgentConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        system_message=system or "You are a helpful and harmless assistant."
    )

    # Apply request_kwargs if provided
    if request_kwargs:
        config = replace(
            config,
            max_tokens=request_kwargs.get('max_tokens', config.max_tokens),
            temperature=request_kwargs.get('temperature', config.temperature),
            top_p=request_kwargs.get('top_p', config.top_p),
            seed=request_kwargs.get('seed', config.seed),
            reasoning_effort=request_kwargs.get('reasoning_effort', config.reasoning_effort)
        )

    return Agent(config)
