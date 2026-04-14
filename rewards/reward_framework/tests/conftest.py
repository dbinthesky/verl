"""Pytest configuration for reward_framework tests."""

import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark async tests with pytest.mark.asyncio."""
    for item in items:
        if item.get_closest_marker("asyncio") is None:
            if hasattr(item, "function") and hasattr(item.function, "__code__"):
                if item.function.__code__.co_flags & 0x0200:  # CO_COROUTINE
                    item.add_marker(pytest.mark.asyncio)
