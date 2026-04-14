"""
Simplified unit tests for Agent class - Core functionality only.

Tests against local deployed model: gpt-oss-120b at 10.102.215.37:28000
"""

import unittest
import asyncio
import time
from reward_framework import Agent, AgentConfig, LLMResponse, create_agent


# Test configuration
# TEST_MODEL = "gpt-oss-120b"
# TEST_BASE_URL = "http://10.102.215.37:28000/v1"
TEST_MODEL = "qwen3.5-35b"
TEST_BASE_URL = "http://10.102.216.23:28000/v1"
TEST_API_KEY = "dummy-key"


class TestAgentConfig(unittest.TestCase):
    """Test AgentConfig validation."""

    def test_valid_config(self):
        """Test creating valid config."""
        config = AgentConfig(
            model=TEST_MODEL,
            base_url=TEST_BASE_URL,
            api_key=TEST_API_KEY,
            temperature=0.7
        )
        self.assertEqual(config.model, TEST_MODEL)
        self.assertEqual(config.temperature, 0.7)

    def test_invalid_temperature(self):
        """Test invalid temperature raises ValueError."""
        with self.assertRaises(ValueError):
            AgentConfig(model=TEST_MODEL, temperature=2.5, api_key="dummy")

    def test_immutable_config(self):
        """Test config is immutable."""
        config = AgentConfig(model=TEST_MODEL, api_key="dummy")
        with self.assertRaises(Exception):
            config.model = "other"


class TestAgentBasic(unittest.TestCase):
    """Test basic Agent functionality."""

    def test_agent_creation(self):
        """Test creating agent."""
        config = AgentConfig(
            model=TEST_MODEL,
            base_url=TEST_BASE_URL,
            api_key=TEST_API_KEY
        )
        agent = Agent(config)
        self.assertEqual(agent.config.model, TEST_MODEL)


class TestAgentGeneration(unittest.TestCase):
    """Test Agent generation with real API."""

    def setUp(self):
        """Set up agent."""
        self.config = AgentConfig(
            model=TEST_MODEL,
            base_url=TEST_BASE_URL,
            api_key=TEST_API_KEY,
            temperature=0.7,
            max_tokens=100
        )
        self.agent = Agent(self.config)

    def test_single_generation(self):
        """Test single generation."""
        async def run():
            response = await self.agent.generate("What is 2+2?")

            self.assertIsNotNone(response)
            self.assertIsInstance(response, LLMResponse)
            self.assertGreater(response.latency, 0)

            print(f"\n✅ Single generation test passed")
            print(f"   Response: {response.content[:50] if response.content else 'None'}...")
            print(f"   Latency: {response.latency:.2f}s")

            await self.agent.close()

        asyncio.run(run())

    def test_batch_generation(self):
        """Test batch generation."""
        async def run():
            prompts = [
                "What is the capital of France?",
                "What is the capital of Germany?",
                "What is the capital of Italy?"
            ]

            results = await self.agent.batch_generate(
                prompts=prompts,
                max_concurrent=2,
                desc="Capitals"
            )

            self.assertEqual(len(results), len(prompts))

            success = sum(1 for _, r in results if r is not None)
            print(f"\n✅ Batch generation test passed")
            print(f"   Success: {success}/{len(prompts)}")

            await self.agent.close()

        asyncio.run(run())

    def test_deduplication(self):
        """Test automatic deduplication."""
        async def run():
            prompts = ["Q1", "Q2", "Q1", "Q3", "Q2"]  # 3 unique

            results = await self.agent.batch_generate(
                prompts=prompts,
                max_concurrent=2,
                deduplicate=True,
                desc="Dedup test"
            )

            self.assertEqual(len(results), len(prompts))
            print(f"\n✅ Deduplication test passed")
            print(f"   Total: {len(prompts)}, Unique: {len(set(prompts))}")

            await self.agent.close()

        asyncio.run(run())

    def test_postprocessing(self):
        """Test postprocessing."""
        async def run():
            def extract_number(text: str) -> str:
                import re
                match = re.search(r'\d+', text)
                return match.group(0) if match else text

            response = await self.agent.generate(
                "10*5=?",
                postprocess_fn=extract_number
            )

            if response:
                print(f"\n✅ Postprocess test passed")
                print(f"   Processed: {response.content}")

            await self.agent.close()

        asyncio.run(run())


class TestAgentConcurrency(unittest.TestCase):
    """Test concurrency control."""

    def test_concurrent_execution(self):
        """Test concurrent requests."""
        async def run():
            config = AgentConfig(
                model=TEST_MODEL,
                base_url=TEST_BASE_URL,
                api_key=TEST_API_KEY,
                max_tokens=20
            )
            agent = Agent(config)

            prompts = ["test"] * 6

            start = time.time()
            results = await agent.batch_generate(
                prompts=prompts,
                max_concurrent=3,
                desc="Concurrent"
            )
            elapsed = time.time() - start

            print(f"\n✅ Concurrency test passed")
            print(f"   Processed {len(prompts)} in {elapsed:.2f}s")

            await agent.close()

        asyncio.run(run())


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility."""

    def test_create_agent(self):
        """Test create_agent helper."""
        agent = create_agent(
            model=TEST_MODEL,
            base_url=TEST_BASE_URL,
            api_key=TEST_API_KEY,
            request_kwargs={'max_tokens': 50}
        )

        self.assertEqual(agent.config.model, TEST_MODEL)
        self.assertEqual(agent.config.max_tokens, 50)

        print("\n✅ Backward compatibility test passed")


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgentConfig))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgentBasic))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgentGeneration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgentConcurrency))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBackwardCompatibility))
    return suite


if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING AGENT WITH REAL API")
    print(f"Model: {TEST_MODEL}")
    print(f"Endpoint: {TEST_BASE_URL}")
    print("="*70)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())

    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {len(result.failures + result.errors)} TEST(S) FAILED")
    print("="*70)
