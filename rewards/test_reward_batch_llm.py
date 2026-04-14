"""
Large-scale batch LLM testing for Agent class.

Tests different batch sizes, concurrency levels, and performance metrics.
Tests against local deployed model: gpt-oss-120b at 10.102.215.37:28000
"""

import unittest
import asyncio
import time
from typing import List
from reward import Agent, AgentConfig, LLMResponse


# Test configuration
TEST_MODEL = "gpt-oss-120b"
TEST_BASE_URL = "http://10.102.215.37:28000/v1"
TEST_API_KEY = "dummy-key"


class TestBatchLLMGeneration(unittest.TestCase):
    """Test large-scale batch LLM generation."""

    def setUp(self):
        """Set up agent."""
        self.config = AgentConfig(
            model=TEST_MODEL,
            base_url=TEST_BASE_URL,
            api_key=TEST_API_KEY,
            temperature=0.7,
            max_tokens=50  # Short responses for speed
        )
        self.agent = Agent(self.config)

    def test_small_batch(self):
        """Test small batch (10 prompts)."""
        async def run():
            prompts = [f"What is {i} + {i}?" for i in range(1, 11)]

            start = time.time()
            results = await self.agent.batch_generate(
                prompts=prompts,
                max_concurrent=3,
                desc="Small batch"
            )
            elapsed = time.time() - start

            success_count = sum(1 for _, r in results if r is not None)

            print(f"\n✅ Small batch test (10 prompts):")
            print(f"   Success: {success_count}/{len(prompts)}")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Throughput: {len(prompts)/elapsed:.2f} prompts/sec")

            self.assertEqual(len(results), 10)
            self.assertGreater(success_count, 8)  # Allow 1-2 failures

            await self.agent.close()

        asyncio.run(run())

    def test_medium_batch(self):
        """Test medium batch (50 prompts)."""
        async def run():
            prompts = [f"Translate 'hello' to language {i}" for i in range(50)]

            start = time.time()
            results = await self.agent.batch_generate(
                prompts=prompts,
                max_concurrent=5,
                desc="Medium batch"
            )
            elapsed = time.time() - start

            success_count = sum(1 for _, r in results if r is not None)

            print(f"\n✅ Medium batch test (50 prompts):")
            print(f"   Success: {success_count}/{len(prompts)}")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Throughput: {len(prompts)/elapsed:.2f} prompts/sec")

            self.assertEqual(len(results), 50)
            self.assertGreater(success_count, 45)  # Allow ~10% failure

            await self.agent.close()

        asyncio.run(run())

    def test_large_batch(self):
        """Test large batch (100 prompts)."""
        async def run():
            prompts = [f"Count from 1 to {i}" for i in range(1, 101)]

            start = time.time()
            results = await self.agent.batch_generate(
                prompts=prompts,
                max_concurrent=10,
                desc="Large batch"
            )
            elapsed = time.time() - start

            success_count = sum(1 for _, r in results if r is not None)

            print(f"\n✅ Large batch test (100 prompts):")
            print(f"   Success: {success_count}/{len(prompts)}")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Throughput: {len(prompts)/elapsed:.2f} prompts/sec")

            self.assertEqual(len(results), 100)
            self.assertGreater(success_count, 90)  # Allow ~10% failure

            await self.agent.close()

        asyncio.run(run())

    def test_concurrency_comparison(self):
        """Test different concurrency levels."""
        async def run():
            prompts = [f"Test {i}" for i in range(30)]

            # Test with concurrency=1
            start = time.time()
            results_1 = await self.agent.batch_generate(
                prompts=prompts,
                max_concurrent=1,
                desc="Concurrency 1"
            )
            time_1 = time.time() - start

            # Test with concurrency=5
            start = time.time()
            results_5 = await self.agent.batch_generate(
                prompts=prompts,
                max_concurrent=5,
                desc="Concurrency 5"
            )
            time_5 = time.time() - start

            # Test with concurrency=10
            start = time.time()
            results_10 = await self.agent.batch_generate(
                prompts=prompts,
                max_concurrent=10,
                desc="Concurrency 10"
            )
            time_10 = time.time() - start

            print(f"\n✅ Concurrency comparison (30 prompts):")
            print(f"   Concurrency=1:  {time_1:.2f}s ({30/time_1:.2f} req/s)")
            print(f"   Concurrency=5:  {time_5:.2f}s ({30/time_5:.2f} req/s)")
            print(f"   Concurrency=10: {time_10:.2f}s ({30/time_10:.2f} req/s)")
            print(f"   Speedup (5 vs 1): {time_1/time_5:.2f}x")
            print(f"   Speedup (10 vs 1): {time_1/time_10:.2f}x")

            # Higher concurrency should be faster
            self.assertLess(time_5, time_1 * 0.8)  # At least 20% faster
            self.assertLess(time_10, time_5 * 0.8)  # At least 20% faster

            await self.agent.close()

        asyncio.run(run())

    def test_deduplication_large_scale(self):
        """Test deduplication with large duplicate set."""
        async def run():
            # 100 prompts, but only 10 unique
            prompts = []
            for i in range(10):
                prompts.extend([f"Question {i}"] * 10)  # Each repeated 10 times

            start = time.time()
            results = await self.agent.batch_generate(
                prompts=prompts,
                max_concurrent=5,
                deduplicate=True,
                desc="Dedup large"
            )
            elapsed = time.time() - start

            success_count = sum(1 for _, r in results if r is not None)

            print(f"\n✅ Deduplication test:")
            print(f"   Total prompts: {len(prompts)}")
            print(f"   Unique prompts: {len(set(prompts))}")
            print(f"   Results: {len(results)}")
            print(f"   Success: {success_count}/{len(results)}")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Dedup efficiency: Processed {len(set(prompts))} instead of {len(prompts)}")

            self.assertEqual(len(results), 100)
            self.assertGreater(success_count, 95)

            await self.agent.close()

        asyncio.run(run())

    def test_mixed_length_prompts(self):
        """Test batch with varying prompt lengths."""
        async def run():
            prompts = [
                "Hi",  # Very short
                "What is 2+2?",  # Short
                "Explain the concept of machine learning in one sentence.",  # Medium
                "Write a detailed explanation of how transformers work in natural language processing, including attention mechanisms.",  # Long
            ] * 10  # Repeat 10 times = 40 prompts

            start = time.time()
            results = await self.agent.batch_generate(
                prompts=prompts,
                max_concurrent=5,
                desc="Mixed length"
            )
            elapsed = time.time() - start

            success_count = sum(1 for _, r in results if r is not None)

            # Calculate average latency for each length
            latencies = {
                "short": [],
                "medium": [],
                "long": []
            }

            for prompt, response in results:
                if response is not None:
                    if len(prompt) < 10:
                        latencies["short"].append(response.latency)
                    elif len(prompt) < 50:
                        latencies["medium"].append(response.latency)
                    else:
                        latencies["long"].append(response.latency)

            print(f"\n✅ Mixed length test (40 prompts):")
            print(f"   Success: {success_count}/{len(prompts)}")
            print(f"   Total time: {elapsed:.2f}s")
            if latencies["short"]:
                print(f"   Avg latency (short):  {sum(latencies['short'])/len(latencies['short']):.2f}s")
            if latencies["medium"]:
                print(f"   Avg latency (medium): {sum(latencies['medium'])/len(latencies['medium']):.2f}s")
            if latencies["long"]:
                print(f"   Avg latency (long):   {sum(latencies['long'])/len(latencies['long']):.2f}s")

            self.assertEqual(len(results), 40)
            self.assertGreater(success_count, 35)

            await self.agent.close()

        asyncio.run(run())

    def test_with_postprocessing(self):
        """Test batch generation with postprocessing."""
        async def run():
            prompts = [f"{i} * 2 = ?" for i in range(1, 51)]

            def extract_number(text: str) -> str:
                """Extract first number from response."""
                import re
                match = re.search(r'\d+', text)
                return match.group(0) if match else text

            start = time.time()
            results = await self.agent.batch_generate(
                prompts=prompts,
                max_concurrent=5,
                postprocess_fn=extract_number,
                desc="With postprocess"
            )
            elapsed = time.time() - start

            success_count = sum(1 for _, r in results if r is not None)

            print(f"\n✅ Postprocessing test (50 prompts):")
            print(f"   Success: {success_count}/{len(prompts)}")
            print(f"   Time: {elapsed:.2f}s")

            # Show sample results
            if success_count > 0:
                sample_results = [(p, r.content) for p, r in results[:3] if r is not None]
                print(f"   Sample results:")
                for prompt, content in sample_results:
                    print(f"     {prompt} → {content}")

            self.assertEqual(len(results), 50)
            self.assertGreater(success_count, 45)

            await self.agent.close()

        asyncio.run(run())


class TestBatchPerformance(unittest.TestCase):
    """Test performance metrics and statistics."""

    def setUp(self):
        """Set up agent."""
        self.config = AgentConfig(
            model=TEST_MODEL,
            base_url=TEST_BASE_URL,
            api_key=TEST_API_KEY,
            temperature=0.7,
            max_tokens=30
        )
        self.agent = Agent(self.config)

    def test_latency_statistics(self):
        """Test and report latency statistics."""
        async def run():
            prompts = [f"Number {i}" for i in range(50)]

            results = await self.agent.batch_generate(
                prompts=prompts,
                max_concurrent=5,
                desc="Latency stats"
            )

            latencies = [r.latency for _, r in results if r is not None]

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                median_latency = sorted(latencies)[len(latencies) // 2]

                print(f"\n✅ Latency statistics (50 prompts):")
                print(f"   Avg:    {avg_latency:.3f}s")
                print(f"   Min:    {min_latency:.3f}s")
                print(f"   Max:    {max_latency:.3f}s")
                print(f"   Median: {median_latency:.3f}s")

                self.assertGreater(avg_latency, 0)
                self.assertLess(avg_latency, 5.0)  # Should be under 5s average

            await self.agent.close()

        asyncio.run(run())

    def test_throughput_scaling(self):
        """Test throughput with increasing batch sizes."""
        async def run():
            batch_sizes = [10, 20, 50]
            results_summary = []

            for size in batch_sizes:
                prompts = [f"Test {i}" for i in range(size)]

                start = time.time()
                results = await self.agent.batch_generate(
                    prompts=prompts,
                    max_concurrent=10,
                    desc=f"Batch {size}"
                )
                elapsed = time.time() - start

                success_count = sum(1 for _, r in results if r is not None)
                throughput = size / elapsed

                results_summary.append({
                    "size": size,
                    "time": elapsed,
                    "throughput": throughput,
                    "success_rate": success_count / size
                })

            print(f"\n✅ Throughput scaling:")
            print(f"   {'Batch Size':<12} {'Time (s)':<10} {'Throughput':<15} {'Success Rate'}")
            print(f"   {'-'*60}")
            for r in results_summary:
                print(f"   {r['size']:<12} {r['time']:<10.2f} {r['throughput']:<15.2f} {r['success_rate']:.1%}")

            # Throughput should generally increase with batch size
            self.assertGreater(results_summary[-1]["throughput"],
                             results_summary[0]["throughput"] * 0.8)

            await self.agent.close()

        asyncio.run(run())


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBatchLLMGeneration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBatchPerformance))
    return suite


if __name__ == '__main__':
    print("\n" + "="*70)
    print("LARGE-SCALE BATCH LLM TESTING")
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
