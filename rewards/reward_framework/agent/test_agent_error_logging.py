"""
Test Agent error logging.

This test deliberately triggers errors to verify logging works.
"""

import unittest
import asyncio
from reward_framework import Agent, AgentConfig, setup_logging, get_logger


# Setup logging first
setup_logging(log_dir="./test_logs", log_level="DEBUG")


class TestAgentErrorLogging(unittest.TestCase):
    """Test Agent error logging."""

    def test_missing_api_key(self):
        """Test logging when API key is missing."""
        print("\n[Test] Testing missing API key error...")

        try:
            config = AgentConfig(model="test-model", api_key=None)
            agent = Agent(config)
            self.fail("Should have raised ValueError")
        except ValueError as e:
            print(f"✅ Caught expected error: {e}")
            # Check that error was logged (log file should contain agent_init_failed)

    def test_postprocess_error(self):
        """Test logging when postprocessing fails."""
        print("\n[Test] Testing postprocess error...")

        async def run():
            def bad_postprocess(text: str) -> str:
                raise ValueError("Intentional postprocess error")

            config = AgentConfig(
                model="gpt-oss-120b",
                base_url="http://10.102.215.37:28000/v1",
                api_key="dummy",
                max_retries=1
            )
            agent = Agent(config)

            response = await agent.generate(
                "Say hello",
                postprocess_fn=bad_postprocess
            )

            self.assertIsNone(response, "Should return None on postprocess error")
            print("✅ Postprocess error handled correctly")

            await agent.close()

        asyncio.run(run())

    def test_invalid_model(self):
        """Test logging when model doesn't exist."""
        print("\n[Test] Testing invalid model error...")

        async def run():
            config = AgentConfig(
                model="nonexistent-model-xyz",
                base_url="http://10.102.215.37:28000/v1",
                api_key="dummy",
                max_retries=1
            )
            agent = Agent(config)

            response = await agent.generate("Test")

            # May or may not fail depending on API behavior
            # Just verify it doesn't crash
            print(f"Response: {response is not None}")

            await agent.close()

        asyncio.run(run())


def verify_error_logs():
    """Verify error logs were written."""
    print("\n" + "="*60)
    print("VERIFYING ERROR LOGS")
    print("="*60)

    import os
    import json

    log_dir = "./test_logs"
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.jsonl')]

    if not log_files:
        print("❌ No log files found")
        return

    latest_log = os.path.join(log_dir, sorted(log_files)[-1])
    print(f"Checking: {latest_log}\n")

    error_events = []
    with open(latest_log, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get('level') == 'error':
                    error_events.append(entry)
            except:
                pass

    print(f"Found {len(error_events)} error log entries:\n")

    for i, event in enumerate(error_events[-10:], 1):  # Show last 10
        print(f"{i}. Event: {event.get('event')}")
        print(f"   Error: {event.get('error_message', event.get('reason', 'N/A'))[:80]}")
        print(f"   Model: {event.get('model', 'N/A')}")
        print()

    # Check for expected error types
    expected_errors = ['agent_init_failed', 'postprocess_failed', 'api_call_failed']
    found_errors = [e.get('event') for e in error_events]

    for expected in expected_errors:
        if expected in found_errors:
            print(f"✅ Found expected error: {expected}")
        else:
            print(f"⚠️  Missing expected error: {expected}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" "*15 + "AGENT ERROR LOGGING TESTS")
    print("="*70)

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAgentErrorLogging)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Verify logs
    verify_error_logs()

    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
