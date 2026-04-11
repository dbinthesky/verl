"""
Test per-log console sampling feature.

This test verifies that each log can specify its own console_sample_rate.
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
from io import StringIO
from pathlib import Path


class TestPerLogSampling(unittest.TestCase):
    """Test per-log console sampling."""

    def setUp(self):
        """Setup test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.test_dir, "logs")

    def tearDown(self):
        """Cleanup test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_per_log_sample_rates(self):
        """Test different sample rates for different logs."""
        from reward import setup_logging, get_logger

        captured_output = StringIO()

        # Setup with default 100% sampling
        setup_logging(
            log_dir=self.log_dir,
            log_level="INFO",
            console_sample_rate=1.0,  # Default no sampling
            console_output=True
        )

        # Replace handler stream
        import logging
        for handler in logging.root.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.stream = captured_output
                break

        logger = get_logger("test")

        # Log 100 messages with different sample rates
        for i in range(100):
            # 0-24: 100% console output (default)
            if i < 25:
                logger.info(f"full_{i}", index=i)

            # 25-49: 10% console output
            elif i < 50:
                logger.info(f"sampled10_{i}", index=i, console_sample_rate=0.1)

            # 50-74: 0% console output (not shown)
            elif i < 75:
                logger.info(f"hidden_{i}", index=i, console_sample_rate=0.0)

            # 75-99: 50% console output
            else:
                logger.info(f"sampled50_{i}", index=i, console_sample_rate=0.5)

        # Check file has all 100 messages
        log_files = list(Path(self.log_dir).glob("*.jsonl"))
        with open(log_files[0], 'r') as f:
            file_lines = [json.loads(line) for line in f]
            file_count = len(file_lines)

        # Check console output
        console_output = captured_output.getvalue()
        full_count = console_output.count("full_")
        sampled10_count = console_output.count("sampled10_")
        hidden_count = console_output.count("hidden_")
        sampled50_count = console_output.count("sampled50_")

        print(f"\n✅ Per-log sampling test:")
        print(f"   File: {file_count} messages (all)")
        print(f"   Console 'full_': {full_count}/25 (100% rate)")
        print(f"   Console 'sampled10_': {sampled10_count}/25 (~10% rate)")
        print(f"   Console 'hidden_': {hidden_count}/25 (0% rate)")
        print(f"   Console 'sampled50_': {sampled50_count}/25 (~50% rate)")

        # Assertions
        self.assertEqual(file_count, 100, "File should have all 100 messages")
        self.assertEqual(full_count, 25, "All 'full_' should be in console")
        self.assertGreater(sampled10_count, 0, "Some 'sampled10_' should be in console")
        self.assertLess(sampled10_count, 10, "Not too many 'sampled10_' in console")
        self.assertEqual(hidden_count, 0, "No 'hidden_' should be in console")
        self.assertGreater(sampled50_count, 5, "Some 'sampled50_' should be in console")
        self.assertLess(sampled50_count, 20, "Not too many 'sampled50_' in console")

    def test_mixed_with_errors(self):
        """Test per-log sampling with ERROR always logged."""
        from reward import setup_logging, get_logger

        captured_output = StringIO()

        setup_logging(
            log_dir=self.log_dir,
            log_level="INFO",
            console_sample_rate=1.0,
            console_output=True
        )

        import logging
        for handler in logging.root.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.stream = captured_output
                break

        logger = get_logger("test")

        # Log INFO with 0% sampling (should not appear)
        for i in range(50):
            logger.info(f"info_{i}", console_sample_rate=0.0)

        # Log ERRORs (should always appear even with 0% sampling)
        for i in range(5):
            logger.error(f"error_{i}", console_sample_rate=0.0)

        # Check console
        console_output = captured_output.getvalue()
        info_count = console_output.count("info_")
        error_count = console_output.count("error_")

        print(f"\n✅ Mixed with errors test:")
        print(f"   Console INFO (0% rate): {info_count} (expected 0)")
        print(f"   Console ERROR (0% rate): {error_count} (expected 5)")

        # File should have all 55
        log_files = list(Path(self.log_dir).glob("*.jsonl"))
        with open(log_files[0], 'r') as f:
            file_count = len(f.readlines())

        self.assertEqual(file_count, 55)
        self.assertEqual(info_count, 0)
        self.assertEqual(error_count, 5, "ERRORs always logged even with 0% rate")

    def test_global_default_rate(self):
        """Test global default rate with per-log override."""
        from reward import setup_logging, get_logger

        captured_output = StringIO()

        # Setup with 10% default sampling
        setup_logging(
            log_dir=self.log_dir,
            log_level="INFO",
            console_sample_rate=0.1,  # Default 10%
            console_output=True
        )

        import logging
        for handler in logging.root.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.stream = captured_output
                break

        logger = get_logger("test")

        # Log 50 with default (10%)
        for i in range(50):
            logger.info(f"default_{i}", index=i)

        # Log 50 with override (100%)
        for i in range(50):
            logger.info(f"override_{i}", index=i, console_sample_rate=1.0)

        console_output = captured_output.getvalue()
        # Count occurrences more carefully
        import re
        default_count = len(re.findall(r'default_\d+', console_output))
        override_count = len(re.findall(r'override_\d+', console_output))

        print(f"\n✅ Global default rate test:")
        print(f"   Console 'default_': {default_count}/50 (~10% default)")
        print(f"   Console 'override_': {override_count}/50 (100% override)")

        # File has all 100
        log_files = list(Path(self.log_dir).glob("*.jsonl"))
        with open(log_files[0], 'r') as f:
            file_count = len(f.readlines())

        self.assertEqual(file_count, 100)
        self.assertGreater(default_count, 0)
        self.assertLess(default_count, 25)  # Roughly 10%, allow variance
        self.assertEqual(override_count, 50)  # All overridden logs


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" "*18 + "PER-LOG SAMPLING TESTS")
    print("="*70)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestPerLogSampling)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {len(result.failures + result.errors)} TEST(S) FAILED")
    print("="*70)
