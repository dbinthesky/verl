"""
Test structured logging functionality.

Requirements:
    pip install structlog

Tests:
1. JSON/JSONL file logging
2. Console output (human-readable)
3. Structured data logging
4. Exception logging
"""

import os
import json
from reward import setup_logging, get_logger


def test_basic_logging():
    """Test basic logging to file and console."""
    print("\n" + "="*60)
    print("TEST: Basic Logging")
    print("="*60)

    # Setup logging
    setup_logging(
        log_dir="./test_logs",
        log_level="DEBUG",
        console_output=True,
        json_indent=None  # JSONL format
    )

    logger = get_logger(__name__)

    # Test different log levels
    logger.debug("debug_event", detail="Debug information")
    logger.info("info_event", user_id=123, action="login")
    logger.warning("warning_event", threshold=0.9, current=0.95)
    logger.error("error_event", error_code=500, retry_count=3)

    print("\n✅ Basic logging completed")


def test_structured_logging():
    """Test structured data logging."""
    print("\n" + "="*60)
    print("TEST: Structured Data Logging")
    print("="*60)

    logger = get_logger("test_module")

    # Log pipeline execution with structured data
    logger.info(
        "pipeline_started",
        pipeline_id="pipe-001",
        batch_size=32,
        model="gpt-4",
        temperature=0.7,
        max_tokens=2048
    )

    # Log node execution
    logger.info(
        "node_executed",
        node_name="parser",
        input_count=100,
        output_count=95,
        duration_ms=234.5
    )

    # Log difficulty evaluation
    logger.info(
        "difficulty_computed",
        weak_pass_rate=0.45,
        adv_pass_rate=0.82,
        final_score=0.67
    )

    print("✅ Structured logging completed")


def test_exception_logging():
    """Test exception logging."""
    print("\n" + "="*60)
    print("TEST: Exception Logging")
    print("="*60)

    logger = get_logger("error_handler")

    try:
        result = 1 / 0
    except ZeroDivisionError:
        logger.exception(
            "division_error",
            operation="divide",
            numerator=1,
            denominator=0
        )

    try:
        data = {"key": "value"}
        _ = data["missing_key"]
    except KeyError as e:
        logger.error(
            "key_error",
            error=str(e),
            available_keys=list(data.keys()),
            exc_info=True
        )

    print("✅ Exception logging completed")


def test_context_logging():
    """Test logging with context."""
    print("\n" + "="*60)
    print("TEST: Context Logging")
    print("="*60)

    logger = get_logger("agent")

    # Log with bound context
    request_logger = logger.bind(request_id="req-12345", user="alice")

    request_logger.info("request_started", method="POST", endpoint="/api/generate")
    request_logger.info("processing", stage="validation")
    request_logger.info("processing", stage="execution")
    request_logger.info("request_completed", status=200, duration_ms=456.7)

    print("✅ Context logging completed")


def verify_jsonl_format():
    """Verify that log file is valid JSONL."""
    print("\n" + "="*60)
    print("TEST: Verify JSONL Format")
    print("="*60)

    # Find the most recent log file
    log_dir = "./test_logs"
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.jsonl')]

    if not log_files:
        print("❌ No log files found")
        return

    latest_log = os.path.join(log_dir, sorted(log_files)[-1])
    print(f"Checking: {latest_log}")

    # Read and parse JSONL
    valid_count = 0
    invalid_count = 0

    with open(latest_log, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())

                # Check required fields
                assert 'event' in entry, "Missing 'event' field"
                assert 'timestamp' in entry, "Missing 'timestamp' field"
                assert 'level' in entry, "Missing 'level' field"

                valid_count += 1

                # Print first entry as example
                if i == 1:
                    print("\nExample log entry:")
                    print(json.dumps(entry, indent=2))

            except json.JSONDecodeError as e:
                print(f"❌ Line {i}: Invalid JSON - {e}")
                invalid_count += 1
            except AssertionError as e:
                print(f"❌ Line {i}: {e}")
                invalid_count += 1

    print(f"\n✅ Parsed {valid_count} valid JSONL entries")
    if invalid_count > 0:
        print(f"❌ Found {invalid_count} invalid entries")
    else:
        print("✅ All entries are valid JSONL")


def show_log_statistics():
    """Show statistics about logged events."""
    print("\n" + "="*60)
    print("LOG STATISTICS")
    print("="*60)

    log_dir = "./test_logs"
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.jsonl')]

    if not log_files:
        print("No log files found")
        return

    latest_log = os.path.join(log_dir, sorted(log_files)[-1])

    # Collect statistics
    event_counts = {}
    level_counts = {}

    with open(latest_log, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                event = entry.get('event', 'unknown')
                level = entry.get('level', 'unknown')

                event_counts[event] = event_counts.get(event, 0) + 1
                level_counts[level] = level_counts.get(level, 0) + 1
            except:
                pass

    print(f"\nLog file: {latest_log}")
    print(f"Total entries: {sum(event_counts.values())}")

    print("\nBy level:")
    for level, count in sorted(level_counts.items()):
        print(f"  {level.upper():8s}: {count}")

    print("\nBy event:")
    for event, count in sorted(event_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {event:30s}: {count}")


def run_all_tests():
    """Run all logging tests."""
    print("\n" + "="*70)
    print(" "*20 + "LOGGING TESTS")
    print("="*70)

    test_basic_logging()
    test_structured_logging()
    test_exception_logging()
    test_context_logging()
    verify_jsonl_format()
    show_log_statistics()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED ✅")
    print("="*70)
    print("\nCheck ./test_logs/ for JSONL log files")


if __name__ == "__main__":
    run_all_tests()
