"""
Test suite for the typed pipeline framework.

This demonstrates:
1. How to implement custom nodes with strong typing
2. How to build a topology graph
3. How to execute a complete pipeline
"""

import asyncio
from typing import List, Optional, Tuple
from reward import (
    Node, NodeConfig, NodeType, NodeResult,
    TopologyGraph, PipelineExecutor, ExecutionContext,
    create_context
)


# ==============================================================================
# Example Node Implementations
# ==============================================================================

class MockParserNode(Node[str, Tuple[str, str]]):
    """Example parser node: splits input by '|'."""

    def __init__(self, config: NodeConfig):
        super().__init__(config)

    async def execute(self,
                     batch_inputs: List[Optional[str]],
                     context: ExecutionContext) -> NodeResult[Tuple[str, str]]:
        """Parse inputs into (question, answer) tuples."""

        outputs: List[Optional[Tuple[str, str]]] = []

        for inp in batch_inputs:
            if inp is None:
                outputs.append(None)
                continue

            try:
                parts = inp.split('|')
                if len(parts) == 2:
                    outputs.append((parts[0].strip(), parts[1].strip()))
                else:
                    outputs.append(None)
            except Exception:
                outputs.append(None)

        return NodeResult(
            outputs=outputs,
            node_name=self.name,
            metadata={'parsed_count': sum(1 for o in outputs if o is not None)}
        )


class MockRuleNode(Node[Tuple[str, str], float]):
    """Example rule node: checks if question is non-empty."""

    def __init__(self, config: NodeConfig, min_score: float, max_score: float):
        super().__init__(config)
        self.min_score = min_score
        self.max_score = max_score

    async def execute(self,
                     batch_inputs: List[Optional[Tuple[str, str]]],
                     context: ExecutionContext) -> NodeResult[float]:
        """Check rule and assign penalty/reward."""

        outputs: List[Optional[float]] = []

        for inp in batch_inputs:
            if inp is None:
                outputs.append(None)
                continue

            question, answer = inp

            # Rule: question must have at least 5 characters
            if len(question) < 5:
                outputs.append(self.min_score)  # Penalty
            else:
                outputs.append(0.0)  # Neutral

        return NodeResult(
            outputs=outputs,
            node_name=self.name
        )


class MockScoreNode(Node[Tuple[str, str], float]):
    """Example scoring node: scores based on answer length."""

    def __init__(self, config: NodeConfig):
        super().__init__(config)

    async def execute(self,
                     batch_inputs: List[Optional[Tuple[str, str]]],
                     context: ExecutionContext) -> NodeResult[float]:
        """Score based on answer quality."""

        outputs: List[Optional[float]] = []

        for inp in batch_inputs:
            if inp is None:
                outputs.append(None)
                continue

            question, answer = inp

            # Simple scoring: longer answers get higher scores
            score = min(len(answer) / 50.0, 1.0)
            outputs.append(score)

        return NodeResult(
            outputs=outputs,
            node_name=self.name,
            metadata={'avg_score': sum(s for s in outputs if s) / max(1, sum(1 for s in outputs if s))}
        )


# ==============================================================================
# Test Functions
# ==============================================================================

def test_topology_construction():
    """Test building a topology graph."""
    print("\n" + "="*60)
    print("TEST: Topology Construction")
    print("="*60)

    topology = TopologyGraph()

    # Add nodes
    parser = MockParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))
    rule_check = MockRuleNode(
        NodeConfig(name="rule_check", node_type=NodeType.RULE, skip_on_negative=True),
        min_score=-1.0, max_score=0.0
    )
    scorer = MockScoreNode(NodeConfig(name="scorer", node_type=NodeType.LLM_JUDGE, weight=1.0))

    topology.add_node(parser)
    topology.add_node(rule_check)
    topology.add_node(scorer)

    # Add edges
    topology.add_edge("parser", "rule_check")
    topology.add_edge("rule_check", "scorer")

    print("\nTopology created successfully!")
    print(topology.visualize())

    # Test topological sort
    levels = topology.topological_sort()
    print(f"\nExecution levels: {levels}")

    assert len(levels) == 3, "Should have 3 levels"
    assert levels[0] == ["parser"], "Level 0 should be parser"

    print("\n✓ Topology construction test passed!")
    return topology


def test_cycle_detection():
    """Test that cycles are detected and prevented."""
    print("\n" + "="*60)
    print("TEST: Cycle Detection")
    print("="*60)

    topology = TopologyGraph()

    node_a = MockParserNode(NodeConfig(name="a", node_type=NodeType.PARSER))
    node_b = MockParserNode(NodeConfig(name="b", node_type=NodeType.PARSER))
    node_c = MockParserNode(NodeConfig(name="c", node_type=NodeType.PARSER))

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    topology.add_edge("a", "b")
    topology.add_edge("b", "c")

    # Try to create a cycle
    try:
        topology.add_edge("c", "a")
        assert False, "Should have raised ValueError for cycle"
    except ValueError as e:
        print(f"\n✓ Cycle correctly detected: {e}")

    print("\n✓ Cycle detection test passed!")


async def test_pipeline_execution():
    """Test executing a complete pipeline."""
    print("\n" + "="*60)
    print("TEST: Pipeline Execution")
    print("="*60)

    # Build topology
    topology = TopologyGraph()

    parser = MockParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))
    rule_check = MockRuleNode(
        NodeConfig(name="rule_check", node_type=NodeType.RULE, skip_on_negative=True, filter_only=True),
        min_score=-1.0, max_score=0.0
    )
    scorer = MockScoreNode(NodeConfig(name="scorer", node_type=NodeType.LLM_JUDGE, weight=1.0))

    topology.add_node(parser).add_node(rule_check).add_node(scorer)
    topology.add_edge("parser", "rule_check")
    topology.add_edge("rule_check", "scorer")

    # Create test data
    batch_inputs = [
        "What is Python? | A programming language",  # Valid
        "Hi | ok",  # Invalid: question too short
        "What is machine learning? | ML is a field of AI that enables computers to learn",  # Valid
        "test | short",  # Invalid: question too short
    ]

    ground_truths = [{'id': i} for i in range(len(batch_inputs))]

    # Create context
    context = create_context(ground_truths=ground_truths, min_reward=-2.0)

    # Execute pipeline
    executor = PipelineExecutor(topology, context)
    scores = await executor.execute(batch_inputs)

    print("\n" + "-"*60)
    print("Results:")
    print("-"*60)
    for i, (inp, score) in enumerate(zip(batch_inputs, scores)):
        print(f"Sample {i}: {inp[:40]:40s} | Score: {score:.3f}")

    executor.print_summary()

    # Verify results
    assert len(scores) == len(batch_inputs), "Should have one score per input"
    assert scores[0] > 0, "Sample 0 should have positive score"
    assert scores[1] == -2.0, "Sample 1 should be skipped (min_reward)"
    assert scores[2] > scores[0], "Sample 2 should score higher (longer answer)"
    assert scores[3] == -2.0, "Sample 3 should be skipped (min_reward)"

    print("\n✓ Pipeline execution test passed!")


async def test_parallel_execution():
    """Test that nodes at same level execute in parallel."""
    print("\n" + "="*60)
    print("TEST: Parallel Execution")
    print("="*60)

    import time

    class SlowNode(Node[str, float]):
        """Node that takes 0.5 seconds to execute."""

        def __init__(self, config: NodeConfig, delay: float = 0.5):
            super().__init__(config)
            self.delay = delay

        async def execute(self, batch_inputs, context):
            await asyncio.sleep(self.delay)
            return NodeResult(
                outputs=[1.0] * len(batch_inputs),
                node_name=self.name
            )

    # Build topology with parallel branches
    topology = TopologyGraph()

    root = MockParserNode(NodeConfig(name="root", node_type=NodeType.PARSER))
    branch_a = SlowNode(NodeConfig(name="branch_a", node_type=NodeType.LLM_JUDGE))
    branch_b = SlowNode(NodeConfig(name="branch_b", node_type=NodeType.LLM_JUDGE))

    topology.add_node(root).add_node(branch_a).add_node(branch_b)
    topology.add_edge("root", "branch_a")
    topology.add_edge("root", "branch_b")

    batch_inputs = ["test|answer"] * 3
    ground_truths = [{'id': i} for i in range(len(batch_inputs))]
    context = create_context(ground_truths=ground_truths)

    # Execute and measure time
    executor = PipelineExecutor(topology, context)
    start = time.time()
    scores = await executor.execute(batch_inputs)
    elapsed = time.time() - start

    print(f"\nElapsed time: {elapsed:.2f}s")
    print(f"Expected: ~0.5s (parallel) vs ~1.0s (sequential)")

    # Should take ~0.5s (parallel) not ~1.0s (sequential)
    assert elapsed < 0.8, f"Parallel execution too slow: {elapsed:.2f}s"

    print("\n✓ Parallel execution test passed!")


# ==============================================================================
# Run All Tests
# ==============================================================================

async def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("TYPED PIPELINE FRAMEWORK TEST SUITE")
    print("="*60)

    # Synchronous tests
    test_topology_construction()
    test_cycle_detection()

    # Asynchronous tests
    await test_pipeline_execution()
    await test_parallel_execution()

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
