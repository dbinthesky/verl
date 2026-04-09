"""
Unit tests for the typed pipeline framework.

Run with: python -m pytest test_reward.py -v
Or: python -m unittest test_reward.py -v
"""

import unittest
import asyncio
from typing import List, Optional, Tuple
from reward import (
    Node, NodeConfig, NodeType, NodeResult,
    TopologyGraph, PipelineExecutor, ExecutionContext,
    create_context, Edge, SolverConfig, DifficultyMetricConfig
)


# ==============================================================================
# Mock Nodes for Testing
# ==============================================================================

class MockParserNode(Node[str, Tuple[str, str]]):
    """Mock parser for testing."""

    def __init__(self, config: NodeConfig):
        super().__init__(config)

    async def execute(self, batch_inputs, context):
        outputs = []
        for inp in batch_inputs:
            if inp is None:
                outputs.append(None)
                continue
            try:
                parts = inp.split('|')
                if len(parts) >= 2:
                    outputs.append((parts[0].strip(), parts[1].strip()))
                else:
                    outputs.append(None)
            except:
                outputs.append(None)

        return NodeResult(
            outputs=outputs,
            node_name=self.name,
            metadata={'parsed': sum(1 for o in outputs if o)}
        )


class MockRuleNode(Node[Tuple[str, str], float]):
    """Mock rule node for testing."""

    def __init__(self, config: NodeConfig, min_score: float, max_score: float):
        super().__init__(config)
        self.min_score = min_score
        self.max_score = max_score

    async def execute(self, batch_inputs, context):
        outputs = []
        for inp in batch_inputs:
            if inp is None:
                outputs.append(None)
            elif len(inp[0]) < 5:
                outputs.append(self.min_score)
            else:
                outputs.append(0.0)
        return NodeResult(outputs=outputs, node_name=self.name)


class MockScoreNode(Node[Tuple[str, str], float]):
    """Mock scoring node for testing."""

    def __init__(self, config: NodeConfig):
        super().__init__(config)

    async def execute(self, batch_inputs, context):
        outputs = []
        for inp in batch_inputs:
            if inp is None:
                outputs.append(None)
            else:
                score = min(len(inp[1]) / 50.0, 1.0)
                outputs.append(score)
        return NodeResult(outputs=outputs, node_name=self.name)


class ErrorNode(Node[str, float]):
    """Node that always raises an error."""

    def __init__(self, config: NodeConfig):
        super().__init__(config)

    async def execute(self, batch_inputs, context):
        raise RuntimeError("Intentional error for testing")


# ==============================================================================
# Test Cases
# ==============================================================================

class TestNodeConfig(unittest.TestCase):
    """Test NodeConfig validation and immutability."""

    def test_valid_config(self):
        """Test creating valid config."""
        config = NodeConfig(
            name="test",
            node_type=NodeType.PARSER,
            skip_on_negative=True,
            filter_only=False,
            weight=1.5
        )
        self.assertEqual(config.name, "test")
        self.assertEqual(config.node_type, NodeType.PARSER)
        self.assertTrue(config.skip_on_negative)
        self.assertFalse(config.filter_only)
        self.assertEqual(config.weight, 1.5)

    def test_empty_name(self):
        """Test that empty name raises ValueError."""
        with self.assertRaises(ValueError):
            NodeConfig(name="", node_type=NodeType.PARSER)

    def test_negative_weight(self):
        """Test that negative weight raises ValueError."""
        with self.assertRaises(ValueError):
            NodeConfig(name="test", node_type=NodeType.PARSER, weight=-1.0)

    def test_immutability(self):
        """Test that config is immutable."""
        config = NodeConfig(name="test", node_type=NodeType.PARSER)
        with self.assertRaises(Exception):  # FrozenInstanceError
            config.name = "modified"


class TestSolverConfig(unittest.TestCase):
    """Test SolverConfig validation."""

    def test_valid_config(self):
        """Test creating valid solver config."""
        config = SolverConfig(
            name="weak",
            agent_key="weak_agent",
            repeat=3,
            prompt_fn_key="respond_wo_context",
            max_concurrent=16
        )
        self.assertEqual(config.name, "weak")
        self.assertEqual(config.repeat, 3)

    def test_invalid_repeat(self):
        """Test that zero or negative repeat raises ValueError."""
        with self.assertRaises(ValueError):
            SolverConfig(
                name="test",
                agent_key="agent",
                repeat=0,
                prompt_fn_key="fn",
                max_concurrent=10
            )

    def test_invalid_max_concurrent(self):
        """Test that zero or negative max_concurrent raises ValueError."""
        with self.assertRaises(ValueError):
            SolverConfig(
                name="test",
                agent_key="agent",
                repeat=1,
                prompt_fn_key="fn",
                max_concurrent=-1
            )


class TestDifficultyMetricConfig(unittest.TestCase):
    """Test DifficultyMetricConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DifficultyMetricConfig(weak_name="weak", adv_name="adv")
        self.assertEqual(config.weak_weight, 0.4)
        self.assertEqual(config.adv_weight, 0.6)
        self.assertEqual(config.weak_overcomplex_threshold, 0.1)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DifficultyMetricConfig(
            weak_name="weak",
            adv_name="adv",
            weak_weight=0.3,
            adv_weight=0.7
        )
        self.assertEqual(config.weak_weight, 0.3)
        self.assertEqual(config.adv_weight, 0.7)


class TestTopologyGraph(unittest.TestCase):
    """Test TopologyGraph operations."""

    def test_add_node(self):
        """Test adding nodes to topology."""
        topology = TopologyGraph()
        node = MockParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))

        topology.add_node(node)
        self.assertIn("parser", topology.nodes)
        self.assertEqual(topology.nodes["parser"], node)

    def test_add_duplicate_node(self):
        """Test that adding duplicate node raises ValueError."""
        topology = TopologyGraph()
        node1 = MockParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))
        node2 = MockParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))

        topology.add_node(node1)
        with self.assertRaises(ValueError):
            topology.add_node(node2)

    def test_add_edge(self):
        """Test adding edges."""
        topology = TopologyGraph()
        node1 = MockParserNode(NodeConfig(name="a", node_type=NodeType.PARSER))
        node2 = MockParserNode(NodeConfig(name="b", node_type=NodeType.PARSER))

        topology.add_node(node1).add_node(node2)
        topology.add_edge("a", "b")

        self.assertEqual(len(topology.edges), 1)
        self.assertIn("b", topology.adjacency["a"])
        self.assertIn("a", topology.reverse_adjacency["b"])

    def test_add_edge_nonexistent_node(self):
        """Test that adding edge with nonexistent node raises ValueError."""
        topology = TopologyGraph()
        node = MockParserNode(NodeConfig(name="a", node_type=NodeType.PARSER))
        topology.add_node(node)

        with self.assertRaises(ValueError):
            topology.add_edge("a", "nonexistent")

    def test_cycle_detection(self):
        """Test that cycles are detected."""
        topology = TopologyGraph()
        for name in ["a", "b", "c"]:
            node = MockParserNode(NodeConfig(name=name, node_type=NodeType.PARSER))
            topology.add_node(node)

        topology.add_edge("a", "b")
        topology.add_edge("b", "c")

        # This should create a cycle
        with self.assertRaises(ValueError):
            topology.add_edge("c", "a")

    def test_topological_sort(self):
        """Test topological sorting."""
        topology = TopologyGraph()

        nodes = {
            "a": MockParserNode(NodeConfig(name="a", node_type=NodeType.PARSER)),
            "b": MockParserNode(NodeConfig(name="b", node_type=NodeType.PARSER)),
            "c": MockParserNode(NodeConfig(name="c", node_type=NodeType.PARSER))
        }

        for node in nodes.values():
            topology.add_node(node)

        topology.add_edge("a", "b")
        topology.add_edge("b", "c")

        levels = topology.topological_sort()

        self.assertEqual(len(levels), 3)
        self.assertEqual(levels[0], ["a"])
        self.assertEqual(levels[1], ["b"])
        self.assertEqual(levels[2], ["c"])

    def test_parallel_nodes(self):
        """Test that independent nodes are in same level."""
        topology = TopologyGraph()

        root = MockParserNode(NodeConfig(name="root", node_type=NodeType.PARSER))
        branch1 = MockParserNode(NodeConfig(name="branch1", node_type=NodeType.PARSER))
        branch2 = MockParserNode(NodeConfig(name="branch2", node_type=NodeType.PARSER))

        topology.add_node(root).add_node(branch1).add_node(branch2)
        topology.add_edge("root", "branch1")
        topology.add_edge("root", "branch2")

        levels = topology.topological_sort()

        self.assertEqual(len(levels), 2)
        self.assertEqual(levels[0], ["root"])
        # branch1 and branch2 should be in same level (parallel)
        self.assertIn("branch1", levels[1])
        self.assertIn("branch2", levels[1])

    def test_get_node(self):
        """Test getting node by name."""
        topology = TopologyGraph()
        node = MockParserNode(NodeConfig(name="test", node_type=NodeType.PARSER))
        topology.add_node(node)

        retrieved = topology.get_node("test")
        self.assertEqual(retrieved, node)

    def test_get_nonexistent_node(self):
        """Test getting nonexistent node raises KeyError."""
        topology = TopologyGraph()
        with self.assertRaises(KeyError):
            topology.get_node("nonexistent")

    def test_validate_isolated_nodes(self):
        """Test validation catches isolated nodes."""
        topology = TopologyGraph()

        node1 = MockParserNode(NodeConfig(name="a", node_type=NodeType.PARSER))
        node2 = MockParserNode(NodeConfig(name="b", node_type=NodeType.PARSER))
        node3 = MockParserNode(NodeConfig(name="isolated", node_type=NodeType.PARSER))

        topology.add_node(node1).add_node(node2).add_node(node3)
        topology.add_edge("a", "b")

        with self.assertRaises(ValueError):
            topology.validate()

    def test_visualize(self):
        """Test topology visualization."""
        topology = TopologyGraph()

        parser = MockParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))
        rule = MockRuleNode(
            NodeConfig(name="rule", node_type=NodeType.RULE, skip_on_negative=True),
            -1.0, 0.0
        )

        topology.add_node(parser).add_node(rule)
        topology.add_edge("parser", "rule")

        viz = topology.visualize()

        self.assertIn("parser", viz)
        self.assertIn("rule", viz)
        self.assertIn("->", viz)


class TestNode(unittest.TestCase):
    """Test Node base class functionality."""

    def test_filter_valid_inputs(self):
        """Test filtering None inputs."""
        node = MockParserNode(NodeConfig(name="test", node_type=NodeType.PARSER))

        inputs = ["a", None, "b", None, "c"]
        valid_indices, valid_inputs = node._filter_valid_inputs(inputs)

        self.assertEqual(valid_indices, [0, 2, 4])
        self.assertEqual(valid_inputs, ["a", "b", "c"])

    def test_reconstruct_batch(self):
        """Test reconstructing batch with None for skipped samples."""
        node = MockParserNode(NodeConfig(name="test", node_type=NodeType.PARSER))

        valid_indices = [0, 2, 4]
        valid_outputs = ["A", "B", "C"]
        total_size = 5

        outputs = node._reconstruct_batch(valid_indices, valid_outputs, total_size)

        self.assertEqual(outputs, ["A", None, "B", None, "C"])


class TestPipelineExecutor(unittest.TestCase):
    """Test PipelineExecutor functionality."""

    def test_simple_execution(self):
        """Test executing a simple pipeline."""

        async def run():
            topology = TopologyGraph()

            parser = MockParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))
            scorer = MockScoreNode(NodeConfig(name="scorer", node_type=NodeType.LLM_JUDGE))

            topology.add_node(parser).add_node(scorer)
            topology.add_edge("parser", "scorer")

            batch_inputs = ["question1 | answer1", "question2 | answer2"]
            context = create_context(ground_truths=[{"id": 0}, {"id": 1}])

            executor = PipelineExecutor(topology, context)
            scores = await executor.execute(batch_inputs)

            self.assertEqual(len(scores), 2)
            self.assertIsInstance(scores[0], float)
            self.assertIsInstance(scores[1], float)

        asyncio.run(run())

    def test_skip_logic(self):
        """Test skip logic with negative scores."""

        async def run():
            topology = TopologyGraph()

            parser = MockParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))
            rule = MockRuleNode(
                NodeConfig(name="rule", node_type=NodeType.RULE,
                          skip_on_negative=True, filter_only=True),
                min_score=-1.0, max_score=0.0
            )
            scorer = MockScoreNode(NodeConfig(name="scorer", node_type=NodeType.LLM_JUDGE))

            topology.add_node(parser).add_node(rule).add_node(scorer)
            topology.add_edge("parser", "rule")
            topology.add_edge("rule", "scorer")

            batch_inputs = [
                "long question | answer1",  # Valid
                "hi | answer2",              # Invalid: question too short
                "another long question | answer3"  # Valid
            ]
            context = create_context(
                ground_truths=[{"id": i} for i in range(3)],
                min_reward=-2.0
            )

            executor = PipelineExecutor(topology, context)
            scores = await executor.execute(batch_inputs)

            self.assertEqual(len(scores), 3)
            self.assertGreater(scores[0], -2.0)  # Valid, scored
            self.assertEqual(scores[1], -2.0)     # Skipped, min_reward
            self.assertGreater(scores[2], -2.0)  # Valid, scored

        asyncio.run(run())

    def test_error_handling(self):
        """Test that node errors don't crash pipeline."""

        async def run():
            topology = TopologyGraph()

            parser = MockParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))
            error_node = ErrorNode(NodeConfig(name="error", node_type=NodeType.LLM_JUDGE))

            topology.add_node(parser).add_node(error_node)
            topology.add_edge("parser", "error")

            batch_inputs = ["question | answer"]
            context = create_context(ground_truths=[{"id": 0}])

            executor = PipelineExecutor(topology, context)
            scores = await executor.execute(batch_inputs)

            # Should complete without crashing
            self.assertEqual(len(scores), 1)

            # Check error was recorded
            result = executor.get_node_result("error")
            self.assertIsNotNone(result)
            self.assertIn('error', result.metadata)

        asyncio.run(run())

    def test_parallel_execution_performance(self):
        """Test that parallel nodes execute concurrently."""

        async def run():
            import time

            class SlowNode(Node[str, float]):
                def __init__(self, config, delay=0.2):
                    super().__init__(config)
                    self.delay = delay

                async def execute(self, batch_inputs, context):
                    await asyncio.sleep(self.delay)
                    return NodeResult(
                        outputs=[1.0] * len(batch_inputs),
                        node_name=self.name
                    )

            topology = TopologyGraph()

            root = MockParserNode(NodeConfig(name="root", node_type=NodeType.PARSER))
            branch1 = SlowNode(NodeConfig(name="b1", node_type=NodeType.LLM_JUDGE), delay=0.2)
            branch2 = SlowNode(NodeConfig(name="b2", node_type=NodeType.LLM_JUDGE), delay=0.2)

            topology.add_node(root).add_node(branch1).add_node(branch2)
            topology.add_edge("root", "b1")
            topology.add_edge("root", "b2")

            batch_inputs = ["test | answer"]
            context = create_context(ground_truths=[{"id": 0}])

            executor = PipelineExecutor(topology, context)

            start = time.time()
            await executor.execute(batch_inputs)
            elapsed = time.time() - start

            # Should take ~0.2s (parallel) not ~0.4s (sequential)
            self.assertLess(elapsed, 0.35, f"Parallel execution too slow: {elapsed:.2f}s")

        asyncio.run(run())

    def test_weighted_aggregation(self):
        """Test weighted score aggregation."""

        async def run():
            topology = TopologyGraph()

            parser = MockParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))
            scorer1 = MockScoreNode(NodeConfig(name="s1", node_type=NodeType.LLM_JUDGE, weight=0.3))
            scorer2 = MockScoreNode(NodeConfig(name="s2", node_type=NodeType.LLM_JUDGE, weight=0.7))

            topology.add_node(parser).add_node(scorer1).add_node(scorer2)
            topology.add_edge("parser", "s1")
            topology.add_edge("parser", "s2")

            # Both scorers return 1.0
            batch_inputs = ["question | " + "a" * 50]  # Max score
            context = create_context(ground_truths=[{"id": 0}])

            executor = PipelineExecutor(topology, context)
            scores = await executor.execute(batch_inputs)

            # Should be 0.3 * 1.0 + 0.7 * 1.0 = 1.0
            self.assertAlmostEqual(scores[0], 1.0, places=2)

        asyncio.run(run())

    def test_filter_only_nodes(self):
        """Test that filter_only nodes don't contribute to score."""

        async def run():
            topology = TopologyGraph()

            parser = MockParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))
            filter_node = MockScoreNode(
                NodeConfig(name="filter", node_type=NodeType.FILTER,
                          filter_only=True, weight=100.0)  # Large weight but ignored
            )
            scorer = MockScoreNode(NodeConfig(name="scorer", node_type=NodeType.LLM_JUDGE))

            topology.add_node(parser).add_node(filter_node).add_node(scorer)
            topology.add_edge("parser", "filter")
            topology.add_edge("filter", "scorer")

            batch_inputs = ["question | answer"]
            context = create_context(ground_truths=[{"id": 0}])

            executor = PipelineExecutor(topology, context)
            scores = await executor.execute(batch_inputs)

            # Score should only come from scorer, not filter_node
            # Even though filter_node has weight=100
            self.assertLess(scores[0], 1.0)

        asyncio.run(run())

    def test_get_node_result(self):
        """Test retrieving node results after execution."""

        async def run():
            topology = TopologyGraph()

            parser = MockParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))
            topology.add_node(parser)

            batch_inputs = ["q1 | a1", "q2 | a2"]
            context = create_context(ground_truths=[{"id": 0}, {"id": 1}])

            executor = PipelineExecutor(topology, context)
            await executor.execute(batch_inputs)

            result = executor.get_node_result("parser")
            self.assertIsNotNone(result)
            self.assertEqual(result.node_name, "parser")
            self.assertEqual(len(result.outputs), 2)
            self.assertIn('parsed', result.metadata)

        asyncio.run(run())


class TestContextCreation(unittest.TestCase):
    """Test context creation utilities."""

    def test_create_context_minimal(self):
        """Test creating context with minimal parameters."""
        context = create_context(ground_truths=[{"id": 0}])

        self.assertIn('ground_truths', context)
        self.assertIn('agents', context)
        self.assertIn('skip_indices', context)

    def test_create_context_full(self):
        """Test creating context with all parameters."""
        def mock_parse(s):
            return s.split('|')

        context = create_context(
            ground_truths=[{"id": 0}],
            agents={"agent1": "mock_agent"},
            max_concurrent={"agent1": 10},
            parse_fn=mock_parse,
            min_reward=-5.0,
            custom_field="custom_value"
        )

        self.assertEqual(context['min_reward'], -5.0)
        self.assertEqual(context['agents']['agent1'], "mock_agent")
        self.assertEqual(context['max_concurrent']['agent1'], 10)
        self.assertEqual(context['parse_fn'], mock_parse)
        self.assertEqual(context['custom_field'], "custom_value")


class TestEdge(unittest.TestCase):
    """Test Edge functionality."""

    def test_unconditional_edge(self):
        """Test unconditional edge always executes."""
        edge = Edge("a", "b")
        context = create_context(ground_truths=[])

        self.assertTrue(edge.should_execute(context))

    def test_conditional_edge(self):
        """Test conditional edge execution."""
        def condition(ctx):
            return ctx.get('flag', False)

        edge = Edge("a", "b", condition=condition)

        context1 = create_context(ground_truths=[], flag=True)
        context2 = create_context(ground_truths=[], flag=False)

        self.assertTrue(edge.should_execute(context1))
        self.assertFalse(edge.should_execute(context2))


# ==============================================================================
# Test Suite
# ==============================================================================

def suite():
    """Create test suite."""
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNodeConfig))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSolverConfig))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDifficultyMetricConfig))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTopologyGraph))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNode))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPipelineExecutor))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestContextCreation))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEdge))

    return suite


if __name__ == '__main__':
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
