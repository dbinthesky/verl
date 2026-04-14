"""
Test suite for the Protocol-based pipeline framework.

This demonstrates:
1. PipelineData Protocol and PipelineDataBase implementation
2. MapNode, ExpandNode, AggregateNode usage
3. In-place modification paradigm
4. Skip propagation (parent → children)
5. Multi-dimensional data expansion (Sample → Parts → Rubrics)
6. Topology graph and pipeline execution
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from reward import (
    # Core Protocol & Base
    PipelineData, PipelineDataBase,

    # Node types
    Node, MapNode, ExpandNode, AggregateNode,

    # Configuration
    NodeConfig, NodeType, ExecutionMetadata,

    # Topology
    TopologyGraph, PipelineExecutor,

    # Utilities
    create_simple_context
)


# ==============================================================================
# Example Data Classes (inheriting from PipelineDataBase)
# ==============================================================================

@dataclass
class SampleData(PipelineDataBase):
    """Root level: A sample with raw text"""
    raw_text: str = ""
    processed_text: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not self.data_id:
            self.data_id = f"sample_{self.sample_idx}"


@dataclass
class PartData(PipelineDataBase):
    """Second level: Parts extracted from a sample"""
    part_name: str = ""
    part_content: str = ""
    quality_score: float = 0.0

    def __post_init__(self):
        super().__post_init__()


@dataclass
class RubricData(PipelineDataBase):
    """Third level: Rubrics for evaluating a part"""
    rubric_name: str = ""
    rubric_score: float = 0.0

    def __post_init__(self):
        super().__post_init__()


# ==============================================================================
# Example Node Implementations
# ==============================================================================

class TextCleanerNode(MapNode[SampleData]):
    """Example MapNode: cleans raw text in-place"""

    async def map_one(self, data: SampleData, context: Dict[str, Any]) -> None:
        """Clean text: strip whitespace and lowercase"""
        data.processed_text = data.raw_text.strip().lower()

        # Mark as skipped if text too short
        if len(data.processed_text) < 5:
            data.mark_skipped("text_too_short", self.name)


class TextFilterNode(MapNode[SampleData]):
    """Example MapNode: filters based on rules"""

    def __init__(self, config: NodeConfig, min_length: int = 10):
        super().__init__(config)
        self.min_length = min_length

    async def map_one(self, data: SampleData, context: Dict[str, Any]) -> None:
        """Filter samples by minimum length"""
        if len(data.processed_text) < self.min_length:
            data.mark_skipped(f"too_short_{len(data.processed_text)}", self.name)


class PartExpanderNode(ExpandNode[SampleData]):
    """Example ExpandNode: splits sample into parts"""

    def expand_one(self, data: SampleData) -> List[PartData]:
        """Split text into words as parts"""
        words = data.processed_text.split()

        parts = []
        for i, word in enumerate(words):
            part = PartData(
                sample_idx=data.sample_idx,
                data_id=f"{data.data_id}/part_{i}",
                parent_id=data.data_id,
                part_name=f"word_{i}",
                part_content=word
            )
            parts.append(part)

        return parts


class RubricExpanderNode(ExpandNode[PartData]):
    """Example ExpandNode: creates rubrics for each part"""

    def expand_one(self, data: PartData) -> List[RubricData]:
        """Create rubrics for length and vowel checks"""
        rubrics = []

        # Rubric 1: Length check
        r1 = RubricData(
            sample_idx=data.sample_idx,
            data_id=f"{data.data_id}/rubric_length",
            parent_id=data.data_id,
            rubric_name="length_check"
        )
        rubrics.append(r1)

        # Rubric 2: Vowel check
        r2 = RubricData(
            sample_idx=data.sample_idx,
            data_id=f"{data.data_id}/rubric_vowel",
            parent_id=data.data_id,
            rubric_name="vowel_check"
        )
        rubrics.append(r2)

        return rubrics


class RubricScorerNode(MapNode[RubricData]):
    """Example MapNode: scores rubrics"""

    async def map_one(self, data: RubricData, context: Dict[str, Any]) -> None:
        """Score rubric based on name"""
        # Find parent PartData
        # In real implementation, you'd traverse the tree
        # For this test, we'll just assign dummy scores
        if data.rubric_name == "length_check":
            data.rubric_score = 1.0
        elif data.rubric_name == "vowel_check":
            data.rubric_score = 0.8
        else:
            data.rubric_score = 0.5


class PartAggregatorNode(AggregateNode[PartData]):
    """Example AggregateNode: aggregates rubric scores to part"""

    def aggregate_children(
        self,
        parent: PartData,
        children: List[RubricData]
    ) -> float:
        """Average rubric scores"""
        if not children:
            return 0.0

        total = sum(c.rubric_score for c in children)
        avg = total / len(children)

        # Store in parent
        parent.quality_score = avg

        return avg


class SampleAggregatorNode(AggregateNode[SampleData]):
    """Example AggregateNode: aggregates part scores to sample"""

    def aggregate_children(
        self,
        parent: SampleData,
        children: List[PartData]
    ) -> float:
        """Average part quality scores"""
        if not children:
            return 0.0

        total = sum(c.quality_score for c in children)
        return total / len(children)


# ==============================================================================
# Test Functions
# ==============================================================================

def test_pipeline_data_protocol():
    """Test PipelineDataBase implements PipelineData Protocol"""
    print("\n" + "="*70)
    print("TEST: PipelineData Protocol Implementation")
    print("="*70)

    # Create a sample
    sample = SampleData(sample_idx=0, raw_text="Hello World")

    # Test Protocol interface
    assert isinstance(sample, PipelineData), "Should implement PipelineData"
    assert sample.data_id == "sample_0"
    assert sample.sample_idx == 0
    assert not sample.is_skipped
    assert sample.parent_id is None

    # Test skip
    sample.mark_skipped("test_reason", "test_node")
    assert sample.is_skipped
    reason, node = sample.get_skip_info()
    assert reason == "test_reason"
    assert node == "test_node"

    # Test metadata
    sample.set_meta("key1", "value1")
    sample.set_meta("key2", 42)
    assert sample.get_meta("key1") == "value1"
    assert sample.get_meta("key2") == 42
    assert sample.get_meta("missing", "default") == "default"

    all_meta = sample.get_all_meta()
    assert "key1" in all_meta
    assert "key2" in all_meta

    print("\n✓ PipelineData Protocol test passed!")


def test_parent_child_relationships():
    """Test parent-child relationships"""
    print("\n" + "="*70)
    print("TEST: Parent-Child Relationships")
    print("="*70)

    # Create sample
    sample = SampleData(sample_idx=0, raw_text="test")

    # Create parts
    part1 = PartData(
        sample_idx=0,
        data_id="sample_0/part_0",
        parent_id="sample_0",
        part_name="part_0"
    )
    part2 = PartData(
        sample_idx=0,
        data_id="sample_0/part_1",
        parent_id="sample_0",
        part_name="part_1"
    )

    # Add children
    sample.add_child(part1)
    sample.add_child(part2)

    children = sample.get_children()
    assert len(children) == 2
    assert children[0].data_id == "sample_0/part_0"
    assert children[1].data_id == "sample_0/part_1"
    assert children[0].parent_id == "sample_0"

    # Test nested children
    rubric1 = RubricData(
        sample_idx=0,
        data_id="sample_0/part_0/rubric_0",
        parent_id="sample_0/part_0",
        rubric_name="test"
    )
    part1.add_child(rubric1)

    # Iterate all descendants
    all_descendants = list(sample.iter_all_descendants())
    assert len(all_descendants) == 3  # 2 parts + 1 rubric

    print("\n✓ Parent-child relationship test passed!")


async def test_map_node():
    """Test MapNode in-place modification"""
    print("\n" + "="*70)
    print("TEST: MapNode (In-place Modification)")
    print("="*70)

    # Create samples
    samples = [
        SampleData(sample_idx=0, raw_text="  Hello World  "),
        SampleData(sample_idx=1, raw_text="Test"),
        SampleData(sample_idx=2, raw_text="Hi"),  # Will be skipped (too short)
    ]

    # Create cleaner node
    cleaner = TextCleanerNode(
        NodeConfig(name="cleaner", node_type=NodeType.TRANSFORMER)
    )

    # Execute
    context = create_simple_context([])
    metadata = await cleaner.execute(samples, context)

    print(f"\nProcessed: {metadata.processed_count}")
    print(f"Skipped: {metadata.skipped_count}")
    print(f"Newly skipped: {metadata.newly_skipped_ids}")

    # Check results (in-place modification)
    assert samples[0].processed_text == "hello world"
    assert samples[1].processed_text == "test"
    assert samples[2].is_skipped
    assert samples[2].processed_text == "hi"

    reason, node = samples[2].get_skip_info()
    assert reason == "text_too_short"
    assert node == "cleaner"

    print("\n✓ MapNode test passed!")


async def test_expand_node():
    """Test ExpandNode creating children"""
    print("\n" + "="*70)
    print("TEST: ExpandNode (Creating Children)")
    print("="*70)

    # Create sample
    samples = [
        SampleData(sample_idx=0, processed_text="hello world test")
    ]

    # Create expander
    expander = PartExpanderNode(
        NodeConfig(name="expander", node_type=NodeType.EXPANDER)
    )

    # Execute
    context = create_simple_context([])
    metadata = await expander.execute(samples, context)

    print(f"\nProcessed: {metadata.processed_count}")

    # Check children were created
    children = samples[0].get_children()
    assert len(children) == 3
    assert children[0].part_content == "hello"
    assert children[1].part_content == "world"
    assert children[2].part_content == "test"
    assert children[0].parent_id == "sample_0"

    print(f"Created {len(children)} parts: {[c.part_content for c in children]}")
    print("\n✓ ExpandNode test passed!")


async def test_aggregate_node():
    """Test AggregateNode"""
    print("\n" + "="*70)
    print("TEST: AggregateNode (Aggregating Children)")
    print("="*70)

    # Create sample with parts that have rubrics
    sample = SampleData(sample_idx=0, processed_text="test")

    # Create parts
    part1 = PartData(
        sample_idx=0,
        data_id="sample_0/part_0",
        parent_id="sample_0"
    )

    # Create rubrics for part1
    rubric1 = RubricData(
        sample_idx=0,
        data_id="sample_0/part_0/rubric_0",
        parent_id="sample_0/part_0",
        rubric_score=0.8
    )
    rubric2 = RubricData(
        sample_idx=0,
        data_id="sample_0/part_0/rubric_1",
        parent_id="sample_0/part_0",
        rubric_score=1.0
    )

    part1.add_child(rubric1)
    part1.add_child(rubric2)
    sample.add_child(part1)

    # Aggregate rubrics to part
    parts = [part1]
    aggregator = PartAggregatorNode(
        NodeConfig(name="part_agg", node_type=NodeType.AGGREGATOR)
    )

    context = create_simple_context([])
    metadata = await aggregator.execute(parts, context)

    print(f"\nProcessed: {metadata.processed_count}")
    print(f"Part quality score: {part1.quality_score}")

    # Check aggregation
    assert part1.quality_score == 0.9  # (0.8 + 1.0) / 2
    assert part1.get_meta('aggregated_score') == 0.9

    print("\n✓ AggregateNode test passed!")


async def test_skip_propagation():
    """Test that skip propagates from parent to children"""
    print("\n" + "="*70)
    print("TEST: Skip Propagation")
    print("="*70)

    # Create sample
    samples = [
        SampleData(sample_idx=0, raw_text="Hi")  # Too short, will be skipped
    ]

    # Clean and expand
    cleaner = TextCleanerNode(
        NodeConfig(name="cleaner", node_type=NodeType.TRANSFORMER)
    )
    expander = PartExpanderNode(
        NodeConfig(name="expander", node_type=NodeType.EXPANDER)
    )

    context = create_simple_context([])

    # Clean (will mark as skipped)
    await cleaner.execute(samples, context)
    assert samples[0].is_skipped

    # Expand (should respect skip)
    await expander.execute(samples, context)

    # Should not create children for skipped samples
    children = samples[0].get_children()
    print(f"\nSkipped sample has {len(children)} children")

    # Expander respects skip by default, so no expansion happens
    assert len(children) == 0

    print("\n✓ Skip propagation test passed!")


async def test_full_pipeline():
    """Test complete pipeline with topology"""
    print("\n" + "="*70)
    print("TEST: Full Pipeline Execution")
    print("="*70)

    # Create samples
    samples = [
        SampleData(sample_idx=0, raw_text="  Hello World  "),
        SampleData(sample_idx=1, raw_text="Machine Learning is Great"),
        SampleData(sample_idx=2, raw_text="Hi"),  # Will be skipped
    ]

    # Build topology
    topology = TopologyGraph()

    cleaner = TextCleanerNode(
        NodeConfig(name="cleaner", node_type=NodeType.TRANSFORMER)
    )
    filter_node = TextFilterNode(
        NodeConfig(name="filter", node_type=NodeType.FILTER),
        min_length=10
    )
    expander = PartExpanderNode(
        NodeConfig(name="expander", node_type=NodeType.EXPANDER)
    )

    topology.add_node(cleaner)
    topology.add_node(filter_node)
    topology.add_node(expander)

    topology.add_edge("cleaner", "filter")
    topology.add_edge("filter", "expander")

    print("\n" + topology.visualize())

    # Execute pipeline
    executor = PipelineExecutor(topology)
    context = create_simple_context([])

    result = await executor.execute(samples, context)

    print("\n" + "-"*70)
    print("Results:")
    print("-"*70)

    for i, sample in enumerate(result):
        status = "SKIPPED" if sample.is_skipped else "OK"
        num_parts = len(sample.get_children())
        print(f"Sample {i}: {sample.raw_text[:30]:30s} | {status:8s} | Parts: {num_parts}")

        if not sample.is_skipped and num_parts > 0:
            parts = sample.get_children()
            print(f"  → Parts: {[p.part_content for p in parts[:3]]}")

    # Summary
    summary = executor.get_execution_summary()
    print("\n" + "-"*70)
    print("Execution Summary:")
    print("-"*70)
    print(f"Total nodes: {summary['total_nodes']}")
    print(f"Total time: {summary['total_time']:.3f}s")

    for node_info in summary['node_summary']:
        print(f"  {node_info['name']:15s} | "
              f"Processed: {node_info['processed']:2d} | "
              f"Skipped: {node_info['skipped']:2d} | "
              f"Time: {node_info['time']:.3f}s")

    # Verify results
    assert samples[0].processed_text == "hello world"
    assert not samples[0].is_skipped
    assert len(samples[0].get_children()) == 2  # "hello", "world"

    assert samples[1].processed_text == "machine learning is great"
    assert not samples[1].is_skipped
    assert len(samples[1].get_children()) == 4

    assert samples[2].is_skipped  # Too short
    assert len(samples[2].get_children()) == 0  # No expansion

    print("\n✓ Full pipeline test passed!")


async def test_multi_level_expansion():
    """Test multi-dimensional expansion (Sample → Parts → Rubrics)"""
    print("\n" + "="*70)
    print("TEST: Multi-Level Expansion (Sample → Parts → Rubrics)")
    print("="*70)

    # Create sample
    samples = [
        SampleData(sample_idx=0, raw_text="hello world")
    ]

    # Build topology
    topology = TopologyGraph()

    cleaner = TextCleanerNode(
        NodeConfig(name="cleaner", node_type=NodeType.TRANSFORMER)
    )
    part_expander = PartExpanderNode(
        NodeConfig(name="part_expander", node_type=NodeType.EXPANDER)
    )

    topology.add_node(cleaner)
    topology.add_node(part_expander)
    topology.add_edge("cleaner", "part_expander")

    # Execute first level
    executor = PipelineExecutor(topology)
    context = create_simple_context([])

    await executor.execute(samples, context)

    # Now expand parts to rubrics
    parts = []
    for sample in samples:
        parts.extend(sample.get_children())

    rubric_expander = RubricExpanderNode(
        NodeConfig(name="rubric_expander", node_type=NodeType.EXPANDER)
    )

    await rubric_expander.execute(parts, context)

    # Verify multi-level structure
    print("\nMulti-level structure:")
    for sample in samples:
        print(f"Sample: {sample.data_id}")
        for part in sample.get_children():
            print(f"  Part: {part.data_id} ({part.part_content})")
            for rubric in part.get_children():
                print(f"    Rubric: {rubric.data_id} ({rubric.rubric_name})")

    # Check structure
    assert len(samples[0].get_children()) == 2  # 2 parts

    part0 = samples[0].get_children()[0]
    assert len(part0.get_children()) == 2  # 2 rubrics per part

    rubric0 = part0.get_children()[0]
    assert rubric0.rubric_name in ["length_check", "vowel_check"]
    assert rubric0.parent_id == part0.data_id

    # Test iter_all_descendants
    all_descendants = list(samples[0].iter_all_descendants())
    assert len(all_descendants) == 6  # 2 parts + 4 rubrics (2 per part)

    print(f"\nTotal descendants: {len(all_descendants)}")
    print("\n✓ Multi-level expansion test passed!")


def test_topology_validation():
    """Test topology validation"""
    print("\n" + "="*70)
    print("TEST: Topology Validation")
    print("="*70)

    # Test cycle detection
    topology = TopologyGraph()

    node_a = TextCleanerNode(NodeConfig(name="a", node_type=NodeType.TRANSFORMER))
    node_b = TextCleanerNode(NodeConfig(name="b", node_type=NodeType.TRANSFORMER))
    node_c = TextCleanerNode(NodeConfig(name="c", node_type=NodeType.TRANSFORMER))

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    topology.add_edge("a", "b")
    topology.add_edge("b", "c")

    try:
        topology.add_edge("c", "a")  # Would create cycle
        assert False, "Should have detected cycle"
    except ValueError as e:
        print(f"\n✓ Cycle detected: {e}")

    # Test topological sort
    topology2 = TopologyGraph()
    topology2.add_node(node_a)
    topology2.add_node(node_b)
    topology2.add_node(node_c)
    topology2.add_edge("a", "b")
    topology2.add_edge("b", "c")

    sorted_nodes = topology2.topological_sort()
    print(f"Topological order: {sorted_nodes}")
    assert sorted_nodes == ["a", "b", "c"]

    print("\n✓ Topology validation test passed!")


# ==============================================================================
# Run All Tests
# ==============================================================================

async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("PROTOCOL-BASED PIPELINE FRAMEWORK TEST SUITE")
    print("="*70)

    # Synchronous tests
    test_pipeline_data_protocol()
    test_parent_child_relationships()
    test_topology_validation()

    # Asynchronous tests
    await test_map_node()
    await test_expand_node()
    await test_aggregate_node()
    await test_skip_propagation()
    await test_full_pipeline()
    await test_multi_level_expansion()

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
