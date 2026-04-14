"""
Test the new auto-collection and type filtering features.

演示新特性：
1. Node 通过 data_type 声明处理的数据类型
2. Executor 自动收集对应类型的节点
3. get_children(type) 按类型过滤子节点
4. mark_skipped_recursive() 递归 skip
"""
import pytest
import asyncio

# 从当前子模块导入数据和节点
from reward_framework.nodes.agentic_task_synthesis import (
    AgenticTaskSample,
    RubricCategory,
    AgenticTaskParserNode,
    RubricCategoryExpanderNode,
    RubricItemExpanderNode,
)

# 从框架导入基础设施
from reward_framework import (
    NodeConfig,
    NodeType,
    TopologyGraph,
    PipelineExecutor,
    create_simple_context,
)


# Test sample
SAMPLE_RESPONSE = """</think>

```json
{
  "task_description": "测试任务：设计在线教育平台",
  "verify_rubrics": {
    "用户体验": [
      {
        "rubric_name": "导航清晰",
        "binary_statement": "主导航不超过3层",
        "justification": ["层级过深影响效率"],
        "traceability": "UX设计原则"
      }
    ],
    "视觉设计": [
      {
        "rubric_name": "配色统一",
        "binary_statement": "使用品牌色系",
        "justification": ["保持一致性"],
        "traceability": "品牌指南"
      }
    ]
  }
}
```"""


@pytest.mark.asyncio
async def test_auto_collection():
    """测试自动收集对应类型的节点"""
    print("\n" + "="*80)
    print("TEST: Auto Node Collection by Type")
    print("="*80)

    # Create samples
    samples = [
        AgenticTaskSample(sample_idx=0, raw_response=SAMPLE_RESPONSE),
        AgenticTaskSample(sample_idx=1, raw_response=SAMPLE_RESPONSE),
    ]

    # Build topology
    topology = TopologyGraph()

    # Parser: 处理 AgenticTaskSample
    parser = AgenticTaskParserNode(
        NodeConfig(name="parser", node_type=NodeType.PARSER)
    )

    # CategoryExpander: 处理 AgenticTaskSample，展开为 RubricCategory
    category_expander = RubricCategoryExpanderNode(
        NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
    )

    # ItemExpander: 处理 RubricCategory，展开为 RubricItem
    item_expander = RubricItemExpanderNode(
        NodeConfig(name="item_expander", node_type=NodeType.EXPANDER)
    )

    # Add nodes
    topology.add_node(parser)
    topology.add_node(category_expander)
    topology.add_node(item_expander)

    # Add edges
    topology.add_edge("parser", "category_expander")
    topology.add_edge("category_expander", "item_expander")

    print("\n" + topology.visualize())

    # Execute pipeline (只传入 root samples)
    executor = PipelineExecutor(topology)
    context = create_simple_context([])

    print("\n" + "="*80)
    print("Pipeline Execution:")
    print("="*80)

    result = await executor.execute(samples, context)

    # Verify results
    print("\n" + "="*80)
    print("Results Verification:")
    print("="*80)

    for i, sample in enumerate(result):
        print(f"\nSample {i}:")
        print(f"  Status: {'SKIPPED' if sample.is_skipped else 'OK'}")

        # Get categories (使用类型过滤)
        categories = sample.get_children(RubricCategory)
        print(f"  Categories: {len(categories)}")

        for j, cat in enumerate(categories):
            print(f"    Category {j}: {cat.category_name}")
            items = cat.get_children()  # Get all children (RubricItem)
            print(f"      Items: {len(items)}")

            for k, item in enumerate(items):
                print(f"        Item {k}: {item.rubric_name}")

    # Test iter_all_descendants with type filter
    print("\n" + "="*80)
    print("Type-filtered descendants:")
    print("="*80)

    sample = result[0]

    # Get all RubricCategory descendants
    all_categories = list(sample.iter_all_descendants(RubricCategory))
    print(f"Total RubricCategory nodes: {len(all_categories)}")

    # Get all descendants (any type)
    all_descendants = list(sample.iter_all_descendants())
    print(f"Total descendants (all types): {len(all_descendants)}")

    print("\n✅ Auto collection test completed!")


@pytest.mark.asyncio
async def test_recursive_skip():
    """测试递归 skip"""
    print("\n" + "="*80)
    print("TEST: Recursive Skip")
    print("="*80)

    # Create and parse sample
    sample = AgenticTaskSample(sample_idx=0, raw_response=SAMPLE_RESPONSE)

    parser = AgenticTaskParserNode(
        NodeConfig(name="parser", node_type=NodeType.PARSER)
    )
    await parser.execute([sample], create_simple_context([]))

    # Expand to categories
    category_expander = RubricCategoryExpanderNode(
        NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
    )
    await category_expander.execute([sample], create_simple_context([]))

    # Expand to items
    categories = sample.get_children(RubricCategory)
    item_expander = RubricItemExpanderNode(
        NodeConfig(name="item_expander", node_type=NodeType.EXPANDER)
    )
    await item_expander.execute(categories, create_simple_context([]))

    print("\nBefore recursive skip:")
    print(f"  Sample skipped: {sample.is_skipped}")

    categories = sample.get_children(RubricCategory)
    print(f"  Categories: {len(categories)}")
    for cat in categories:
        items = cat.get_children()
        print(f"    {cat.category_name}: skipped={cat.is_skipped}, items={len(items)}")
        for item in items:
            print(f"      {item.rubric_name}: skipped={item.is_skipped}")

    # Now recursively skip the sample
    print("\n🔴 Calling mark_skipped_recursive()...")
    sample.mark_skipped_recursive("test_recursive_skip", "test_node")

    print("\nAfter recursive skip:")
    print(f"  Sample skipped: {sample.is_skipped}")

    categories = sample.get_children(RubricCategory)
    for cat in categories:
        items = cat.get_children()
        print(f"    {cat.category_name}: skipped={cat.is_skipped}")
        for item in items:
            print(f"      {item.rubric_name}: skipped={item.is_skipped}")

    # Verify all are skipped
    all_descendants = list(sample.iter_all_descendants())
    all_skipped = all(d.is_skipped for d in all_descendants)
    print(f"\n✅ All descendants skipped: {all_skipped}")


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("NEW FRAMEWORK FEATURES TESTS")
    print("="*80)

    await test_auto_collection()
    await test_recursive_skip()

    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)


if __name__ == '__main__':
    asyncio.run(main())
