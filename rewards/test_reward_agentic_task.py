"""
Test Agentic Task Synthesis Reward.

Tests all components of the Agentic Task Synthesis implementation
with the new Protocol-based framework.
"""

import unittest
import asyncio
from typing import List
from reward import (
    # Data structures
    AgenticTaskSample,
    RubricCategory,
    RubricItem,

    # Nodes
    AgenticTaskParserNode,
    RubricCategoryExpanderNode,
    RubricItemExpanderNode,

    # Framework
    NodeConfig,
    NodeType,
    create_simple_context
)


# ==============================================================================
# Test Data Samples
# ==============================================================================

# 完整的 LLM 响应样例（包含 <think>、```json```）
SAMPLE_LLM_RESPONSE = """这是思考过程...
需要分析任务需求...
</think>

```json
{
  "task_description": "你将基于以下全量业务约束与背景信息，完成「DTI Outfit Reality Television」专业内容架构设计任务：1.背景：该节目2018年首播，为时尚竞技类真人秀，核心是新锐设计师竞技，兼具娱乐、教育与行业科普属性，拍摄于时尚之都，评委为时尚行业资深专家；2.核心约束：内容架构不可为纯行业流水账概述，必须锚定「幕后揭秘+创意竞技+行业教育」三重核心，杜绝纯娱乐化叙事；3.结构约束：必须覆盖节目传记、创作理念、幕后制作、评委、选手历程、设计挑战、行业影响、受众反响、创新引领、教育价值、全球影响力、技术融合、未来规划、常见问答14个核心维度；4.合规约束：需体现多元文化包容性、可持续时尚倡导、数字技术（3D打印/数字设计）应用三大隐性要求，信息需公平分布无缺失；5.受众约束：需同时适配时尚专业爱好者与普通大众，兼顾设计技术细节与选手情感故事线；6.叙事约束：需形成「引发好奇→行业理解→情感共鸣→价值行动」的认知递进闭环，无逻辑捷径可走。指令：输出符合所有约束的完整内容架构逻辑方案，确保每一部分均服务于节目核心价值主张。",
  "verify_rubrics": {
    "核心价值主张锚定": [
      {
        "rubric_name": "三重核心价值明确标注",
        "binary_statement": "内容架构明确锚定节目「幕后揭秘、创意竞技、行业教育」三重核心价值，未偏向纯娱乐化叙事",
        "justification": [
          "步骤1：从task_description核心约束中提取「锚定幕后揭秘+创意竞技+行业教育三重核心、杜绝纯娱乐化」的强制规则",
          "步骤2：结合内容架构设计的任务目标，该三重核心为唯一核心定位方向",
          "步骤3：唯一必然推导出架构需明确体现该三重核心价值"
        ],
        "traceability": "对应专家工作流节点B1（确立节目独特性：教育性vs纯娱乐化）、B2（排除干扰项：非流水账行业概述）；预训练语料明确节目为娱乐+教育融合的时尚竞技真人秀"
      }
    ],
    "叙事锚点与认知递进构建": [
      {
        "rubric_name": "四阶认知递进链路完整",
        "binary_statement": "内容架构完整呈现「引发好奇→行业理解→情感共鸣→价值行动」的认知递进链路",
        "justification": [
          "步骤1：从task_description叙事约束中提取四阶认知递进的强制要求",
          "步骤2：该叙事链路为唯一指定闭环逻辑，无替代方案",
          "步骤3：必然得出架构需完整包含该四阶递进链路"
        ],
        "traceability": "对应专家工作流节点H1（检查认知递进：从好奇→理解→共鸣→行动）；预训练语料以幕后好奇开场构建受众认知"
      }
    ],
    "强制结构维度覆盖": [
      {
        "rubric_name": "14项核心维度无遗漏",
        "binary_statement": "内容架构完整覆盖节目传记、创作理念、幕后制作、评委、选手历程、设计挑战、行业影响、受众反响、创新引领、教育价值、全球影响力、技术融合、未来规划、常见问答14个指定维度",
        "justification": [
          "步骤1：从task_description结构约束中提取14项强制覆盖维度列表",
          "步骤2：该维度列表为刚性结构要求，无删减或替换空间",
          "步骤3：唯一推导出架构需包含全部14个维度"
        ],
        "traceability": "对应专家工作流节点D1（罗列必要维度列表）；预训练语料正文包含全部14个对应章节"
      }
    ]
  }
}
```"""

# 带有 EOS 标记的样例
SAMPLE_WITH_EOS = SAMPLE_LLM_RESPONSE + "\n<|im_end|>"

# 无 <think> 标签的样例
SAMPLE_NO_THINK = """```json
{
  "task_description": "简单任务描述",
  "verify_rubrics": {
    "测试类别": [
      {
        "rubric_name": "测试检查点",
        "binary_statement": "测试语句",
        "justification": ["步骤1"],
        "traceability": "测试溯源"
      }
    ]
  }
}
```"""

# 格式错误的样例（缺少必填字段）
SAMPLE_INVALID_SCHEMA = """```json
{
  "task_description": "任务描述",
  "verify_rubrics": {
    "类别": [
      {
        "rubric_name": "名称"
      }
    ]
  }
}
```"""


# ==============================================================================
# Utility Functions
# ==============================================================================

def print_tree_structure(samples: List[AgenticTaskSample], max_depth: int = 3):
    """Print complete tree structure of samples with children.

    Args:
        samples: List of AgenticTaskSample
        max_depth: Maximum depth to print
    """
    print("\n" + "="*80)
    print("TREE STRUCTURE")
    print("="*80)

    for sample in samples:
        # Level 1: Sample
        status = "[SKIPPED]" if sample.is_skipped else "[OK]"
        print(f"\n{status} Sample: {sample.data_id}")
        print(f"  └─ task_desc: {sample.task_description[:80]}...")
        print(f"  └─ raw_response_len: {len(sample.raw_response)}")

        if sample.is_skipped:
            reason, node = sample.get_skip_info()
            print(f"  └─ skip_reason: {reason} (at {node})")
            continue

        # Level 2: Categories
        categories = sample.get_children()
        print(f"  └─ categories: {len(categories)}")

        for i, category in enumerate(categories):
            cat_status = "[SKIPPED]" if category.is_skipped else "[OK]"
            print(f"\n     {cat_status} Category {i}: {category.data_id}")
            print(f"        ├─ name: {category.category_name}")
            print(f"        ├─ parent_id: {category.parent_id}")

            if category.is_skipped:
                reason, node = category.get_skip_info()
                print(f"        └─ skip_reason: {reason}")
                continue

            # Level 3: Rubric Items
            items = category.get_children()
            print(f"        └─ rubric_items: {len(items)}")

            for j, item in enumerate(items):
                item_status = "[SKIPPED]" if item.is_skipped else "[OK]"
                print(f"\n           {item_status} RubricItem {j}: {item.data_id}")
                print(f"              ├─ name: {item.rubric_name}")
                print(f"              ├─ parent_id: {item.parent_id}")
                print(f"              ├─ binary_statement: {item.binary_statement[:60]}...")
                print(f"              ├─ justification_steps: {len(item.justification)}")
                print(f"              ├─ traceability: {item.traceability[:60]}...")
                print(f"              └─ judge_score: {item.judge_score}")

    print("\n" + "="*80)


# ==============================================================================
# Test Cases
# ==============================================================================

class TestAgenticTaskSample(unittest.TestCase):
    """Test AgenticTaskSample data structure."""

    def test_create_sample(self):
        """Test creating AgenticTaskSample instance."""
        sample = AgenticTaskSample(
            sample_idx=0,
            raw_response="test response",
            task_description="Test task"
        )

        self.assertEqual(sample.sample_idx, 0)
        self.assertEqual(sample.data_id, "sample_0")
        self.assertEqual(sample.raw_response, "test response")
        self.assertEqual(sample.task_description, "Test task")

    def test_parent_child_relationships(self):
        """Test parent-child relationships."""
        sample = AgenticTaskSample(sample_idx=0)

        category = RubricCategory(
            sample_idx=0,
            data_id="sample_0/category_0",
            parent_id="sample_0",
            category_name="Test Category"
        )

        sample.add_child(category)

        self.assertEqual(len(sample.get_children()), 1)
        self.assertEqual(category.parent_id, sample.data_id)


class TestAgenticTaskParserNode(unittest.TestCase):
    """Test AgenticTaskParserNode."""

    def setUp(self):
        """Set up parser node."""
        self.parser = AgenticTaskParserNode(
            NodeConfig(name="parser", node_type=NodeType.PARSER)
        )

    def test_postprocess_solution(self):
        """Test removing EOS markers."""
        # Test with <|im_end|>
        text = "Hello world<|im_end|>extra"
        result = self.parser._postprocess_solution(text)
        self.assertEqual(result, "Hello world")

        # Test with <｜end▁of▁sentence｜>
        text = "Test<｜end▁of▁sentence｜>extra"
        result = self.parser._postprocess_solution(text)
        self.assertEqual(result, "Test")

        # Test without markers
        text = "No markers here"
        result = self.parser._postprocess_solution(text)
        self.assertEqual(result, "No markers here")

    def test_extract_json_from_response(self):
        """Test extracting JSON from LLM response."""
        # Test with <think> tag
        json_str = self.parser._extract_json_from_response(SAMPLE_LLM_RESPONSE)
        self.assertIn("task_description", json_str)
        self.assertIn("verify_rubrics", json_str)
        self.assertNotIn("<think>", json_str)
        self.assertNotIn("```", json_str)

        # Test without <think> tag
        json_str = self.parser._extract_json_from_response(SAMPLE_NO_THINK)
        self.assertIn("task_description", json_str)

        # Test with EOS marker
        json_str = self.parser._extract_json_from_response(SAMPLE_WITH_EOS)
        self.assertNotIn("<|im_end|>", json_str)

    def test_extract_json_failure(self):
        """Test JSON extraction failure."""
        with self.assertRaises(ValueError):
            self.parser._extract_json_from_response("No JSON here")

    def test_validate_schema_valid(self):
        """Test schema validation with valid data."""
        valid_data = {
            "task_description": "Test task",
            "verify_rubrics": {
                "类别1": [
                    {
                        "rubric_name": "检查点1",
                        "binary_statement": "语句1",
                        "justification": ["步骤1", "步骤2"],
                        "traceability": "溯源1"
                    }
                ]
            }
        }

        self.assertTrue(self.parser._validate_schema(valid_data))

    def test_validate_schema_invalid(self):
        """Test schema validation with invalid data."""
        # Missing task_description
        invalid_data1 = {
            "verify_rubrics": {}
        }
        self.assertFalse(self.parser._validate_schema(invalid_data1))

        # Wrong rubrics structure
        invalid_data2 = {
            "task_description": "Test",
            "verify_rubrics": "not a dict"
        }
        self.assertFalse(self.parser._validate_schema(invalid_data2))

    def test_parse_success(self):
        """Test parsing single response successfully."""
        async def run():
            # Create sample
            sample = AgenticTaskSample(
                sample_idx=0,
                raw_response=SAMPLE_LLM_RESPONSE
            )

            samples = [sample]

            # Parse
            context = create_simple_context([])
            metadata = await self.parser.execute(samples, context)

            # Check metadata
            self.assertEqual(metadata.processed_count, 1)
            self.assertEqual(metadata.skipped_count, 0)

            # Check sample was modified in-place
            self.assertIn("DTI Outfit Reality Television", sample.task_description)
            self.assertIn("verify_rubrics", sample.parsed_json)
            self.assertEqual(len(sample.parsed_json["verify_rubrics"]), 3)

            # Print tree structure
            print_tree_structure(samples)

        asyncio.run(run())

    def test_parse_failure(self):
        """Test parsing failure with invalid input."""
        async def run():
            # Create sample with invalid schema
            sample = AgenticTaskSample(
                sample_idx=0,
                raw_response=SAMPLE_INVALID_SCHEMA
            )

            samples = [sample]

            # Parse (should fail and mark as skipped)
            context = create_simple_context([])
            metadata = await self.parser.execute(samples, context)

            # Check it was skipped
            self.assertTrue(sample.is_skipped)
            reason, node = sample.get_skip_info()
            self.assertIn("parse_error", reason)
            self.assertEqual(node, "parser")

        asyncio.run(run())

    def test_parse_batch(self):
        """Test batch parsing."""
        async def run():
            # Create batch
            samples = [
                AgenticTaskSample(sample_idx=0, raw_response=SAMPLE_LLM_RESPONSE),
                AgenticTaskSample(sample_idx=1, raw_response=SAMPLE_NO_THINK),
                AgenticTaskSample(sample_idx=2, raw_response=SAMPLE_INVALID_SCHEMA),  # Should fail
            ]

            # Parse
            context = create_simple_context([])
            metadata = await self.parser.execute(samples, context)

            # Check results
            self.assertEqual(metadata.processed_count, 2)  # First two succeed
            self.assertEqual(len(metadata.newly_skipped_ids), 1)  # Third fails

            # Check individual samples
            self.assertFalse(samples[0].is_skipped)
            self.assertFalse(samples[1].is_skipped)
            self.assertTrue(samples[2].is_skipped)

            print_tree_structure(samples)

        asyncio.run(run())


class TestRubricExpansion(unittest.TestCase):
    """Test RubricCategory and RubricItem expansion."""

    def test_category_expansion(self):
        """Test expanding sample into categories."""
        async def run():
            # Create and parse sample
            sample = AgenticTaskSample(
                sample_idx=0,
                raw_response=SAMPLE_LLM_RESPONSE
            )

            # Parse first
            parser = AgenticTaskParserNode(
                NodeConfig(name="parser", node_type=NodeType.PARSER)
            )
            await parser.execute([sample], create_simple_context([]))

            # Expand into categories
            expander = RubricCategoryExpanderNode(
                NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
            )
            metadata = await expander.execute([sample], create_simple_context([]))

            # Check expansion
            self.assertEqual(metadata.processed_count, 1)
            categories = sample.get_children()
            self.assertEqual(len(categories), 3)  # 3 categories

            # Check category details
            self.assertEqual(categories[0].category_name, "核心价值主张锚定")
            self.assertEqual(categories[1].category_name, "叙事锚点与认知递进构建")
            self.assertEqual(categories[2].category_name, "强制结构维度覆盖")

            # Check parent relationship
            self.assertEqual(categories[0].parent_id, sample.data_id)

            print_tree_structure([sample])

        asyncio.run(run())

    def test_rubric_item_expansion(self):
        """Test expanding categories into rubric items."""
        async def run():
            # Setup: Parse and expand to categories
            sample = AgenticTaskSample(
                sample_idx=0,
                raw_response=SAMPLE_LLM_RESPONSE
            )

            parser = AgenticTaskParserNode(
                NodeConfig(name="parser", node_type=NodeType.PARSER)
            )
            await parser.execute([sample], create_simple_context([]))

            category_expander = RubricCategoryExpanderNode(
                NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
            )
            await category_expander.execute([sample], create_simple_context([]))

            # Now expand categories to rubric items
            categories = sample.get_children()

            item_expander = RubricItemExpanderNode(
                NodeConfig(name="item_expander", node_type=NodeType.EXPANDER)
            )
            metadata = await item_expander.execute(categories, create_simple_context([]))

            # Check expansion
            self.assertEqual(metadata.processed_count, 3)  # 3 categories processed

            # Check first category's items
            items = categories[0].get_children()
            self.assertEqual(len(items), 1)  # 1 rubric in first category

            # Check item details
            item = items[0]
            self.assertEqual(item.rubric_name, "三重核心价值明确标注")
            self.assertIn("幕后揭秘", item.binary_statement)
            self.assertEqual(len(item.justification), 3)
            self.assertEqual(item.parent_id, categories[0].data_id)

            # Print full tree
            print_tree_structure([sample])

            # Test iter_all_descendants
            all_descendants = list(sample.iter_all_descendants())
            print(f"\n总共 {len(all_descendants)} 个后代节点:")
            print(f"  - 3 个 Category")
            print(f"  - {len(all_descendants) - 3} 个 RubricItem")

        asyncio.run(run())


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgenticTaskSample))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgenticTaskParserNode))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRubricExpansion))
    return suite


if __name__ == '__main__':
    print("\n" + "="*80)
    print(" "*20 + "AGENTIC TASK SYNTHESIS TESTS")
    print("="*80)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())

    print("\n" + "="*80)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {len(result.failures + result.errors)} TEST(S) FAILED")
    print("="*80)
