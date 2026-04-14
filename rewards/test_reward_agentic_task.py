"""
Test Agentic Task Synthesis Reward.

Tests all components of the Agentic Task Synthesis reward implementation.
"""

import unittest
import asyncio
from reward import (
    ParsedAgenticTask,
    AgenticTaskParserNode,
    NodeConfig,
    NodeType,
    create_context
)


# ========== 共用测试样例 ==========

# 完整的 LLM 响应样例（包含 <think>、```json```）
SAMPLE_LLM_RESPONSE = """<think>
这是思考过程...
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


# ========== 测试用例 ==========

class TestParsedAgenticTask(unittest.TestCase):
    """Test ParsedAgenticTask data structure."""

    def test_create_parsed_task(self):
        """Test creating ParsedAgenticTask instance."""
        task = ParsedAgenticTask(
            task_description="Create a travel planning agent",
            subtasks=["Book flights", "Find hotels", "Create itinerary"],
            expected_output="JSON with travel plan",
            raw_text="Task: Create a travel planning agent..."
        )

        self.assertEqual(task.task_description, "Create a travel planning agent")
        self.assertEqual(len(task.subtasks), 3)
        self.assertEqual(task.expected_output, "JSON with travel plan")
        self.assertIn("travel planning", task.raw_text)

    def test_parsed_task_immutable(self):
        """Test that ParsedAgenticTask is immutable."""
        task = ParsedAgenticTask(
            task_description="Test task",
            subtasks=[],
            expected_output="Output",
            raw_text="Raw"
        )

        # Should not be able to modify
        with self.assertRaises(Exception):
            task.task_description = "Modified"

    def test_empty_subtasks(self):
        """Test task with no subtasks."""
        task = ParsedAgenticTask(
            task_description="Simple task",
            subtasks=[],
            expected_output="Result",
            raw_text="Simple task"
        )

        self.assertEqual(len(task.subtasks), 0)
        self.assertEqual(task.task_description, "Simple task")


class TestAgenticTaskParserNode(unittest.TestCase):
    """Test AgenticTaskParserNode."""

    def setUp(self):
        """Set up parser node."""
        self.parser = AgenticTaskParserNode(
            NodeConfig(name="parser", node_type=NodeType.PARSER)
        )

    def test_parser_node_creation(self):
        """Test creating parser node."""
        self.assertEqual(self.parser.config.name, "parser")
        self.assertEqual(self.parser.config.node_type, NodeType.PARSER)

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

        # Test with <|endoftext|>
        text = "Content<|endoftext|>extra"
        result = self.parser._postprocess_solution(text)
        self.assertEqual(result, "Content")

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

        # Missing rubric fields
        invalid_data3 = {
            "task_description": "Test",
            "verify_rubrics": {
                "category": [
                    {"rubric_name": "test"}  # Missing other fields
                ]
            }
        }
        self.assertFalse(self.parser._validate_schema(invalid_data3))

    def test_parse_single_success(self):
        """Test parsing single response successfully."""
        parsed = self.parser._parse_single(SAMPLE_LLM_RESPONSE)

        self.assertIsNotNone(parsed)
        self.assertIn("DTI Outfit Reality Television", parsed.task_description)
        self.assertIn("核心价值主张锚定", parsed.subtasks)
        self.assertIn("叙事锚点与认知递进构建", parsed.subtasks)
        self.assertIn("强制结构维度覆盖", parsed.subtasks)
        self.assertIn("验证标准包含", parsed.expected_output)

    def test_parse_single_failure(self):
        """Test parsing failure with invalid input."""
        parsed = self.parser._parse_single(SAMPLE_INVALID_SCHEMA)
        self.assertIsNone(parsed)

        parsed = self.parser._parse_single("No JSON at all")
        self.assertIsNone(parsed)

    def test_execute_batch(self):
        """Test batch parsing."""
        async def run():
            batch_inputs = [
                SAMPLE_LLM_RESPONSE,
                SAMPLE_NO_THINK,
                SAMPLE_INVALID_SCHEMA,  # Should fail
                None  # Skip
            ]

            context = create_context(batch_inputs)
            result = await self.parser.execute(batch_inputs, context)

            # Check results
            self.assertEqual(len(result.outputs), 4)
            self.assertIsNotNone(result.outputs[0])  # First should succeed
            self.assertIsNotNone(result.outputs[1])  # Second should succeed
            self.assertIsNone(result.outputs[2])     # Third should fail
            self.assertIsNone(result.outputs[3])     # Fourth is None

            # Check metadata
            self.assertEqual(result.metadata["parser_success"], 2)

            # Check first parsed result
            first = result.outputs[0]
            self.assertIsInstance(first, ParsedAgenticTask)
            self.assertEqual(len(first.subtasks), 3)

        asyncio.run(run())

    def test_execute_empty_batch(self):
        """Test parsing empty batch."""
        async def run():
            batch_inputs = []
            context = create_context(batch_inputs)
            result = await self.parser.execute(batch_inputs, context)

            self.assertEqual(len(result.outputs), 0)
            self.assertEqual(result.metadata["parser_success"], 0)

        asyncio.run(run())


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        TestParsedAgenticTask
    ))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        TestAgenticTaskParserNode
    ))
    return suite


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" "*15 + "AGENTIC TASK SYNTHESIS TESTS")
    print("="*70)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())

    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {len(result.failures + result.errors)} TEST(S) FAILED")
    print("="*70)
