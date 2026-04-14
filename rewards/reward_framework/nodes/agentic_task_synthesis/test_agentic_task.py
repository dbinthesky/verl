"""
Test Agentic Task Synthesis Reward.

Tests all components of the Agentic Task Synthesis implementation
with the new Protocol-based framework.
"""

import unittest
import asyncio
from typing import List

# 从当前子模块导入数据和节点
from reward_framework.nodes.agentic_task_synthesis import (
    AgenticTaskSample,
    RubricCategory,
    RubricItem,
    AgenticTaskParserNode,
    RubricCategoryExpanderNode,
    RubricItemExpanderNode,
    CategoryOrthogonalityCheckNode,
)

# 从框架导入基础设施
from reward_framework import (
    NodeConfig,
    NodeType,
    create_simple_context,
    Agent,
    AgentConfig,
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
        "rubric_name": "三重核心价值标注完整性",
        "binary_statement": "内容架构中出现「幕后揭秘」这一文字表述",
        "justification": [
          "步骤1：从task_description中提取「幕后揭秘」为强制核心价值关键词",
          "步骤2：判卷仅需在架构文本中检索该固定词汇",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点B1；预训练语料定义节目核心为幕后揭秘类时尚真人秀"
      },
      {
        "rubric_name": "创意竞技价值标注",
        "binary_statement": "内容架构中出现「创意竞技」这一文字表述",
        "justification": [
          "步骤1：从task_description中提取「创意竞技」为强制核心价值关键词",
          "步骤2：判卷仅需在架构文本中检索该固定词汇",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点B1；预训练语料定义节目为创意竞技类真人秀"
      },
      {
        "rubric_name": "行业教育价值标注",
        "binary_statement": "内容架构中出现「行业教育」这一文字表述",
        "justification": [
          "步骤1：从task_description中提取「行业教育」为强制核心价值关键词",
          "步骤2：判卷仅需在架构文本中检索该固定词汇",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点B1；预训练语料强调节目教育属性"
      },
      {
        "rubric_name": "纯娱乐化叙事规避",
        "binary_statement": "内容架构未将节目定位描述为纯娱乐化节目",
        "justification": [
          "步骤1：从task_description中提取「杜绝纯娱乐化叙事」约束",
          "步骤2：判卷仅需核查架构定位描述",
          "步骤3：仅依据文本表述即可完成判定"
        ],
        "traceability": "对应专家工作流节点B2"
      }
    ],
    "叙事认知链路构建": [
      {
        "rubric_name": "认知递进链路完整性",
        "binary_statement": "内容架构中出现「引发好奇→行业理解→情感共鸣→价值行动」完整文字序列",
        "justification": [
          "步骤1：从task_description中提取固定认知递进序列",
          "步骤2：判卷仅需检索该完整固定序列",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点H1"
      }
    ],
    "结构维度完整覆盖": [
      {
        "rubric_name": "节目传记维度覆盖",
        "binary_statement": "内容架构包含节目传记模块",
        "justification": [
          "步骤1：从task_description中提取节目传记为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "创作理念维度覆盖",
        "binary_statement": "内容架构包含创作理念模块",
        "justification": [
          "步骤1：从task_description中提取创作理念为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "幕后制作维度覆盖",
        "binary_statement": "内容架构包含幕后制作模块",
        "justification": [
          "步骤1：从task_description中提取幕后制作为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "评委维度覆盖",
        "binary_statement": "内容架构包含评委模块",
        "justification": [
          "步骤1：从task_description中提取评委为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "选手历程维度覆盖",
        "binary_statement": "内容架构包含选手历程模块",
        "justification": [
          "步骤1：从task_description中提取选手历程为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "设计挑战维度覆盖",
        "binary_statement": "内容架构包含设计挑战模块",
        "justification": [
          "步骤1：从task_description中提取设计挑战为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "行业影响维度覆盖",
        "binary_statement": "内容架构包含行业影响模块",
        "justification": [
          "步骤1：从task_description中提取行业影响为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "受众反响维度覆盖",
        "binary_statement": "内容架构包含受众反响模块",
        "justification": [
          "步骤1：从task_description中提取受众反响为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "创新引领维度覆盖",
        "binary_statement": "内容架构包含创新引领模块",
        "justification": [
          "步骤1：从task_description中提取创新引领为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "教育价值维度覆盖",
        "binary_statement": "内容架构包含教育价值模块",
        "justification": [
          "步骤1：从task_description中提取教育价值为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "全球影响力维度覆盖",
        "binary_statement": "内容架构包含全球影响力模块",
        "justification": [
          "步骤1：从task_description中提取全球影响力为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "技术融合维度覆盖",
        "binary_statement": "内容架构包含技术融合模块",
        "justification": [
          "步骤1：从task_description中提取技术融合为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "未来规划维度覆盖",
        "binary_statement": "内容架构包含未来规划模块",
        "justification": [
          "步骤1：从task_description中提取未来规划为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      },
      {
        "rubric_name": "常见问答维度覆盖",
        "binary_statement": "内容架构包含常见问答模块",
        "justification": [
          "步骤1：从task_description中提取常见问答为指定维度",
          "步骤2：判卷仅需核查模块名称存在性",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料含对应章节"
      }
    ],
    "隐性合规要素落地": [
      {
        "rubric_name": "多元文化包容要素呈现",
        "binary_statement": "内容架构中出现「多元文化包容性」这一文字表述",
        "justification": [
          "步骤1：从task_description中提取「多元文化包容性」固定表述",
          "步骤2：判卷仅需检索该固定文字",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点F4；预训练语料强调多元文化背景"
      },
      {
        "rubric_name": "可持续时尚要素呈现",
        "binary_statement": "内容架构中出现「可持续时尚」这一文字表述",
        "justification": [
          "步骤1：从task_description中提取「可持续时尚」固定表述",
          "步骤2：判卷仅需检索该固定文字",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点F2；预训练语料提及可持续时尚倡导"
      },
      {
        "rubric_name": "3D打印技术提及",
        "binary_statement": "内容架构中出现「3D打印」这一文字表述",
        "justification": [
          "步骤1：从task_description中提取「3D打印」固定技术名词",
          "步骤2：判卷仅需检索该固定文字",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点F6；预训练语料提及3D打印应用"
      },
      {
        "rubric_name": "数字设计技术提及",
        "binary_statement": "内容架构中出现「数字设计」这一文字表述",
        "justification": [
          "步骤1：从task_description中提取「数字设计」固定技术名词",
          "步骤2：判卷仅需检索该固定文字",
          "步骤3：仅依据文本存在性即可完成判定"
        ],
        "traceability": "对应专家工作流节点F6；预训练语料提及数字设计应用"
      }
    ],
    "受众分层适配落地": [
      {
        "rubric_name": "专业术语呈现",
        "binary_statement": "内容架构中出现时尚设计专业术语",
        "justification": [
          "步骤1：从task_description中提取需包含设计专业术语的约束",
          "步骤2：判卷仅需核查专业术语存在性",
          "步骤3：仅依据文本特征即可完成判定"
        ],
        "traceability": "对应专家工作流节点G3；预训练语料含设计专业内容"
      },
      {
        "rubric_name": "设计流程细节呈现",
        "binary_statement": "内容架构中出现时尚设计流程相关描述",
        "justification": [
          "步骤1：从task_description中提取需包含设计流程细节的约束",
          "步骤2：判卷仅需核查流程描述存在性",
          "步骤3：仅依据文本特征即可完成判定"
        ],
        "traceability": "对应专家工作流节点G3；预训练语料讲解设计流程"
      },
      {
        "rubric_name": "选手故事线呈现",
        "binary_statement": "内容架构中出现选手个人成长故事相关描述",
        "justification": [
          "步骤1：从task_description中提取需包含选手故事线的约束",
          "步骤2：判卷仅需核查故事描述存在性",
          "步骤3：仅依据文本特征即可完成判定"
        ],
        "traceability": "对应专家工作流节点G4；预训练语料聚焦选手故事"
      }
    ],
    "叙事逻辑无捷径合规": [
      {
        "rubric_name": "流水账形式规避",
        "binary_statement": "内容架构未采用纯行业流水账式文本结构",
        "justification": [
          "步骤1：从task_description中提取「禁止流水账」约束",
          "步骤2：判卷仅需核查文本结构形式",
          "步骤3：仅依据文本结构即可完成判定"
        ],
        "traceability": "对应专家工作流节点B2"
      },
      {
        "rubric_name": "约束推导合规性",
        "binary_statement": "内容架构中所有模块均来自task_description指定维度列表",
        "justification": [
          "步骤1：从task_description中提取固定维度列表",
          "步骤2：判卷仅需比对模块与列表一致性",
          "步骤3：仅依据文本比对即可完成判定"
        ],
        "traceability": "对应专家工作流节点H2"
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
            result = await self.parser.process_one(sample, context)

            # Check result
            self.assertFalse(result.is_skipped)

            # Check sample was modified in-place
            self.assertIn("DTI Outfit Reality Television", sample.task_description)
            self.assertIn("verify_rubrics", sample.parsed_json)
            self.assertEqual(len(sample.parsed_json["verify_rubrics"]), 6)

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
            result = await self.parser.process_one(sample, context)

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

            # Parse (concurrent)
            context = create_simple_context([])
            results = await asyncio.gather(*[self.parser.process_one(s, context) for s in samples])

            # Check results
            self.assertEqual(len([r for r in results if not r.is_skipped]), 2)  # First two succeed
            self.assertEqual(len([r.data_id for r in results if r.is_skipped]), 1)  # Third fails

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
            await parser.process_one(sample, create_simple_context([]))

            # Expand into categories
            expander = RubricCategoryExpanderNode(
                NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
            )
            result = await expander.process_one(sample, create_simple_context([]))

            # Check expansion
            self.assertFalse(result.is_skipped)
            categories = sample.get_children()
            self.assertEqual(len(categories), 6)  # 6 categories

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
            await parser.process_one(sample, create_simple_context([]))

            category_expander = RubricCategoryExpanderNode(
                NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
            )
            await category_expander.process_one(sample, create_simple_context([]))

            # Now expand categories to rubric items
            categories = sample.get_children()

            item_expander = RubricItemExpanderNode(
                NodeConfig(name="item_expander", node_type=NodeType.EXPANDER)
            )
            context = create_simple_context([])
            results = await asyncio.gather(*[item_expander.process_one(item, context) for item in categories])

            # Check expansion
            self.assertEqual(len([r for r in results if not r.is_skipped]), 6)  # 3 categories processed

            # Check first category's items
            items = categories[0].get_children()
            self.assertEqual(len(items), 1)  # 1 rubric in first category

            # Check item details
            item = items[0]
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


class TestCategoryOrthogonalityCheck(unittest.TestCase):
    """Test CategoryOrthogonalityCheckNode with real LLM."""

    # Test configuration
    TEST_MODEL = "gpt-oss-120b"
    TEST_BASE_URL = "http://10.102.215.37:28000/v1"
    TEST_API_KEY = "dummy-key"

    # Test sample with potentially overlapping categories
    SAMPLE_BAD_ORTHOGONALITY = """</think>

```json
{
  "task_description": "设计一个电商网站的性能优化方案，需要提升页面加载速度、响应时间和用户体验。",
  "verify_rubrics": {
    "用户体验优化": [
      {
        "rubric_name": "页面加载快",
        "binary_statement": "首屏加载时间<2秒",
        "justification": ["提升用户体验"],
        "traceability": "UX标准"
      }
    ],
    "性能提升": [
      {
        "rubric_name": "响应速度",
        "binary_statement": "页面响应时间<1秒",
        "justification": ["性能优化"],
        "traceability": "性能指标"
      }
    ],
    "速度优化": [
      {
        "rubric_name": "快速加载",
        "binary_statement": "资源加载快速",
        "justification": ["提升速度"],
        "traceability": "速度要求"
      }
    ]
  }
}
```"""

    def setUp(self):
        """Set up agent and nodes."""
        self.agent_config = AgentConfig(
            model=self.TEST_MODEL,
            base_url=self.TEST_BASE_URL,
            api_key=self.TEST_API_KEY,
            temperature=0.7,
            max_tokens=4096,  # 增加到4096以支持详细推理
            reasoning_effort="high"
        )
        self.agent = Agent(self.agent_config)

    def test_orthogonality_check_with_llm(self):
        """Test orthogonality check with real LLM call."""
        async def run():
            # Parse and expand sample
            sample = AgenticTaskSample(
                sample_idx=0,
                raw_response=SAMPLE_LLM_RESPONSE  # 使用真实的完整样本
            )

            parser = AgenticTaskParserNode(
                NodeConfig(name="parser", node_type=NodeType.PARSER)
            )
            await parser.process_one(sample, create_simple_context([]))

            category_expander = RubricCategoryExpanderNode(
                NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
            )
            await category_expander.process_one(sample, create_simple_context([]))

            # Get categories
            categories = sample.get_children(RubricCategory)

            print("\n" + "="*100)
            print(" " * 35 + "ORTHOGONALITY CHECK TEST")
            print("="*100)

            print(f"\n📋 任务描述:")
            print(f"    {sample.task_description}")

            print(f"\n📊 Rubric 大类 ({len(categories)} 个):")
            for i, cat in enumerate(categories):
                print(f"    {i+1}. {cat.category_name}")

            # Create orthogonality checker
            checker = CategoryOrthogonalityCheckNode(
                config=NodeConfig(
                    name="orthogonality_check",
                    node_type=NodeType.LLM_JUDGE,
                    skip_on_failure=False  # Don't skip, just record
                ),
                agent=self.agent
            )

            # Build and print prompt
            prompt = checker._build_prompt(sample.task_description, categories)

            print("\n" + "="*100)
            print(" " * 40 + "生成的 PROMPT")
            print("="*100)
            print(prompt)
            print("="*100)

            # Execute orthogonality check
            print("\n🤖 调用 LLM 进行正交性检查...")
            print(f"   Model: {self.TEST_MODEL}")
            print(f"   Endpoint: {self.TEST_BASE_URL}")
            print(f"   Reasoning effort: high")

            context = create_simple_context([])
            result = await checker.process_one(sample, context)

            print(f"\n✅ LLM 调用完成:")
            print(f"   Processed: {'Yes' if not result.is_skipped else 'No'}")
            print(f"   Skipped: {'Yes' if result.is_skipped else 'No'}")

            # Get the raw LLM response (now stored in metadata)
            raw_response = sample.get_meta('rubric_orthogonality_raw_response', None)

            print("\n" + "="*100)
            print(" " * 38 + "LLM 原始响应")
            print("="*100)

            if raw_response:
                print(raw_response)
            else:
                print("(未找到原始响应)")

            # 打印 token usage 和 finish_reason
            print("\n" + "="*100)
            print(" " * 35 + "LLM 调用详细信息")
            print("="*100)

            # 尝试获取更多调试信息
            all_meta = sample.get_all_meta()
            if 'rubric_orthogonality_raw_response' in all_meta:
                response_len = len(all_meta['rubric_orthogonality_raw_response'])
                print(f"响应长度: {response_len} 字符")

            if 'rubric_orthogonality_finish_reason' in all_meta:
                finish_reason = all_meta['rubric_orthogonality_finish_reason']
                print(f"Finish reason: {finish_reason}")
                if finish_reason == 'length':
                    print("⚠️  响应因 max_tokens 限制被截断！")

            if 'rubric_orthogonality_token_usage' in all_meta:
                usage = all_meta['rubric_orthogonality_token_usage']
                print(f"Token usage: {usage}")

            if sample.is_skipped:
                reason, node = sample.get_skip_info()
                print(f"Sample 被 skip: {reason} (at {node})")

            if 'rubric_orthogonality_error' in all_meta:
                print(f"错误信息: {all_meta['rubric_orthogonality_error']}")

            print("\n" + "="*100)
            print(" " * 40 + "解析后的结果")
            print("="*100)

            # Get parsed judgment
            judgment = sample.get_meta('rubric_orthogonality_judgment', {})
            print(f"{judgment}")

            print("\n" + "="*100)

            await self.agent.close()

            print("\n✅ 测试完成!")
            print("="*100)

        asyncio.run(run())


class TestCategoryClassificationCheck(unittest.TestCase):
    """测试 CategoryClassificationCheckNode"""

    # Test configuration
    TEST_MODEL = "gpt-oss-120b"
    TEST_BASE_URL = "http://10.102.215.37:28000/v1"
    TEST_API_KEY = "dummy-key"

    def setUp(self):
        """Set up agent and nodes."""
        self.agent_config = AgentConfig(
            model=self.TEST_MODEL,
            base_url=self.TEST_BASE_URL,
            api_key=self.TEST_API_KEY,
            temperature=0.7,
            max_tokens=4096,
            reasoning_effort="high"
        )
        self.agent = Agent(self.agent_config)

    def test_classification_check_prompt(self):
        """测试分类检查的 Prompt 生成（不调用 LLM，只打印 prompt）"""
        async def run():
            print("\n" + "="*100)
            print(" " * 30 + "CATEGORY CLASSIFICATION CHECK TEST")
            print("="*100)

            # Parse and expand sample
            sample = AgenticTaskSample(
                sample_idx=0,
                raw_response=SAMPLE_LLM_RESPONSE
            )

            parser = AgenticTaskParserNode(
                NodeConfig(name="parser", node_type=NodeType.PARSER)
            )
            await parser.process_one(sample, create_simple_context([]))

            category_expander = RubricCategoryExpanderNode(
                NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
            )
            await category_expander.process_one(sample, create_simple_context([]))

            # Expand items for all categories
            categories = sample.get_children(RubricCategory)
            item_expander = RubricItemExpanderNode(
                NodeConfig(name="item_expander", node_type=NodeType.EXPANDER)
            )
            context = create_simple_context([])
            await asyncio.gather(*[item_expander.process_one(cat, context) for cat in categories])

            print(f"\n📊 Sample 信息:")
            print(f"   Categories: {len(categories)}")
            for i, cat in enumerate(categories, 1):
                items = cat.get_children(RubricItem)
                print(f"   {i}. {cat.category_name}: {len(items)} items")

            # Create classifier node (without calling LLM)
            from reward_framework.nodes.agentic_task_synthesis.validator import CategoryClassificationCheckNode

            classifier = CategoryClassificationCheckNode(
                NodeConfig(
                    name="category_classifier",
                    node_type=NodeType.LLM_JUDGE,
                    skip_on_failure=False  # Don't skip, just record
                ),
                agent=None  # We won't call LLM
            )

            # Pick first category to check
            test_category = categories[0]
            all_category_names = [cat.category_name for cat in categories]

            print(f"\n🔍 检查的 Category:")
            print(f"   名称: {test_category.category_name}")
            print(f"   Items 数量: {len(test_category.get_children(RubricItem))}")

            # Build and print prompt
            prompt = classifier._build_prompt(
                current_category=test_category,
                all_category_names=all_category_names,
                task_description=sample.task_description
            )

            print("\n" + "="*100)
            print(" " * 40 + "生成的 PROMPT")
            print("="*100)
            print(prompt)
            print("="*100)

            print("\n✅ Prompt 生成测试完成!")
            print("   请检查 prompt 格式是否正确，是否包含所有必要信息。")
            print("="*100)

        asyncio.run(run())

    def test_classification_check_with_llm(self):
        """测试分类检查（调用真实 LLM，打印 prompt 和 response）"""
        async def run():
            # Parse and expand sample
            sample = AgenticTaskSample(
                sample_idx=0,
                raw_response=SAMPLE_LLM_RESPONSE
            )

            parser = AgenticTaskParserNode(
                NodeConfig(name="parser", node_type=NodeType.PARSER)
            )
            await parser.process_one(sample, create_simple_context([]))

            category_expander = RubricCategoryExpanderNode(
                NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
            )
            await category_expander.process_one(sample, create_simple_context([]))

            # Expand items for all categories
            categories = sample.get_children(RubricCategory)
            item_expander = RubricItemExpanderNode(
                NodeConfig(name="item_expander", node_type=NodeType.EXPANDER)
            )
            context = create_simple_context([])
            await asyncio.gather(*[item_expander.process_one(cat, context) for cat in categories])

            print("\n" + "="*100)
            print(" " * 30 + "CATEGORY CLASSIFICATION CHECK WITH LLM")
            print("="*100)

            print(f"\n📋 任务描述:")
            print(f"    {sample.task_description[:200]}...")

            print(f"\n📊 Rubric 大类 ({len(categories)} 个):")
            for i, cat in enumerate(categories, 1):
                items = cat.get_children(RubricItem)
                print(f"    {i}. {cat.category_name}: {len(items)} items")

            # Create classifier node
            from reward_framework.nodes.agentic_task_synthesis.validator import CategoryClassificationCheckNode

            classifier = CategoryClassificationCheckNode(
                NodeConfig(
                    name="category_classifier",
                    node_type=NodeType.LLM_JUDGE,
                    skip_on_failure=False,
                    skip_on_negative=False
                ),
                agent=self.agent
            )

            # Pick first category to check
            test_category = categories[0]
            all_category_names = [cat.category_name for cat in categories]

            print(f"\n🔍 检查的 Category:")
            print(f"   名称: {test_category.category_name}")
            print(f"   Items 数量: {len(test_category.get_children(RubricItem))}")

            # Build and print prompt
            prompt = classifier._build_prompt(
                current_category=test_category,
                all_category_names=all_category_names,
                task_description=sample.task_description
            )

            print("\n" + "="*100)
            print(" " * 40 + "生成的 PROMPT")
            print("="*100)
            print(prompt)
            print("="*100)

            # Execute classification check
            print("\n🤖 调用 LLM 进行分类检查...")
            print(f"   Model: {self.TEST_MODEL}")
            print(f"   Endpoint: {self.TEST_BASE_URL}")
            print(f"   Reasoning effort: high")

            # Prepare context with parent sample and task description
            check_context = create_simple_context([])
            check_context['parent_sample'] = sample
            check_context['task_description'] = sample.task_description

            result = await classifier.process_one(test_category, check_context)

            print(f"\n✅ LLM 调用完成:")
            print(f"   Processed: {'Yes' if not result.is_skipped else 'No'}")
            print(f"   Skipped: {'Yes' if result.is_skipped else 'No'}")

            # Get the raw LLM response
            raw_response = test_category.get_meta('category_classification_raw_response', None)

            print("\n" + "="*100)
            print(" " * 38 + "LLM 原始响应")
            print("="*100)

            if raw_response:
                print(raw_response)
            else:
                print("(未找到原始响应)")

            # Print token usage and finish_reason
            print("\n" + "="*100)
            print(" " * 35 + "LLM 调用详细信息")
            print("="*100)

            all_meta = test_category.get_all_meta()
            if 'category_classification_raw_response' in all_meta:
                response_len = len(all_meta['category_classification_raw_response'])
                print(f"响应长度: {response_len} 字符")

            if 'category_classification_finish_reason' in all_meta:
                finish_reason = all_meta['category_classification_finish_reason']
                print(f"Finish reason: {finish_reason}")
                if finish_reason == 'length':
                    print("⚠️  响应因 max_tokens 限制被截断！")

            if 'category_classification_token_usage' in all_meta:
                usage = all_meta['category_classification_token_usage']
                print(f"Token usage: {usage}")

            if test_category.is_skipped:
                reason, node = test_category.get_skip_info()
                print(f"Category 被 skip: {reason} (at {node})")

            if 'category_classification_error' in all_meta:
                print(f"错误信息: {all_meta['category_classification_error']}")

            print("\n" + "="*100)
            print(" " * 40 + "解析后的结果")
            print("="*100)

            # Get parsed judgment
            judgment = test_category.get_meta('category_classification_judgment', {})
            print(f"{judgment}")

            print("\n" + "="*100)

            await self.agent.close()

            print("\n✅ 测试完成!")
            print("="*100)

        asyncio.run(run())


class TestRubricRigidityCheck(unittest.TestCase):
    """测试 RubricRigidityCheckNode"""

    # Test configuration
    TEST_MODEL = "gpt-oss-120b"
    TEST_BASE_URL = "http://10.102.215.37:28000/v1"
    TEST_API_KEY = "dummy-key"

    def setUp(self):
        """Set up agent and nodes."""
        self.agent_config = AgentConfig(
            model=self.TEST_MODEL,
            base_url=self.TEST_BASE_URL,
            api_key=self.TEST_API_KEY,
            temperature=0.7,
            max_tokens=4096,
            reasoning_effort="high"
        )
        self.agent = Agent(self.agent_config)

    def test_rigidity_check_prompt(self):
        """测试刚性检查的 Prompt 生成（不调用 LLM，只打印 prompt）"""
        async def run():
            print("\n" + "="*100)
            print(" " * 35 + "RUBRIC RIGIDITY CHECK TEST")
            print("="*100)

            # Parse and expand sample to get rubric items
            sample = AgenticTaskSample(
                sample_idx=0,
                raw_response=SAMPLE_LLM_RESPONSE
            )

            parser = AgenticTaskParserNode(
                NodeConfig(name="parser", node_type=NodeType.PARSER)
            )
            await parser.process_one(sample, create_simple_context([]))

            category_expander = RubricCategoryExpanderNode(
                NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
            )
            await category_expander.process_one(sample, create_simple_context([]))

            # Expand items for all categories
            categories = sample.get_children(RubricCategory)
            item_expander = RubricItemExpanderNode(
                NodeConfig(name="item_expander", node_type=NodeType.EXPANDER)
            )
            context = create_simple_context([])
            await asyncio.gather(*[item_expander.process_one(cat, context) for cat in categories])

            # Get first rubric item for testing
            test_rubric_item = None
            for cat in categories:
                items = cat.get_children(RubricItem)
                if items:
                    test_rubric_item = items[0]
                    break

            if not test_rubric_item:
                print("❌ No rubric items found for testing")
                return

            print(f"\n📊 测试 Rubric 条目:")
            print(f"   名称: {test_rubric_item.rubric_name}")
            print(f"   二元判断: {test_rubric_item.binary_statement}")

            # Create rigidity check node
            from reward_framework.nodes.agentic_task_synthesis.validator import RubricRigidityCheckNode

            rigidity_checker = RubricRigidityCheckNode(
                NodeConfig(
                    name="rigidity_check",
                    node_type=NodeType.LLM_JUDGE,
                    skip_on_failure=False
                ),
                agent=None  # We won't call LLM
            )

            # Build and print prompt
            prompt = rigidity_checker._build_prompt(
                test_rubric_item,
                {'task_description': sample.task_description}
            )

            print("\n" + "="*100)
            print(" " * 40 + "生成的 PROMPT")
            print("="*100)
            print(prompt)
            print("="*100)

            print("\n✅ Prompt 生成测试完成!")
            print("   请检查 prompt 格式是否正确，是否包含三项检查规则。")
            print("="*100)

        asyncio.run(run())

    def test_rigidity_check_with_llm(self):
        """测试刚性检查（调用真实 LLM，打印 prompt 和 response）"""
        async def run():
            # Parse and expand sample to get rubric items
            sample = AgenticTaskSample(
                sample_idx=0,
                raw_response=SAMPLE_LLM_RESPONSE
            )

            parser = AgenticTaskParserNode(
                NodeConfig(name="parser", node_type=NodeType.PARSER)
            )
            await parser.process_one(sample, create_simple_context([]))

            category_expander = RubricCategoryExpanderNode(
                NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
            )
            await category_expander.process_one(sample, create_simple_context([]))

            # Expand items for all categories
            categories = sample.get_children(RubricCategory)
            item_expander = RubricItemExpanderNode(
                NodeConfig(name="item_expander", node_type=NodeType.EXPANDER)
            )
            context = create_simple_context([])
            await asyncio.gather(*[item_expander.process_one(cat, context) for cat in categories])

            print("\n" + "="*100)
            print(" " * 30 + "RUBRIC RIGIDITY CHECK WITH LLM (ALL ITEMS)")
            print("="*100)

            # Collect all rubric items from all categories
            all_rubric_items = []
            for cat in categories:
                items = cat.get_children(RubricItem)
                for item in items:
                    all_rubric_items.append((cat, item))

            print(f"\n📊 总共 {len(all_rubric_items)} 个 Rubric 条目:")
            for i, (cat, item) in enumerate(all_rubric_items, 1):
                print(f"   {i}. [{cat.category_name}] {item.rubric_name}")

            # Create rigidity check node
            from reward_framework.nodes.agentic_task_synthesis.validator import RubricRigidityCheckNode

            rigidity_checker = RubricRigidityCheckNode(
                NodeConfig(
                    name="rigidity_check",
                    node_type=NodeType.LLM_JUDGE,
                    skip_on_failure=False,
                    skip_on_negative=False
                ),
                agent=self.agent
            )

            # Show first item's prompt as example
            if all_rubric_items:
                first_cat, first_item = all_rubric_items[0]
                prompt = rigidity_checker._build_prompt(
                    first_item,
                    {'task_description': sample.task_description}
                )
                print("\n" + "="*100)
                print(" " * 35 + "Prompt 示例（第一条）")
                print("="*100)
                print(prompt)
                print("="*100)

            # Execute rigidity check for all items (concurrently)
            print(f"\n🤖 并发调用 LLM 进行刚性检查...")
            print(f"   Model: {self.TEST_MODEL}")
            print(f"   Endpoint: {self.TEST_BASE_URL}")
            print(f"   Reasoning effort: high")
            print(f"   并发数量: {len(all_rubric_items)} 个 items")

            check_context = create_simple_context([])
            check_context['task_description'] = sample.task_description

            # Concurrent execution
            await asyncio.gather(*[
                rigidity_checker.process_one(item, check_context)
                for cat, item in all_rubric_items
            ])

            print(f"\n✅ 所有 LLM 调用完成!")

            # Print results for all items
            print("\n" + "="*100)
            print(" " * 35 + "所有条目的检查结果")
            print("="*100)

            for i, (cat, item) in enumerate(all_rubric_items, 1):
                print(f"\n{'='*100}")
                print(f"【{i}/{len(all_rubric_items)}】{cat.category_name} - {item.rubric_name}")
                print(f"{'='*100}")

                print(f"\n📝 二元判断:")
                print(f"   {item.binary_statement}")

                # Get stored metadata
                pass_value = item.get_meta('rubric_rigidity_pass', 'N/A')
                reason_value = item.get_meta('rubric_rigidity_reason', 'N/A')
                judgment = item.get_meta('rubric_rigidity_judgment', {})

                print(f"\n✅ 判定结果:")
                print(f"   通过: {pass_value}")

                if judgment:
                    is_atomic = judgment.get('is_atomic', 'N/A')
                    is_self_contained = judgment.get('is_self_contained', 'N/A')
                    is_rigid = judgment.get('is_rigid', 'N/A')

                    print(f"\n📋 详细检查:")
                    print(f"   原子性 (is_atomic): {is_atomic}")
                    print(f"   自闭环 (is_self_contained): {is_self_contained}")
                    print(f"   二元刚性 (is_rigid): {is_rigid}")

                print(f"\n💬 理由:")
                print(f"   {reason_value}")

                # Show raw response for debugging
                raw_response = item.get_meta('rubric_rigidity_raw_response', None)
                if raw_response:
                    print(f"\n🔍 LLM 原始响应:")
                    print(f"   {raw_response[:200]}..." if len(raw_response) > 200 else f"   {raw_response}")

                # Check for errors
                if item.is_skipped:
                    skip_reason, node = item.get_skip_info()
                    print(f"\n⚠️  被跳过: {skip_reason} (at {node})")

                error = item.get_meta('rubric_rigidity_error', None)
                if error:
                    print(f"\n❌ 错误: {error}")

            # Summary statistics
            print("\n" + "="*100)
            print(" " * 40 + "汇总统计")
            print("="*100)

            total = len(all_rubric_items)
            passed = sum(1 for _, item in all_rubric_items if item.get_meta('rubric_rigidity_pass', False))
            failed = total - passed

            print(f"\n总计: {total} 个 Rubric 条目")
            print(f"✅ 通过: {passed} 个 ({passed/total*100:.1f}%)")
            print(f"❌ 未通过: {failed} 个 ({failed/total*100:.1f}%)")

            # Breakdown by violation type
            atomic_violations = sum(1 for _, item in all_rubric_items
                                   if not item.get_meta('rubric_rigidity_judgment', {}).get('is_atomic', True))
            self_contained_violations = sum(1 for _, item in all_rubric_items
                                           if not item.get_meta('rubric_rigidity_judgment', {}).get('is_self_contained', True))
            rigid_violations = sum(1 for _, item in all_rubric_items
                                  if not item.get_meta('rubric_rigidity_judgment', {}).get('is_rigid', True))

            print(f"\n违规类型统计:")
            print(f"  - 违背原子性: {atomic_violations} 个")
            print(f"  - 违背自闭环: {self_contained_violations} 个")
            print(f"  - 违背二元刚性: {rigid_violations} 个")

            print("\n" + "="*100)

            await self.agent.close()

            print("\n✅ 测试完成!")
            print("="*100)

        asyncio.run(run())


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgenticTaskSample))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgenticTaskParserNode))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRubricExpansion))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCategoryOrthogonalityCheck))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCategoryClassificationCheck))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRubricRigidityCheck))
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