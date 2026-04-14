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
        "rubric_name": "三重核心价值显性标注",
        "binary_statement": "内容架构明确标注节目「幕后揭秘、创意竞技、行业教育」三重核心价值，未偏向纯娱乐化叙事",
        "justification": [
          "步骤1：从task_description核心约束提取「锚定三重核心、杜绝纯娱乐化」强制规则",
          "步骤2：该三重核心为节目唯一核心定位，无替代表述",
          "步骤3：唯一推导出架构需显性体现该三重核心"
        ],
        "traceability": "对应专家工作流节点B1、B2；预训练语料定义节目为创意+竞争+教育融合体"
      }
    ],
    "叙事认知链路构建": [
      {
        "rubric_name": "四阶认知递进完整呈现",
        "binary_statement": "内容架构完整包含「引发好奇→行业理解→情感共鸣→价值行动」的认知递进链路",
        "justification": [
          "步骤1：从task_description叙事约束提取四阶认知递进强制要求",
          "步骤2：该链路为唯一指定叙事闭环逻辑",
          "步骤3：必然得出架构需完整覆盖该递进链路"
        ],
        "traceability": "对应专家工作流节点H1；预训练语料以幕后好奇开场构建受众认知"
      }
    ],
    "结构维度完整覆盖": [
      {
        "rubric_name": "14项指定维度无遗漏",
        "binary_statement": "内容架构完整覆盖任务指定的14个核心维度，无任何维度删减",
        "justification": [
          "步骤1：从task_description结构约束提取14项刚性维度列表",
          "步骤2：该维度列表无替换或删减空间",
          "步骤3：唯一推导出架构需包含全部14个维度"
        ],
        "traceability": "对应专家工作流节点D1；预训练语料正文包含全部对应章节"
      }
    ],
    "隐性合规要素落地": [
      {
        "rubric_name": "多元文化包容要素呈现",
        "binary_statement": "内容架构中明确体现节目多元文化背景包容性设计",
        "justification": [
          "步骤1：从task_description合规约束提取多元文化包容隐性要求",
          "步骤2：该要素为强制合规项，需独立呈现",
          "步骤3：必然得出架构需包含该要素"
        ],
        "traceability": "对应专家工作流节点F4；预训练语料强调选手多元文化背景"
      },
      {
        "rubric_name": "可持续时尚要素呈现",
        "binary_statement": "内容架构中明确体现节目对可持续时尚设计的倡导",
        "justification": [
          "步骤1：从task_description合规约束提取可持续时尚倡导要求",
          "步骤2：该要素为独立隐性约束，不可省略",
          "步骤3：唯一推导出架构需包含该要素"
        ],
        "traceability": "对应专家工作流节点F2；预训练语料行业影响章节提及可持续时尚"
      },
      {
        "rubric_name": "数字技术应用呈现",
        "binary_statement": "内容架构中明确提及3D打印、数字设计类技术融合应用",
        "justification": [
          "步骤1：从task_description合规约束提取数字技术应用要求",
          "步骤2：该技术项为独立合规要素",
          "步骤3：必然得出架构需包含该技术表述"
        ],
        "traceability": "对应专家工作流节点F6；预训练语料技术整合章节提及数字设计技术"
      },
      {
        "rubric_name": "隐性信息分布均衡",
        "binary_statement": "三类隐性合规要素在架构中均有体现，无单一要素缺失",
        "justification": [
          "步骤1：从task_description合规约束提取「隐性信息公平分布」规则",
          "步骤2：三类要素为并列强制项，需均衡呈现",
          "步骤3：唯一推导出三类要素均需存在"
        ],
        "traceability": "对应专家工作流节点F1"
      }
    ],
    "受众分层适配落地": [
      {
        "rubric_name": "专业受众技术适配",
        "binary_statement": "内容架构包含时尚设计专业术语与流程细节内容",
        "justification": [
          "步骤1：从task_description受众约束提取专业爱好者适配要求",
          "步骤2：专业技术细节为适配该群体的独立要素",
          "步骤3：必然得出架构需包含专业技术内容"
        ],
        "traceability": "对应专家工作流节点G3；预训练语料教育章节讲解设计专业流程"
      },
      {
        "rubric_name": "大众受众情感适配",
        "binary_statement": "内容架构包含选手个人成长故事线相关内容",
        "justification": [
          "步骤1：从task_description受众约束提取普通大众适配要求",
          "步骤2：选手故事线为适配该群体的独立要素",
          "步骤3：唯一推导出架构需包含该故事内容"
        ],
        "traceability": "对应专家工作流节点G4；预训练语料选手历程章节聚焦个人故事"
      }
    ],
    "叙事逻辑无捷径合规": [
      {
        "rubric_name": "非流水账架构落地",
        "binary_statement": "内容架构未采用纯行业流水账式概述形式",
        "justification": [
          "步骤1：从task_description核心约束提取「禁止流水账」刚性规则",
          "步骤2：该规则无任何豁免情形",
          "步骤3：必然得出架构需规避流水账形式"
        ],
        "traceability": "对应专家工作流节点B2"
      },
      {
        "rubric_name": "逻辑捷径完全规避",
        "binary_statement": "内容架构所有核心结论均由任务约束推导，无主观臆造内容",
        "justification": [
          "步骤1：从task_description叙事约束提取「无逻辑捷径」规则",
          "步骤2：架构内容需完全依托给定约束，不可凭空添加",
          "步骤3：唯一推导出所有关键点均需由约束触发"
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
            results = await self.parser.process_batch(samples, context)

            # Check metadata
            self.assertEqual(len([r for r in results if not r.is_skipped]), 1)
            self.assertEqual(len([r for r in results if r.is_skipped]), 0)

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
            results = await self.parser.process_batch(samples, context)

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
            results = await self.parser.process_batch(samples, context)

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
            await parser.process_batch([sample], create_simple_context([]))

            # Expand into categories
            expander = RubricCategoryExpanderNode(
                NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
            )
            results = await expander.process_batch([sample], create_simple_context([]))

            # Check expansion
            self.assertEqual(len([r for r in results if not r.is_skipped]), 1)
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
            await parser.process_batch([sample], create_simple_context([]))

            category_expander = RubricCategoryExpanderNode(
                NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
            )
            await category_expander.process_batch([sample], create_simple_context([]))

            # Now expand categories to rubric items
            categories = sample.get_children()

            item_expander = RubricItemExpanderNode(
                NodeConfig(name="item_expander", node_type=NodeType.EXPANDER)
            )
            results = await item_expander.process_batch(categories, create_simple_context([]))

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
            await parser.process_batch([sample], create_simple_context([]))

            category_expander = RubricCategoryExpanderNode(
                NodeConfig(name="category_expander", node_type=NodeType.EXPANDER)
            )
            await category_expander.process_batch([sample], create_simple_context([]))

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
            results = await checker.process_batch([sample], context)

            print(f"\n✅ LLM 调用完成:")
            print(f"   Processed: {len([r for r in results if not r.is_skipped])}")
            print(f"   Skipped: {len([r for r in results if r.is_skipped])}")
            print(f"   Execution time: {0.0:.2f}s")

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


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgenticTaskSample))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgenticTaskParserNode))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRubricExpansion))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCategoryOrthogonalityCheck))
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
