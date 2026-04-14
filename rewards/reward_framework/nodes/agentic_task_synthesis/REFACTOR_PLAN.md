# 重构计划：从批处理到单条处理

## 背景

**问题**：verl 0.7.0 使用单条数据接口，而我们的框架是批处理模式

```python
# verl 0.7.0 接口
def compute_score(data_source: str, solution_str: str,
                  ground_truth: str, extra_info: dict) -> dict:
    """处理单条数据，返回单条结果"""
    pass

# 我们的框架（当前）
class MapNode:
    async def execute(self, batch: List[PipelineDataBase], context):
        """处理批量数据"""
        pass
```

**核心冲突**：
1. verl 每次调用传入单条数据
2. verl 通过异步并发控制多条数据处理
3. 我们的框架假设批处理输入

---

## 重构方案：单条处理 + 可选批处理

### 设计原则

1. **单条优先**：节点默认处理单条数据
2. **可选批处理**：特殊节点（如 LLM）可以优化批处理
3. **向后兼容**：保留批处理接口，内部转换为单条

### 新架构

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Node(ABC):
    """节点基类（单条处理）"""

    @abstractmethod
    async def process_one(self, data: PipelineDataBase, context: Dict[str, Any]) -> PipelineDataBase:
        """处理单条数据

        Args:
            data: 单条数据
            context: 执行上下文

        Returns:
            处理后的数据（可以原地修改，也可以返回新对象）
        """
        pass

    async def process_batch(self, batch: List[PipelineDataBase], context: Dict[str, Any]) -> List[PipelineDataBase]:
        """批处理（默认实现：并发处理单条）

        子类可以重写以实现批处理优化（如 LLM 批量 API）
        """
        return await asyncio.gather(*[self.process_one(data, context) for data in batch])


class MapNode(Node):
    """Map 节点：处理数据但不改变结构"""

    @abstractmethod
    async def map_one(self, data: PipelineDataBase, context: Dict[str, Any]) -> None:
        """处理单条数据（原地修改）"""
        pass

    async def process_one(self, data: PipelineDataBase, context: Dict[str, Any]) -> PipelineDataBase:
        # Skip 逻辑
        if data.is_skipped and self.config.respect_skip:
            return data

        try:
            await self.map_one(data, context)
        except Exception as e:
            if self.config.skip_on_failure:
                data.mark_skipped(f"map_error: {e}", self.name)

        return data


class ExpandNode(Node):
    """Expand 节点：一条数据展开为多条"""

    @abstractmethod
    async def expand_one(self, data: PipelineDataBase, context: Dict[str, Any]) -> List[PipelineDataBase]:
        """展开单条数据"""
        pass

    async def process_one(self, data: PipelineDataBase, context: Dict[str, Any]) -> PipelineDataBase:
        if data.is_skipped and self.config.respect_skip:
            return data

        children = await self.expand_one(data, context)
        for child in children:
            data.add_child(child)

        return data


class LLMNode(MapNode):
    """LLM 节点：支持批处理优化"""

    async def map_one(self, data: PipelineDataBase, context: Dict[str, Any]):
        prompt = self._build_prompt(data)

        # 通过限流器调用 LLM
        async with context['llm_limiter']:
            response = await self.agent.generate(prompt)

        self._process_result(data, response)

    async def process_batch(self, batch: List[PipelineDataBase], context: Dict[str, Any]):
        """批处理优化：利用 LLM 批量 API"""
        # 过滤 skipped
        active = [d for d in batch if not (d.is_skipped and self.config.respect_skip)]
        skipped = [d for d in batch if d.is_skipped and self.config.respect_skip]

        if not active:
            return batch

        # 批量生成 prompts
        prompts = [self._build_prompt(d) for d in active]

        # 批量调用 LLM（带限流）
        async with context['llm_limiter']:
            responses = await self.agent.generate_batch(prompts)

        # 批量处理结果
        for data, response in zip(active, responses):
            self._process_result(data, response)

        return batch
```

### Pipeline 改造

```python
class Pipeline(ABC):
    """Pipeline 基类"""

    @abstractmethod
    async def run_one(self, data: PipelineDataBase, context: Dict[str, Any]) -> PipelineDataBase:
        """处理单条数据（符合 verl 接口）"""
        pass

    async def run_batch(self, batch: List[PipelineDataBase], context: Dict[str, Any]) -> List[PipelineDataBase]:
        """批量处理（并发调用 run_one）"""
        return await asyncio.gather(*[self.run_one(data, context) for data in batch])


class AgenticTaskPipeline(Pipeline):
    """Agentic Task 处理流程"""

    def __init__(self):
        self.parser = AgenticTaskParserNode(...)
        self.category_expander = RubricCategoryExpanderNode(...)
        self.item_expander = RubricItemExpanderNode(...)
        self.orthogonality_checker = CategoryOrthogonalityCheckNode(...)

    async def run_one(self, data: AgenticTaskSample, context: Dict[str, Any]) -> AgenticTaskSample:
        """处理单条样本"""
        # Step 1: 解析
        await self.parser.process_one(data, context)

        # Step 2: 展开 categories
        await self.category_expander.process_one(data, context)

        # Step 3: 并行处理
        categories = data.get_children(RubricCategory)

        await asyncio.gather(
            # 展开 items（批处理优化）
            self.item_expander.process_batch(categories, context),
            # 检查正交性（单条）
            self.orthogonality_checker.process_one(data, context),
        )

        return data
```

### verl 接口适配

```python
# 全局 pipeline 实例（复用）
_pipeline = AgenticTaskPipeline()
_context = create_simple_context(llm_concurrency=10)

async def compute_score(data_source: str, solution_str: str,
                       ground_truth: str, extra_info: dict) -> dict:
    """verl 0.7.0 接口（单条数据）"""
    # 创建单条数据
    sample = AgenticTaskSample(
        sample_idx=0,
        raw_response=solution_str
    )
    sample.set_meta('data_source', data_source)
    sample.set_meta('ground_truth', ground_truth)
    sample.set_meta('extra_info', extra_info)

    # 处理单条数据
    result = await _pipeline.run_one(sample, _context)

    # 提取分数
    return {
        'score': result.get_meta('final_score', 0.0),
        'orthogonality_score': result.get_meta('rubric_orthogonality_score', 0.0),
        'details': result.metadata,
    }

# 如果需要批量处理（测试/离线评估）
async def compute_scores_batch(samples: List[dict]) -> List[dict]:
    """批量处理接口"""
    data_batch = [
        AgenticTaskSample(sample_idx=i, raw_response=s['solution_str'])
        for i, s in enumerate(samples)
    ]

    # 批量并发处理
    results = await _pipeline.run_batch(data_batch, _context)

    return [
        {
            'score': r.get_meta('final_score', 0.0),
            'details': r.metadata,
        }
        for r in results
    ]
```

---

## 重构步骤

### Phase 1: 核心基类改造（0.5天）

1. 修改 `Node` 基类：
   - 添加 `process_one()` 抽象方法
   - 添加 `process_batch()` 默认实现
   - 保留 `execute()` 向后兼容

2. 修改 `MapNode` / `ExpandNode` / `AggregateNode`：
   - 实现 `process_one()`
   - 原有 `map_one()` / `expand_one()` 等保持不变
   - `execute()` 内部调用 `process_batch()`

**文件**：
- `reward_framework/nodes/base.py`

### Phase 2: Pipeline 基类（0.5天）

1. 创建 `Pipeline` 抽象基类：
   - `run_one()` - 处理单条
   - `run_batch()` - 批量并发

2. 实现 `AgenticTaskPipeline`

**文件**：
- `reward_framework/pipeline/base.py`
- `reward_framework/nodes/agentic_task_synthesis/pipeline.py`

### Phase 3: verl 接口适配（0.5天）

1. 创建 verl 接口层
2. 添加测试验证单条/批量模式

**文件**：
- `reward_framework/integrations/verl.py`
- `reward_framework/integrations/test_verl_interface.py`

### Phase 4: LLM 批处理优化（可选，1天）

1. 为 LLM 节点实现 `process_batch()` 优化
2. 添加批量 API 支持

**文件**：
- `reward_framework/agent/agent.py` (添加 `generate_batch()`)
- `reward_framework/nodes/agentic_task_synthesis/validator.py` (重写 `process_batch()`)

---

## 总时间估算

| Phase | 任务 | 时间 |
|-------|------|------|
| 1 | 核心基类改造 | 0.5 天 |
| 2 | Pipeline 基类 | 0.5 天 |
| 3 | verl 接口适配 | 0.5 天 |
| 4 | LLM 批处理优化（可选） | 1 天 |
| **总计** | | **1.5-2.5 天** |

---

## 向后兼容

保留现有 `execute(batch)` 接口：

```python
class Node:
    async def execute(self, batch: List[PipelineDataBase], context):
        """向后兼容接口"""
        return await self.process_batch(batch, context)
```

现有测试继续工作，新代码使用 `process_one()` / `process_batch()`。

---

## 优势

1. ✅ **符合 verl 0.7.0 接口**：单条数据处理
2. ✅ **灵活并发控制**：verl 控制或我们的 `run_batch` 控制
3. ✅ **可选批处理优化**：LLM 节点可以批量调用
4. ✅ **向后兼容**：保留 `execute()` 接口
5. ✅ **清晰架构**：单条处理为主，批处理为辅

---

**文档创建时间**: 2026-04-14
**状态**: 设计方案待确认
