# Pipeline 架构重构设计

## 背景与动机

当前使用 TopologyGraph + Executor 的自动 DAG 执行模式存在以下问题：
1. 节点只能串行执行，无法利用并发
2. 自动调度缺乏灵活性，用户无法控制执行顺序
3. LLM 调用分散在各个节点，无法统一管理并发

**目标**：
- 替换 TopologyGraph/Executor 为 Pipeline 模式（用户手动编排）
- 实现 LLM 调用的并发控制
- 提升执行性能 5-10 倍

---

## 设计演进

### 方案 1: 统一 LLM 队列（复杂，已放弃）

**核心思路**：
- 所有节点将 LLM 请求提交到统一队列
- Worker 池根据不同 LLM 配置创建消费者
- 批量处理和智能调度

**复杂点**：
1. 多个 producer（节点）同时生产请求
2. 请求归属和顺序保证
3. 批处理调度策略（哪些请求可以合并？）
4. 回调机制设计

**代码示例**：
```python
# 每个节点配置 LLM
node = CategoryOrthogonalityCheckNode(
    config=NodeConfig(...),
    llm_config=LLMConfig(model="gpt-4", base_url="...", max_workers=5)
)

# 批量提交
requests = [LLMRequest(prompt=..., config=self.llm_config) for data in batch]
results = await llm_queue.submit_batch(requests)

# Worker 池
queue = LLMQueue()
queue.register_llm_config(llm_config)  # 创建对应数量的 worker
```

**问题**：
- 实现复杂度高（3-5天）
- 多 producer 情况下批处理逻辑复杂
- 请求时序和归属难以管理

---

### 方案 2: Pipeline + 全局限流器（简化，推荐）

**核心思路**：
1. Pipeline 基类：用户手动编排节点执行顺序（获得并行控制权）
2. 全局限流器：使用 `asyncio.Semaphore` 控制 LLM 并发数
3. 节点直接调用 LLM，但走统一限流器

**架构**：
```
Pipeline (用户编排)
  ├─ Node 1 ──┐
  ├─ Node 2 ──┼──> LLM Limiter (Semaphore) ──> LLM API
  └─ Node 3 ──┘
```

**代码示例**：

```python
# 1. 全局限流器（放在 context 中）
context = {
    'llm_limiter': asyncio.Semaphore(10)  # 最多 10 个并发 LLM 调用
}

# 2. 节点调用 LLM 时自动限流
class CategoryOrthogonalityCheckNode(MapNode):
    async def map_one(self, data: AgenticTaskSample, context: Dict[str, Any]):
        prompt = self._build_prompt(data)

        # 通过限流器调用 LLM
        async with context['llm_limiter']:
            response = await self.agent.generate(prompt)

        # 处理结果
        self._process_result(data, response)

# 3. Pipeline 手动编排（用户代码）
class AgenticTaskPipeline(Pipeline):
    def __init__(self):
        self.parser = AgenticTaskParserNode(...)
        self.category_expander = RubricCategoryExpanderNode(...)
        self.item_expander = RubricItemExpanderNode(...)
        self.orthogonality_checker = CategoryOrthogonalityCheckNode(...)

    async def run(self, batch: List[AgenticTaskSample], context: Dict[str, Any]):
        # Step 1: 串行解析
        await self.parser.execute(batch, context)

        # Step 2: 展开为 categories
        await self.category_expander.execute(batch, context)

        # Step 3: 并行处理（用户显式控制）
        categories = []
        for sample in batch:
            categories.extend(sample.get_children(RubricCategory))

        await asyncio.gather(
            # 展开 rubric items
            self.item_expander.execute(categories, context),
            # 检查正交性（在 sample 层面）
            self.orthogonality_checker.execute(batch, context),
        )

        return batch
```

**Pipeline 基类定义**：
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Pipeline(ABC):
    """Pipeline base class for manual orchestration.

    Users inherit from this class and implement the run() method
    to explicitly control node execution order and parallelism.
    """

    @abstractmethod
    async def run(self, batch: List[PipelineDataBase], context: Dict[str, Any]) -> List[PipelineDataBase]:
        """Execute the pipeline.

        Args:
            batch: Input data batch
            context: Execution context (must include 'llm_limiter')

        Returns:
            Processed data batch
        """
        pass
```

---

## 实现计划

### Phase 1: Pipeline 基类（1天）

**文件**：
- `reward_framework/pipeline/base.py` - Pipeline 抽象基类
- `reward_framework/pipeline/__init__.py` - 模块导出

**改动**：
- 创建 Pipeline 抽象基类
- 更新文档和示例

**不删除**：
- TopologyGraph/Executor 暂时保留（向后兼容）
- 但推荐新代码使用 Pipeline

### Phase 2: 限流器集成（0.5天）

**改动**：
- 在 `create_simple_context()` 中添加 `llm_limiter`
- 更新需要调用 LLM 的节点（使用 `async with context['llm_limiter']`）

**涉及节点**：
- `CategoryOrthogonalityCheckNode` (validator.py)
- 未来所有调用 LLM 的节点

### Phase 3: 示例 Pipeline（0.5天）

**文件**：
- `reward_framework/nodes/agentic_task_synthesis/pipeline.py` - 示例 Pipeline
- `reward_framework/nodes/agentic_task_synthesis/test_pipeline.py` - 测试

**内容**：
- 实现 `AgenticTaskPipeline` 展示最佳实践
- 添加测试验证并行执行和限流

---

## 优点

1. **实现简单**：
   - Pipeline 基类只需 20 行代码
   - Semaphore 是 Python 标准库
   - 无需复杂的队列和批处理逻辑

2. **性能提升**：
   - 用户可以手动并行化独立节点
   - 全局限流防止 API rate limit
   - 预期性能提升 5-10 倍

3. **灵活可控**：
   - 用户完全控制执行顺序
   - 可以根据业务逻辑动态调整
   - 易于调试（执行流程清晰）

4. **向后兼容**：
   - TopologyGraph/Executor 保留（但不推荐）
   - 渐进式迁移

---

## 待解决问题

1. **限流器粒度**：
   - 当前设计是全局限流（所有 LLM 调用共享）
   - 是否需要按 LLM 类型分别限流？（例如 GPT-4 最多 5 个，GPT-3.5 最多 20 个）
   - **建议**：先实现全局限流，按需添加

2. **错误处理**：
   - 并行执行时，一个节点失败是否影响其他？
   - **建议**：使用 `asyncio.gather(return_exceptions=True)` 收集错误

3. **进度显示**：
   - 多个节点并行执行时，如何显示进度？
   - **建议**：每个节点独立的 tqdm，或者使用 `tqdm.asyncio.gather()`

---

## 代码位置

- **当前框架**：`/mnt/shared-storage-user/ailab-hx/tongjian/verl/rewards/reward_framework/`
- **节点实现**：`reward_framework/nodes/agentic_task_synthesis/`
- **测试**：`reward_framework/nodes/agentic_task_synthesis/test_*.py`

---

## 时间估算

| Phase | 任务 | 时间 | 优先级 |
|-------|------|------|--------|
| 1 | Pipeline 基类 | 1 天 | P0 |
| 2 | 限流器集成 | 0.5 天 | P0 |
| 3 | 示例 Pipeline | 0.5 天 | P1 |
| **总计** | | **2 天** | |

---

## 下一步

1. 实现 `Pipeline` 基类
2. 添加 `llm_limiter` 到 context
3. 更新 `CategoryOrthogonalityCheckNode` 使用限流器
4. 编写 `AgenticTaskPipeline` 示例
5. 添加测试验证性能提升

---

**文档创建时间**: 2026-04-14
**状态**: 设计方案已确定，等待实现
