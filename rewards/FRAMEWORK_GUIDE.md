"""
Typed Pipeline Framework - Usage Guide & Documentation

## Overview

This framework provides a strongly-typed, declarative approach to building
multi-stage LLM evaluation pipelines. It's designed specifically for RL-based
question generation and evaluation tasks with the constraint that all code
must reside in a single file (for verl framework compatibility).

## Key Features

### 1. Strong Type Safety
- Generic type parameters for nodes: Node[InputT, OutputT]
- Protocol-based interfaces for functions
- TypedDict for context with clear documentation
- IDE auto-completion and type checking support

### 2. Declarative Topology
- Explicit DAG structure as first-class citizen
- No more inheritance chains
- Visual representation of pipeline
- Automatic cycle detection

### 3. Smart Data Flow
- Automatic routing to nearest parser/transformer output
- Maintains type consistency across nodes
- Skip logic for failed samples
- Context-based access to all results

### 4. Parallel Execution
- Automatic parallelization within levels
- Efficient resource utilization
- Async/await throughout

## Core Concepts

### Node
Atomic processing unit with typed inputs/outputs.

Types of nodes:
- PARSER: Parse raw strings into structured data
- RULE: Apply rule-based checks (e.g., format validation)
- LLM_GENERATOR: Generate content using LLM (e.g., questions)
- LLM_JUDGE: Judge/verify content using LLM
- AGGREGATOR: Combine multiple inputs
- FILTER: Filter samples based on criteria
- TRANSFORMER: Transform data structure

### Topology
Directed Acyclic Graph defining node dependencies.

Features:
- Topological sort for execution order
- Parallel execution within levels
- Conditional edges (optional)
- Validation and visualization

### Executor
Orchestrates node execution according to topology.

Features:
- Manages skip logic
- Aggregates final scores
- Provides execution summary
- Error handling per node

## Usage Examples

### Example 1: Basic Pipeline

```python
from reward import (
    Node, NodeConfig, NodeType, NodeResult,
    TopologyGraph, PipelineExecutor, create_context
)

# 1. Define custom node
class MyParserNode(Node[str, Tuple[str, str]]):
    async def execute(self, batch_inputs, context):
        outputs = []
        for inp in batch_inputs:
            if inp is None:
                outputs.append(None)
            else:
                # Your parsing logic
                question, answer = parse_input(inp)
                outputs.append((question, answer))

        return NodeResult(outputs=outputs, node_name=self.name)

# 2. Build topology
topology = (TopologyGraph()
    .add_node(MyParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER)))
    .add_node(MyRuleNode(NodeConfig(name="rule", node_type=NodeType.RULE,
                                     skip_on_negative=True)))
    .add_node(MyScorerNode(NodeConfig(name="scorer", node_type=NodeType.LLM_JUDGE)))
    .add_edge("parser", "rule")
    .add_edge("rule", "scorer"))

# 3. Create context
context = create_context(
    ground_truths=[...],
    agents={'judge': my_agent},
    max_concurrent={'judge': 16},
    min_reward=-2.0
)

# 4. Execute
executor = PipelineExecutor(topology, context)
scores = await executor.execute(batch_inputs)
```

### Example 2: Generator-Solver-Judge Pattern

This is the core pattern for your use case:
- Generator: Creates questions (from RL rollout)
- Solvers: Multiple agents attempt questions (weak/adv)
- Judge: Verifies correctness and computes difficulty

```python
from reward import SolverConfig, DifficultyMetricConfig

# Define solver configurations
solver_configs = [
    SolverConfig(
        name='weak',
        agent_key='weak_agent',
        repeat=3,
        prompt_fn_key='respond_wo_context',
        max_concurrent=16
    ),
    SolverConfig(
        name='adv',
        agent_key='adv_agent',
        repeat=3,
        prompt_fn_key='respond_wo_context',
        max_concurrent=16
    )
]

# Define difficulty metric
difficulty_metric = DifficultyMetricConfig(
    weak_name='weak',
    adv_name='adv',
    weak_weight=0.4,
    adv_weight=0.6,
    weak_overcomplex_threshold=0.1,
    adv_overcomplex_threshold=0.3,
    weak_oversimple_threshold=0.9,
    adv_oversimple_threshold=0.95,
    advantage_gap_threshold=0.2
)

# Build topology with MultiSolver and DifficultyJudge nodes
# (Implementation in next section)
```

### Example 3: Parallel Branches

```python
topology = TopologyGraph()

# Root parser
parser = ParserNode(NodeConfig(name="parser", node_type=NodeType.PARSER))

# Parallel evaluation branches
similarity_check = SimilarityNode(NodeConfig(name="similarity", ...))
quality_check = QualityNode(NodeConfig(name="quality", ...))
difficulty_eval = DifficultyNode(NodeConfig(name="difficulty", ...))

# Build
topology.add_node(parser)
topology.add_node(similarity_check)
topology.add_node(quality_check)
topology.add_node(difficulty_eval)

# Parser feeds all three (executed in parallel)
topology.add_edge("parser", "similarity")
topology.add_edge("parser", "quality")
topology.add_edge("parser", "difficulty")

# All three contribute to final score
```

### Example 4: Conditional Execution

```python
# Add conditional edge
def should_run_expensive_check(context: ExecutionContext) -> bool:
    # Only run if in validation mode
    return context.get('split') == 'valid'

topology.add_edge(
    "quick_check",
    "expensive_check",
    condition=should_run_expensive_check
)
```

## Node Implementation Patterns

### Pattern 1: Parser Node

```python
class MyParserNode(Node[str, ParsedResult]):
    def __init__(self, config: NodeConfig, parse_fn: Callable):
        super().__init__(config)
        self.parse_fn = parse_fn

    async def execute(self, batch_inputs, context):
        outputs = [self.parse_fn(inp) if inp else None
                  for inp in batch_inputs]

        return NodeResult(
            outputs=outputs,
            node_name=self.name,
            metadata={'parsed_count': sum(1 for o in outputs if o)}
        )
```

### Pattern 2: Rule Check Node

```python
class RuleCheckNode(Node[ParsedT, float]):
    def __init__(self, config: NodeConfig,
                 penalty_module: PenaltyOrRewardModule):
        super().__init__(config)
        self.penalty_module = penalty_module

    async def execute(self, batch_inputs, context):
        outputs = []
        ground_truths = context['ground_truths']

        for inp, gt in zip(batch_inputs, ground_truths):
            if inp is None:
                outputs.append(None)
            else:
                score = self.penalty_module.get_penalty_or_reward(inp, gt)
                outputs.append(score)

        return NodeResult(outputs=outputs, node_name=self.name)
```

### Pattern 3: LLM Judge Node

```python
class LLMJudgeNode(Node[ParsedT, float]):
    def __init__(self, config: NodeConfig,
                 judge_task: BatchCallOpenAPI,
                 agent_key: str):
        super().__init__(config)
        self.judge_task = judge_task
        self.agent_key = agent_key

    async def execute(self, batch_inputs, context):
        # Filter valid inputs
        valid_indices, valid_inputs = self._filter_valid_inputs(batch_inputs)

        if not valid_inputs:
            return NodeResult(outputs=[None] * len(batch_inputs),
                            node_name=self.name)

        # Call LLM judge
        agent = context['agents'][self.agent_key]
        max_concurrent = context['max_concurrent'][self.agent_key]

        results = await self.judge_task.do_job(
            agent=agent,
            batch_inputs=valid_inputs,
            max_concurrent_requests=max_concurrent
        )

        # Reconstruct full batch
        outputs = self._reconstruct_batch(valid_indices, results,
                                         len(batch_inputs))

        return NodeResult(outputs=outputs, node_name=self.name)
```

## Advanced Features

### Accessing Previous Node Results

Nodes can access results from any previous node via context:

```python
class CombinerNode(Node[ParsedT, float]):
    async def execute(self, batch_inputs, context):
        # Access executor's results
        executor = context.get('executor')  # Need to add this
        similarity_results = executor.get_node_result('similarity')

        # Combine with current processing
        ...
```

### Custom Aggregation

Override the default weighted sum:

```python
class CustomAggregator(Node[Any, float]):
    async def execute(self, batch_inputs, context):
        # Implement custom aggregation logic
        # E.g., geometric mean instead of weighted sum
        ...
```

### Visualization

```python
# Print topology structure
print(topology.visualize(show_types=True))

# Output:
# ============================================================
# Pipeline Topology
# ============================================================
#
# Level 0:
#   - parser [parser]
#
# Level 1:
#   - rule_check [rule] (skip_on_neg)
#   - format_check [rule] (skip_on_neg)
#
# Level 2:
#   - scorer [llm_judge]
#
# Edges:
#   parser -> rule_check
#   parser -> format_check
#   rule_check -> scorer
#   format_check -> scorer
# ============================================================
```

## Configuration Best Practices

### 1. NodeConfig Usage

```python
# For filter nodes (don't contribute to score)
NodeConfig(
    name="format_check",
    node_type=NodeType.RULE,
    skip_on_negative=True,  # Skip sample if negative
    filter_only=True,       # Don't add to final score
    weight=0.0              # Redundant with filter_only but explicit
)

# For scoring nodes
NodeConfig(
    name="difficulty",
    node_type=NodeType.LLM_JUDGE,
    skip_on_negative=False,  # Don't skip
    filter_only=False,       # Include in final score
    weight=1.0               # Full weight
)

# For bonus/minor rewards
NodeConfig(
    name="style_bonus",
    node_type=NodeType.LLM_JUDGE,
    weight=0.1  # 10% weight
)
```

### 2. Context Organization

```python
context = create_context(
    ground_truths=batch_gt,

    # Agents
    agents={
        'weak_agent': Agent(...),
        'adv_agent': Agent(...),
        'verify_agent': Agent(...),
        'judge_agent': Agent(...)
    },

    # Concurrency limits
    max_concurrent={
        'weak_agent': 16,
        'adv_agent': 16,
        'verify_agent': 8,
        'judge_agent': 4
    },

    # Parsing functions
    parse_fn=doc2query_parse_fn,

    # Task-specific config
    split='train',
    task_name='doc2query_v2',
    min_reward=-2.0,

    # Prompt functions
    respond_wo_context=lambda p, gt, _: f"{p[0]}\n\nAnswer:",
    respond_w_context=lambda p, gt, _: f"Context: {gt['doc']}\n\n{p[0]}"
)
```

## Migration from Old Code

### Before (Inheritance-based)

```python
class Doc2QueryV3ComputeScore(Doc2QueryV2ComputeScore):
    def coarse_process(self):
        return [
            Process(name="Quality", function=self.check_quality, ...),
            Process(name="Similarity", function=self.check_similarity, ...)
        ]

    def finegrain_process(self):
        return Process(name="Difficulty", function=self.get_difficulty, ...)
```

### After (Topology-based)

```python
def create_doc2query_v3_topology(args, parse_fn):
    topology = TopologyGraph()

    # Add nodes
    topology.add_node(ParserNode(...))
    topology.add_node(QualityNode(...))
    topology.add_node(SimilarityNode(...))
    topology.add_node(DifficultyNode(...))

    # Define flow
    topology.add_edge("parser", "quality")
    topology.add_edge("parser", "similarity")
    topology.add_edge("quality", "difficulty")
    topology.add_edge("similarity", "difficulty")

    return topology

# Usage
topology = create_doc2query_v3_topology(args, parse_fn)
executor = PipelineExecutor(topology, context)
scores = await executor.execute(batch_inputs)
```

## Type Safety Benefits

### 1. IDE Support
- Auto-completion for all methods and fields
- Jump-to-definition works correctly
- Inline documentation

### 2. Early Error Detection
```python
# This will be caught by type checker:
parser = ParserNode[str, int](...)  # Returns int
scorer = ScorerNode[Tuple[str, str], float](...)  # Expects tuple
topology.add_edge("parser", "scorer")  # Type mismatch!
```

### 3. Self-Documenting Code
```python
# Clear input/output types
class MyNode(Node[InputType, OutputType]):
    ...

# Clear protocol requirements
def my_parse_fn(solution: str) -> Optional[ParsedResult]:
    ...
```

## Performance Considerations

### 1. Batch Size
- Larger batches → better GPU utilization
- But: more memory, longer to first result
- Recommended: 8-32 samples per batch

### 2. Concurrency
- Set max_concurrent based on API limits
- Monitor rate limits
- Consider exponential backoff (in Agent class)

### 3. Caching
- Implement caching in nodes for repeated inputs
- Use LRU cache for expensive operations
- Cache at prompt level, not response level

## Debugging Tips

### 1. Execution Summary
```python
executor.print_summary()
# Shows execution time and success rate per node
```

### 2. Visualize Topology
```python
print(topology.visualize())
# Shows DAG structure before execution
```

### 3. Access Intermediate Results
```python
result = executor.get_node_result('parser')
print(f"Parser outputs: {result.outputs}")
print(f"Parser metadata: {result.metadata}")
```

### 4. Enable Detailed Logging
```python
# Framework already prints execution progress
# Add more logging in your nodes:
print(f"[{self.name}] Processing {len(valid_inputs)} samples")
```

## Next Steps

To complete the framework, you need to implement:

1. **MultiSolverNode**: Multi-agent solving with repeats
2. **DifficultyJudgeNode**: Verify + compute difficulty score
3. **Concrete penalty/reward modules**: Adapt from fabricate_qa.py
4. **Integration with existing Agent class**: From fabricate_qa.py
5. **BatchCallOpenAPI adapters**: Wrap existing judge tasks

See the implementation guide in the next document.
"""
