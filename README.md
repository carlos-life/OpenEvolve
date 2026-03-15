# OpenEvolve

**Evolve algorithms with LLMs. Open-source AlphaEvolve alternative.**

OpenEvolve uses large language models as mutation and crossover operators inside an evolutionary algorithm loop. Describe a problem, provide test cases, and let AI evolve increasingly better solutions — potentially discovering approaches that humans haven't thought of.

```
          ┌──────────────────────────────────────────┐
          │           Evolutionary Loop               │
          │                                            │
          │   ┌─────────┐    ┌──────────┐              │
   LLM ──▶│   │ Generate │──▶│ Evaluate │──┐           │
          │   └─────────┘    │ (Sandbox)│  │           │
          │                  └──────────┘  │           │
          │   ┌─────────┐    ┌──────────┐  │           │
          │   │  Mutate  │◀──│  Select  │◀─┘           │
          │   │Crossover │    │ (Tourney)│              │
   LLM ──▶│   └────┬────┘    └──────────┘              │
          │        │                                    │
          │        └──── Next Generation ──────▶ ...   │
          └──────────────────────────────────────────┘
                              │
                              ▼
                     Best Solution Found
```

## How It Works

1. **Seed** — LLM generates an initial population of candidate solutions
2. **Evaluate** — Each candidate runs in a sandboxed subprocess against your test cases
3. **Select** — Tournament selection picks the fittest individuals
4. **Evolve** — LLM mutates or crosses over selected candidates to create offspring
5. **Repeat** — The cycle continues for N generations, progressively improving solutions

Unlike traditional evolutionary programming, OpenEvolve uses an LLM to understand *why* a solution works or fails, producing intelligent mutations instead of random ones.

## Features

- **Any OpenAI-compatible API** — Works with Claude, GPT, DeepSeek, Ollama, or any local model with an OpenAI-compatible endpoint.
- **Sandboxed execution** — Candidate code runs in isolated subprocesses with strict timeouts. No network access.
- **Weighted test cases** — Assign higher weight to critical test cases.
- **Elitism** — Top performers carry over to the next generation unchanged.
- **Progress callbacks** — Monitor fitness per generation in real time.
- **Zero-API-key mode** — Ships with `MockLLMClient` for testing and CI without any API calls.

## Quick Start

### Install

```bash
pip install -e .
```

### Basic Usage

```python
import asyncio
from openevolve import Evolution, EvolutionConfig

config = EvolutionConfig(
    population_size=10,
    generations=5,
    llm_model="claude-sonnet-4-6",          # or "gpt-4o-mini", "deepseek-chat", etc.
    llm_api_key="sk-xxx",
    llm_base_url="https://api.anthropic.com/v1",  # optional, defaults to OpenAI
)

engine = Evolution(config)

result = asyncio.run(engine.run(
    problem_description="Write a function that sorts a list of integers efficiently.",
    function_signature="def solution(arr: list) -> list:",
    test_cases=[
        {"input": ([3, 1, 2],), "expected": [1, 2, 3]},
        {"input": ([],), "expected": []},
        {"input": ([5, 4, 3, 2, 1],), "expected": [1, 2, 3, 4, 5]},
        {"input": ([1, 1, 1],), "expected": [1, 1, 1]},
    ],
))

print(f"Best fitness: {result.best_individual.fitness}")
print(f"Best code:\n{result.best_individual.code}")
```

### With Progress Callback

```python
def on_generation(gen: int, best_fitness: float, avg_fitness: float):
    print(f"Gen {gen}: best={best_fitness:.3f}  avg={avg_fitness:.3f}")

result = asyncio.run(engine.run(
    problem_description="...",
    function_signature="...",
    test_cases=[...],
    on_generation=on_generation,
))
```

### With a Starting Solution

```python
result = asyncio.run(engine.run(
    problem_description="Improve this sorting function.",
    function_signature="def solution(arr: list) -> list:",
    test_cases=[...],
    initial_code="def solution(arr: list) -> list:\n    return sorted(arr)",
))
```

### CLI

```bash
# Runs a default sorting evolution demo
openevolve --generations 10 --population-size 20 --model gpt-4o-mini
```

## Examples

```bash
# Evolve a sorting algorithm
python examples/sort_algorithm.py

# Evolve a fuzzy string matcher
python examples/string_match.py
```

## Configuration

```python
EvolutionConfig(
    population_size=20,       # Candidates per generation
    generations=10,           # Number of generations
    mutation_rate=0.3,        # Probability of mutation vs crossover
    elite_ratio=0.2,          # Top 20% carry over unchanged
    tournament_size=3,        # Tournament selection group size
    timeout_seconds=10,       # Max execution time per candidate
    llm_model="gpt-4o-mini",  # Model name
    llm_base_url="",          # Custom API endpoint
    llm_api_key="",           # API key (empty = MockLLMClient)
)
```

## Architecture

```
openevolve/
├── evolve.py       # Evolution engine — orchestrates the full loop
├── llm.py          # LLM client (OpenAI-compatible) + MockLLMClient
├── sandbox.py      # Subprocess-based sandboxed code execution
├── evaluator.py    # Fitness scoring against weighted test cases
├── population.py   # Population management, selection, elitism
└── models.py       # Individual, EvolutionConfig, EvolutionResult
```

### How the LLM is Used

- **Initial generation**: *"Write N different solutions for: {problem}. Function signature: {signature}"*
- **Mutation**: *"Here's a solution with fitness {score}. Improve it. Here's the code: {code}"*
- **Crossover**: *"Combine the best aspects of these two solutions into a new one."*

The LLM sees the problem description, the candidate code, and its fitness score — enabling it to make informed, targeted improvements rather than random perturbations.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

32 tests. All tests use `MockLLMClient` — **no API keys needed** to run the test suite.

## Ideas for Evolution Targets

Beyond sorting, you can evolve:

- **Hashing functions** — Minimize collisions on a given dataset
- **Compression algorithms** — Maximize compression ratio
- **Regex patterns** — Find the shortest regex that matches all positive examples and rejects all negative ones
- **Scheduling algorithms** — Optimize task scheduling with constraints
- **Trading strategies** — Evolve buy/sell logic against historical data
- **Neural network architectures** — Evolve activation functions or layer configurations
- **Prompt engineering** — Evolve prompts that maximize task performance

## Contributors

| | |
|---|---|
| [**Carlos**](https://github.com/carlos-life) | Creator & maintainer |
| [**Claude Opus 4.6**](https://claude.ai) | AI pair programmer |

## License

MIT

---

# OpenEvolve

**用 LLM 进化算法。开源版 AlphaEvolve。**

OpenEvolve 将大语言模型作为进化算法中的变异和交叉算子。你只需要描述问题、提供测试用例，AI 就会进化出越来越好的解决方案——甚至可能发现人类没想到的方法。

## 工作原理

1. **播种** —— LLM 生成初始候选解群体
2. **评估** —— 每个候选解在沙箱子进程中运行，对照测试用例评分
3. **选择** —— 锦标赛选择法挑出适应度最高的个体
4. **进化** —— LLM 对选中的个体进行变异或交叉，生成后代
5. **循环** —— 持续 N 代，解决方案逐代改进

和传统进化编程不同，OpenEvolve 利用 LLM 理解方案*为什么*有效或失败，从而产生智能的变异，而非随机的扰动。

## 核心特性

- **兼容任何 OpenAI 格式的 API** —— 支持 Claude、GPT、DeepSeek、Ollama 或任何本地模型。
- **沙箱执行** —— 候选代码在隔离的子进程中运行，有严格的超时限制，无网络访问。
- **加权测试用例** —— 可以给关键测试用例分配更高的权重。
- **精英保留** —— 每代的最优个体直接保留到下一代。
- **进度回调** —— 实时监控每代的适应度变化。
- **无 API Key 模式** —— 内置 `MockLLMClient`，测试和 CI 不需要任何 API 调用。

## 快速开始

### 安装

```bash
pip install -e .
```

### 基本使用

```python
import asyncio
from openevolve import Evolution, EvolutionConfig

config = EvolutionConfig(
    population_size=10,
    generations=5,
    llm_model="claude-sonnet-4-6",
    llm_api_key="sk-xxx",
)

engine = Evolution(config)

result = asyncio.run(engine.run(
    problem_description="写一个高效的整数列表排序函数。",
    function_signature="def solution(arr: list) -> list:",
    test_cases=[
        {"input": ([3, 1, 2],), "expected": [1, 2, 3]},
        {"input": ([],), "expected": []},
        {"input": ([5, 4, 3, 2, 1],), "expected": [1, 2, 3, 4, 5]},
    ],
))

print(f"最佳适应度: {result.best_individual.fitness}")
print(f"最佳代码:\n{result.best_individual.code}")
```

### 命令行

```bash
# 运行默认的排序进化演示
openevolve --generations 10 --population-size 20 --model gpt-4o-mini
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `population_size` | 20 | 每代的候选解数量 |
| `generations` | 10 | 进化代数 |
| `mutation_rate` | 0.3 | 变异概率（vs 交叉）|
| `elite_ratio` | 0.2 | 精英保留比例 |
| `tournament_size` | 3 | 锦标赛选择的组大小 |
| `timeout_seconds` | 10 | 每个候选解的最大执行时间 |
| `llm_model` | gpt-4o-mini | 使用的模型 |
| `llm_api_key` | 空 | API Key（为空则使用 MockLLMClient）|

## LLM 是如何被使用的

- **初始生成**：*"针对这个问题写 N 个不同的解决方案：{问题描述}"*
- **变异**：*"这是一个适应度为 {分数} 的方案，请改进它"*
- **交叉**：*"结合这两个方案的优点，生成一个新方案"*

LLM 能看到问题描述、候选代码和适应度分数，因此可以做出有针对性的改进，而不是随机试探。

## 进化目标的灵感

除了排序，你还可以进化：

- **哈希函数** —— 最小化特定数据集上的碰撞率
- **压缩算法** —— 最大化压缩比
- **正则表达式** —— 找到匹配所有正例且拒绝所有反例的最短正则
- **调度算法** —— 在约束条件下优化任务调度
- **交易策略** —— 在历史数据上进化买卖逻辑
- **Prompt 工程** —— 进化出最大化任务表现的提示词

## 测试

```bash
pip install pytest
pytest tests/ -v
```

共 32 个测试，全部使用 `MockLLMClient`，**运行测试不需要任何 API Key**。

## 贡献者

| | |
|---|---|
| [**Carlos**](https://github.com/carlos-life) | 作者 & 维护者 |
| [**Claude Opus 4.6**](https://claude.ai) | AI 协作开发 |

## 许可证

MIT
