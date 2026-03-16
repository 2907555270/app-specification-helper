# 结构化输出

结构化输出（Structured Output）是指让大语言模型（LLM）的输出严格符合指定的格式（如 JSON、Pydantic 对象、XML 等），而不是自由文本。当前主流的结构化输出方式可以分为以下几类，从稳定性、强制程度、实现成本和适用场景来对比：

| 方式                                       | 实现方式                                                                          | 强制程度  | 稳定性（多轮/长上下文） | 支持模型                                                     | 优点                                | 缺点                              | 推荐场景                                      |
| ---------------------------------------- | ----------------------------------------------------------------------------- | ----- | ------------ | -------------------------------------------------------- | --------------------------------- | ------------------------------- | ----------------------------------------- |
| 1. 原生 Structured Outputs（最高推荐）           | `with_structured_output(schema, method="json_schema", strict=True)`           | ★★★★★ | ★★★★★        | OpenAI (gpt-4o系列、o1系列)Claude 3.5+Gemini 1.5+             | 模型生成时就被强制约束，几乎不会偏离 schema；多轮也非常稳定 | 必须模型本身支持（老模型不支持）                | **生产首选**，尤其是需要高可靠性的场景（如 NL2SQL、工具调用、数据提取） |
| 2. Function Calling / Tool Calling       | `llm.bind_tools(tools)` 或 `with_structured_output(method="function_calling")` | ★★★★☆ | ★★★★☆        | OpenAI、Claude、Gemini、Groq、DeepSeek 等几乎所有支持 tool call 的模型 | 兼容性最广；模型返回 tool\_calls 结构，解析成本低   | 比 json\_schema 略弱，可能偶尔加多余文字     | 兼容性要求高、模型不支持 json\_schema 时首选             |
| 3. JSON Mode（response\_format）           | `llm.bind(response_format={"type": "json_object"})` + prompt 约束               | ★★★☆☆ | ★★★☆☆        | OpenAI 兼容模型（包括很多开源代理）                                    | 强制输出是合法 JSON，简单易用                 | 不强制 schema 结构，容易字段缺失/类型错        | 轻量场景、老模型、快速原型                             |
| 4. Prompt Engineering + JsonOutputParser | 在 prompt 中反复强调 JSON 格式 + `JsonOutputParser(pydantic_model)`                   | ★★☆☆☆ | ★★☆☆☆        | 几乎所有模型                                                   | 通用性最强，无需模型原生支持                    | 多轮/长上下文下极易崩（加解释、markdown、控制字符等） | 极早期实验、模型不支持任何结构化时 fallback                |
| 5. Grammar / Constrained Decoding        | 使用 llama.cpp、outlines、Guidance、LMQL 等库强制语法                                    | ★★★★★ | ★★★★★        | 开源模型（Llama 3、Mistral、Qwen 等）                             | 理论上最强，几乎 100% 符合格式                | 需要本地部署或特定推理引擎；速度慢、显存占用高         | 本地部署、对格式要求极高的场景（如医疗、金融合规）                 |

### 2026 年主流推荐组合（生产排序）

| 优先级 | 组合方式                                                        | 适用模型                                           | 稳定性 | 适用场景                   |
| --- | ----------------------------------------------------------- | ---------------------------------------------- | --- | ---------------------- |
| 1   | `with_structured_output(method="json_schema", strict=True)` | OpenAI gpt-4o / o1 / Claude 3.5+ / Gemini 1.5+ | 最高  | 99% 的生产场景首选            |
| 2   | `bind_tools` + 解析 tool\_calls                               | 几乎所有支持 tool 的模型                                | 非常高 | 模型不支持 json\_schema 时次选 |
| 3   | JSON Mode + 强 prompt 约束 + 后处理                               | OpenAI 兼容模型                                    | 中高  | 快速上线、老模型过渡             |
| 4   | Prompt + JsonOutputParser + retry                           | 所有模型                                           | 较低  | 极端 fallback 或实验阶段      |
| 5   | Grammar/Constrained Decoding                                | 开源本地模型                                         | 极高  | 本地部署 + 合规要求极高的场景       |

# Stream

### LangGraph StateSnapshot 访问注意

LangGraph 0.1.x+ 版本中，`invoke()` 和 `get_state()` 返回的是 `StateSnapshot` 对象，需要用 `.values` 属性访问状态：

```python
# 正确方式
final_state = self.graph.invoke(inputs, config)
messages = final_state.values.get("messages", [])  # 注意 .values

# 或者兼容处理
if hasattr(final_state, 'values'):
    messages = final_state.values.get("messages", [])
else:
    messages = final_state.get("messages", [])
```

LangChain 的 streaming 目前支持多种模式（modes），以下是官方文档中提到的主要几种，以及它们各自的适用场景和输出特点（基于最新文档）：

| 模式名称     | stream\_mode 值 | 每次 chunk 输出的是什么                     | 典型使用场景                              | 优点 / 缺点简评             |
| -------- | -------------- | ----------------------------------- | ----------------------------------- | --------------------- |
| values   | "values"       | 完整的当前状态（state）快照                    | 需要看到整个状态逐步变化（例如 Agent 的完整 messages） | 信息最完整，但数据量大、重复多       |
| updates  | "updates"      | 每个节点执行后产生的状态增量（delta / patch）       | 关注“哪个节点做了什么改变”                      | 数据量最小、最精确，常用于日志/调试    |
| messages | "messages"     | (message, metadata) 元组              | 想实时拿到每条新消息（最常用于聊天 UI）               | 适合显示聊天记录，但需要自己处理换行/类型 |
| custom   | "custom"       | 节点自定义的输出（需节点实现 emit\_custom\_event） | 高度定制化场景（很少用）                        | 灵活性最高，但实现成本高          |
| debug    | "debug"        | 非常详细的内部执行信息（节点进出、状态变化等）             | 深度调试 LangGraph 执行流程                 | 信息爆炸，生产环境慎用           |

### 最常用的三种对比（实际开发中 90% 会用到这三种）

| 想实现的效果                    | 推荐 stream\_mode | 示例代码片段（简版）                                                                                  | 备注                    |
| ------------------------- | --------------- | ------------------------------------------------------------------------------------------- | --------------------- |
| 实时显示聊天消息（最常见）             | "messages"      | for msg, meta in graph.stream(..., stream\_mode="messages"): print(msg.content)             | 适合前端流式输出，但需要自己判断换行和类型 |
| 只想看到状态增量（哪个节点改了什么）        | "updates"       | for update in graph.stream(..., stream\_mode="updates"): print(update)                      | 最轻量，推荐用于日志和后端处理       |
| 需要看到完整逐步状态（messages 列表变化） | "values"        | for snapshot in graph.stream(..., stream\_mode="values"): print(snapshot\["messages"]\[-1]) | 数据量较大，但最直观            |

### 实际使用建议（2025-2026 年主流实践）

1. **前端聊天界面 / 实时显示** → 用 `stream_mode="messages"`，然后自己处理不同类型消息的显示逻辑（AI 回复、工具调用、工具结果）
2. **后端日志 / 调试** → 用 `stream_mode="updates"` 或 `stream_mode="debug"`
3. **需要完整上下文的场景**（例如交互式 debug 或复杂 agent） → 用 `stream_mode="values"`
4. **混合使用**：可以同时订阅多种 mode（LangGraph 支持多路 stream），但一般不必要。

# 交互式 Agent 架构

## 我们解决的核心问题（努力方向）

### 1. 多轮对话下结构化输出遵守度下降

- **问题根源**：历史消息（尤其是上一轮的 JSON 输出）污染上下文，导致模型忘记 schema 约束。
- **解决路径**：把多轮交互和结构化 SQL 生成**彻底分离**，让 NL2SQL Agent 只看到单次、干净、完整、自洽的 query。

### 2. 检索系统在多轮对话中的歧义

- **问题根源**：如果直接把多轮历史塞给检索器，会导致召回噪声大、意图漂移。如果直接用追问进行检索，会失去很多上下文信息，会产生歧义。
- **解决路径**：检索只针对交互 Agent 总结后的最终 query 执行，既包含了历史上下文，又减少了对召回的噪声。

### 3. 工具调用后不必要的二次 LLM 包装

- **问题根源**：ReAct 循环默认会让工具结果回 agent 再总结，导致多余延迟/成本。
- **解决路径**：手动搭建 LangGraph + 自定义 post-tool router，让 nl2sql\_tool 执行完直接路由到 END，不回 agent。

## 核心架构设计

### 文件职责划分

| 文件                | 职责               | 关键类/函数                      |
| ----------------- | ---------------- | --------------------------- |
| `graph.py`        | LangGraph 状态图构建  | `build_interactive_graph()` |
| `leader_agent.py` | 交互式 BI Agent 业务层 | `InteractiveNL2SQLAgent`    |
| `nl2sql_agent.py` | 单次 NL2SQL 生成器    | `NL2SQLAgent`               |

### 调用链路

```
用户输入 → InteractiveNL2SQLAgent.run()
           ↓
         LangGraph (graph.py)
           ↓
    ┌──────┴──────┐
    ↓             ↓
  agent        tools
  (LLM决策)     (nl2sql_tool)
    ↑             ↓
    └──────┴──────┘
           ↓
      NL2SQLAgent.run()
           ↓
    ┌──────┴──────┐
    ↓             ↓
  召回          生成SQL
           ↓
        返回结果
```

<br />