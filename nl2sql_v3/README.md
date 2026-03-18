# nl2sql_v3

NL2SQL 智能对话系统 - 基于 LangGraph 的多轮对话式自然语言转 SQL 生成工具

## 项目简介

nl2sql_v3 是一个完整的 NL2SQL（自然语言转 SQL）系统，支持多轮对话式 SQL 生成。它采用多路混合检索策略进行表召回，结合 LangGraph Agent 架构实现意图澄清与智能 SQL 生成。

## 核心特性

- **多路混合召回**：融合 BM25 关键词检索、Sparse 向量检索、Dense 向量检索
- **智能重排**：支持 Cross-Encoder 重排，提升召回精度
- **灵活配置**：通过 YAML 文件配置各模块参数
- **CLI 交互**：支持交互式查询、批量评估等多种模式
- **智能 Agent**：基于 LangGraph 的多轮对话 Agent，支持意图澄清、意图总结
- **性能优异**：在已确定数据库范围场景下，Hit\@1 达到 87%+

## 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户查询                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   HybridRetriever                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │  BM25       │ │  Sparse     │ │  Dense      │          │
│  │  (关键词)    │ │  (稀疏向量)  │ │  (稠密向量)  │          │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘          │
│         └───────────────┼───────────────┘                  │
│                         ▼                                   │
│              ES Hybrid Search                               │
│                         ▼                                   │
│              可选：Rerank 重排                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Top-K 相关表                              │
└─────────────────────────────────────────────────────────────┘
```

### Agent 架构（多轮对话）

```
用户输入 → InteractiveNL2SQLAgent (LangGraph)
                    │
           ┌────────┴────────┐
           │                 │
        agent           tools
      (LLM决策)        (nl2sql_tool)
           │                 │
           └────────┬────────┘
                    ▼
           NL2SQLAgent.run()
                    │
           ┌────────┴────────┐
           │                 │
         召回            生成SQL
```

### 核心文件职责

| 文件 | 职责 | 关键类/函数 |
|------|------|-------------|
| `agent/graph.py` | LangGraph 状态图构建 | `build_interactive_graph()` |
| `agent/leader_agent.py` | 交互式 BI Agent 业务层 | `InteractiveNL2SQLAgent` |
| `agent/nl2sql_agent.py` | 单次 NL2SQL 生成器 | `NL2SQLAgent` |
| `main.py` | CLI 统一入口 | `chat`, `agent`, `recall`, `evaluate` |

## 快速开始

### 环境要求

- Python 3.9+
- Elasticsearch 8.x
- 向量编码服务（可选）
- 重排服务（可选）

### 安装

```bash
# 克隆项目
cd nl2sql_v3

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -e .
```

### 配置

编辑 `config.yaml` 文件，配置各服务地址和召回参数：

```yaml
services:
  sparse_vector:
    url: "http://127.0.0.1:8000/api/v1/sparse_vector"
  dense_vector:
    url: "http://127.0.0.1:8000/api/v1/dense_vector"
  rerank:
    url: "http://127.0.0.1:8000/api/v1/rerank"
  elasticsearch:
    hosts:
      - "http://192.168.116.5:9200"
    index: "tables-metadata"

recall:
  weights:
    keyword: 0.1
    sparse: 0.6
    dense: 0.3
  top_k: 20
  hybrid_search_top_k: 100
  rerank_enabled: true
  rerank_top_k: 20
  rerank_threshold: -20
```

### 使用方法

#### 1. 构建索引

```bash
python main.py build-index
```

#### 2. 多轮对话（推荐）

```bash
# 启动交互式多轮对话
python main.py chat
```

#### 3. 单次 SQL 生成

```bash
# 基本用法
python main.py agent "how many clubs are there"

# 禁用 few-shot 示例
python main.py agent "how many clubs are there" --no-fewshot

# 指定 LLM temperature
python main.py agent "how many clubs are there" --temperature 0.2

# 禁用 SQL 执行
python main.py agent "how many clubs are there" --no-execute
```

#### 4. 表召回

```bash
# 基本用法
python main.py recall "how many clubs are there"

# 指定返回数量
python main.py recall "how many clubs are there" --top-k 5

# 显示详细分数
python main.py recall "how many clubs are there" --show-scores
```

#### 5. 性能评估

```bash
# 评估召回效果
python main.py evaluate

# 指定输出文件
python main.py evaluate --output result.json
```

## 命令详解

### build-index

构建 Elasticsearch 索引，将表元数据导入搜索引擎。

```bash
python main.py build-index [--force]  # --force 强制重建
```

### recall

执行表召回查询。

| 参数                    | 说明                          | 默认值           |
| --------------------- | --------------------------- | ------------- |
| `query`               | 自然语言查询（必填）                  | -             |
| `--top-k`, `-k`       | 返回结果数量                      | 5             |
| `--weights`, `-w`     | 融合权重 (keyword,sparse,dense) | "0.3,0.3,0.4" |
| `--show-scores`, `-s` | 显示详细分数                      | False         |
| `--no-keyword`        | 禁用关键词检索                     | False         |
| `--no-sparse`         | 禁用稀疏向量检索                    | False         |
| `--no-dense`          | 禁用稠密向量检索                    | False         |

### evaluate

运行召回评估，输出 Hit Rate、MRR 等指标。

| 参数                | 说明       | 默认值   |
| ----------------- | -------- | ----- |
| `--db`, `-d`      | 指定数据库范围  | 全部    |
| `--output`, `-o`  | 结果输出文件   | -     |
| `--verbose`, `-v` | 显示详细信息   | False |
| `--no-keyword`    | 禁用关键词检索  | False |
| `--no-sparse`     | 禁用稀疏向量检索 | False |
| `--no-dense`      | 禁用稠密向量检索 | False |

## 召回参数调优

### 权重配置

| 场景     | keyword | sparse | dense | 说明     |
| ------ | ------- | ------ | ----- | ------ |
| 精确匹配为主 | 0.1     | 0.6    | 0.3   | 推荐配置   |
| 语义理解为主 | 0.1     | 0.3    | 0.6   | 侧重语义理解 |

### Top-K 参数

| 参数                    | 说明        | 推荐值    |
| --------------------- | --------- | ------ |
| `hybrid_search_top_k` | ES 粗筛召回数量 | 50-100 |
| `rerank_top_k`        | 重排输出数量    | 10-20  |
| `top_k`               | 最终返回数量    | 5-10   |

### 重排阈值

| 参数                 | 说明          | 推荐值       |
| ------------------ | ----------- | --------- |
| `rerank_threshold` | 重排分数阈值（BGE Reranker 输出范围 [-1,1]） | -0.1 或 0.0 |

## 项目结构

```
nl2sql_v3/
├── agent/                    # Agent 模块
│   ├── graph.py            # LangGraph 状态图构建
│   ├── leader_agent.py    # 交互式 BI Agent 业务层
│   ├── nl2sql_agent.py   # 单次 NL2SQL 生成器
│   ├── nl2sql_prompts.py # Prompt 模板
│   └── schema_builder.py  # Schema 构建器
├── client/                 # 客户端模块
│   ├── api_client.py      # API 客户端（向量服务、重排服务）
│   └── es_client.py       # Elasticsearch 客户端
├── recall/                # 召回模块
│   ├── base.py           # 基础数据结构
│   ├── fusion.py         # 混合召回实现
│   ├── keyword.py        # 关键词检索
│   ├── sparse.py         # 稀疏向量检索
│   └── dense.py          # 稠密向量检索
├── technical_doc/         # 技术文档
│   ├── rag.md           # RAG 技术指南
│   ├── agent.md         # Agent 技术文档
│   └── benchmark.md     # 基准测试报告
├── main.py               # CLI 入口
├── config.py             # 配置加载
├── config.yaml           # 配置文件
└── README.md            # 项目说明
```

## 性能基准

基于实际测试数据（2147 查询）：

### 召回性能（Determined 场景 - 已确定数据库范围）

| 配置 | Hit\@1 | Hit\@3 | Hit\@5 | MRR | 平均耗时 |
| --- | ------ | ------ | ------ | ---- | ---- |
| 无重排 (keyword:0.1, sparse:0.6, dense:0.3) | 81.4% | 88.0% | 88.0% | 0.84 | 46ms |
| Weighted RRF + 重排 | **87.8%** | **99.3%** | **100%** | **0.93** | 95ms |
| **BGE-M3 + Weighted RRF + 重排** | **89.9%** | **99.3%** | **99.8%** | **0.95** | 100ms |

### 召回性能（Not Determined 场景 - 全库搜索）

| 配置 | Hit\@1 | Hit\@3 | Hit\@5 | MRR | 平均耗时 |
| --- | ------ | ------ | ------ | ---- | ---- |
| 无重排 (keyword:0.1, sparse:0.6, dense:0.3) | 67.1% | 87.2% | 93.0% | 0.77 | 42ms |
| Weighted RRF + 重排 | 57.2% | 78.3% | 85.3% | 0.70 | 242ms |
| **BGE-M3 + Weighted RRF + 重排** | **63.3%** | **84.8%** | **90.9%** | **0.75** | 324ms |

**结论**：
- Weighted RRF + 重排在 Determined 场景下达 **Hit\@5 ≈ 100%**
- BGE-M3 向量模型在全库搜索场景下提升显著（Hit\@1 +10.7%）

## 相关文档

- [RAG 技术指南](./technical_doc/rag.md) - 检索与重排技术详解
- [Agent 架构设计](./technical_doc/agent.md) - LangGraph 多轮对话 Agent 设计
- [表召回-问题记录](./technical_doc/rag_explore.md) - 检索与重排技术问题记录
- [基准测试报告](./technical_doc/benchmark.md) - 性能评估与优化建议

## 许可证

MIT License
