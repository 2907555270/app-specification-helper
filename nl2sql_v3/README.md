# nl2sql\_v3

NL2SQL 表召回系统 - 基于混合检索与重排的智能表选择工具

## 项目简介

nl2sql\_v3 是一个用于 NL2SQL（自然语言转 SQL）场景的表召回工具。它采用多路混合检索策略，结合关键词检索、稀疏向量检索、稠密向量检索以及重排模型，从大量数据库表中精准召回与用户查询相关的表。

## 核心特性

- **多路混合召回**：融合 BM25 关键词检索、Sparse 向量检索、Dense 向量检索
- **智能重排**：支持 Cross-Encoder 重排，提升召回精度
- **灵活配置**：通过 YAML 文件配置各模块参数
- **CLI 交互**：支持交互式查询、批量评估等多种模式
- **性能优异**：在已确定数据库范围场景下，Hit\@1 达到 87%+

## 系统架构

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

#### 2. 交互式查询

```bash
python main.py interactive
```

#### 3. 单次查询

```bash
# 基本用法
python main.py recall "查询销售额最高的商品"

# 指定返回数量
python main.py recall "查询销售额最高的商品" --top-k 5

# 显示详细分数
python main.py recall "查询销售额最高的商品" --show-scores
```

#### 4. 性能评估

```bash
# 评估召回效果
python main.py evaluate

# 指定输出文件
python main.py evaluate --output result.json

# 指定数据库范围
python main.py evaluate --db my_database
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
| `rerank_threshold` | 重排分数阈值，暂时无用 | 存在负数（不使用） |

## 项目结构

```
nl2sql_v3/
├── client/                 # 客户端模块
│   ├── api_client.py      # API 客户端（向量服务、重排服务）
│   └── es_client.py       # Elasticsearch 客户端
├── recall/                # 召回模块
│   ├── base.py           # 基础数据结构
│   ├── fusion.py         # 混合召回实现
│   ├── keyword.py        # 关键词检索
│   ├── sparse.py         # 稀疏向量检索
│   └── dense.py         # 稠密向量检索
├── technical_doc/         # 技术文档
│   ├── rag.md           # RAG 技术指南
│   └── benchmark.md     # 基准测试报告
├── main.py               # CLI 入口
├── config.py             # 配置加载
├── config.yaml           # 配置文件
└── README.md            # 项目说明
```

## 性能基准

基于实际测试数据（2100+ 查询）：

| 场景  | Hit\@1    | Hit\@3    | Hit\@5    | MRR      | 平均耗时 |
| --- | --------- | --------- | --------- | -------- | ---- |
| 无重排 | 81.4%     | 88.0%     | 88.0%     | 0.84     | 46ms |
| 有重排 | **87.6%** | **99.0%** | **99.6%** | **0.93** | 85ms |

## 相关文档

- [RAG 技术指南](./technical_doc/rag.md) - 检索与重排技术详解
- [基准测试报告](./technical_doc/benchmark.md) - 性能评估与优化建议

## 许可证

MIT License
