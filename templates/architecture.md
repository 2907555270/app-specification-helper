# 技术架构设计

## 1. 技术栈选择

### 1.1 核心语言与框架
| 类别 | 技术选型 | 版本 | 选择理由 |
|------|----------|------|----------|
| 语言 | Python | 3.8+ | 用户指定，易于集成LLM和向量服务 |
| 框架 | LangChain | 0.1.x | 用户指定，简化LLM集成 |
| CLI框架 | Click/Typer | - | 构建命令行工具 |
| 配置管理 | PyYAML | - | 配置文件支持 |

### 1.2 外部服务集成
| 类别 | 技术选型 | 说明 |
|------|----------|------|
| 向量服务-稀疏 | HTTP API | POST /api/v1/sparse_vector |
| 向量服务-稠密 | HTTP API | POST /api/v1/dense_vector |
| 翻译服务 | HTTP API | POST /api/v1/translate |
| LLM服务 | OpenRouter | x-ai/grok-4.1-fast |
| 向量存储 | Elasticsearch | 192.168.116.5:9200 |

### 1.3 数据处理
| 类别 | 技术选型 | 说明 |
|------|----------|------|
| JSON处理 | Python json | 解析metadatas.json和query_and_tables.json |
| 向量计算 | numpy | 稠密向量相似度计算 |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI 入口                                 │
│                    (main.py / cli.py)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      表召回引擎 (TableRetriever)                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ 关键词召回   │  │ 稀疏向量召回  │  │ 稠密向量召回  │          │
│  │(KeywordMatch)│  │(SparseRecall)│  │(DenseRecall) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│          │                  │                  │               │
│          └──────────────────┼──────────────────┘               │
│                             ▼                                   │
│                  ┌──────────────────┐                          │
│                  │ 结果融合 (Fusion) │                          │
│                  │ Score = α*K + β*S + γ*D                     │
│                  └──────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        外部服务                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ 稀疏向量API │  │ 稠密向量API │  │  翻译API    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐                             │
│  │ Elasticsearch│  │ OpenRouter  │  (暂不使用)                │
│  └─────────────┘  └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块划分

| 模块名称 | 职责 | 文件 |
|----------|------|------|
| CLI入口 | 命令行参数解析、交互入口 | cli.py |
| 配置管理 | 加载配置文件、管理服务地址 | config.py |
| 元数据加载 | 解析metadatas.json，构建索引 | metadata_loader.py |
| 关键词召回 | 基于表名/字段名的匹配 | keyword_recaller.py |
| 稀疏向量召回 | 调用稀疏向量API + ES检索 | sparse_recaller.py |
| 稠密向量召回 | 调用稠密向量API + ES检索 | dense_recaller.py |
| 结果融合 | 加权融合三种召回结果 | fusion.py |
| 批量评估 | 使用QA对评估召回效果 | evaluator.py |
| ES客户端 | 封装ES操作 | es_client.py |
| API客户端 | 封装向量/翻译服务调用 | api_client.py |

---

## 3. 数据流设计

### 3.1 索引构建流程
```
metadatas.json
      │
      ▼
┌─────────────────┐
│ 解析元数据      │ → db_name, table_name, columns[]
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ 构建索引文档    │
│ - text: 组合文本 │
│ - sparse_vector │ ← 调用稀疏向量API
│ - dense_vector  │ ← 调用稠密向量API
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ 写入ES索引      │
│ 索引名: tables  │
└─────────────────┘
```

### 3.2 召回流程
```
用户输入问题
      │
      ▼
┌─────────────────┐
│ 文本预处理      │
│ - 翻译(可选)    │
│ - 分词/清理     │
└─────────────────┘
      │
      ├──────────────────┬──────────────────┐
      ▼                  ▼                  ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ 关键词召回  │   │ 稀疏向量召回 │   │ 稠密向量召回 │
│ 匹配得分    │   │ ES检索得分  │   │ ES检索得分  │
└─────────────┘   └─────────────┘   └─────────────┘
      │                  │                  │
      └──────────────────┼──────────────────┘
                         ▼
              ┌─────────────────────┐
              │ 加权融合 + 排序      │
              │ Result = α*K + β*S + γ*D
              └─────────────────────┘
                         │
                         ▼
              返回 Top-K 相关表
```

---

## 4. ES索引设计

### 4.1 索引名称
```
tables_metadata
```

### 4.2 索引Mapping
```json
{
  "mappings": {
    "properties": {
      "db_name": {
        "type": "keyword"
      },
      "table_name": {
        "type": "keyword"
      },
      "all_names": {
        "type": "text",
        "analyzer": "standard"
      },
      "sparse_vector": {
        "type": "rank_features"
      },
      "dense_vector": {
        "type": "dense_vector",
        "dims": 1024,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}
```

### 4.3 文档结构
```json
{
  "db_name": "soccer_3",
  "table_name": "player",
  "all_names": "soccer_3 player player_id player_name team_id position goals assists",
  "sparse_vector": {...},
  "dense_vector": [...]
}
```

---

## 5. 配置设计

### 5.1 配置文件结构 (config.yaml)
```yaml
# 服务配置
services:
  # 向量服务
  sparse_vector:
    url: "http://192.168.116.5:8080/api/v1/sparse_vector"
  dense_vector:
    url: "http://192.168.116.5:8080/api/v1/dense_vector"
  translate:
    url: "http://192.168.116.5:8080/api/v1/translate"
  
  # Elasticsearch
  elasticsearch:
    hosts: ["http://192.168.116.5:9200"]
    index: "tables_metadata"
  
  # LLM (暂不使用)
  llm:
    base_url: "https://openrouter.ai/api/v1"
    api_key: "${OPENROUTER_API_KEY}"
    model: "x-ai/grok-4.1-fast"

# 召回配置
recall:
  # 融合权重
  weights:
    keyword: 0.3
    sparse: 0.3
    dense: 0.4
  
  # 召回数量
  top_k: 5
  
  # 关键词匹配阈值
  keyword_threshold: 0.5

# 数据路径
data:
  metadata: "data/metadatas.json"
  queries: "data/query_and_tables.json"

# 日志
logging:
  level: "INFO"
```

### 5.2 环境变量
```bash
# 必须设置
export OPENROUTER_API_KEY="your-api-key"

# 可选，配置文件中已包含默认值
export ES_HOSTS='["http://192.168.116.5:9200"]'
```

---

## 6. 核心算法设计

### 6.1 关键词召回
```python
def keyword_recall(query: str, tables: List[TableInfo]) -> List[RecallResult]:
    # 1. 提取查询中的关键词 (简单实现: 分词 + 名词提取)
    keywords = extract_keywords(query)
    
    # 2. 对每个表计算匹配得分
    results = []
    for table in tables:
        score = 0.0
        # 表名匹配
        if table.table_name in query:
            score += 1.0
        # 字段名匹配
        for col in table.columns:
            if col in query:
                score += 0.5
        # 关键词匹配
        for kw in keywords:
            if kw in table.table_name or kw in ' '.join(table.columns):
                score += 0.3
        
        if score > 0:
            results.append(RecallResult(...))
    
    return normalize_scores(results)
```

### 6.2 稀疏向量召回
```python
def sparse_recall(query: str) -> List[RecallResult]:
    # 1. 调用稀疏向量API
    sparse_vec = call_sparse_vector_api(query)
    
    # 2. ES检索
    response = es.search(
        index="tables_metadata",
        knn={
            "field": "sparse_vector",
            "query_vector": sparse_vec,
            "k": 10
        }
    )
    
    # 3. 解析结果
    return parse_es_response(response)
```

### 6.3 稠密向量召回
```python
def dense_recall(query: str) -> List[RecallResult]:
    # 1. 调用稠密向量API
    dense_vec = call_dense_vector_api(query)
    
    # 2. ES检索 (使用dense_vector字段)
    response = es.search(
        index="tables_metadata",
        knn={
            "field": "dense_vector",
            "query_vector": dense_vec,
            "k": 10,
            "similarity": 0.7
        }
    )
    
    # 3. 解析结果
    return parse_es_response(response)
```

### 6.4 结果融合
```python
def fusion(
    keyword_results: List[RecallResult],
    sparse_results: List[RecallResult],
    dense_results: List[RecallResult],
    weights: dict = {"keyword": 0.3, "sparse": 0.3, "dense": 0.4}
) -> List[RecallResult]:
    # 1. 合并所有结果 (按db_name+table_name)
    all_tables = {}
    for r in keyword_results + sparse_results + dense_results:
        key = f"{r.db_name}.{r.table_name}"
        if key not in all_tables:
            all_tables[key] = {"keyword": 0, "sparse": 0, "dense": 0, "db_name": r.db_name, "table_name": r.table_name}
        all_tables[key][r.match_type] = r.score
    
    # 2. 加权计算
    final_results = []
    for key, scores in all_tables.items():
        final_score = (
            weights["keyword"] * scores["keyword"] +
            weights["sparse"] * scores["sparse"] +
            weights["dense"] * scores["dense"]
        )
        final_results.append(RecallResult(
            db_name=scores["db_name"],
            table_name=scores["table_name"],
            score=final_score,
            match_type="hybrid"
        ))
    
    # 3. 排序返回
    return sorted(final_results, key=lambda x: x.score, reverse=True)[:top_k]
```

---

## 7. CLI命令设计

### 7.1 命令结构
```bash
nl2sql_v3 <command> [options]
```

### 7.2 命令列表
| 命令 | 说明 | 示例 |
|------|------|------|
| recall | 单次召回 | `nl2sql_v3 recall "哪些球员得分最高"` |
| build-index | 构建ES索引 | `nl2sql_v3 build-index` |
| evaluate | 批量评估 | `nl2sql_v3 evaluate` |
| serve | 启动API服务(可选) | `nl2sql_v3 serve --port 8080` |

### 7.3 命令详情

#### recall 命令
```bash
nl2sql_v3 recall "球员信息" --top-k 5 --show-scores
```

输出:
```
Query: 球员信息

Top 5 Related Tables:
┌──────────────┬───────────┬──────────┬─────────┐
│ DB Name      │ Table Name│ Score    │ Match  │
├──────────────┼───────────┼──────────┼────────┤
│ soccer_3     │ player    │ 0.92     │ hybrid │
│ soccer_3     │ club      │ 0.65     │ sparse │
│ soccer_3     │ match     │ 0.58     │ dense  │
│ soccer_3     │ team      │ 0.45     │ keyword│
│ soccer_3     │ coach     │ 0.32     │ dense  │
└──────────────┴───────────┴──────────┴────────┘
```

#### evaluate 命令
```bash
nl2sql_v3 evaluate
```

输出:
```
Evaluation Results:
┌──────────────┬────────────┬────────────┬────────────┐
│ Total Queries│ Hit@1     │ Hit@3      │ Hit@5      │
├──────────────┼────────────┼────────────┼────────────┤
│ 186          │ 0.72       │ 0.88       │ 0.93       │
└──────────────┴────────────┴────────────┴────────────┘

MRR: 0.82
```

---

## 8. 项目结构

```
nl2sql_v3/
├── config.yaml           # 配置文件
├── requirements.txt      # 依赖
├── pyproject.toml        # 项目配置
├── main.py               # 入口
├── nl2sql_v3/
│   ├── __init__.py
│   ├── cli.py            # CLI命令
│   ├── config.py         # 配置管理
│   ├── client/
│   │   ├── __init__.py
│   │   ├── api_client.py # API调用封装
│   │   └── es_client.py  # ES操作封装
│   ├── recall/
│   │   ├── __init__.py
│   │   ├── base.py       # 基类
│   │   ├── keyword.py    # 关键词召回
│   │   ├── sparse.py     # 稀疏向量召回
│   │   ├── dense.py      # 稠密向量召回
│   │   └── fusion.py     # 结果融合
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py     # 元数据加载
│   │   └── evaluator.py  # 评估
│   └── utils/
│       ├── __init__.py
│       └── logger.py     # 日志
├── data/
│   ├── metadatas.json    # 表元数据
│   └── query_and_tables.json  # 评估数据
└── tests/
    └── ...
```

---

*技术架构设计完成*
