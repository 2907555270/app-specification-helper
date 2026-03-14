# API 接口设计

## 1. 接口说明

本项目为CLI工具，主要通过命令行交互。API文档主要描述：
1. 内部模块接口定义
2. 外部服务集成接口

---

## 2. 内部模块接口

### 2.1 表召回引擎接口

#### ITableRetriever
```python
class ITableRetriever(Protocol):
    """表召回器接口"""
    
    def recall(self, query: str, top_k: int = 5) -> List[TableMatch]:
        """
        根据自然语言问题召回相关表
        
        Args:
            query: 自然语言问题
            top_k: 返回前k个结果
            
        Returns:
            相关表列表，按得分降序排列
        """
        ...
```

#### IRecallMethod
```python
class IRecallMethod(Protocol):
    """召回方法接口"""
    
    def recall(self, query: str) -> List[RecallResult]:
        """
        执行召回
        
        Args:
            query: 查询文本
            
        Returns:
            召回结果列表
        """
        ...
```

### 2.2 数据模型

#### TableMatch
```python
class TableMatch(BaseModel):
    db_name: str          # 数据库名
    table_name: str       # 表名
    score: float          # 得分 (0-1)
    match_type: str       # 召回类型: "keyword", "sparse", "dense", "hybrid"
```

#### RecallResult
```python
class RecallResult(BaseModel):
    db_name: str
    table_name: str
    score: float
    match_type: str
```

#### EvaluationResult
```python
class EvaluationResult(BaseModel):
    total_queries: int
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mrr: float
    details: List[Dict]
```

---

## 3. 外部服务接口

### 3.1 稀疏向量服务

**端点**: `POST /api/v1/sparse_vector`

**请求**:
```json
{
    "text": "要向量化的文本",
    "query_mode": true
}
```

**响应**:
```json
{
    "sparse_id_vector": {"1234": 0.5, "5678": 0.3},
    "sparse_word_vector": {"hello": 0.5, "world": 0.3}
}
```

### 3.2 稠密向量服务

**端点**: `POST /api/v1/dense_vector`

**请求**:
```json
{
    "text": "要向量化的文本"
}
```

**响应**:
```json
{
    "dense_vector": [0.1, 0.2, 0.3, ...]
}
```

### 3.3 翻译服务

**端点**: `POST /api/v1/translate`

**请求**:
```json
{
    "text": "要翻译的中文文本"
}
```

**响应**:
```json
{
    "translated_text": "Translated English text"
}
```

---

## 4. CLI命令接口

### 4.1 recall 命令

```bash
nl2sql_v3 recall <query> [OPTIONS]
```

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| query | STRING | 是 | 自然语言问题 |

**选项**:
| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --top-k | INTEGER | 5 | 返回前k个结果 |
| --show-scores | FLAG | False | 显示详细得分 |
| --translate | FLAG | False | 是否翻译 |
| --weights | TEXT | "0.3,0.3,0.4" | 融合权重(逗号分隔) |

### 4.2 build-index 命令

```bash
nl2sql_v3 build-index [OPTIONS]
```

**选项**:
| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --force | FLAG | False | 强制重建索引 |
| --batch-size | INTEGER | 100 | 批量写入大小 |

### 4.3 evaluate 命令

```bash
nl2sql_v3 evaluate [OPTIONS]
```

**选项**:
| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --output | TEXT | None | 结果输出文件 |
| --verbose | FLAG | False | 显示详细信息 |

---

## 5. 错误码定义

### 5.1 CLI错误码
| 错误码 | 说明 |
|--------|------|
| 0 | 成功 |
| 1 | 参数错误 |
| 2 | 服务不可用 |
| 3 | 数据加载失败 |
| 4 | ES操作失败 |
| 5 | API调用失败 |

### 5.2 服务错误码
| 错误码 | 说明 |
|--------|------|
| 1001 | 向量服务超时 |
| 1002 | ES连接失败 |
| 1003 | 索引不存在 |
| 1004 | 文档解析错误 |

---

## 6. 日志接口

### 6.1 日志级别
| 级别 | 说明 |
|------|------|
| DEBUG | 详细调试信息 |
| INFO | 一般信息 |
| WARNING | 警告 |
| ERROR | 错误 |

### 6.2 日志格式
```json
{
    "timestamp": "2024-01-01T00:00:00.000Z",
    "level": "INFO",
    "module": "recall.keyword",
    "message": "召回完成",
    "query": "球员信息",
    "result_count": 5,
    "latency_ms": 125
}
```

---

*API接口设计完成*
