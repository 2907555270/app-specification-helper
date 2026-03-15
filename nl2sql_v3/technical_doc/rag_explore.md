# 表召回问题与解决方案汇总

本文档记录 NL2SQL 系统在表召回实践中遇到的各种问题、根本原因分析及解决方案，供后续开发参考。

---

## 一、混合检索架构概述

### 1.1 三路召回流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        表召回完整流程                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   输入：自然语言查询 "销售人员的年度业绩报表"                       │
│                         │                                           │
│         ┌───────────────┼───────────────┐                          │
│         ▼               ▼               ▼                          │
│   ┌───────────┐   ┌───────────┐   ┌───────────┐                    │
│   │ BM25 召回 │   │ Sparse    │   │ Dense     │                    │
│   │ 关键词    │   │ 向量检索  │   │ 向量检索  │                    │
│   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘                    │
│         │               │               │                          │
│         └───────────────┼───────────────┘                          │
│                         ▼                                           │
│               ┌─────────────────────┐                                │
│               │   RRF 融合排序      │                                │
│               │ ( Reciprocal Rank  │                                │
│               │   Fusion )         │                                │
│               └─────────┬───────────┘                                │
│                         │                                            │
│                         ▼                                           │
│               ┌─────────────────────┐                                │
│               │  Cross-Encoder 重排 │                                │
│               │  (Rerank)           │                                │
│               └─────────┬───────────┘                                │
│                         │                                            │
│                         ▼                                            │
│              输出：Top-K 相关表列表                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心配置参数

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `hybrid_search_top_k` | RRF 融合后输出数量 | 100 |
| `rerank_threshold` | Rerank 分数阈值，低于此值丢弃 | 0.5 |
| `weights` | 三路召回权重（keyword/sparse/dense） | {0.1, 0.6, 0.3} |
| `k` | RRF 平滑常数 | 60 |

---

## 二、Elasticsearch 原生 RRF 限制

### 2.1 问题描述

在尝试使用 Elasticsearch 原生的 RRF 功能时，遇到授权错误：

```
AuthorizationException: current license is non-compliant 
for [Reciprocal Rank Fusion (RRF)]
```

### 2.2 原因分析

Elasticsearch 的 RRF（Reciprocal Rank Fusion）功能属于 **X-Pack 高级特性**，需要 Platinum 或更高版本的许可证。开源版（Basic/OSS）无法使用此功能。

### 2.3 解决方案

采用 **Python 端实现 RRF**：

1. 分别调用 3 次 ES 查询（BM25、Sparse、Dense）
2. 收集各路返回的文档 ID 和排名
3. 在 Python 端手动计算 RRF 分数
4. 排序输出融合结果

```python
def manual_rrf(self, all_runs: List[Dict[str, float]], k: int = 60, weights: Optional[List[float]] = None) -> List[tuple]:
    doc_scores: Dict[str, float] = {}
    weights = weights or [1.0] * len(all_runs)
    for run_dict, weight in zip(all_runs, weights):
        if not run_dict:
            continue
        sorted_items = sorted(run_dict.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(sorted_items, 1):
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            doc_scores[doc_id] += weight * 10 * 1.0 / (k + rank)
    
    if not doc_scores:
        return []
    
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:k]
```

---

## 三、RRF 性能问题

### 3.1 问题描述

初次实现 RRF 时，使用 `ranx` 库进行融合，融合时间高达 **28 秒**，严重影响查询性能。

### 3.2 原因分析

1. `ranx` 库内部实现复杂，有额外的预处理开销
2. 数据格式转换开销大
3. 对小规模数据（表召回通常只需融合几百个表）而言，库的开销远大于计算本身

### 3.3 解决方案

移除 `ranx` 依赖，改用纯 Python 实现 RRF 算法：

```python
def manual_rrf(self, all_runs: List[Dict[str, float]], k: int = 60, weights: Optional[List[float]] = None) -> List[tuple]:
    doc_scores: Dict[str, float] = {}
    weights = weights or [1.0] * len(all_runs)
    for run_dict, weight in zip(all_runs, weights):
        if not run_dict:
            continue
        sorted_items = sorted(run_dict.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(sorted_items, 1):
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            doc_scores[doc_id] += weight * 10 * 1.0 / (k + rank)
    
    if not doc_scores:
        return []
    
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:k]
```

### 3.4 优化效果

| 指标 | 优化前 (ranx) | 优化后 (纯 Python) |
|------|-------------|-------------------|
| 融合时间 | ~28 秒 | < 50 毫秒 |
| 依赖 | ranx | 无 |

---

## 四、Reranker 模型输出负数分数问题

### 4.1 问题描述

在使用 BGE Reranker 模型进行重排序时，发现输出的分数包含**负数**。由于系统设置了**按最小分数阈值过滤**（`rerank_threshold`），导致重排序后输出结果异常：

- 大量本应保留的表因分数为负数而被过滤掉
- 阈值设置困难：设置过高会丢失相关表，设置过低会保留过多不相关表

### 4.2 BGE Reranker 分数范围

BGE Reranker 模型的输出分数范围是 **[-1, 1]**：

| 分数范围 | 语义含义 |
|----------|----------|
| > 0 | 正相关（Query 与 Document 相关） |
| = 0 | 无相关性 |
| < 0 | 负相关（Query 与 Document 不相关） |

**示例**：
- `0.9` - 非常相关
- `0.5` - 相关
- `0.0` - 无相关性
- `-0.5` - 不相关
- `-0.9` - 非常不相关

### 4.3 原因分析

Cross-Encoder 类型的 Reranker 模型（如 BGE-Reranker-v2-m3）基于 **Logitech/Sigmoid** 输出，天然包含负数：

1. **模型输出机制**：Cross-Encoder 将 Query 和 Document 拼接后联合编码，输出一个 logit 值
2. **Sigmoid 变换**：通过 Sigmoid 函数将 logit 映射到 (0, 1) 区间，但某些实现可能直接输出 logit（未经过 Sigmoid）
3. **训练目标**：模型训练时使用对比学习，正样本分数高，负样本分数低，自然包含负数区间

### 4.4 解决方案

#### 方案一：调整阈值策略（推荐）

使用 **动态阈值** 或 **负数阈值**：

```python
# 方案1：使用负数阈值，允许保留负分数的结果
rerank_threshold = -0.1  # 改为负数阈值

# 方案2：使用百分位动态计算阈值
import numpy as np
rerank_threshold = np.percentile(all_rerank_scores, 25)  # 使用第25百分位
```

#### 方案二：分数归一化

将分数映射到 [0, 1] 或 [0, 100] 的正数区间：

```python
# Min-Max 归一化
def normalize_scores(scores: List[float]) -> List[float]:
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]

# 使用归一化后的分数
normalized_scores = normalize_scores(rerank_scores)
```

#### 方案三：基于排名的阈值

不依赖绝对分数值，而是基于排名位置：

```python
# 只保留 Top-K，不依赖阈值
top_k = 10
final_results = reranked_results[:top_k]
```

### 4.5 配置建议

| 场景 | 推荐配置 |
|------|---------|
| 保守（只保留高相关） | `rerank_threshold: 0.3` |
| 平衡（默认推荐） | `rerank_threshold: 0.0` 或 `-0.1` |
| 宽松（保留更多候选） | `rerank_threshold: -0.3` |
| 完全不过滤 | `rerank_threshold: -1.0` |

### 4.6 最佳实践

1. **默认使用负数阈值**：建议初始设置 `rerank_threshold: -0.1` 或 `0.0`
2. **结合业务场景调整**：根据召回率/精确率要求灵活调整
3. **监控分数分布**：定期分析 rerank 分数分布，优化阈值
4. **考虑使用 Ranker 而非 Reranker**：如果分数问题难以处理，可考虑使用 Bi-Encoder + 简单相似度作为备选

---

## 五、KNN 检索过滤问题

### 5.1 问题描述

在 `dense_search` 方法中，使用 `filter_db_name` 参数过滤指定数据库的表时，发现过滤不生效，返回结果中包含了所有数据库的表。

### 5.2 原因分析

KNN（K-Nearest Neighbors）查询与普通查询不同，有自己独立的执行路径。最初尝试在 `query` 中添加 filter：

```python
# 错误写法 - filter 不会生效
body["query"] = {
    "bool": {
        "filter": [{"term": {"db_name": filter_db_name}}]
    }
}
```

这种写法只对普通查询有效，KNN 查询会忽略 `query` 中的 filter。

### 5.3 解决方案

#### 方案一：使用 post_filter（后置过滤）

```python
body["post_filter"] = {"term": {"db_name": filter_db_name}}
```

**缺点**：先执行 KNN 搜索，再对结果过滤，性能较低。

#### 方案二：使用 KNN 内置 filter（前置过滤）✅ 推荐

```python
knn_config: Dict[str, Any] = {
    "field": "dense_vector",
    "query_vector": dense_vector,
    "k": size,
    "num_candidates": size * 2,
}

if filter_db_name:
    knn_config["filter"] = {"term": {"db_name": filter_db_name}}

body = {
    "size": size,
    "knn": knn_config
}
```

### 5.4 两种过滤方式对比

| 方式 | 位置 | 执行顺序 | 性能 |
|------|------|---------|------|
| **pre-filter** | `knn.filter` | KNN 搜索**之前**过滤 | ✅ 更优 |
| **post-filter** | `post_filter` | KNN 搜索**之后**过滤 | 较低 |

---

## 六、权重配置问题

### 6.1 分数尺度差异

**问题**：三路召回（BM25、Sparse、Dense）返回的分数尺度完全不同，直接加权融合不合理。

**解决方案**：使用 RRF 融合，忽略绝对分数值，只使用排名信息。

### 6.2 权重设置建议

根据不同场景调整权重：

| 场景 | keyword | sparse | dense |
|------|---------|--------|-------|
| 表名/列名精确匹配多 | 0.4 | 0.3 | 0.3 |
| 自然语言查询多 | 0.1 | 0.4 | 0.5 |
| 混合场景 | 0.2 | 0.4 | 0.4 |

---

## 七、评估指标与阈值

### 7.1 评估指标

| 指标 | 说明 |
|------|------|
| Hit@K | 前 K 个结果中包含正确表的概率 |
| MRR | 平均倒数排名（Mean Reciprocal Rank） |

### 7.2 阈值确定方法

动态计算 `rerank_threshold`：

```python
min_rerank_threshold = calculate_min_threshold(
    scores=rerank_scores,
    percentile=25  # 使用第25百分位作为最低阈值
)
```

---

## 八、总结与最佳实践

### 8.1 问题分类

| 类别 | 问题数 | 典型例子 |
|------|--------|---------|
| 许可限制 | 1 | ES RRF 需要付费版 |
| 性能问题 | 1 | ranx 融合耗时 28 秒 |
| Reranker 分数 | 1 | 负数分数导致过滤异常 |
| KNN 过滤 | 1 | KNN filter 不生效 |
| 权重配置 | 1 | 分数尺度差异 |
| 评估 | 1 | 阈值计算 |

### 8.2 最佳实践

1. **优先使用排名信息**：不同召回源分数尺度不同，RRF 只用排名更鲁棒
2. **做好空值保护**：所有外部数据都需要做 None 检查
3. **避免过度依赖库**：简单算法可自己实现，减少依赖和性能开销
4. **配置集中管理**：所有可调参数放在 config.yaml 中
5. **添加详细日志**：便于排查线上问题
6. **KNN 过滤使用 pre-filter**：在 knn 配置内添加 filter，性能更优
7. **Reranker 阈值设为负数**：BGE Reranker 输出范围是 [-1, 1]，默认阈值建议设为 0.0 或 -0.1

### 8.3 技术选型建议

| 场景 | 推荐方案 |
|------|---------|
| 多路召回融合 | Python 端 RRF（避免 license 问题） |
| 重排模型 | BGE-Reranker-v2-m3 |
| 向量检索 | BGE-M3（Sparse + Dense 双输出） |
| 混合搜索 | BM25 + Sparse + Dense → RRF → Rerank |
