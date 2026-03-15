# App Specification Helper

AI 驱动的应用规格说明书生成与项目初始化工具

## 概述

本项目是一个基于 AI 的应用开发辅助工具，通过引导用户描述需求，自动生成规范的项目文档，并支持直接初始化项目代码。

## 核心能力

### 1. 项目开发指导（SKILL）

通过激活 `project-guide` Skill，AI 可以引导用户完成项目需求收集：

**触发条件**：
- 用户想要开发新项目
- 用户表达项目需求
- 用户想要创建应用

**工作流程**：

```
用户描述需求 → AI 引导提问 → 收集项目信息 → 生成文档模板
```

**引导内容**：

| 阶段 | 内容 |
|------|------|
| 项目概览 | 名称、类型、简介、核心目标 |
| 详细需求 | 用户群体、核心功能、业务流程、数据需求 |
| 技术约束 | 技术偏好、已有资源、约束条件 |

### 2. 模板生成

根据收集的信息，自动生成以下文档：

| 模板文件 | 说明 |
|---------|------|
| `templates/main.md` | 项目概览和索引 |
| `templates/requirements.md` | 详细需求文档 |
| `templates/architecture.md` | 技术架构设计 |
| `templates/api.md` | API 接口设计 |
| `templates/update.md` | 需求变更记录 |

### 3. 项目初始化

AI 阅读 `templates/` 目录下的所有模板文件，理解项目需求和技术规格后，自动生成项目代码。

```
templates/*.md → AI 阅读理解 → 项目代码生成
```

## 项目示例：nl2sql_v3

[nl2sql_v3](./nl2sql_v3/) 是基于本工具生成的 NL2SQL 表召回系统。

### 生成过程

1. **用户描述需求**：想要一个 NL2SQL 表召回工具
2. **AI 引导提问**：了解检索方式、向量模型、重排需求等
3. **生成模板文档**：创建了 requirements.md、architecture.md 等
4. **AI 阅读模板**：理解混合检索、RRF 融合、重排等技术方案
5. **生成项目代码**：完成 ES 客户端、召回模块、CLI 接口等

### 技术特性

- 多路混合检索（BM25 + Sparse + Dense）
- Weighted RRF 融合排序
- Cross-Encoder 重排
- CLI 交互式查询
- 批量评估功能

### 性能指标

| 场景 | Hit@5 | MRR |
|------|-------|-----|
| Determined | **100%** | 0.93 |
| Not Determined | 85.3% | 0.70 |

## 快速开始

### 使用 SKILL

1. 告诉 AI 你想开发什么项目
2. AI 会引导你回答一系列问题
3. 查看生成的模板文档
4. 确认后 AI 开始生成项目代码

### 运行 nl2sql_v3

```bash
cd nl2sql_v3

# 安装依赖
pip install -e .

# 配置服务
编辑 config.yaml

# 构建索引
python main.py build-index

# 交互式查询
python main.py interactive
```

## 目录结构

```
.
├── .trae/                      # Trae AI 配置
│   └── skills/
│       └── project-guide/       # 项目开发指导 Skill
│           └── SKILL.md
├── templates/                   # 项目模板目录
│   ├── main.md
│   ├── requirements.md
│   ├── architecture.md
│   ├── api.md
│   └── update.md
├── nl2sql_v3/                  # 生成的 NL2SQL 项目
│   ├── client/
│   ├── recall/
│   ├── technical_doc/
│   ├── main.py
│   └── config.yaml
└── README.md                   # 本文件
```

## 相关文档

- [nl2sql_v3 项目文档](./nl2sql_v3/README.md)
- [RAG 技术指南](./nl2sql_v3/technical_doc/rag.md)
- [基准测试报告](./nl2sql_v3/technical_doc/benchmark.md)
- [问题记录](./nl2sql_v3/technical_doc/rag_explore.md)
