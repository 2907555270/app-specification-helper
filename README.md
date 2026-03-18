# App Specification Helper

> AI 驱动的应用规格说明书生成与项目初始化工具

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

根据收集的信息，自动生成以下文档模板：

| 模板文件 | 说明 |
|---------|------|
| `templates/main.md` | 项目概览和索引 |
| `templates/requirements.md` | 详细需求文档 |
| `templates/architecture.md` | 技术架构设计 |
| `templates/api.md` | API 接口设计 |
| `templates/update.md` | 需求变更记录 |

### 3. 项目初始化

AI 阅读 `templates/` 目录下的所有模板文件，理解项目需求和技术规格后，生成初始项目代码。

```
templates/*.md → AI 阅读理解 → 项目代码生成
```

## 工作原理

1. **用户描述**：告诉 AI 你想开发什么项目
2. **AI 引导**：通过一系列问题帮助你梳理需求
3. **生成模板**：自动创建规范的项目文档
4. **初始化项目**：AI 阅读模板并生成基础代码框架

## 局限性

⚠️ **重要提示**：

- **初始框架**：Skill 帮助你搭建项目的**初始框架**，生成规范的项目文档
- **方案设计**：具体的技术方案、架构设计、实现路径需要**你自己思考和完善**
- **持续迭代**：项目的完整实现需要在开发过程中**不断调整和优化**

也就是说，AI 可以帮你：
- 整理思路，形成规范文档
- 生成基础代码结构
- 提供技术选型建议

但最终的技术决策和实现细节，需要结合你的业务场景和实践经验来确定。

## 项目示例

[nl2sql_v3](./nl2sql_v3/) 是基于本工具生成的 NL2SQL 智能对话系统，展示了从需求描述到项目初始化的完整过程。该系统支持多轮对话式 SQL 生成，采用 LangGraph Agent 架构实现意图澄清与智能 SQL 生成。

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
├── nl2sql_v3/                  # 生成的示例项目
└── README.md                   # 本文件
```

## 相关文档

- [nl2sql_v3 项目文档](./nl2sql_v3/README.md)
- [RAG 技术指南](./nl2sql_v3/technical_doc/rag.md)
- [基准测试报告](./nl2sql_v3/technical_doc/benchmark.md)
