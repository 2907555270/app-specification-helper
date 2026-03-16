import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SQLGenerationResult(BaseModel):
    sql: str
    confidence: float
    explanation: str
    selected_tables: List[str]
    used_columns: List[str]


class PromptTemplate:
    SYSTEM_PROMPT = """你是一个专业的SQL查询生成专家。你的任务是根据用户提出的自然语言问题，从给定的表结构中选择合适的表和字段，生成对应的SQL查询语句。

## 任务要求：
1. 仔细分析用户问题，理解其意图
2. 从提供的表结构中选择与问题最相关的表
3. 生成准确、高效的SQL语句
4. 评估你的SQL生成置信度

## 输出格式：
请以JSON格式返回结果，包含以下字段：
- sql: 生成的SQL语句
- confidence: 置信度 (0.0-1.0)
- explanation: 简要解释你的SQL生成逻辑
- selected_tables: 使用的表列表
- used_columns: 使用的字段列表

## 注意事项：
- 只选择与问题相关的表，避免不必要的JOIN
- 确保SQL语法正确
- 考虑字段之间的关联关系
- 如果无法生成有效的SQL，confidence应设置较低"""

    USER_PROMPT_TEMPLATE = """## 表结构信息：
{schema}

## 用户问题：
{query}

## 要求：
请生成对应的SQL查询语句。"""

    FEWSHOT_EXAMPLES = [
        {
            "schema": """## database.user_table - 用户表
user_id: int [PK] - 用户ID, username: varchar - 用户名, email: varchar - 邮箱, created_at: datetime - 创建时间

## database.order_table - 订单表
order_id: int [PK] - 订单ID, user_id: int [FK] - 用户ID, amount: decimal - 金额, status: varchar - 订单状态, order_date: datetime - 订单日期""",
            "query": "查询所有用户的订单总金额",
            "result": {
                "sql": "SELECT u.user_id, u.username, SUM(o.amount) as total_amount FROM database.user_table u LEFT JOIN database.order_table o ON u.user_id = o.user_id GROUP BY u.user_id, u.username",
                "confidence": 0.95,
                "explanation": "通过user_id将用户表和订单表关联，使用SUM函数计算每个用户的订单总金额，使用LEFT JOIN确保包含没有订单的用户",
                "selected_tables": ["database.user_table", "database.order_table"],
                "used_columns": ["user_id", "username", "amount"]
            }
        },
        {
            "schema": """## sales.products - 产品表
product_id: int [PK] - 产品ID, product_name: varchar - 产品名称, category: varchar - 类别, price: decimal - 价格

## sales.inventory - 库存表
inventory_id: int [PK] - 库存ID, product_id: int [FK] - 产品ID, quantity: int - 数量, warehouse: varchar - 仓库""",
            "query": "查找库存少于10件的产品",
            "result": {
                "sql": "SELECT p.product_id, p.product_name, p.category, i.quantity, i.warehouse FROM sales.products p INNER JOIN sales.inventory i ON p.product_id = i.product_id WHERE i.quantity < 10",
                "confidence": 0.9,
                "explanation": "通过product_id关联产品和库存表，使用WHERE条件筛选数量少于10的产品",
                "selected_tables": ["sales.products", "sales.inventory"],
                "used_columns": ["product_id", "product_name", "category", "quantity", "warehouse"]
            }
        }
    ]

    @classmethod
    def build_prompt(
        cls,
        query: str,
        schema: str,
        include_fewshot: bool = True,
        custom_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        messages = []

        messages.append({
            "role": "system",
            "content": cls.SYSTEM_PROMPT
        })

        if include_fewshot:
            examples = custom_examples or cls.FEWSHOT_EXAMPLES
            for example in examples:
                messages.append({
                    "role": "user",
                    "content": cls.USER_PROMPT_TEMPLATE.format(
                        schema=example["schema"],
                        query=example["query"]
                    )
                })
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(example["result"], ensure_ascii=False)
                })

        messages.append({
            "role": "user",
            "content": cls.USER_PROMPT_TEMPLATE.format(
                schema=schema,
                query=query
            )
        })

        return messages


class CompactPromptTemplate(PromptTemplate):
    SYSTEM_PROMPT = """你是一个SQL查询生成专家。根据表结构和用户问题，生成SQL。

输出JSON格式：
{"sql": "...", "confidence": 0.0-1.0, "explanation": "...", "selected_tables": [...], "used_columns": [...]}"""

    USER_PROMPT_TEMPLATE = """表结构:
{schema}

问题: {query}

SQL:"""

    FEWSHOT_EXAMPLES = [
        {
            "schema": "## db.users\nuser_id: int [PK], username: varchar, email: varchar\n\n## db.orders\norder_id: int [PK], user_id: int [FK], amount: decimal, status: varchar",
            "query": "查询每个用户的订单数量",
            "result": {
                "sql": "SELECT u.user_id, u.username, COUNT(o.order_id) as order_count FROM db.users u LEFT JOIN db.orders o ON u.user_id = o.user_id GROUP BY u.user_id, u.username",
                "confidence": 0.95,
                "explanation": "通过user_id关联users和orders表，使用COUNT统计订单数量",
                "selected_tables": ["db.users", "db.orders"],
                "used_columns": ["user_id", "username", "order_id"]
            }
        }
    ]
