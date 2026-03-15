import json
import sys
from pathlib import Path
from typing import Any, Dict, List

script_root = Path(__file__).parent.parent.parent
input_file = script_root / "data" / "test.json"
tables_file = script_root / "data" / "test_tables.json"
output_file = script_root / "data" / "test_query_and_tables.json"


def load_db_table_mapping(tables_path: str) -> Dict[str, List[str]]:
    """加载 db_id -> table_names 的映射"""
    with open(tables_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    mapping = {}
    for db in tables_data:
        db_id = db.get("db_id", "")
        table_names = db.get("table_names", [])
        mapping[db_id] = table_names

    return mapping


def extract_tables_from_sql(sql: Dict[str, Any], table_names: List[str]) -> List[str]:
    """从 SQL 结构中提取表名"""
    tables = []
    from_clause = sql.get("from", {})
    table_units = from_clause.get("table_units", [])

    for unit in table_units:
        if isinstance(unit, list) and len(unit) >= 2:
            unit_type = unit[0]
            idx = unit[1]
            if unit_type == "table_unit" and isinstance(idx, int) and idx < len(table_names):
                tables.append(table_names[idx])

    return tables


def convert_test_to_query_and_tables(input_path: str, tables_path: str, output_path: str):
    db_table_mapping = load_db_table_mapping(tables_path)

    with open(input_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []

    for item in test_data:
        db_id = item.get("db_id", "")
        question = item.get("question", "")
        sql = item.get("sql", {})

        table_names = db_table_mapping.get(db_id, [])
        tables = extract_tables_from_sql(sql, table_names)

        results.append({
            "db_name": db_id,
            "question": question,
            "tables": tables,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(results)} records to {output_path}")

    if results:
        print(f"Sample: {results[0]}")


if __name__ == "__main__":
    convert_test_to_query_and_tables(str(input_file), str(tables_file), str(output_file))
