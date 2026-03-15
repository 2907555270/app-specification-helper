import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Set

script_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_root))

from nl2sql_v3.models.table_info import TableInfo, ColumnInfo, ForeignKeyRelation, RelatedTable


def load_tables_json(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_structured_tables(file_path: str) -> List[TableInfo]:
    """加载已转换的结构化表数据 (tables_structured.json)"""
    data = load_tables_json(file_path)
    
    tables = []
    for item in data:
        columns = []
        for col in item.get("columns", []):
            columns.append(ColumnInfo(
                name=col.get("name", ""),
                name_cn=col.get("name_cn", ""),
                type=col.get("type", "text"),
                is_primary_key=col.get("is_primary_key", False),
                is_foreign_key=col.get("is_foreign_key", False),
            ))
        
        foreign_keys = []
        for fk in item.get("foreign_keys", []):
            foreign_keys.append(ForeignKeyRelation(
                source_db=fk.get("source_db", ""),
                source_table=fk.get("source_table", ""),
                source_column=fk.get("source_column", ""),
                target_db=fk.get("target_db", ""),
                target_table=fk.get("target_table", ""),
                target_column=fk.get("target_column", ""),
            ))
        
        related_tables = []
        for rt in item.get("related_tables", []):
            related_tables.append(RelatedTable(
                table_name=rt.get("table_name", ""),
                db_name=rt.get("db_name", ""),
                join_columns=rt.get("join_columns", []),
            ))
        
        tables.append(TableInfo(
            db_name=item.get("db_name", ""),
            table_name=item.get("table_name", ""),
            table_name_cn=item.get("table_name_cn", ""),
            columns=columns,
            primary_keys=item.get("primary_keys", []),
            foreign_keys=foreign_keys,
            related_tables=related_tables,
        ))
    
    return tables


def extract_table_info(db_data: Dict[str, Any]) -> List[TableInfo]:
    db_name = db_data.get("db_id", "")
    table_names = db_data.get("table_names", [])
    table_names_original = db_data.get("table_names_original", [])
    column_names = db_data.get("column_names", [])
    column_types = db_data.get("column_types", [])
    foreign_keys = db_data.get("foreign_keys", [])
    primary_keys = db_data.get("primary_keys", [])

    table_column_map: Dict[int, List[tuple]] = defaultdict(list)
    for idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx >= 0:
            col_type = column_types[idx] if idx < len(column_types) else "text"
            table_column_map[table_idx].append((idx, col_name, col_type))

    fk_relations: Dict[int, List[tuple]] = defaultdict(list)
    for fk in foreign_keys:
        if len(fk) == 2:
            src_idx, tgt_idx = fk
            if src_idx < len(column_names) and tgt_idx < len(column_names):
                src_table_idx = column_names[src_idx][0]
                tgt_table_idx = column_names[tgt_idx][0]
                fk_relations[src_table_idx].append((src_idx, tgt_idx))

    table_info_list: List[TableInfo] = []
    for table_idx, table_name in enumerate(table_names):
        table_name_cn = ""
        if table_idx < len(table_names_original):
            table_name_original = table_names_original[table_idx]
            table_name_cn = _extract_chinese_name(table_name_original)

        columns: List[ColumnInfo] = []
        if table_idx in table_column_map:
            for col_idx, col_name, col_type in table_column_map[table_idx]:
                col_name_cn = _extract_chinese_name(col_name)
                is_primary = col_idx in primary_keys
                is_foreign = any(
                    src_idx == col_idx or tgt_idx == col_idx
                    for src_idx, tgt_idx in fk_relations.get(table_idx, [])
                )
                columns.append(ColumnInfo(
                    name=col_name,
                    name_cn=col_name_cn,
                    type=col_type,
                    is_primary_key=is_primary,
                    is_foreign_key=is_foreign,
                ))

        fk_relations_list: List[ForeignKeyRelation] = []
        if table_idx in fk_relations:
            for src_idx, tgt_idx in fk_relations[table_idx]:
                if src_idx < len(column_names) and tgt_idx < len(column_names):
                    src_col = column_names[src_idx][1]
                    tgt_col = column_names[tgt_idx][1]
                    tgt_table_idx = column_names[tgt_idx][0]
                    tgt_table_name = table_names[tgt_table_idx] if tgt_table_idx < len(table_names) else ""
                    fk_relations_list.append(ForeignKeyRelation(
                        source_db=db_name,
                        source_table=table_name,
                        source_column=src_col,
                        target_db=db_name,
                        target_table=tgt_table_name,
                        target_column=tgt_col,
                    ))

        primary_key_cols = [
            column_names[i][1] for i in primary_keys
            if i < len(column_names) and column_names[i][0] == table_idx
        ]

        table_info = TableInfo(
            db_name=db_name,
            table_name=table_name,
            table_name_cn=table_name_cn,
            columns=columns,
            primary_keys=primary_key_cols,
            foreign_keys=fk_relations_list,
            related_tables=[],
        )
        table_info_list.append(table_info)

    return table_info_list


def _extract_chinese_name(name: str) -> str:
    chinese_chars = []
    for char in name:
        if "\u4e00" <= char <= "\u9fff":
            chinese_chars.append(char)
    return "".join(chinese_chars)


def build_related_tables(table_info_list: List[TableInfo]) -> List[TableInfo]:
    db_name = ""
    if table_info_list:
        db_name = table_info_list[0].db_name

    fk_graph: Dict[str, List[ForeignKeyRelation]] = defaultdict(list)
    for table_info in table_info_list:
        for fk in table_info.foreign_keys:
            fk_graph[table_info.table_name].append(fk)

    for table_info in table_info_list:
        related: List[RelatedTable] = []

        for fk in table_info.foreign_keys:
            related.append(RelatedTable(
                db_name=fk.target_db,
                table_name=fk.target_table,
                join_columns=[fk.source_column],
            ))

        reverse_fks = [fk for fk in fk_graph.get(table_info.table_name, []) if fk.target_table == table_info.table_name]

        for fk in fk_graph.get(table_info.table_name, []):
            if fk.target_table != table_info.table_name:
                continue
            for other_table in table_info_list:
                if other_table.table_name == fk.source_table:
                    related.append(RelatedTable(
                        db_name=other_table.db_name,
                        table_name=other_table.table_name,
                        join_columns=[fk.target_column],
                    ))

        seen = set()
        unique_related: List[RelatedTable] = []
        for rt in related:
            key = f"{rt.db_name}.{rt.table_name}"
            if key not in seen:
                seen.add(key)
                unique_related.append(rt)

        table_info.related_tables = unique_related

    return table_info_list


def extract_all_tables(input_file: str) -> List[TableInfo]:
    db_list = load_tables_json(input_file)
    all_table_info: List[TableInfo] = []

    for db_data in db_list:
        table_info_list = extract_table_info(db_data)
        table_info_list = build_related_tables(table_info_list)
        all_table_info.extend(table_info_list)

    return all_table_info


def save_to_json(table_info_list: List[TableInfo], output_file: str):
    from nl2sql_v3.models.table_info import USE_PYDANTIC
    
    data = []
    for info in table_info_list:
        if USE_PYDANTIC:
            data.append(info.model_dump())
        else:
            data.append(info.to_es_doc())
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} tables to {output_file}")


def get_related_tables(
    table_info_list: List[TableInfo],
    db_name: str,
    table_name: str,
) -> List[TableInfo]:
    target = None
    for t in table_info_list:
        if t.db_name == db_name and t.table_name == table_name:
            target = t
            break

    if not target:
        return []

    related_table_names: Set[str] = {table_name}
    for rt in target.related_tables:
        related_table_names.add(rt.table_name)

    result = [
        t for t in table_info_list
        if t.db_name == db_name and t.table_name in related_table_names
    ]
    return result


if __name__ == "__main__":
    input_file = script_root / "data" / "test_tables.json"
    output_file = script_root / "data" / "test_tables_structured.json"

    print("Extracting table structure from tables.json...")
    table_info_list = extract_all_tables(str(input_file))
    save_to_json(table_info_list, str(output_file))

    print(f"\nTotal tables extracted: {len(table_info_list)}")

    if table_info_list:
        sample = table_info_list[0]
        print(f"\nSample table: {sample.db_name}.{sample.table_name}")
        print(f"  Columns: {len(sample.columns)}")
        print(f"  Primary keys: {sample.primary_keys}")
        print(f"  Foreign keys: {len(sample.foreign_keys)}")
        print(f"  Related tables: {[rt.table_name for rt in sample.related_tables]}")

        if sample.columns:
            print(f"  First column: {sample.columns[0].name} ({sample.columns[0].type})")
