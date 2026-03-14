#!/usr/bin/env python
import sys
sys.path.insert(0, '.')

from nl2sql_v3.data.loader import MetadataLoader
from nl2sql_v3.recall.keyword import keyword_recall

loader = MetadataLoader()
tables = loader.load()
print(f"Loaded {len(tables)} tables")

query = "How many clubs are there?"
results = keyword_recall(query, tables, threshold=0.3)

print(f"\nQuery: {query}")
print(f"Top results:")
for r in results[:10]:
    print(f"  {r.db_name}.{r.table_name}: {r.score:.2f} ({r.match_type})")
