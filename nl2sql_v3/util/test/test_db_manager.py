from nl2sql_v3.util.db_manager import DatabaseManager, db_manager

# 测试列出所有数据库
dbs = db_manager.list_databases()
print('可用数据库:', dbs[:5], '...' if len(dbs) > 5 else '')

# 测试连接并查询 car_1
print('\n=== 测试 car_1 数据库 ===')
result = db_manager.connect('car_1')
print('连接结果:', result)

# 查询所有表
tables = db_manager.get_tables('car_1')
print('表列表:', tables[:5])

# 执行SQL查询
print('\n执行 SQL: SELECT * FROM car_names LIMIT 3')
result = db_manager.execute('car_1', 'SELECT * FROM car_names LIMIT 3', fetch_all=True)
print('查询结果:', result)

# 测试查询单个结果
print('\n执行 SQL: SELECT name FROM car_names LIMIT 1 (fetch_one)')
result = db_manager.execute('car_1', 'SELECT name FROM car_names LIMIT 1', fetch_one=True)
print('单条结果:', result)

# 测试获取表结构
print('\n获取表结构: car_names')
schema = db_manager.get_table_schema('car_1', 'car_names')
print('表结构:', schema)

# 测试插入/更新操作
print('\n测试 SELECT COUNT')
result = db_manager.execute('car_1', 'SELECT COUNT(*) as cnt FROM car_names')
print('count结果:', result)

# 测试另一个数据库
print('\n=== 测试 chinook_1 数据库 ===')
tables2 = db_manager.get_tables('chinook_1')
print('chinook_1 表列表:', tables2[:5])

# 测试不存在的数据库
print('\n=== 测试不存在的数据库 ===')
result = db_manager.connect('non_existent_db')
print('连接不存在数据库:', result)

# 关闭连接
db_manager.close()
print('\n连接已关闭')
