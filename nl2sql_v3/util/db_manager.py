import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import sqlite3

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, database_dir: Optional[str] = None):
        if database_dir is None:
            database_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
        self.database_dir = Path(database_dir)
        self._connections: Dict[str, sqlite3.Connection] = {}

    def _get_db_path(self, db_name: str) -> Optional[Path]:
        db_dir = self.database_dir / db_name
        if not db_dir.exists():
            logger.warning(f"Database directory not found: {db_dir}")
            return None

        sqlite_files = list(db_dir.glob("*.sqlite"))
        if not sqlite_files:
            logger.warning(f"No .sqlite file found in: {db_dir}")
            return None

        return sqlite_files[0]

    def _get_connection(self, db_name: str) -> Optional[sqlite3.Connection]:
        if db_name in self._connections:
            try:
                self._connections[db_name].execute("SELECT 1")
                return self._connections[db_name]
            except Exception:
                del self._connections[db_name]

        db_path = self._get_db_path(db_name)
        if db_path is None:
            return None

        try:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._connections[db_name] = conn
            logger.info(f"Connected to database: {db_name} at {db_path}")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database {db_name}: {e}")
            return None

    def connect(self, db_name: str) -> bool:
        return self._get_connection(db_name) is not None

    def execute(
        self,
        db_name: str,
        sql: str,
        params: Optional[Tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = True,
    ) -> Optional[Union[Dict, List[Dict], Tuple]]:
        conn = self._get_connection(db_name)
        if conn is None:
            logger.error(f"Database not connected: {db_name}")
            return None

        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            if sql.strip().upper().startswith(("SELECT", "PRAGMA", "EXPLAIN")):
                if fetch_one:
                    row = cursor.fetchone()
                    if row:
                        return dict(row) if isinstance(row, sqlite3.Row) else row
                    return None
                elif fetch_all:
                    rows = cursor.fetchall()
                    return [dict(row) if isinstance(row, sqlite3.Row) else row for row in rows]
                else:
                    return None
            else:
                conn.commit()
                return {"rows_affected": cursor.rowcount, "lastrowid": cursor.lastrowid}

        except sqlite3.Error as e:
            logger.error(f"SQL execution error on {db_name}: {e}")
            conn.rollback()
            return None

    def execute_many(
        self,
        db_name: str,
        sql: str,
        params_list: List[Tuple],
    ) -> Optional[int]:
        conn = self._get_connection(db_name)
        if conn is None:
            logger.error(f"Database not connected: {db_name}")
            return None

        try:
            cursor = conn.cursor()
            cursor.executemany(sql, params_list)
            conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"SQL executemany error on {db_name}: {e}")
            conn.rollback()
            return None

    def get_tables(self, db_name: str) -> List[str]:
        result = self.execute(
            db_name,
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            fetch_all=True,
        )
        if result is None:
            return []
        return [row["name"] for row in result]

    def get_table_schema(self, db_name: str, table_name: str) -> Optional[List[Dict]]:
        sql = f"PRAGMA table_info('{table_name}')"
        return self.execute(db_name, sql, fetch_all=True)

    def get_table_indexes(self, db_name: str, table_name: str) -> Optional[List[Dict]]:
        sql = f"PRAGMA index_list('{table_name}')"
        return self.execute(db_name, sql, fetch_all=True)

    def get_foreign_keys(self, db_name: str, table_name: str) -> Optional[List[Dict]]:
        sql = f"PRAGMA foreign_key_list('{table_name}')"
        return self.execute(db_name, sql, fetch_all=True)

    def get_table_info(self, db_name: str, table_name: str) -> Optional[Dict[str, Any]]:
        schema = self.get_table_schema(db_name, table_name)
        indexes = self.get_table_indexes(db_name, table_name)
        fks = self.get_foreign_keys(db_name, table_name)

        if schema is None:
            return None

        return {
            "table_name": table_name,
            "columns": schema,
            "indexes": indexes or [],
            "foreign_keys": fks or [],
        }

    def close(self, db_name: Optional[str] = None):
        if db_name:
            if db_name in self._connections:
                self._connections[db_name].close()
                del self._connections[db_name]
                logger.info(f"Closed connection to: {db_name}")
        else:
            for name, conn in self._connections.items():
                conn.close()
            self._connections.clear()
            logger.info("Closed all database connections")

    def list_databases(self) -> List[str]:
        if not self.database_dir.exists():
            return []
        return [d.name for d in self.database_dir.iterdir() if d.is_dir()]

    def disconnect_all(self):
        self.close()

    def __del__(self):
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass


db_manager = DatabaseManager()
