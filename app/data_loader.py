"""Data loading helpers for the Streamlit app.

Minimal, well-documented functions to read from a database engine or CSV fallback.
"""
from typing import List, Optional

import pandas as pd
import sqlalchemy


def engine_from_connection_string(conn_str: str) -> sqlalchemy.engine.Engine:
    """Return a SQLAlchemy engine from a connection string."""
    return sqlalchemy.create_engine(conn_str)


def list_tables(engine: sqlalchemy.engine.Engine) -> List[str]:
    inspector = sqlalchemy.inspect(engine)
    return inspector.get_table_names()


def read_table(engine: sqlalchemy.engine.Engine, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
    sql = f"SELECT * FROM {table_name}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return pd.read_sql(sql, engine)
