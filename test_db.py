from src.db import get_mssql_engine

engine = get_mssql_engine()

with engine.connect() as conn:
    result = conn.exec_driver_sql("SELECT 1")
    print("Database connection successful:", result.scalar())