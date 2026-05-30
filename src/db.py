import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def get_mssql_engine():
    server = os.getenv("MSSQL_SERVER", "localhost")
    database = os.getenv("MSSQL_DATABASE", "Finance_Project")
    driver = os.getenv("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server")

    odbc_str = (
        f"DRIVER={driver};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )

    connect_str = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_str)}"
    engine = create_engine(connect_str, fast_executemany=True)
    return engine