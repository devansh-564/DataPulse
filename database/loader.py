import pandas as pd
from database.connection import check_db_connection

def load_to_postgresql(df: pd.DataFrame, table_name: str = "datapulse_cleaned_data", connection_string: str = None) -> dict:
    """
    Attempts to load cleaned DataFrame into PostgreSQL database table.
    """
    db_check = check_db_connection(connection_string)
    if not db_check["is_configured"]:
        return {
            "success": False,
            "table_name": table_name,
            "rows_loaded": 0,
            "message": db_check["status_message"]
        }

    try:
        engine = db_check["engine"]
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
        return {
            "success": True,
            "table_name": table_name,
            "rows_loaded": len(df),
            "message": f"Successfully loaded {len(df)} row(s) into PostgreSQL table '{table_name}'."
        }
    except Exception as e:
        return {
            "success": False,
            "table_name": table_name,
            "rows_loaded": 0,
            "message": f"Database ingestion error: {str(e)}"
        }
