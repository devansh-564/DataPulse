import os

def check_db_connection(connection_string: str = None) -> dict:
    """
    Checks if PostgreSQL database is configured and accessible.
    Returns status dict. Never crashes if DB is unavailable or SQLAlchemy is absent.
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        return {
            "is_configured": False,
            "engine": None,
            "status_message": "PostgreSQL driver (SQLAlchemy) not installed — database export unavailable."
        }

    if not connection_string:
        connection_string = os.getenv("DATAPULSE_DB_URL", "")

    if not connection_string or not connection_string.startswith("postgresql"):
        return {
            "is_configured": False,
            "engine": None,
            "status_message": "PostgreSQL not configured — database export unavailable. (CSV Export active)"
        }

    try:
        engine = create_engine(connection_string, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "is_configured": True,
            "engine": engine,
            "status_message": "PostgreSQL database connected successfully."
        }
    except Exception as e:
        return {
            "is_configured": False,
            "engine": None,
            "status_message": f"PostgreSQL connection offline: {str(e)[:80]}. (CSV Export active)"
        }

