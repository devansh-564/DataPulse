import datetime
import pandas as pd
from core.cleaner import clean_dataset
from core.transformer import transform_schema_and_text
from core.validator import validate_cleaning
from database.loader import load_to_postgresql

def run_etl_pipeline(df_raw: pd.DataFrame, domain: str = "Customer", table_name: str = "datapulse_cleaned_data", db_url: str = None) -> dict:
    """
    Executes the full Extract-Transform-Load pipeline and produces an ETL Execution Report.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if df_raw is None or df_raw.empty:
        return {
            "timestamp": timestamp,
            "status": "Failed",
            "message": "Input raw DataFrame is empty."
        }

    # 1. EXTRACT
    input_rows = len(df_raw)
    input_cols = len(df_raw.columns)

    # 2. TRANSFORM & CLEAN
    df_cleaned, clean_logs = clean_dataset(df_raw)
    df_transformed, schema_logs = transform_schema_and_text(df_cleaned)

    # 3. VALIDATE DQI
    val_results = validate_cleaning(df_raw, df_transformed, domain)

    # 4. LOAD (PostgreSQL optional attempt)
    db_result = load_to_postgresql(df_transformed, table_name=table_name, connection_string=db_url)

    # Convert cleaned DataFrame to CSV string for download
    csv_bytes = df_transformed.to_csv(index=False).encode('utf-8')

    report = {
        "execution_timestamp": timestamp,
        "pipeline_status": "Success",
        "domain": domain,
        "input_rows": input_rows,
        "input_cols": input_cols,
        "cleaned_rows": len(df_transformed),
        "output_cols": len(df_transformed.columns),
        "initial_dqi": val_results["initial_dqi"],
        "final_dqi": val_results["cleaned_dqi"],
        "dqi_improvement": val_results["dqi_improvement"],
        "database_status": db_result["message"],
        "db_success": db_result["success"],
        "rows_loaded_to_db": db_result["rows_loaded"],
        "destination_table": table_name if db_result["success"] else "N/A (CSV Mode)",
        "csv_download_payload": csv_bytes,
        "transformation_logs": clean_logs + schema_logs,
        "processed_df": df_transformed
    }

    return report
