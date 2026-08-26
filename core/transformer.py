import pandas as pd
import re

def transform_schema_and_text(df: pd.DataFrame, options: dict = None) -> tuple:
    """
    Transforms DataFrame schema and standardizes text fields for database ETL loading.
    
    Options:
    - standardize_headers: Convert column names to lowercase_snake_case
    - drop_empty_cols: Remove columns where all values are null
    """
    if df is None or df.empty:
        return df, ["DataFrame is empty."]

    if options is None:
        options = {
            "standardize_headers": True,
            "drop_empty_cols": True
        }

    transformed_df = df.copy()
    logs = []

    # 1. Header Standardization
    if options.get("standardize_headers", True):
        new_cols = []
        for c in transformed_df.columns:
            clean_c = re.sub(r'[^\w\s]', '', str(c)).strip()
            clean_c = re.sub(r'[\s]+', '_', clean_c).lower()
            new_cols.append(clean_c)
        transformed_df.columns = new_cols
        logs.append("Standardized column headers to lowercase_snake_case format.")

    # 2. Drop Completely Empty Columns
    if options.get("drop_empty_cols", True):
        empty_cols = [c for c in transformed_df.columns if transformed_df[c].isnull().all()]
        if empty_cols:
            transformed_df = transformed_df.drop(columns=empty_cols)
            logs.append(f"Dropped {len(empty_cols)} completely empty column(s): {', '.join(empty_cols)}")
        else:
            logs.append("No completely empty columns found.")

    return transformed_df, logs
