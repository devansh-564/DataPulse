import pandas as pd
import numpy as np

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Analyzes a DataFrame and returns key dataset-level metrics
    and column-level statistics.
    """
    if df is None or df.empty:
        return {
            "total_rows": 0,
            "total_cols": 0,
            "total_cells": 0,
            "missing_cells": 0,
            "missing_pct": 0.0,
            "duplicate_rows": 0,
            "duplicate_pct": 0.0,
            "column_profile": pd.DataFrame()
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    missing_cells = int(df.isnull().sum().sum())
    missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

    col_stats = []
    for col in df.columns:
        null_cnt = int(df[col].isnull().sum())
        null_pct = round((null_cnt / total_rows) * 100, 2) if total_rows > 0 else 0.0
        unique_cnt = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)
        
        # Non-null sample value
        sample_vals = df[col].dropna().values
        sample_val = str(sample_vals[0]) if len(sample_vals) > 0 else "N/A"

        stat_summary = ""
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0:
                stat_summary = f"Min: {non_null.min()}, Max: {non_null.max()}, Mean: {round(non_null.mean(), 2)}"
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # check leading/trailing spaces
            str_series = df[col].dropna().astype(str)
            space_count = sum(1 for s in str_series if s != s.strip())
            if space_count > 0:
                stat_summary = f"{space_count} values have leading/trailing whitespace"

        col_stats.append({
            "Column Name": col,
            "Data Type": dtype_str,
            "Non-Null Count": total_rows - null_cnt,
            "Null Count": null_cnt,
            "Null %": f"{null_pct}%",
            "Unique Count": unique_cnt,
            "Sample Value": sample_val[:30] + ("..." if len(sample_val) > 30 else ""),
            "Details / Notes": stat_summary
        })

    col_df = pd.DataFrame(col_stats)

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "column_profile": col_df
    }
