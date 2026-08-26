import pandas as pd
import numpy as np

def clean_dataset(df: pd.DataFrame, options: dict = None) -> tuple:
    """
    Executes safe automated data cleaning operations on a DataFrame.
    
    Default options:
    - trim_whitespace: True
    - drop_duplicates: True
    - impute_numeric: True
    - impute_categorical: True
    - standardize_dates: True
    
    Returns (cleaned_df, audit_logs)
    """
    if df is None or df.empty:
        return df, ["Dataset is empty. No operations performed."]

    if options is None:
        options = {
            "trim_whitespace": True,
            "drop_duplicates": True,
            "impute_numeric": True,
            "impute_categorical": True,
            "standardize_dates": True
        }

    cleaned_df = df.copy()
    audit_logs = []

    # 1. Whitespace Trimming
    if options.get("trim_whitespace", True):
        trimmed_cols = 0
        for col in cleaned_df.select_dtypes(include=['object', 'string']).columns:
            # Only trim string cells
            cleaned_df[col] = cleaned_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            trimmed_cols += 1
        audit_logs.append(f"Trimmed leading/trailing whitespace across {trimmed_cols} string column(s).")

    # 2. Duplicate Removal
    if options.get("drop_duplicates", True):
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        dropped_rows = initial_rows - len(cleaned_df)
        if dropped_rows > 0:
            audit_logs.append(f"Removed {dropped_rows} redundant duplicate row(s). (Rows: {initial_rows} -> {len(cleaned_df)})")
        else:
            audit_logs.append("No duplicate rows found.")

    # 3. Numeric Median Imputation
    if options.get("impute_numeric", True):
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            # Skip identifier columns
            if "id" in str(col).lower() or "key" in str(col).lower():
                continue
            null_cnt = cleaned_df[col].isnull().sum()
            if null_cnt > 0:
                med_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(med_val)
                audit_logs.append(f"Imputed {null_cnt} missing value(s) in numeric column '{col}' using Median ({med_val}).")

    # 4. Categorical Mode Imputation
    if options.get("impute_categorical", True):
        for col in cleaned_df.select_dtypes(include=['object', 'string']).columns:
            if "id" in str(col).lower() or "email" in str(col).lower():
                continue  # Skip key & email fields for manual safety
            null_cnt = cleaned_df[col].isnull().sum()
            if null_cnt > 0:
                modes = cleaned_df[col].mode()
                mode_val = modes.iloc[0] if not modes.empty else "Unknown"
                cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                audit_logs.append(f"Imputed {null_cnt} missing value(s) in categorical column '{col}' using Mode ('{mode_val}').")

    # 5. Safe Date Standardization with Safeguard
    if options.get("standardize_dates", True):
        for col in cleaned_df.columns:
            if "date" in str(col).lower() or "time" in str(col).lower():
                non_null_before = cleaned_df[col].dropna().count()
                if non_null_before > 0:
                    converted = pd.to_datetime(cleaned_df[col], errors='coerce')
                    non_null_after = converted.dropna().count()
                    
                    # SAFEGUARD: If date conversion creates new NaT/nulls, DO NOT apply it automatically
                    if non_null_after < non_null_before:
                        lost_cnt = non_null_before - non_null_after
                        audit_logs.append(f"SAFEGUARD TRIGGERED: Date conversion on '{col}' would create {lost_cnt} new null(s). Preserved original string values and flagged for review.")
                    else:
                        cleaned_df[col] = converted.dt.strftime('%Y-%m-%d').fillna(cleaned_df[col])
                        audit_logs.append(f"Standardized date formatting in column '{col}' to YYYY-MM-DD.")

    return cleaned_df, audit_logs
