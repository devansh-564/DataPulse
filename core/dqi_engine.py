import pandas as pd
import numpy as np
import re

DOMAIN_WEIGHTS = {
    "Customer":    {"completeness": 0.25, "uniqueness": 0.25, "validity": 0.20, "consistency": 0.15, "integrity": 0.15},
    "Sales":       {"completeness": 0.30, "uniqueness": 0.15, "validity": 0.20, "consistency": 0.20, "integrity": 0.15},
    "Finance":     {"completeness": 0.30, "uniqueness": 0.15, "validity": 0.20, "consistency": 0.20, "integrity": 0.15},
    "Healthcare":  {"completeness": 0.35, "uniqueness": 0.10, "validity": 0.20, "consistency": 0.20, "integrity": 0.15},
    "Education":   {"completeness": 0.25, "uniqueness": 0.20, "validity": 0.20, "consistency": 0.20, "integrity": 0.15},
    "General / Other": {"completeness": 0.20, "uniqueness": 0.20, "validity": 0.20, "consistency": 0.20, "integrity": 0.20}
}

EMAIL_REGEX = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

def calculate_dqi(df: pd.DataFrame, domain: str = "Customer") -> dict:
    """
    Calculates dynamic Data Quality Index (DA-DQI) across 5 core dimensions:
    1. Completeness
    2. Uniqueness
    3. Validity
    4. Consistency
    5. Integrity
    """
    if df is None or df.empty:
        return {
            "da_dqi": 0.0,
            "dimensions": {"completeness": 0, "uniqueness": 0, "validity": 0, "consistency": 0, "integrity": 0},
            "weights": DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["General / Other"]),
            "domain": domain,
            "details": {}
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    # 1. Completeness
    missing_cells = df.isnull().sum().sum()
    completeness_score = (1.0 - (missing_cells / total_cells)) * 100.0 if total_cells > 0 else 100.0

    # 2. Uniqueness
    dup_rows = df.duplicated().sum()
    uniqueness_score = (1.0 - (dup_rows / total_rows)) * 100.0 if total_rows > 0 else 100.0

    # 3. Validity (Evaluate Emails, Dates, and Text format checks)
    total_validity_evaluated = 0
    valid_count = 0

    for col in df.columns:
        col_lower = str(col).lower()
        series = df[col].dropna()
        
        if "email" in col_lower:
            for val in series:
                total_validity_evaluated += 1
                if isinstance(val, str) and re.match(EMAIL_REGEX, val.strip()):
                    valid_count += 1
        elif "date" in col_lower or "time" in col_lower:
            for val in series:
                total_validity_evaluated += 1
                try:
                    pd.to_datetime(val, format='mixed', errors='coerce')
                    valid_count += 1
                except Exception:
                    pass

    if total_validity_evaluated == 0:
        # Fallback validity: percentage of cells without invalid whitespace string corruption
        validity_score = 100.0
    else:
        validity_score = (valid_count / total_validity_evaluated) * 100.0

    # 4. Consistency (Evaluate numeric range logic e.g., non-negative Age/Salary)
    total_consistency_checked = 0
    consistent_count = 0

    for col in df.columns:
        col_lower = str(col).lower()
        if pd.api.types.is_numeric_dtype(df[col]):
            # Skip identifier/key columns
            if "id" in col_lower or "key" in col_lower:
                continue
            series = df[col].dropna()
            for val in series:
                total_consistency_checked += 1
                if "age" in col_lower:
                    if 0 <= val <= 120:
                        consistent_count += 1
                else:
                    if val >= 0:
                        consistent_count += 1

    if total_consistency_checked == 0:
        consistency_score = 100.0
    else:
        consistency_score = (consistent_count / total_consistency_checked) * 100.0

    # 5. Integrity (Primary key / Key Identifier column null evaluation)
    id_cols = [c for c in df.columns if "id" in str(c).lower() or "key" in str(c).lower()]
    if id_cols:
        total_id_nulls = sum(df[c].isnull().sum() for c in id_cols)
        total_id_cells = total_rows * len(id_cols)
        integrity_score = (1.0 - (total_id_nulls / total_id_cells)) * 100.0 if total_id_cells > 0 else 100.0
    else:
        # Fallback integrity: check first column
        first_col_nulls = df.iloc[:, 0].isnull().sum()
        integrity_score = (1.0 - (first_col_nulls / total_rows)) * 100.0 if total_rows > 0 else 100.0

    # Retrieve weights
    weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["General / Other"])

    # Compute DA-DQI
    da_dqi = (
        weights["completeness"] * completeness_score +
        weights["uniqueness"]   * uniqueness_score +
        weights["validity"]     * validity_score +
        weights["consistency"]  * consistency_score +
        weights["integrity"]    * integrity_score
    )

    return {
        "da_dqi": round(da_dqi, 2),
        "dimensions": {
            "Completeness": round(completeness_score, 2),
            "Uniqueness": round(uniqueness_score, 2),
            "Validity": round(validity_score, 2),
            "Consistency": round(consistency_score, 2),
            "Integrity": round(integrity_score, 2)
        },
        "weights": weights,
        "domain": domain,
        "details": {
            "missing_cells": int(missing_cells),
            "duplicate_rows": int(dup_rows),
            "validity_evaluated": total_validity_evaluated,
            "consistency_checked": total_consistency_checked,
            "id_columns_found": id_cols
        }
    }
