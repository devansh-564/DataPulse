import pandas as pd
import re
from core.cleaning_policy import POLICY_MATRIX

EMAIL_REGEX = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

def detect_issues_and_recommend(df: pd.DataFrame) -> list:
    """
    Detects dataset quality issues and generates policy-aligned recommendations.
    Returns a list of structured issue dictionaries.
    """
    if df is None or df.empty:
        return []

    issues = []
    total_rows = len(df)

    # 1. Check Duplicate Rows
    dup_cnt = int(df.duplicated().sum())
    if dup_cnt > 0:
        policy = POLICY_MATRIX["DUPLICATE_ROWS"]
        issues.append({
            "column": "Dataset Level",
            "issue_type": "Duplicate Rows",
            "detected_count": dup_cnt,
            "severity": "Medium",
            "recommendation": policy["description"],
            "action": policy["action"],
            "mode": policy["mode"]
        })

    # 2. Check Column-Level Issues
    for col in df.columns:
        col_lower = str(col).lower()
        null_cnt = int(df[col].isnull().sum())
        
        # Check Missing Values
        if null_cnt > 0:
            is_id_col = "id" in col_lower or "key" in col_lower
            if is_id_col:
                policy = POLICY_MATRIX["MISSING_IDENTIFIER"]
                issues.append({
                    "column": col,
                    "issue_type": "Missing Identifier",
                    "detected_count": null_cnt,
                    "severity": "High",
                    "recommendation": policy["description"],
                    "action": policy["action"],
                    "mode": policy["mode"]
                })
            elif pd.api.types.is_numeric_dtype(df[col]):
                policy = POLICY_MATRIX["MISSING_NUMERIC"]
                issues.append({
                    "column": col,
                    "issue_type": "Missing Numeric Values",
                    "detected_count": null_cnt,
                    "severity": "Medium",
                    "recommendation": f"{policy['description']} (Median: {df[col].median()})",
                    "action": policy["action"],
                    "mode": policy["mode"]
                })
            else:
                policy = POLICY_MATRIX["MISSING_CATEGORICAL"]
                mode_val = df[col].mode().values[0] if not df[col].mode().empty else "Unknown"
                issues.append({
                    "column": col,
                    "issue_type": "Missing Categorical Values",
                    "detected_count": null_cnt,
                    "severity": "Low",
                    "recommendation": f"{policy['description']} (Mode: '{mode_val}')",
                    "action": policy["action"],
                    "mode": policy["mode"]
                })

        # Check Whitespace Padding in text columns
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            series = df[col].dropna().astype(str)
            padded_cnt = sum(1 for s in series if s != s.strip())
            if padded_cnt > 0:
                policy = POLICY_MATRIX["WHITESPACE_PADDING"]
                issues.append({
                    "column": col,
                    "issue_type": "Leading/Trailing Spaces",
                    "detected_count": padded_cnt,
                    "severity": "Low",
                    "recommendation": policy["description"],
                    "action": policy["action"],
                    "mode": policy["mode"]
                })

        # Check Invalid Email Formats
        if "email" in col_lower:
            series = df[col].dropna().astype(str)
            invalid_email_cnt = sum(1 for s in series if not re.match(EMAIL_REGEX, s.strip()))
            if invalid_email_cnt > 0:
                policy = POLICY_MATRIX["INVALID_EMAIL_FORMAT"]
                issues.append({
                    "column": col,
                    "issue_type": "Invalid Email Syntax",
                    "detected_count": invalid_email_cnt,
                    "severity": "High",
                    "recommendation": policy["description"],
                    "action": policy["action"],
                    "mode": policy["mode"]
                })

        # Check Negative Numeric Values (Consistency Flaw)
        if pd.api.types.is_numeric_dtype(df[col]):
            is_id_col = "id" in col_lower or "key" in col_lower
            if not is_id_col:
                series = df[col].dropna()
                neg_vals = [v for v in series if v < 0]
                if len(neg_vals) > 0:
                    policy = POLICY_MATRIX["NEGATIVE_NUMERIC"]
                    min_neg = min(neg_vals)
                    severity = "High" if any(kw in col_lower for kw in ["income", "salary", "revenue", "amount", "price", "age", "cost", "fee", "pay"]) else "Medium"
                    issues.append({
                        "column": col,
                        "issue_type": "Negative Numeric Flaw",
                        "detected_count": len(neg_vals),
                        "severity": severity,
                        "recommendation": f"Negative value detected in {col} (e.g. {min_neg}). Review or correct the value before using the dataset.",
                        "action": policy["action"],
                        "mode": policy["mode"]
                    })

    return issues
