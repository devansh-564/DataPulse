"""
DataPulse Cleaning Policy Matrix
Maps detected data quality flaws and column types to recommended cleaning policies
and safety approval modes (AUTO vs MANUAL REVIEW).
"""

POLICY_MATRIX = {
    "MISSING_NUMERIC": {
        "action": "Median Imputation",
        "mode": "AUTO",
        "description": "Fill missing numeric values with the column median.",
        "safety_level": "Safe"
    },
    "MISSING_CATEGORICAL": {
        "action": "Mode Imputation",
        "mode": "AUTO",
        "description": "Fill missing text/categorical values with the most frequent value.",
        "safety_level": "Safe"
    },
    "MISSING_IDENTIFIER": {
        "action": "Flag Primary Key Null",
        "mode": "MANUAL REVIEW",
        "description": "Missing values in primary key or identifier columns require manual verification.",
        "safety_level": "High Risk"
    },
    "DUPLICATE_ROWS": {
        "action": "Duplicate Removal",
        "mode": "AUTO",
        "description": "Drop redundant duplicate rows while preserving the first instance.",
        "safety_level": "Safe"
    },
    "WHITESPACE_PADDING": {
        "action": "Text Trimming",
        "mode": "AUTO",
        "description": "Strip leading and trailing spaces from text columns.",
        "safety_level": "Safe"
    },
    "INVALID_EMAIL_FORMAT": {
        "action": "Validate & Flag",
        "mode": "MANUAL REVIEW",
        "description": "Email address does not match standard syntax (missing @ or domain). Flag for review.",
        "safety_level": "Medium Risk"
    },
    "INVALID_DATE_FORMAT": {
        "action": "Safe Date Parsing",
        "mode": "AUTO",
        "description": "Standardize date strings to YYYY-MM-DD where safely parseable.",
        "safety_level": "Safe"
    },
    "NEGATIVE_NUMERIC": {
        "action": "Flag Negative Value",
        "mode": "MANUAL REVIEW",
        "description": "Negative values in non-negative numeric columns (e.g., Income, Salary, Age) require manual verification.",
        "safety_level": "High Risk"
    }
}
