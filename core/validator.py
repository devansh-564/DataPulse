import pandas as pd
from core.dqi_engine import calculate_dqi
from core.profiler import profile_dataset

def validate_cleaning(df_initial: pd.DataFrame, df_cleaned: pd.DataFrame, domain: str = "Customer") -> dict:
    """
    Compares initial and cleaned DataFrames, evaluates DA-DQI progression,
    and returns quantitative validation metrics.
    """
    if df_initial is None or df_cleaned is None:
        return {"status": "Error", "message": "Invalid input DataFrames."}

    initial_dqi_res = calculate_dqi(df_initial, domain)
    cleaned_dqi_res = calculate_dqi(df_cleaned, domain)

    initial_score = initial_dqi_res["da_dqi"]
    cleaned_score = cleaned_dqi_res["da_dqi"]
    improvement = round(cleaned_score - initial_score, 2)

    initial_prof = profile_dataset(df_initial)
    cleaned_prof = profile_dataset(df_cleaned)

    missing_reduced = initial_prof["missing_cells"] - cleaned_prof["missing_cells"]
    duplicates_reduced = initial_prof["duplicate_rows"] - cleaned_prof["duplicate_rows"]

    if improvement > 0:
        status_msg = f"Quality Improved (+{improvement} pts)"
        quality_passed = True
    elif improvement == 0:
        status_msg = "Quality Unchanged (0.00 pts)"
        quality_passed = True
    else:
        status_msg = f"Quality Decreased ({improvement} pts) — Review Recommended"
        quality_passed = False

    return {
        "status": status_msg,
        "passed": quality_passed,
        "initial_dqi": initial_score,
        "cleaned_dqi": cleaned_score,
        "dqi_improvement": improvement,
        "initial_rows": initial_prof["total_rows"],
        "cleaned_rows": cleaned_prof["total_rows"],
        "rows_delta": cleaned_prof["total_rows"] - initial_prof["total_rows"],
        "initial_missing": initial_prof["missing_cells"],
        "cleaned_missing": cleaned_prof["missing_cells"],
        "missing_reduced": missing_reduced,
        "initial_duplicates": initial_prof["duplicate_rows"],
        "cleaned_duplicates": cleaned_prof["duplicate_rows"],
        "duplicates_reduced": duplicates_reduced,
        "initial_dimensions": initial_dqi_res["dimensions"],
        "cleaned_dimensions": cleaned_dqi_res["dimensions"]
    }
