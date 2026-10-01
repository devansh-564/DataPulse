import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.1) -> dict:
    """
    Applies Scikit-Learn Isolation Forest on numerical features to identify potential anomalies.
    Anomalies are flagged for review and NOT automatically deleted.
    """
    if df is None or df.empty:
        return {
            "anomaly_count": 0,
            "anomaly_pct": 0.0,
            "anomalies_df": pd.DataFrame(),
            "full_df_with_flags": df,
            "numerical_cols": [],
            "status": "No data provided."
        }

    # Select numerical columns (excluding ID columns)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if "id" not in str(c).lower()]
    
    if len(num_cols) == 0:
        return {
            "anomaly_count": 0,
            "anomaly_pct": 0.0,
            "anomalies_df": pd.DataFrame(),
            "full_df_with_flags": df,
            "numerical_cols": [],
            "status": "No numerical features found for Isolation Forest detection."
        }

    # Handle missing values temporarily for Isolation Forest fitting
    X = df[num_cols].copy()
    for col in num_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())

    # Initialize Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(X)
    scores = model.decision_function(X)

    full_df = df.copy()
    full_df["anomaly_flag"] = ["Anomaly (-1)" if p == -1 else "Normal (1)" for p in predictions]
    full_df["anomaly_score"] = np.round(scores, 4)

    anomalies_df = full_df[full_df["anomaly_flag"] == "Anomaly (-1)"]
    anomaly_cnt = len(anomalies_df)
    anomaly_pct = round((anomaly_cnt / len(df)) * 100, 2)

    return {
        "anomaly_count": anomaly_cnt,
        "anomaly_pct": anomaly_pct,
        "anomalies_df": anomalies_df,
        "full_df_with_flags": full_df,
        "numerical_cols": num_cols,
        "status": f"Identified {anomaly_cnt} potential anomalous record(s) ({anomaly_pct}% of dataset)."
    }
