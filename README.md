# ⚡ DataPulse: A Domain-Adaptive Data Quality Index Driven Intelligent Data Engineering & Analytics Framework

DataPulse is a B.Tech major project application designed to evaluate, clean, validate, and manage tabular datasets using transparent mathematical formulas and domain-adaptive quality indexing.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```

---

## 🏗️ Project Architecture

```
DataPulse/
├── app.py                      # Main Streamlit Dashboard & Navigation UI
├── core/
│   ├── profiler.py             # Dataset Statistics & Column Profiling
│   ├── domain_detector.py      # Rule-Based Domain Detection Engine
│   ├── dqi_engine.py           # Dynamic DA-DQI Engine (5 Dimensions + Domain Weights)
│   ├── recommendation_engine.py# Rule-Based Recommendation Generator
│   ├── cleaning_policy.py      # Policy Matrix (AUTO vs MANUAL decision matrix)
│   ├── decision_engine.py      # Explainable Decision Layer
│   ├── cleaner.py              # Safe Data Cleaning Operations
│   ├── validator.py            # Quality Validation & Before/After Comparison
│   ├── anomaly_detector.py     # Scikit-Learn Isolation Forest Anomaly Detection
│   ├── transformer.py          # Schema & Text Transformations
│   └── etl_engine.py           # ETL Execution Engine & CSV/PostgreSQL Export
├── database/
│   ├── connection.py           # PostgreSQL Connection Manager (Optional + Fallback)
│   └── loader.py               # Table Schema Generator & Ingestion
├── data/
│   └── demo_dataset.csv        # 11-row Flawed Demo Dataset
├── utils/
│   └── ui_components.py        # Custom UI Cards, Badges, Plotly Theme
├── requirements.txt            # Dependencies
└── README.md                   # Setup & Instructions
```

---

## 📐 5 Quality Dimensions & Formulas

1. **Completeness ($D_1$)**: $(1 - \frac{\text{Missing Cells}}{\text{Total Cells}}) \times 100$
2. **Uniqueness ($D_2$)**: $(1 - \frac{\text{Duplicate Rows}}{\text{Total Rows}}) \times 100$
3. **Validity ($D_3$)**: Ratio of format-compliant Email/Date cells.
4. **Consistency ($D_4$)**: Ratio of logical non-negative range numerical cells.
5. **Integrity ($D_5$)**: Ratio of non-null key identifier columns (`*_ID`).

**DA-DQI Formula**:
$$\text{DA-DQI} = \sum_{i=1}^{5} w_i \times D_i$$
Where weights $w_i$ adapt based on the detected dataset domain (*Customer, Sales, Finance, Healthcare, Education, General*).
