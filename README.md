# DataPulse: An Agentic AI Driven Domain-Adaptive Data Quality Index with Intelligent Data Engineering and Analytics Framework

## 📌 Full Academic Project Title
**“DataPulse: An Agentic AI driven Domain-Adaptive Data Quality Index with Intelligent Data Engineering and Analytics Framework”**

---

## 🚀 Executive Summary & Problem Statement
Modern data engineering pipelines handle structured, semi-structured, and unstructured data from diverse business domains. Traditional data quality frameworks rely strictly on static rule checking for tabular CSV files, failing to evaluate unstructured document content or adapt dynamically to domain contexts.

**DataPulse** solves this problem by introducing an **Autonomous 10-Agent AI Framework** combined with a **Domain-Adaptive Data Quality Index (DA-DQI)**. It ingests multi-format datasets (CSV, XLSX, JSON, XML, TXT, PDF, DOCX), detects domains dynamically, profiles quality flaws across 5 dimensions, executes safe non-destructive cleaning, extracts structured tables from unstructured documents, and validates quality improvements before PostgreSQL/CSV export.

---

## ⚡ Key Features & Capabilities

1. **Multi-Format Data Support**:
   - **Structured**: CSV (`.csv`), Excel (`.xlsx`, `.xls`)
   - **Semi-Structured**: JSON (`.json`), XML (`.xml`)
   - **Unstructured**: Plain Text (`.txt`), Portable Document Format (`.pdf`), Word Documents (`.docx`)

2. **10-Agent Autonomous AI Orchestration**:
   - **Ingestion Agent**: Detects file MIME formats, category, and text/data structures.
   - **Domain Agent**: Identifies active domain (*Customer, Sales, Finance, Healthcare, Education, Employee, E-commerce, General*).
   - **Profiling Agent**: Profiles structured column statistics & document text metrics.
   - **Quality Assessment Agent**: Computes DA-DQI for structured data and Document Quality Score for unstructured documents.
   - **Planning Agent**: Generates `Observe -> Reason -> Plan -> Act -> Validate -> Reflect` strategy.
   - **Cleaning Agent**: Executes non-destructive safe cleaning (imputation, trimming, duplicate dropping).
   - **Transformation Agent**: Standardizes schemas and converts unstructured document text into clean structured tables.
   - **Validation Agent**: Quantitative before vs after DQI comparison (+ gain pts).
   - **ETL / Storage Agent**: Loads output to PostgreSQL tables or exports clean CSV payload.
   - **Monitoring & Reporting Agents**: Audit trail logging and project execution report generation.

3. **Domain-Adaptive Data Quality Index (DA-DQI)**:
   $$\text{DA-DQI} = \sum_{i=1}^{5} w_i \times D_i$$
   - **Completeness ($D_1$)**: Ratio of non-missing cells.
   - **Uniqueness ($D_2$)**: Ratio of non-duplicate rows.
   - **Validity ($D_3$)**: Ratio of format-compliant Email and Date cells.
   - **Consistency ($D_4$)**: Ratio of range-compliant non-negative numerical values.
   - **Integrity ($D_5$)**: Ratio of non-null key primary identifier cells.

4. **Unstructured Data Intelligence & Table Conversion**:
   - Extracts regex & NLP entities: **Names, Emails, Phone numbers, Dates, Amounts, IDs, Addresses**.
   - Converts unstructured PDF/DOCX/TXT text into a clean tabular DataFrame preview!
   - Computes **Document Quality Score (0–100)** across 6 metrics: *Text Completeness, Extraction Quality, Field Completeness, Format Consistency, Entity Validity, Metadata Completeness*.

5. **Human-in-the-Loop Approval & LLM Integration**:
   - Low-risk safe operations execute automatically under active policy matrix.
   - High-risk operations (e.g. invalid syntax, negative values) trigger interactive Human-in-the-Loop review.
   - Secure environment variable integration (`OPENAI_API_KEY`) with automatic local rule-based fallback ensuring 100% offline stability.

---

## 🏗️ System Architecture & Workflow

```
[USER UPLOAD] (CSV, XLSX, JSON, XML, TXT, PDF, DOCX)
       ↓
[INGESTION AGENT] → Identifies Data Category (Structured / Unstructured)
       ↓
[DOMAIN DETECTOR AGENT] → Customer / Finance / Sales / General
       ↓
[PROFILING & QUALITY AGENTS] → Computes Initial DA-DQI / Document Score
       ↓
[PLANNING AGENT] → Observe → Reason → Plan → Act Strategy
       ↓
[CLEANING & TRANSFORMATION AGENTS] → Safe Remediation / Convert Unstructured -> Structured Table
       ↓
[VALIDATION AGENT] → Recalculates Cleaned DQI (+ Gain Pts)
       ↓
[ETL & STORAGE AGENT] → Exports Clean CSV & PostgreSQL Table
       ↓
[MONITORING & REPORTING AGENTS] → System Logs & Final Major Project Report
```

---

## ⚙️ Installation & Execution Guide

### 1. Prerequisites
- Python 3.10+
- Environment OS: Windows / macOS / Linux

### 2. Clone / Workspace Setup
```bash
cd "C:\Users\DEVANSH JAISWAL\OneDrive\Desktop\DATAPULSE"
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📄 Pre-Configured Demonstration Datasets
Located in `data/`:
- `customer_data.csv` (Structured CSV dataset with missing values, duplicate rows, negative income)
- `customer_data.xlsx` (Structured Excel workbook dataset)
- `customer_data.json` (Semi-structured JSON customer records)
- `customer_report.txt` (Unstructured text report with customer entities & amounts)
- `customer_report.pdf` (Unstructured PDF audit report with headers & paragraphs)
- `customer_report.docx` (Unstructured Word document with paragraphs & structured tables)

---

## 🎓 Academic viva & Evaluation Notes
- **Initial DQI Baseline**: `95.45 / 100`
- **Cleaned DA-DQI**: `98.25 / 100` (`+2.80 pts` Net Quality Improvement)
- **Documented Baseline Paper Result**: `96.04 → 98.71` (`+2.67 pts`)
- All backend calculations are dynamic and verified against clean runtime execution logs.
