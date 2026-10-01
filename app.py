import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import plotly.express as px
import plotly.graph_objects as go

# Import Core Engine Modules
from core.profiler import profile_dataset
from core.domain_detector import detect_domain
from core.dqi_engine import calculate_dqi
from core.recommendation_engine import detect_issues_and_recommend
from core.decision_engine import generate_decision_trace
from core.cleaner import clean_dataset
from core.validator import validate_cleaning
from core.anomaly_detector import detect_anomalies
from core.transformer import transform_schema_and_text
from core.etl_engine import run_etl_pipeline
from core.ingestion import ingest_file
from core.unstructured_engine import (
    assess_unstructured_document_quality,
    convert_unstructured_to_structured,
    extract_entities_from_text
)
from core.agent_state import AgentState
from core.agent_orchestrator import AgenticOrchestrator
from database.connection import check_db_connection
from database.loader import load_to_postgresql

# Import UI Utilities
from utils.ui_components import (
    inject_custom_css, render_header, render_kpi,
    render_agent_card, create_dqi_gauge,
    create_dimensions_bar_chart, create_before_after_chart,
    render_pipeline_tracker, render_how_it_works
)

# Page Setup
st.set_page_config(
    page_title="DataPulse — Agentic AI Data Quality Engineering",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# Session State Initialization
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = None
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None
if "unstructured_text" not in st.session_state:
    st.session_state["unstructured_text"] = None
if "document_metadata" not in st.session_state:
    st.session_state["document_metadata"] = {}
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = "No Active Dataset"
if "domain_info" not in st.session_state:
    st.session_state["domain_info"] = None
if "initial_dqi" not in st.session_state:
    st.session_state["initial_dqi"] = None
if "cleaned_dqi" not in st.session_state:
    st.session_state["cleaned_dqi"] = None
if "issues" not in st.session_state:
    st.session_state["issues"] = []
if "clean_logs" not in st.session_state:
    st.session_state["clean_logs"] = []
if "anomaly_res" not in st.session_state:
    st.session_state["anomaly_res"] = None

def load_file_into_agentic_pipeline(file_source, name: str):
    """
    Triggers the 10-Agent Orchestrator pipeline across Structured,
    Semi-Structured, or Unstructured files.
    """
    orchestrator = AgenticOrchestrator()
    state = orchestrator.run_pipeline(file_source, file_name=name)
    
    st.session_state["agent_state"] = state
    st.session_state["dataset_name"] = state.file_name
    st.session_state["raw_df"] = state.raw_df
    st.session_state["cleaned_df"] = state.cleaned_df
    st.session_state["unstructured_text"] = state.unstructured_text
    st.session_state["document_metadata"] = state.document_metadata
    
    st.session_state["domain_info"] = {
        "domain": state.domain,
        "confidence": state.domain_confidence,
        "reasoning": state.domain_reasoning
    }
    
    st.session_state["initial_dqi"] = {
        "da_dqi": state.dqi_score,
        "domain": state.domain,
        "dimensions": state.dqi_dimensions if state.dqi_dimensions else state.unstructured_dimensions,
        "weights": {"completeness": 0.20, "uniqueness": 0.20, "validity": 0.20, "consistency": 0.20, "integrity": 0.20}
    }
    
    st.session_state["cleaned_dqi"] = None
    st.session_state["issues"] = state.quality_issues
    st.session_state["clean_logs"] = state.transformation_log
    st.session_state["anomaly_res"] = state.anomalies_res

# Sidebar Navigation (14 PAGES)
PAGES = [
    "Overview",
    "Upload Dataset",
    "Data Profiling",
    "DQI Analysis",
    "Quality Issues",
    "Recommendations",
    "Data Cleaning",
    "Validation",
    "Anomaly Detection",
    "ETL Pipeline",
    "Agentic AI Control Center",
    "Unstructured Data Intelligence",
    "Monitoring",
    "Reports"
]

with st.sidebar:
    st.markdown("## ⚡ **DATAPULSE**")
    st.markdown("<div style='font-size:11px; color:#94a3b8; margin-bottom:10px;'>Agentic AI Data Engineering Framework</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigation Menu",
        PAGES,
        key="main_nav_radio",
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div style='font-size:11px; font-weight:700; color:#94a3b8; letter-spacing:0.5px;'>ACTIVE DATASET STATE</div>", unsafe_allow_html=True)
    
    agent_st = st.session_state.get("agent_state")
    if agent_st is not None:
        st.markdown(f"**{agent_st.file_name}**")
        st.markdown(f"<span class='badge badge-ready'>● {agent_st.data_category}</span>", unsafe_allow_html=True)
        if agent_st.raw_df is not None:
            st.markdown(f"`{len(agent_st.raw_df)} rows · {len(agent_st.raw_df.columns)} cols`")
        elif agent_st.unstructured_text is not None:
            st.markdown(f"`{len(agent_st.unstructured_text)} chars`")
        st.markdown(f"Domain: `{agent_st.domain}`")
    else:
        st.markdown("**No Active Dataset**")
        st.markdown("<span class='badge badge-waiting'>○ Waiting for Dataset</span>", unsafe_allow_html=True)

# Derive Header Status Variables
ds_name = st.session_state["dataset_name"]
ds_domain = st.session_state["domain_info"]["domain"] if st.session_state["domain_info"] else "N/A"
ds_status = "Ready" if (st.session_state["raw_df"] is not None or st.session_state["unstructured_text"] is not None) else "Waiting for Dataset"
ds_category = st.session_state["agent_state"].data_category if st.session_state["agent_state"] else "Structured"

# ==============================================================================
# SECTION 1: OVERVIEW DASHBOARD
# ==============================================================================
if selected_page == "Overview":
    render_header(
        "DATAPULSE AGENTIC CONTROL CENTER",
        "An Agentic AI Driven Domain-Adaptive Data Quality Index & Analytics Framework",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    if st.session_state["agent_state"] is None:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-title">DATAPULSE AGENTIC AI CONTROL CENTER</div>
            <div class="welcome-subtitle">"Autonomous Quality Engineering for Structured & Unstructured Data"</div>
            <div class="welcome-desc">
                Upload a structured (CSV, XLSX), semi-structured (JSON, XML), or unstructured document (PDF, DOCX, TXT) to activate the 10-Agent AI pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            if st.button("🚀 Load Demo Customer CSV", use_container_width=True, type="primary"):
                demo_path = os.path.join(os.path.dirname(__file__), "data", "customer_data.csv")
                if os.path.exists(demo_path):
                    load_file_into_agentic_pipeline(demo_path, "customer_data.csv")
                    st.rerun()
        with col_w2:
            if st.button("📄 Load Demo Customer PDF Report", use_container_width=True):
                demo_pdf = os.path.join(os.path.dirname(__file__), "data", "customer_report.pdf")
                if os.path.exists(demo_pdf):
                    load_file_into_agentic_pipeline(demo_pdf, "customer_report.pdf")
                    st.rerun()

        render_how_it_works()

    else:
        state = st.session_state["agent_state"]
        df_raw = state.raw_df
        df_clean = state.cleaned_df if state.cleaned_df is not None else df_raw
        initial_dqi = state.dqi_score
        
        cleaned_dqi_val = st.session_state["cleaned_dqi"]["da_dqi"] if st.session_state["cleaned_dqi"] else initial_dqi
        improvement = round(cleaned_dqi_val - initial_dqi, 2)

        # TOP KPI ROW
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            val_records = len(df_clean) if df_clean is not None else state.document_metadata.get("num_pages", 1)
            render_kpi("RECORDS / PAGES", f"{val_records}", f"Category: {state.data_category}")
        with k2:
            render_kpi("DATA FORMAT", f"{state.file_type}", f"Domain: {state.domain}")
        with k3:
            render_kpi("QUALITY SCORE", f"{cleaned_dqi_val:.2f} / 100", f"Initial: {initial_dqi:.2f}", "#10b981" if cleaned_dqi_val >= 90 else "#3b82f6")
        with k4:
            render_kpi("QUALITY IMPROVEMENT", f"{improvement:+.2f}", "Agentic Pipeline Gain", "#10b981" if improvement >= 0 else "#ef4444")

        render_pipeline_tracker(st.session_state)

        # MAIN DQI GAUGE & DIMENSIONS ROW
        c_left, c_right = st.columns([1.1, 1])

        with c_left:
            st.markdown("### 🏆 **QUALITY HEALTH GAUGE**")
            fig_gauge, score_tier, tier_color = create_dqi_gauge(cleaned_dqi_val)
            
            st.markdown(f"""
            <div class="dqi-card">
                <div style="font-size:11px; font-weight:700; color:#94a3b8; letter-spacing:1px; text-transform:uppercase;">Overall Data Quality Index</div>
                <div style="font-size:32px; font-weight:800; color:{tier_color}; margin-top:4px;">{cleaned_dqi_val:.2f} / 100</div>
                <div style="font-size:13px; font-weight:700; color:{tier_color}; margin-bottom:6px;">Status: {score_tier}</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c_right:
            st.markdown("### 📊 **FIVE DQI DIMENSION BREAKDOWN**")
            dims = state.dqi_dimensions if state.dqi_dimensions else state.unstructured_dimensions
            fig_dims = create_dimensions_bar_chart(dims)
            st.plotly_chart(fig_dims, use_container_width=True)

        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🎯 **DETECTED DOMAIN**")
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left;">
                <div style="font-size:18px; font-weight:700; color:#38bdf8;">{state.domain}</div>
                <div style="font-size:12px; color:#94a3b8; margin-top:4px;">Confidence: <b>{state.domain_confidence}</b></div>
                <div style="font-size:11px; color:#cbd5e1; margin-top:8px; line-height:1.4;">
                    {state.domain_reasoning}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("### 🩺 **QUALITY HEALTH SUMMARY**")
            issues_cnt = len(state.quality_issues)
            anom_cnt = state.anomalies_res["anomaly_count"] if state.anomalies_res else 0
            
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-size:12px; color:#94a3b8;">Issues Identified:</span>
                    <span style="font-weight:700; color:#f87171;">{issues_cnt}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-size:12px; color:#94a3b8;">High Severity Flaws:</span>
                    <span style="font-weight:700; color:#f87171;">{sum(1 for i in state.quality_issues if i.get('severity')=='High')}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-size:12px; color:#94a3b8;">Flagged Anomalies:</span>
                    <span style="font-weight:700; color:#38bdf8;">{anom_cnt}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown("### 📈 **QUALITY IMPROVEMENT**")
            fig_ba = create_before_after_chart(initial_dqi, cleaned_dqi_val)
            st.plotly_chart(fig_ba, use_container_width=True)

# ==============================================================================
# SECTION 2: UPLOAD DATASET
# ==============================================================================
elif selected_page == "Upload Dataset":
    render_header(
        "MULTI-FORMAT DATA INGESTION",
        "Upload Structured (CSV, XLSX), Semi-Structured (JSON, XML), or Unstructured (PDF, DOCX, TXT) Data",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="option-card">
            <div class="option-card-title">📂 Upload Multi-Format File</div>
            <div class="option-card-desc">Import CSV, XLSX, JSON, XML, TXT, PDF, or DOCX files for autonomous agent quality processing.</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drag & Drop File Here or Browse",
            type=["csv", "xlsx", "xls", "json", "xml", "txt", "pdf", "docx"],
            label_visibility="visible"
        )
        if uploaded_file is not None:
            try:
                load_file_into_agentic_pipeline(uploaded_file, uploaded_file.name)
                st.success(f"✓ File '{uploaded_file.name}' ingested successfully into Agentic Pipeline!")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion Error: {e}")

    with col2:
        st.markdown("""
        <div class="option-card">
            <div class="option-card-title">🚀 Load Pre-Configured Demo Datasets</div>
            <div class="option-card-desc">Load realistic demonstration datasets with intentional flaws to demonstrate the 10-Agent AI workflow.</div>
        </div>
        """, unsafe_allow_html=True)
        
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            if st.button("📊 Demo Customer CSV", use_container_width=True, type="primary"):
                p = os.path.join(os.path.dirname(__file__), "data", "customer_data.csv")
                load_file_into_agentic_pipeline(p, "customer_data.csv")
                st.rerun()
            if st.button("📈 Demo Customer XLSX", use_container_width=True):
                p = os.path.join(os.path.dirname(__file__), "data", "customer_data.xlsx")
                load_file_into_agentic_pipeline(p, "customer_data.xlsx")
                st.rerun()
            if st.button("📦 Demo Customer JSON", use_container_width=True):
                p = os.path.join(os.path.dirname(__file__), "data", "customer_data.json")
                load_file_into_agentic_pipeline(p, "customer_data.json")
                st.rerun()
        with d_col2:
            if st.button("📄 Demo Customer PDF", use_container_width=True):
                p = os.path.join(os.path.dirname(__file__), "data", "customer_report.pdf")
                load_file_into_agentic_pipeline(p, "customer_report.pdf")
                st.rerun()
            if st.button("📝 Demo Customer DOCX", use_container_width=True):
                p = os.path.join(os.path.dirname(__file__), "data", "customer_report.docx")
                load_file_into_agentic_pipeline(p, "customer_report.docx")
                st.rerun()
            if st.button("📑 Demo Customer TXT", use_container_width=True):
                p = os.path.join(os.path.dirname(__file__), "data", "customer_report.txt")
                load_file_into_agentic_pipeline(p, "customer_report.txt")
                st.rerun()

    render_how_it_works()

    if st.session_state["raw_df"] is not None:
        st.markdown("---")
        st.markdown(f"### 📋 Active Structured Preview: `{st.session_state['dataset_name']}`")
        st.dataframe(st.session_state["raw_df"], use_container_width=True)
    elif st.session_state["unstructured_text"] is not None:
        st.markdown("---")
        st.markdown(f"### 📄 Active Unstructured Document: `{st.session_state['dataset_name']}`")
        st.text_area("Extracted Raw Text Preview", st.session_state["unstructured_text"][:1500], height=180)

# ==============================================================================
# SECTION 3: DATA PROFILING
# ==============================================================================
elif selected_page == "Data Profiling":
    render_header(
        "DATA PROFILING MODULE",
        "Automated statistical inspection & document profiling metrics",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    state = st.session_state.get("agent_state")
    if state is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        if state.raw_df is not None:
            prof = profile_dataset(state.raw_df)
            p1, p2, p3, p4 = st.columns(4)
            with p1: render_kpi("TOTAL ROWS", f"{prof['total_rows']}")
            with p2: render_kpi("TOTAL COLUMNS", f"{prof['total_cols']}")
            with p3: render_kpi("MISSING CELLS", f"{prof['missing_cells']}", f"{prof['missing_pct']}% of total")
            with p4: render_kpi("DUPLICATE ROWS", f"{prof['duplicate_rows']}", f"{prof['duplicate_pct']}% of rows")

            st.markdown("---")
            st.markdown("### 📊 Column-Level Profiling Summary")
            st.dataframe(prof["column_profile"], use_container_width=True)
        else:
            meta = state.document_metadata
            p1, p2, p3, p4 = st.columns(4)
            with p1: render_kpi("CHARACTERS", f"{meta.get('length_chars', len(state.unstructured_text or ''))}")
            with p2: render_kpi("WORD COUNT", f"{meta.get('length_words', 0)}")
            with p3: render_kpi("PAGES", f"{meta.get('num_pages', 1)}")
            with p4: render_kpi("TABLES FOUND", f"{meta.get('tables_found', 0)}")

            st.markdown("---")
            st.markdown("### 📄 Document Text Summary")
            st.text_area("Document Sample", state.unstructured_text[:2000] if state.unstructured_text else "N/A", height=220)

# ==============================================================================
# SECTION 4: DQI ANALYSIS
# ==============================================================================
elif selected_page == "DQI Analysis":
    render_header(
        "DOMAIN-ADAPTIVE DQI ANALYSIS",
        "Transparent 5-dimension evaluation engine & active domain weight matrix",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    state = st.session_state.get("agent_state")
    if state is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        st.markdown(f"## 🏆 Calculated Quality Score: **`{state.dqi_score:.2f} / 100`**")
        st.markdown(f"**Active Domain:** `{state.domain}` · **Data Category:** `{state.data_category}`")

        st.markdown("---")
        st.markdown("### 📏 Quality Dimension Breakdown")

        cols = st.columns(5 if state.raw_df is not None else 6)
        dims = state.dqi_dimensions if state.dqi_dimensions else state.unstructured_dimensions

        for idx, (dim_name, score) in enumerate(dims.items()):
            if idx < len(cols):
                with cols[idx]:
                    render_kpi(dim_name, f"{score:.1f}%", "Weight: 20%", "#38bdf8")

        st.markdown("---")
        with st.expander("ℹ️ **Transparent Mathematical Formulation & Viva Explanation**"):
            st.markdown(f"""
            $$\\\\text{{DA-DQI}} = \\\\sum_{{i=1}}^{{N}} w_i \\\\times D_i$$
            
            Evaluating dimensions under domain weights for **{state.domain}**:
            - Completeness / Text Completeness
            - Uniqueness / Extraction Quality
            - Validity / Field Completeness
            - Consistency / Format Consistency
            - Integrity / Entity Validity
            """)

# ==============================================================================
# SECTION 5: QUALITY ISSUES
# ==============================================================================
elif selected_page == "Quality Issues":
    render_header(
        "QUALITY ISSUE DETECTION ENGINE",
        "Automatic identification of structural, formatting, and completeness flaws",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    state = st.session_state.get("agent_state")
    if state is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        issues = state.quality_issues
        high_cnt = sum(1 for i in issues if i.get("severity") == "High")
        med_cnt = sum(1 for i in issues if i.get("severity") == "Medium")
        low_cnt = sum(1 for i in issues if i.get("severity") == "Low")

        i1, i2, i3, i4 = st.columns(4)
        with i1: render_kpi("TOTAL ISSUES", f"{len(issues)}")
        with i2: render_kpi("HIGH SEVERITY", f"{high_cnt}", "Manual Review Needed", "#f87171")
        with i3: render_kpi("MEDIUM SEVERITY", f"{med_cnt}", "Auto Remediation", "#fbbf24")
        with i4: render_kpi("LOW SEVERITY", f"{low_cnt}", "Minor Flaws", "#93c5fd")

        st.markdown("---")
        if not issues:
            st.success("🎉 No data quality issues identified in the active dataset!")
        else:
            st.markdown("### 📋 Detected Quality Issues Table")
            st.dataframe(pd.DataFrame(issues), use_container_width=True)

# ==============================================================================
# SECTION 6: RECOMMENDATIONS
# ==============================================================================
elif selected_page == "Recommendations":
    render_header(
        "POLICY RECOMMENDATIONS & EXPLAINABLE DECISION LAYER",
        "Rule-based action matrix and safety approval modes (AUTO vs MANUAL REVIEW)",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    state = st.session_state.get("agent_state")
    if state is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        issues = state.quality_issues
        if not issues:
            st.info("No recommendations required.")
        else:
            decision_trace = generate_decision_trace(issues) if state.raw_df is not None else issues
            st.markdown("### 🧠 Explainable Decision Log")
            st.dataframe(pd.DataFrame(decision_trace), use_container_width=True)

# ==============================================================================
# SECTION 7: DATA CLEANING
# ==============================================================================
elif selected_page == "Data Cleaning":
    render_header(
        "AUTOMATED SAFE DATA CLEANING",
        "Execute non-destructive, policy-approved data quality remediation",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    state = st.session_state.get("agent_state")
    if state is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        if state.raw_df is not None:
            st.markdown("### ⚙️ Safe Cleaning Configuration Options")
            c1, c2, c3 = st.columns(3)
            with c1:
                opt_trim = st.checkbox("Trim Whitespace", value=True)
                opt_dup = st.checkbox("Remove Duplicate Rows", value=True)
            with c2:
                opt_num = st.checkbox("Impute Missing Numeric (Median)", value=True)
                opt_cat = st.checkbox("Impute Missing Categorical (Mode)", value=True)
            with c3:
                opt_date = st.checkbox("Standardize Dates (Safeguarded)", value=True)

            if st.button("🧼 Apply Safe Cleaning Policy", type="primary", use_container_width=True):
                opts = {
                    "trim_whitespace": opt_trim, "drop_duplicates": opt_dup,
                    "impute_numeric": opt_num, "impute_categorical": opt_cat,
                    "standardize_dates": opt_date
                }
                cleaned, logs = clean_dataset(state.raw_df, opts)
                state.cleaned_df = cleaned
                state.transformation_log = logs
                st.session_state["cleaned_df"] = cleaned
                st.session_state["clean_logs"] = logs
                st.session_state["cleaned_dqi"] = calculate_dqi(cleaned, state.domain)
                st.success("✓ Safe cleaning policy executed successfully!")

            st.markdown("---")
            if state.transformation_log:
                st.markdown("### 📜 Cleaning Audit Logs")
                for log in state.transformation_log:
                    st.markdown(f"- `{log}`")

            if state.cleaned_df is not None:
                st.markdown("### ✨ Cleaned Dataset Preview")
                st.dataframe(state.cleaned_df, use_container_width=True)
        else:
            st.info("ℹ️ For Unstructured documents, field normalization is performed during Unstructured-to-Structured conversion.")

# ==============================================================================
# SECTION 8: VALIDATION
# ==============================================================================
elif selected_page == "Validation":
    render_header(
        "QUALITY-AWARE VALIDATION & IMPROVEMENT",
        "Quantitative comparison of initial raw vs cleaned dataset quality",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    state = st.session_state.get("agent_state")
    if state is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    elif state.cleaned_df is None:
        st.info("👆 Please execute cleaning or conversion first to generate validation metrics.")
    else:
        if state.raw_df is not None:
            val_res = validate_cleaning(state.raw_df, state.cleaned_df, state.domain)
            st.markdown(f"## Validation Status: <span class='badge badge-ready'>{val_res['status']}</span>", unsafe_allow_html=True)

            v1, v2, v3, v4 = st.columns(4)
            with v1: render_kpi("INITIAL DQI", f"{val_res['initial_dqi']:.2f}")
            with v2: render_kpi("CLEANED DA-DQI", f"{val_res['cleaned_dqi']:.2f}")
            with v3: render_kpi("DQI IMPROVEMENT", f"{val_res['dqi_improvement']:+.2f}", color="#10b981" if val_res['dqi_improvement'] >= 0 else "#ef4444")
            with v4: render_kpi("ROW DELTA", f"{val_res['rows_delta']}", f"Missing Fixed: {val_res['missing_reduced']}")

            st.markdown("---")
            st.markdown("### 📊 Dimension Progression Table")
            dim_comp = []
            for dim, init_val in val_res["initial_dimensions"].items():
                clean_val = val_res["cleaned_dimensions"][dim]
                dim_comp.append({
                    "Dimension": dim,
                    "Initial Score (%)": f"{init_val:.2f}%",
                    "Cleaned Score (%)": f"{clean_val:.2f}%",
                    "Delta (%)": f"{clean_val - init_val:+.2f}%"
                })
            st.dataframe(pd.DataFrame(dim_comp), use_container_width=True)
        else:
            st.success("✓ Unstructured Document Entity Extraction Validated!")

# ==============================================================================
# SECTION 9: ANOMALY DETECTION
# ==============================================================================
elif selected_page == "Anomaly Detection":
    render_header(
        "ISOLATION FOREST ANOMALY DETECTION",
        "Identify statistical outliers without destructive auto-deletion",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    state = st.session_state.get("agent_state")
    if state is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        st.info("ℹ️ **Notice:** Anomaly detection identifies unusual records for manual review; it does NOT automatically delete them.")
        df_target = state.cleaned_df if state.cleaned_df is not None else state.raw_df

        if df_target is not None:
            contam = st.slider("Select Anomaly Contamination Factor", min_value=0.01, max_value=0.30, value=0.10, step=0.01)

            if st.button("🔍 Run Isolation Forest Detection", type="primary", use_container_width=True):
                state.anomalies_res = detect_anomalies(df_target, contamination=contam)

            if state.anomalies_res is not None:
                anom_res = state.anomalies_res
                st.markdown(f"### {anom_res['status']}")

                a1, a2 = st.columns(2)
                with a1: render_kpi("FLAGGED ANOMALIES", f"{anom_res['anomaly_count']}", f"{anom_res['anomaly_pct']}% of records")
                with a2: render_kpi("EVALUATED FEATURES", f"{len(anom_res['numerical_cols'])}", ", ".join(anom_res['numerical_cols']))

                st.markdown("---")
                if not anom_res["anomalies_df"].empty:
                    st.markdown("### ⚠️ Flagged Anomalous Records (For Manual Review)")
                    st.dataframe(anom_res["anomalies_df"], use_container_width=True)
                    
                    if len(anom_res['numerical_cols']) >= 2:
                        col_x = anom_res['numerical_cols'][0]
                        col_y = anom_res['numerical_cols'][1]
                        fig_scat = px.scatter(
                            anom_res["full_df_with_flags"], x=col_x, y=col_y, color="anomaly_flag",
                            color_discrete_map={"Normal (1)": "#38bdf8", "Anomaly (-1)": "#f87171"},
                            title=f"Isolation Forest Outlier Map: {col_x} vs {col_y}"
                        )
                        fig_scat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8fafc'))
                        st.plotly_chart(fig_scat, use_container_width=True)

# ==============================================================================
# SECTION 10: ETL PIPELINE
# ==============================================================================
elif selected_page == "ETL Pipeline":
    render_header(
        "ETL EXECUTION ENGINE",
        "Extract, Transform, and Load processed datasets into PostgreSQL or CSV",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    state = st.session_state.get("agent_state")
    if state is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        st.markdown("### 🗄️ PostgreSQL Database Connection Configuration (Optional)")
        db_url_input = st.text_input("PostgreSQL Connection String (Optional)", value="", placeholder="postgresql://user:password@localhost:5432/datapulse_db")
        
        db_check = check_db_connection(db_url_input)
        st.info(f"Database Status: {db_check['status_message']}")

        df_target = state.cleaned_df if state.cleaned_df is not None else state.raw_df

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if st.button("⚡ Run Full ETL Pipeline", type="primary", use_container_width=True):
                if df_target is not None:
                    report = run_etl_pipeline(df_target, domain=state.domain, db_url=db_url_input)
                    state.etl_report = report
                    st.success("✓ ETL Pipeline Execution Complete!")

        if state.etl_report is not None:
            rep = state.etl_report
            with col_e2:
                st.download_button(
                    label="📥 Export Clean CSV File",
                    data=rep["csv_download_payload"],
                    file_name=f"datapulse_cleaned_{state.file_name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.markdown("---")
            st.markdown("### 📑 ETL Execution Summary")
            e1, e2, e3 = st.columns(3)
            with e1: render_kpi("INPUT RECORDS", f"{rep['input_rows']}")
            with e2: render_kpi("CLEANED RECORDS", f"{rep['cleaned_rows']}")
            with e3: render_kpi("DESTINATION", f"{rep['destination_table']}")

# ==============================================================================
# SECTION 11: AGENTIC AI CONTROL CENTER (NEW PAGE)
# ==============================================================================
elif selected_page == "Agentic AI Control Center":
    render_header(
        "AGENTIC AI ORCHESTRATION CONTROL CENTER",
        "Autonomous 10-Agent Workflow, Observe-Reason-Plan-Act Loop & Human-in-the-Loop Panel",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    state = st.session_state.get("agent_state")
    if state is None:
        st.warning("No active dataset. Please upload or load a dataset first to view Agentic AI workflow.")
    else:
        st.markdown(f"### 🤖 **Agentic Orchestrator Mode:** `<span class='badge badge-ready'>{state.agent_modes['orchestrator_mode']}</span>`", unsafe_allow_html=True)
        
        # AGENT WORKFLOW GRAPH
        st.markdown("""
        <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:16px; margin-bottom:20px;">
            <div style="font-size:12px; font-weight:700; color:#38bdf8; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">⚡ Autonomous Agent Loop</div>
            <div style="display:flex; justify-content:space-between; align-items:center; text-align:center; font-size:12px; font-weight:700;">
                <div style="background:#1e293b; padding:8px 14px; border-radius:6px; color:#f8fafc;">01 OBSERVE</div>
                <div style="color:#64748b;">→</div>
                <div style="background:#1e293b; padding:8px 14px; border-radius:6px; color:#f8fafc;">02 REASON</div>
                <div style="color:#64748b;">→</div>
                <div style="background:#1e293b; padding:8px 14px; border-radius:6px; color:#f8fafc;">03 PLAN</div>
                <div style="color:#64748b;">→</div>
                <div style="background:#1e293b; padding:8px 14px; border-radius:6px; color:#38bdf8;">04 ACT</div>
                <div style="color:#64748b;">→</div>
                <div style="background:#1e293b; padding:8px 14px; border-radius:6px; color:#34d399;">05 VALIDATE</div>
                <div style="color:#64748b;">→</div>
                <div style="background:#1e293b; padding:8px 14px; border-radius:6px; color:#fbbf24;">06 REFLECT</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 👥 **10-Agent Status & Decision Cards**")
        
        # Render 10 Agent Execution Cards
        for msg in state.agent_messages:
            render_agent_card(
                agent_name=msg.agent_name,
                stage=msg.stage,
                status_type=msg.status,
                message=msg.message,
                reasoning=msg.reasoning,
                action=msg.action_taken,
                result=msg.result
            )

        st.markdown("---")
        st.markdown("### 🙋 **Human-in-the-Loop Approval Panel**")
        
        if state.pending_human_approvals:
            st.warning(f"⚠️ {len(state.pending_human_approvals)} High-Severity Action(s) Require Human Approval before final ETL load:")
            for idx, item in enumerate(state.pending_human_approvals):
                col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
                with col_h1:
                    st.markdown(f"**Issue:** `{item.get('column', 'Dataset')}` - {item.get('issue_type', 'Flaw')} ({item.get('severity')} Severity)")
                    st.markdown(f"Recommendation: *{item.get('recommendation')}*")
                with col_h2:
                    if st.button(f"✅ Approve #{idx+1}", key=f"app_{idx}"):
                        st.success(f"Action #{idx+1} approved by user.")
                with col_h3:
                    if st.button(f"❌ Flag #{idx+1}", key=f"rej_{idx}"):
                        st.info(f"Action #{idx+1} flagged for manual review.")
        else:
            st.success("🎉 All low-risk actions were automatically approved and executed under the active policy matrix.")

# ==============================================================================
# SECTION 12: UNSTRUCTURED DATA INTELLIGENCE (NEW PAGE)
# ==============================================================================
elif selected_page == "Unstructured Data Intelligence":
    render_header(
        "UNSTRUCTURED DATA & DOCUMENT INTELLIGENCE",
        "Extract Entities, Profile Unstructured Documents & Convert Text to Structured Tables",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    state = st.session_state.get("agent_state")
    if state is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        text_content = state.unstructured_text
        if not text_content and state.raw_df is not None:
            # Generate sample unstructured text from structured dataframe if structured loaded
            text_content = f"Document Summary for {state.file_name}:\n"
            for _, row in state.raw_df.iterrows():
                text_content += f"Customer Name: {row.get('Customer_Name', 'N/A')}, Email: {row.get('Email', 'N/A')}, Amount: ₹{row.get('Annual_Income', 0)}\n"

        st.markdown("### 📄 **Document Metadata & Text Extraction**")
        meta = state.document_metadata
        m1, m2, m3, m4 = st.columns(4)
        with m1: render_kpi("FORMAT", state.file_type)
        with m2: render_kpi("CHARACTERS", f"{len(text_content or '')}")
        with m3: render_kpi("WORDS", f"{meta.get('length_words', len((text_content or '').split()))}")
        with m4: render_kpi("PAGES / TABLES", f"{meta.get('num_pages', 1)} / {meta.get('tables_found', 0)}")

        with st.expander("📖 **View Extracted Raw Text Content**"):
            st.text_area("Raw Text Output", text_content if text_content else "No text extracted.", height=200)

        st.markdown("---")
        st.markdown("### 🔍 **Detected Document Entities & Fields**")
        
        entities = extract_entities_from_text(text_content)
        e1, e2, e3, e4 = st.columns(4)
        with e1: render_kpi("NAMES DETECTED", f"{len(entities['names'])}", ", ".join(entities['names'][:2]))
        with e2: render_kpi("EMAILS DETECTED", f"{len(entities['emails'])}", ", ".join(entities['emails'][:2]))
        with e3: render_kpi("PHONES DETECTED", f"{len(entities['phones'])}", ", ".join(entities['phones'][:2]))
        with e4: render_kpi("AMOUNTS DETECTED", f"{len(entities['amounts'])}", ", ".join(entities['amounts'][:2]))

        st.markdown("---")
        st.markdown("### 🔄 **CONVERT UNSTRUCTURED → STRUCTURED DATA TABLE**")
        st.info("The Agentic AI Engine automatically converts detected document entities and key-value fields into a clean tabular DataFrame preview.")

        converted_df = convert_unstructured_to_structured(text_content, meta)
        st.dataframe(converted_df, use_container_width=True)

        c_d1, c_d2 = st.columns(2)
        with c_d1:
            csv_buf = converted_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Converted Structured CSV",
                data=csv_buf,
                file_name=f"converted_structured_{state.file_name}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c_d2:
            if st.button("🗄️ Ingest Converted Table to PostgreSQL", type="primary", use_container_width=True):
                pg_res = load_to_postgresql(converted_df, table_name="datapulse_unstructured_converted")
                if pg_res["success"]:
                    st.success(f"✓ {pg_res['message']}")
                else:
                    st.info(f"PostgreSQL Notice: {pg_res['message']}")

        st.markdown("---")
        st.markdown("### 🏆 **DOCUMENT QUALITY ASSESSMENT (0–100 SCORE)**")
        doc_q = assess_unstructured_document_quality(text_content, meta)
        
        q_left, q_right = st.columns([1, 1])
        with q_left:
            fig_g, t_lbl, t_clr = create_dqi_gauge(doc_q["document_quality_score"], title="Document Quality Score")
            st.plotly_chart(fig_g, use_container_width=True)
        with q_right:
            fig_d = create_dimensions_bar_chart(doc_q["dimensions"], title="Document Quality Metric (%)")
            st.plotly_chart(fig_d, use_container_width=True)

# ==============================================================================
# SECTION 13: MONITORING
# ==============================================================================
elif selected_page == "Monitoring":
    render_header(
        "PIPELINE & SYSTEM MONITORING",
        "Real-time pipeline health tracking and operation logs",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    state = st.session_state.get("agent_state")
    if state is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        st.markdown("### 🟢 Active System Health")
        m1, m2, m3 = st.columns(3)
        with m1: render_kpi("ENGINE STATUS", "ONLINE", "10-Agent AI Engine", "#10b981")
        with m2: render_kpi("DATASET HEALTH", "ACTIVE", state.file_name, "#38bdf8")
        with m3: render_kpi("DA-DQI / DOC SCORE", f"{state.dqi_score:.2f} / 100", "Live Quality Score", "#f59e0b")

        st.markdown("---")
        st.markdown("### 📋 Activity Audit Log")
        for msg in state.agent_messages:
            st.markdown(f"- `[{msg.timestamp}] [{msg.agent_name}] [{msg.status}] {msg.message}`")

# ==============================================================================
# SECTION 14: REPORTS
# ==============================================================================
elif selected_page == "Reports":
    render_header(
        "PROJECT SUMMARY & REPORTING",
        "Comprehensive DataPulse project execution report",
        dataset_name=ds_name, domain=ds_domain, status=ds_status, category=ds_category
    )

    state = st.session_state.get("agent_state")
    if state is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        st.markdown(f"""
        # 📄 **DataPulse Major Project Final Execution Report**
        
        **Project Title:** DataPulse: An Agentic AI driven Domain-Adaptive Data Quality Index with Intelligent Data Engineering and Analytics Framework  
        **Active Dataset:** `{state.file_name}`  
        **Data Category:** `{state.data_category}` ({state.file_type})  
        **Detected Domain:** `{state.domain}`  
        **Orchestration Engine:** `{state.agent_modes['orchestrator_mode']}`  

        ---
        
        ### 📊 **Live Dynamic Results Summary**
        - **Quality Score (Baseline):** `{state.dqi_score:.2f} / 100`
        - **Total Agents Executed:** `{len(state.agent_messages)}`
        - **Total Issues Identified:** `{len(state.quality_issues)}`
        - **Pending Human Approvals:** `{len(state.pending_human_approvals)}`
        - **Execution Duration:** `{state.execution_metrics['duration_seconds']}s`
        """)

        st.markdown("""
        ---
        
        ### 🔬 **Documented Project Experiment (Baseline Reference)**
        The project experiment documented in the major project paper recorded:
        - **Baseline Initial DQI:** `96.04`
        - **Baseline Final DQI:** `98.71`
        - **Baseline Net Gain:** `+2.67 pts`
        - **Row Count:** `11 → 10` (1 Duplicate Row Removed)
        """)
