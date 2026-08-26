import streamlit as st
import pandas as pd
import numpy as np
import os
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
from database.connection import check_db_connection

# Import UI Utilities
from utils.ui_components import (
    inject_custom_css, render_header, render_kpi,
    create_dqi_gauge, create_dimensions_bar_chart,
    create_before_after_chart, render_pipeline_tracker,
    render_how_it_works
)

# Page Setup
st.set_page_config(
    page_title="DataPulse — Intelligent Data Quality Framework",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# Session State Initialization
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None
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

def load_dataset_into_session(df: pd.DataFrame, name: str):
    st.session_state["raw_df"] = df
    st.session_state["cleaned_df"] = None
    st.session_state["dataset_name"] = name
    
    # Run Initial Processing
    st.session_state["domain_info"] = detect_domain(df)
    domain = st.session_state["domain_info"]["domain"]
    st.session_state["initial_dqi"] = calculate_dqi(df, domain)
    st.session_state["issues"] = detect_issues_and_recommend(df)
    st.session_state["cleaned_dqi"] = None
    st.session_state["clean_logs"] = []
    st.session_state["anomaly_res"] = None

# Sidebar Navigation (SINGLE RADIO CONTROL FOR 100% UNAMBIGUOUS ACTIVE STATE)
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
    "Monitoring",
    "Reports"
]

with st.sidebar:
    st.markdown("## ⚡ **DATAPULSE**")
    st.markdown("<div style='font-size:12px; color:#94a3b8; margin-bottom:12px;'>Intelligent Data Quality & Engineering</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigation Menu",
        PAGES,
        key="main_nav_radio",
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div style='font-size:11px; font-weight:700; color:#94a3b8; letter-spacing:0.5px;'>ACTIVE DATASET</div>", unsafe_allow_html=True)
    
    if st.session_state["raw_df"] is not None:
        df_active = st.session_state["raw_df"]
        st.markdown(f"**{st.session_state['dataset_name']}**")
        st.markdown("<span class='badge badge-ready'>● Ready</span>", unsafe_allow_html=True)
        st.markdown(f"`{len(df_active)} rows · {len(df_active.columns)} columns`")
        st.markdown(f"Domain: `{st.session_state['domain_info']['domain']}`")
    else:
        st.markdown("**No Active Dataset**")
        st.markdown("<span class='badge badge-waiting'>○ Waiting for Dataset</span>", unsafe_allow_html=True)

# Derive Header Status Variables
ds_name = st.session_state["dataset_name"]
ds_domain = st.session_state["domain_info"]["domain"] if st.session_state["domain_info"] else "N/A"
ds_status = "Ready" if st.session_state["raw_df"] is not None else "Waiting for Dataset"

# ==============================================================================
# SECTION 1: OVERVIEW DASHBOARD
# ==============================================================================
if selected_page == "Overview":
    render_header(
        "DATAPULSE CONTROL CENTER",
        "Data Quality Intelligence & Engineering Dashboard",
        dataset_name=ds_name,
        domain=ds_domain,
        status=ds_status
    )

    # EMPTY STATE (When no dataset loaded)
    if st.session_state["raw_df"] is None:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-title">DATAPULSE CONTROL CENTER</div>
            <div class="welcome-subtitle">"Measure. Improve. Trust Your Data."</div>
            <div class="welcome-desc">
                Upload a dataset or load the DataPulse demonstration dataset to begin data quality analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            if st.button("🚀 Load Demo Dataset", use_container_width=True, type="primary"):
                demo_path = os.path.join(os.path.dirname(__file__), "data", "demo_dataset.csv")
                if os.path.exists(demo_path):
                    demo_df = pd.read_csv(demo_path)
                    load_dataset_into_session(demo_df, "demo_dataset.csv")
                    st.rerun()
                else:
                    st.error("Demo dataset file not found.")
        with col_w2:
            st.info("💡 Navigate to **'Upload Dataset'** in the sidebar to select your own custom CSV dataset.")

        render_how_it_works()

    # ACTIVE DATASET STATE
    else:
        df_raw = st.session_state["raw_df"]
        df_clean = st.session_state["cleaned_df"] if st.session_state["cleaned_df"] is not None else df_raw
        
        prof_raw = profile_dataset(df_raw)
        domain = st.session_state["domain_info"]["domain"]
        initial_dqi = st.session_state["initial_dqi"]["da_dqi"]
        
        cleaned_dqi_val = st.session_state["cleaned_dqi"]["da_dqi"] if st.session_state["cleaned_dqi"] else initial_dqi
        improvement = round(cleaned_dqi_val - initial_dqi, 2)

        # TOP KPI ROW
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi("TOTAL ROWS", f"{len(df_clean)}", f"Initial: {len(df_raw)} rows")
        with k2:
            render_kpi("COLUMNS", f"{len(df_clean.columns)}", f"Domain: {domain}")
        with k3:
            render_kpi("DA-DQI SCORE", f"{cleaned_dqi_val:.2f} / 100", f"Initial: {initial_dqi:.2f}", "#10b981" if cleaned_dqi_val >= 90 else "#3b82f6")
        with k4:
            render_kpi("DQI IMPROVEMENT", f"{improvement:+.2f}", "Post-Cleaning Gain", "#10b981" if improvement >= 0 else "#ef4444")

        # PIPELINE STAGE TRACKER
        render_pipeline_tracker(st.session_state)

        # MAIN DQI HEALTH GAUGE & DIMENSIONS ROW
        c_left, c_right = st.columns([1.1, 1])

        with c_left:
            st.markdown("### 🏆 **DOMAIN-ADAPTIVE DATA QUALITY INDEX**")
            dqi_data = st.session_state["cleaned_dqi"] if st.session_state["cleaned_dqi"] else st.session_state["initial_dqi"]
            score_val = dqi_data["da_dqi"]
            
            fig_gauge, score_tier, tier_color = create_dqi_gauge(score_val)
            
            st.markdown(f"""
            <div class="dqi-card">
                <div style="font-size:12px; font-weight:700; color:#94a3b8; letter-spacing:1px; text-transform:uppercase;">Overall Dataset Quality Health</div>
                <div style="font-size:32px; font-weight:800; color:{tier_color}; margin-top:4px;">{score_val:.2f} / 100</div>
                <div style="font-size:14px; font-weight:700; color:{tier_color}; margin-bottom:8px;">Status: {score_tier}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("""
            <div style="font-size:11px; color:#64748b; text-align:center;">
                <b>Score Scale:</b> 90–100 (Excellent) &nbsp;|&nbsp; 75–89 (Good) &nbsp;|&nbsp; 60–74 (Fair) &nbsp;|&nbsp; Below 60 (Needs Attention)
            </div>
            """, unsafe_allow_html=True)

        with c_right:
            st.markdown("### 📊 **FIVE DQI DIMENSION BREAKDOWN**")
            fig_dims = create_dimensions_bar_chart(dqi_data["dimensions"])
            st.plotly_chart(fig_dims, use_container_width=True)

        st.markdown("---")

        # DOMAIN CARD, QUALITY HEALTH SUMMARY & BEFORE/AFTER CHART
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("### 🎯 **DETECTED DOMAIN**")
            dom_info = st.session_state["domain_info"]
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left;">
                <div style="font-size:18px; font-weight:700; color:#38bdf8;">{dom_info['domain']}</div>
                <div style="font-size:12px; color:#94a3b8; margin-top:4px;">Confidence: <b>{dom_info['confidence']}</b></div>
                <div style="font-size:11px; color:#cbd5e1; margin-top:8px; line-height:1.4;">
                    {dom_info['reasoning']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("### 🩺 **QUALITY HEALTH SUMMARY**")
            issues_cnt = len(st.session_state["issues"])
            anom_cnt = st.session_state["anomaly_res"]["anomaly_count"] if st.session_state["anomaly_res"] else 0
            
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-size:12px; color:#94a3b8;">Issues Detected:</span>
                    <span style="font-weight:700; color:#f87171;">{issues_cnt}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-size:12px; color:#94a3b8;">Missing Cells:</span>
                    <span style="font-weight:700; color:#fbbf24;">{prof_raw['missing_cells']}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-size:12px; color:#94a3b8;">Duplicate Rows:</span>
                    <span style="font-weight:700; color:#fbbf24;">{prof_raw['duplicate_rows']}</span>
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
        "UPLOAD DATASET",
        "Import a CSV dataset or load the DataPulse demonstration dataset",
        dataset_name=ds_name, domain=ds_domain, status=ds_status
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="option-card">
            <div class="option-card-title">📂 Upload Your CSV</div>
            <div class="option-card-desc">Option A: Import a tabular CSV dataset from your local drive for quality evaluation.</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Drag & Drop CSV File Here or Browse", type=["csv"], label_visibility="visible")
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                load_dataset_into_session(user_df, uploaded_file.name)
                st.success(f"✓ Dataset '{uploaded_file.name}' loaded successfully! ({len(user_df)} rows, {len(user_df.columns)} columns)")
                st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    with col2:
        st.markdown("""
        <div class="option-card">
            <div class="option-card-title">🚀 Use Demo Dataset</div>
            <div class="option-card-desc">Option B: Load the pre-configured 11-row demonstration dataset with intentional flaws.</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Load Demo Dataset", type="primary", use_container_width=True):
            demo_path = os.path.join(os.path.dirname(__file__), "data", "demo_dataset.csv")
            if os.path.exists(demo_path):
                demo_df = pd.read_csv(demo_path)
                load_dataset_into_session(demo_df, "demo_dataset.csv")
                st.success("✓ Demonstration dataset loaded!")
                st.rerun()
            else:
                st.error("Demo dataset file not found.")

    render_how_it_works()

    if st.session_state["raw_df"] is not None:
        st.markdown("---")
        st.markdown(f"### 📋 Active Dataset Preview: `{st.session_state['dataset_name']}`")
        st.dataframe(st.session_state["raw_df"], use_container_width=True)

# ==============================================================================
# SECTION 3: DATA PROFILING
# ==============================================================================
elif selected_page == "Data Profiling":
    render_header(
        "DATA PROFILING MODULE",
        "Automated statistical inspection and structural quality analysis",
        dataset_name=ds_name, domain=ds_domain, status=ds_status
    )

    if st.session_state["raw_df"] is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        df = st.session_state["raw_df"]
        prof = profile_dataset(df)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            render_kpi("TOTAL ROWS", f"{prof['total_rows']}")
        with p2:
            render_kpi("TOTAL COLUMNS", f"{prof['total_cols']}")
        with p3:
            render_kpi("MISSING CELLS", f"{prof['missing_cells']}", f"{prof['missing_pct']}% of total")
        with p4:
            render_kpi("DUPLICATE ROWS", f"{prof['duplicate_rows']}", f"{prof['duplicate_pct']}% of rows")

        st.markdown("---")
        st.markdown("### 📊 Column-Level Profiling Summary")
        st.dataframe(prof["column_profile"], use_container_width=True)

# ==============================================================================
# SECTION 4: DQI ANALYSIS
# ==============================================================================
elif selected_page == "DQI Analysis":
    render_header(
        "DOMAIN-ADAPTIVE DQI ANALYSIS",
        "Transparent 5-dimension evaluation engine and active domain weight matrix",
        dataset_name=ds_name, domain=ds_domain, status=ds_status
    )

    if st.session_state["raw_df"] is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        dqi_res = st.session_state["initial_dqi"]
        st.markdown(f"## 🏆 Calculated DA-DQI Score: **`{dqi_res['da_dqi']} / 100`**")
        st.markdown(f"**Active Domain:** `{dqi_res['domain']}`")

        st.markdown("---")
        st.markdown("### 📏 5 Core Quality Dimensions & Domain Weights")

        cols = st.columns(5)
        dims = dqi_res["dimensions"]
        weights = dqi_res["weights"]

        for idx, (dim_name, score) in enumerate(dims.items()):
            w = weights.get(dim_name.lower(), 0.20)
            with cols[idx]:
                render_kpi(dim_name, f"{score:.1f}%", f"Weight: {w*100:.0f}%", "#38bdf8")

        st.markdown("---")
        with st.expander("ℹ️ **Transparent Mathematical Formulation & Viva Explanation**"):
            st.markdown(f"""
            $$\\\\text{{DA-DQI}} = \\\\sum_{{i=1}}^{{5}} w_i \\\\times D_i$$
            
            Where:
            - **Completeness ($D_1$)**: Ratio of non-missing cells = `{(dims['Completeness']):.2f}%` (Weight: `{weights['completeness']}`)
            - **Uniqueness ($D_2$)**: Ratio of non-duplicate rows = `{(dims['Uniqueness']):.2f}%` (Weight: `{weights['uniqueness']}`)
            - **Validity ($D_3$)**: Ratio of format-compliant Email & Date cells = `{(dims['Validity']):.2f}%` (Weight: `{weights['validity']}`)
            - **Consistency ($D_4$)**: Ratio of logical range-compliant numeric cells = `{(dims['Consistency']):.2f}%` (Weight: `{weights['consistency']}`)
            - **Integrity ($D_5$)**: Ratio of non-null key identifier cells = `{(dims['Integrity']):.2f}%` (Weight: `{weights['integrity']}`)
            """)

# ==============================================================================
# SECTION 5: QUALITY ISSUES
# ==============================================================================
elif selected_page == "Quality Issues":
    render_header(
        "QUALITY ISSUE DETECTION ENGINE",
        "Automatic identification of structural, formatting, and completeness flaws",
        dataset_name=ds_name, domain=ds_domain, status=ds_status
    )

    if st.session_state["raw_df"] is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        issues = st.session_state["issues"]
        
        high_cnt = sum(1 for i in issues if i["severity"] == "High")
        med_cnt = sum(1 for i in issues if i["severity"] == "Medium")
        low_cnt = sum(1 for i in issues if i["severity"] == "Low")

        i1, i2, i3, i4 = st.columns(4)
        with i1:
            render_kpi("TOTAL ISSUES", f"{len(issues)}")
        with i2:
            render_kpi("HIGH SEVERITY", f"{high_cnt}", "Requires Manual Review", "#f87171")
        with i3:
            render_kpi("MEDIUM SEVERITY", f"{med_cnt}", "Auto Remediation", "#fbbf24")
        with i4:
            render_kpi("LOW SEVERITY", f"{low_cnt}", "Minor Flaws", "#93c5fd")

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
        dataset_name=ds_name, domain=ds_domain, status=ds_status
    )

    if st.session_state["raw_df"] is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        issues = st.session_state["issues"]
        if not issues:
            st.info("No recommendations required.")
        else:
            decision_trace = generate_decision_trace(issues)
            st.markdown("### 🧠 Explainable Decision Log")
            st.dataframe(pd.DataFrame(decision_trace), use_container_width=True)

# ==============================================================================
# SECTION 7: DATA CLEANING
# ==============================================================================
elif selected_page == "Data Cleaning":
    render_header(
        "AUTOMATED SAFE DATA CLEANING",
        "Execute non-destructive, policy-approved data quality remediation",
        dataset_name=ds_name, domain=ds_domain, status=ds_status
    )

    if st.session_state["raw_df"] is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        st.markdown("### ⚙️ Cleaning Configuration Options")
        
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
                "trim_whitespace": opt_trim,
                "drop_duplicates": opt_dup,
                "impute_numeric": opt_num,
                "impute_categorical": opt_cat,
                "standardize_dates": opt_date
            }
            cleaned, logs = clean_dataset(st.session_state["raw_df"], opts)
            st.session_state["cleaned_df"] = cleaned
            st.session_state["clean_logs"] = logs
            
            domain = st.session_state["domain_info"]["domain"]
            st.session_state["cleaned_dqi"] = calculate_dqi(cleaned, domain)
            st.success("✓ Safe cleaning policy executed successfully!")

        st.markdown("---")
        if st.session_state["clean_logs"]:
            st.markdown("### 📜 Cleaning Execution Audit Logs")
            for log in st.session_state["clean_logs"]:
                st.markdown(f"- `{log}`")

        if st.session_state["cleaned_df"] is not None:
            st.markdown("### ✨ Cleaned Dataset Preview")
            st.dataframe(st.session_state["cleaned_df"], use_container_width=True)

# ==============================================================================
# SECTION 8: VALIDATION
# ==============================================================================
elif selected_page == "Validation":
    render_header(
        "QUALITY-AWARE VALIDATION & IMPROVEMENT",
        "Quantitative comparison of initial raw vs cleaned dataset quality",
        dataset_name=ds_name, domain=ds_domain, status=ds_status
    )

    if st.session_state["raw_df"] is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    elif st.session_state["cleaned_df"] is None:
        st.info("👆 Please execute cleaning in **'Data Cleaning'** first to generate validation metrics.")
    else:
        domain = st.session_state["domain_info"]["domain"]
        val_res = validate_cleaning(st.session_state["raw_df"], st.session_state["cleaned_df"], domain)

        st.markdown(f"## Validation Status: <span class='badge badge-ready'>{val_res['status']}</span>", unsafe_allow_html=True)

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            render_kpi("INITIAL DQI", f"{val_res['initial_dqi']:.2f}")
        with v2:
            render_kpi("CLEANED DA-DQI", f"{val_res['cleaned_dqi']:.2f}")
        with v3:
            render_kpi("DQI IMPROVEMENT", f"{val_res['dqi_improvement']:+.2f}", color="#10b981" if val_res['dqi_improvement'] >= 0 else "#ef4444")
        with v4:
            render_kpi("ROW DELTA", f"{val_res['rows_delta']}", f"Missing Fixed: {val_res['missing_reduced']}")

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

# ==============================================================================
# SECTION 9: ANOMALY DETECTION
# ==============================================================================
elif selected_page == "Anomaly Detection":
    render_header(
        "ISOLATION FOREST ANOMALY DETECTION",
        "Identify statistical outliers without destructive auto-deletion",
        dataset_name=ds_name, domain=ds_domain, status=ds_status
    )

    if st.session_state["raw_df"] is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        st.info("ℹ️ **Notice:** Anomaly detection identifies unusual records for manual review; it does NOT automatically delete them.")

        df_target = st.session_state["cleaned_df"] if st.session_state["cleaned_df"] is not None else st.session_state["raw_df"]

        contam = st.slider("Select Anomaly Contamination Factor", min_value=0.01, max_value=0.30, value=0.10, step=0.01)

        if st.button("🔍 Run Isolation Forest Detection", type="primary", use_container_width=True):
            st.session_state["anomaly_res"] = detect_anomalies(df_target, contamination=contam)

        if st.session_state["anomaly_res"] is not None:
            anom_res = st.session_state["anomaly_res"]
            st.markdown(f"### {anom_res['status']}")

            a1, a2 = st.columns(2)
            with a1:
                render_kpi("FLAGGED ANOMALIES", f"{anom_res['anomaly_count']}", f"{anom_res['anomaly_pct']}% of records")
            with a2:
                render_kpi("EVALUATED FEATURES", f"{len(anom_res['numerical_cols'])}", ", ".join(anom_res['numerical_cols']))

            st.markdown("---")
            if not anom_res["anomalies_df"].empty:
                st.markdown("### ⚠️ Flagged Anomalous Records (For Manual Review)")
                st.dataframe(anom_res["anomalies_df"], use_container_width=True)
                
                if len(anom_res['numerical_cols']) >= 2:
                    col_x = anom_res['numerical_cols'][0]
                    col_y = anom_res['numerical_cols'][1]
                    fig_scat = px.scatter(
                        anom_res["full_df_with_flags"],
                        x=col_x,
                        y=col_y,
                        color="anomaly_flag",
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
        dataset_name=ds_name, domain=ds_domain, status=ds_status
    )

    if st.session_state["raw_df"] is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        st.markdown("### 🗄️ PostgreSQL Database Connection Configuration (Optional)")
        db_url_input = st.text_input("PostgreSQL Connection String (Optional)", value="", placeholder="postgresql://user:password@localhost:5432/datapulse_db")
        
        db_check = check_db_connection(db_url_input)
        st.info(f"Database Status: {db_check['status_message']}")

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if st.button("⚡ Run Full ETL Pipeline", type="primary", use_container_width=True):
                domain = st.session_state["domain_info"]["domain"]
                report = run_etl_pipeline(st.session_state["raw_df"], domain=domain, db_url=db_url_input)
                st.session_state["etl_report"] = report
                st.success("✓ ETL Pipeline Execution Complete!")

        if "etl_report" in st.session_state:
            rep = st.session_state["etl_report"]
            with col_e2:
                st.download_button(
                    label="📥 Export Clean CSV File",
                    data=rep["csv_download_payload"],
                    file_name=f"datapulse_cleaned_{st.session_state['dataset_name']}",
                    mime="text/csv",
                    use_container_width=True
                )

            st.markdown("---")
            st.markdown("### 📑 ETL Execution Summary")
            
            e1, e2, e3 = st.columns(3)
            with e1:
                render_kpi("INPUT RECORDS", f"{rep['input_rows']}")
            with e2:
                render_kpi("CLEANED RECORDS", f"{rep['cleaned_rows']}")
            with e3:
                render_kpi("DESTINATION", f"{rep['destination_table']}")

# ==============================================================================
# SECTION 11: MONITORING
# ==============================================================================
elif selected_page == "Monitoring":
    render_header(
        "PIPELINE & SYSTEM MONITORING",
        "Real-time pipeline health tracking and operation logs",
        dataset_name=ds_name, domain=ds_domain, status=ds_status
    )

    if st.session_state["raw_df"] is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        st.markdown("### 🟢 Active System Health")
        m1, m2, m3 = st.columns(3)
        with m1:
            render_kpi("ENGINE STATUS", "ONLINE", "Local Python Engine", "#10b981")
        with m2:
            render_kpi("DATASET HEALTH", "ACTIVE", st.session_state['dataset_name'], "#38bdf8")
        with m3:
            dqi_val = st.session_state['cleaned_dqi']['da_dqi'] if st.session_state['cleaned_dqi'] else st.session_state['initial_dqi']['da_dqi']
            render_kpi("DA-DQI SCORE", f"{dqi_val:.2f} / 100", "Live Quality Score", "#f59e0b")

        st.markdown("---")
        st.markdown("### 📋 Activity Audit Log")
        st.info("✓ System engine operating normally.")

# ==============================================================================
# SECTION 12: REPORTS
# ==============================================================================
elif selected_page == "Reports":
    render_header(
        "PROJECT SUMMARY & REPORTING",
        "Comprehensive DataPulse project execution report",
        dataset_name=ds_name, domain=ds_domain, status=ds_status
    )

    if st.session_state["raw_df"] is None:
        st.warning("No active dataset. Please upload or load a dataset first.")
    else:
        domain = st.session_state["domain_info"]["domain"]
        initial_dqi = st.session_state["initial_dqi"]["da_dqi"]
        cleaned_dqi = st.session_state["cleaned_dqi"]["da_dqi"] if st.session_state["cleaned_dqi"] else initial_dqi
        
        st.markdown(f"""
        # 📄 **DataPulse Major Project Final Execution Report**
        
        **Project Title:** DataPulse: A Domain-Adaptive Data Quality Index Driven Intelligent Data Engineering and Analytics Framework  
        **Active Dataset:** `{st.session_state['dataset_name']}`  
        **Detected Domain:** `{domain}`  

        ---
        
        ### 📊 **Live Dynamic Results Summary**
        - **Initial Raw DQI:** `{initial_dqi:.2f} / 100`
        - **Post-Cleaning DA-DQI:** `{cleaned_dqi:.2f} / 100`
        - **Net Quality Gain:** `{cleaned_dqi - initial_dqi:+.2f} pts`
        - **Raw Record Count:** `{len(st.session_state['raw_df'])}`
        - **Cleaned Record Count:** `{len(st.session_state['cleaned_df']) if st.session_state['cleaned_df'] is not None else len(st.session_state['raw_df'])}`
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
