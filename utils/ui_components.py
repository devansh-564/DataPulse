import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

CUSTOM_CSS = """
<style>
    /* Dark Enterprise Dashboard Theme */
    .stApp {
        background-color: #090d16;
        color: #f1f5f9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Main Content Container Adjustments */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }

    /* Global Header Banner */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 18px 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        border-left: 5px solid #38bdf8;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }
    
    .main-header h1 {
        font-size: 22px;
        font-weight: 700;
        margin: 0 0 4px 0;
        color: #f8fafc;
        letter-spacing: -0.3px;
    }

    .main-header p {
        font-size: 13px;
        color: #94a3b8;
        margin: 0 0 10px 0;
    }

    .status-strip {
        display: flex;
        gap: 16px;
        font-size: 12px;
        color: #cbd5e1;
        background: rgba(15, 23, 42, 0.6);
        padding: 6px 14px;
        border-radius: 6px;
        border: 1px solid #1e293b;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        height: 100%;
    }

    .kpi-card:hover {
        border-color: #38bdf8;
    }

    .kpi-card .kpi-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }

    .kpi-card .kpi-val {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.2;
    }

    .kpi-card .kpi-sub {
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
    }

    /* Agentic Status Card */
    .agent-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-left: 4px solid #38bdf8;
    }

    .agent-card.success { border-left-color: #10b981; }
    .agent-card.warning { border-left-color: #f59e0b; }
    .agent-card.action { border-left-color: #3b82f6; }

    .agent-card .agent-title {
        font-size: 14px;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        justify-content: space-between;
    }

    .agent-card .agent-body {
        font-size: 12px;
        color: #cbd5e1;
        margin-top: 6px;
        line-height: 1.4;
    }

    /* Option Cards on Upload Page */
    .option-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 24px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .option-card-title {
        font-size: 16px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .option-card-desc {
        font-size: 13px;
        color: #94a3b8;
        margin-bottom: 16px;
        line-height: 1.4;
    }

    /* DQI Main Gauge Card Container */
    .dqi-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 700;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-ready { background-color: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-waiting { background-color: #78350f; color: #fbbf24; border: 1px solid #d97706; }
    .badge-auto { background-color: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-manual { background-color: #78350f; color: #fbbf24; border: 1px solid #d97706; }
    .badge-high { background-color: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }
    .badge-medium { background-color: #7c2d12; color: #fdba74; border: 1px solid #ea580c; }
    .badge-low { background-color: #1e3a8a; color: #93c5fd; border: 1px solid #2563eb; }

    /* Pipeline Stage Tracker UI */
    .pipeline-tracker {
        display: flex;
        justify-content: space-between;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .pipeline-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        font-size: 11px;
        font-weight: 600;
        color: #64748b;
    }

    .pipeline-step.active { color: #38bdf8; }
    .pipeline-step.done { color: #34d399; }

    .step-icon {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 4px;
        font-size: 12px;
        background: #1e293b;
        border: 1px solid #334155;
    }

    .pipeline-step.done .step-icon {
        background: #064e3b;
        border-color: #059669;
        color: #34d399;
    }

    .pipeline-step.active .step-icon {
        background: #0c4a6e;
        border-color: #0284c7;
        color: #38bdf8;
    }

    /* How DataPulse Works Pipeline Box */
    .how-it-works-container {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 24px;
    }

    .how-it-works-title {
        font-size: 14px;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: 0.5px;
        margin-bottom: 14px;
        text-transform: uppercase;
    }

    .how-flow {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }

    .how-step {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 12px;
        font-weight: 600;
        color: #38bdf8;
        text-align: center;
    }

    .how-arrow {
        color: #64748b;
        font-weight: 700;
    }

    .how-footer {
        display: flex;
        gap: 16px;
        font-size: 11px;
        color: #94a3b8;
        margin-top: 10px;
    }

    /* Welcome Container */
    .welcome-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
    }

    .welcome-title {
        font-size: 28px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .welcome-subtitle {
        font-size: 15px;
        color: #38bdf8;
        margin-bottom: 16px;
    }

    .welcome-desc {
        font-size: 14px;
        color: #94a3b8;
        max-width: 600px;
        margin: 0 auto 24px auto;
        line-height: 1.5;
    }

    /* Streamlit Button Overrides */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }

    /* Sidebar Radio Clean Focus Styles */
    section[data-testid="stSidebar"] div.stRadio > div {
        gap: 4px !important;
    }
    
    section[data-testid="stSidebar"] div.stRadio label {
        border-radius: 6px !important;
        padding: 6px 12px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #cbd5e1 !important;
        transition: background 0.15s ease !important;
    }

    section[data-testid="stSidebar"] div.stRadio label:hover {
        background-color: #1e293b !important;
        color: #f8fafc !important;
    }
</style>
"""

def inject_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_header(title: str, subtitle: str, dataset_name: str = "No Active Dataset", domain: str = "N/A", status: str = "Waiting for Dataset", category: str = "Structured"):
    status_class = "badge-ready" if status in ["Ready", "Analyzed", "Completed"] else "badge-waiting"
    status_icon = "ΓùÅ" if status in ["Ready", "Analyzed", "Completed"] else "Γùï"

    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
        <div class="status-strip">
            <div class="status-pill">≡ƒôè <b>Dataset:</b> {dataset_name}</div>
            <div class="status-pill">≡ƒôª <b>Category:</b> {category}</div>
            <div class="status-pill">≡ƒÄ» <b>Domain:</b> {domain}</div>
            <div class="status-pill"><span class="badge {status_class}">{status_icon} Status: {status}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi(label: str, value: str, subtext: str = "", color: str = "#38bdf8"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-val" style="color: {color};">{value}</div>
        {"<div class='kpi-sub'>" + subtext + "</div>" if subtext else ""}
    </div>
    """, unsafe_allow_html=True)

def render_agent_card(agent_name: str, stage: str, status_type: str, message: str, reasoning: str, action: str, result: str):
    card_cls = status_type.lower() if status_type.lower() in ['success', 'warning', 'action'] else ''
    st.markdown(f"""
    <div class="agent-card {card_cls}">
        <div class="agent-title">
            <span>≡ƒñû <b>{agent_name}</b> <small style="color:#94a3b8;">[{stage}]</small></span>
            <span class="badge badge-ready">{status_type}</span>
        </div>
        <div class="agent-body">
            <div><b>Detected/Message:</b> {message}</div>
            <div><b>Reasoning:</b> <i>"{reasoning}"</i></div>
            <div><b>Action Taken:</b> {action}</div>
            <div style="color:#38bdf8; font-weight:600; margin-top:2px;"><b>Result:</b> {result}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_dqi_gauge(score: float, title: str = "Overall Dataset Quality Health"):
    if score >= 90:
        bar_color = "#10b981"
        label_text = "EXCELLENT"
    elif score >= 75:
        bar_color = "#3b82f6"
        label_text = "GOOD"
    elif score >= 60:
        bar_color = "#f59e0b"
        label_text = "FAIR"
    else:
        bar_color = "#ef4444"
        label_text = "NEEDS ATTENTION"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': " / 100", 'font': {'size': 36, 'color': '#f8fafc', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'dtick': 20},
            'bar': {'color': bar_color, 'thickness': 0.25},
            'bgcolor': "#0f172a",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 60], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [60, 75], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [75, 90], 'color': 'rgba(59, 130, 246, 0.15)'},
                {'range': [90, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc', family='Inter'),
        margin=dict(l=20, r=20, t=30, b=10),
        height=220
    )

    return fig, label_text, bar_color

def create_dimensions_bar_chart(dimensions: dict, title: str = "Dimension Score (%)"):
    categories = list(dimensions.keys())
    values = [float(v) for v in dimensions.values()]

    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation='h',
        marker=dict(
            color=values,
            colorscale=[[0, '#ef4444'], [0.6, '#f59e0b'], [0.75, '#3b82f6'], [1.0, '#10b981']],
            cmin=0,
            cmax=100
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition='inside',
        insidetextanchor='end',
        textfont=dict(color='#ffffff', size=11, family='Inter')
    ))

    fig.update_layout(
        xaxis=dict(
            range=[0, 100],
            gridcolor='#1e293b',
            title=dict(text=title, font=dict(color='#94a3b8', size=11)),
            tickfont=dict(color='#94a3b8', size=11)
        ),
        yaxis=dict(
            gridcolor='#1e293b',
            tickfont=dict(color='#f8fafc', size=12)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=20, t=10, b=30),
        height=240,
        showlegend=False
    )

    return fig

def create_before_after_chart(initial_dqi: float, cleaned_dqi: float):
    delta = cleaned_dqi - initial_dqi

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Initial Raw DQI', 'Cleaned DA-DQI'],
        y=[initial_dqi, cleaned_dqi],
        marker_color=['#f59e0b', '#10b981'],
        text=[f"{initial_dqi:.2f}", f"{cleaned_dqi:.2f}"],
        textposition='auto',
        width=0.35,
        textfont=dict(size=14, color='#ffffff', family='Inter')
    ))

    fig.update_layout(
        title=dict(text=f"Quality Improvement Progression (Delta: {delta:+.2f} pts)", font=dict(size=13, color='#94a3b8')),
        yaxis=dict(
            range=[0, 100],
            title=dict(text="DA-DQI Score", font=dict(color='#94a3b8', size=11)),
            gridcolor='#1e293b',
            tickfont=dict(color='#94a3b8', size=11)
        ),
        xaxis=dict(
            tickfont=dict(color='#f8fafc', size=12)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=30),
        height=240,
        showlegend=False
    )

    return fig

def render_pipeline_tracker(session_state: dict):
    has_raw = session_state.get("raw_df") is not None or session_state.get("unstructured_text") is not None
    has_cleaned = session_state.get("cleaned_df") is not None
    has_agent_state = session_state.get("agent_state") is not None

    stages = [
        {"name": "INGEST", "icon": "ΓåÑ", "done": has_raw},
        {"name": "PROFILE", "icon": "Γù½", "done": has_raw},
        {"name": "ASSESS", "icon": "Γùë", "done": has_raw},
        {"name": "PLAN", "icon": "Γ£ª", "done": has_agent_state},
        {"name": "CLEAN", "icon": "ΓÖ╗", "done": has_cleaned},
        {"name": "VALIDATE", "icon": "Γ£ô", "done": has_cleaned},
        {"name": "TRANSFORM", "icon": "Γîü", "done": has_cleaned},
        {"name": "LOAD", "icon": "Γçä", "done": has_cleaned},
        {"name": "MONITOR", "icon": "≡ƒ⌐║", "done": has_agent_state}
    ]

    html_steps = ""
    for s in stages:
        cls = "done" if s["done"] else ("active" if has_raw else "")
        mark = "Γ£ô" if s["done"] else s["icon"]
        html_steps += f"""<div class="pipeline-step {cls}"><div class="step-icon">{mark}</div><div>{s['name']}</div></div>"""

    st.markdown(f"""<div class="pipeline-tracker">{html_steps}</div>""", unsafe_allow_html=True)

def render_how_it_works():
    st.markdown("""
    <div class="how-it-works-container">
        <div class="how-it-works-title">ΓÜí How DataPulse Works (Agentic AI Workflow)</div>
        <div class="how-flow">
            <div class="how-step">01 Upload</div>
            <div class="how-arrow">ΓåÆ</div>
            <div class="how-step">02 Detect & Profile</div>
            <div class="how-arrow">ΓåÆ</div>
            <div class="how-step">03 Assess Quality</div>
            <div class="how-arrow">ΓåÆ</div>
            <div class="how-step">04 Agentic Plan</div>
            <div class="how-arrow">ΓåÆ</div>
            <div class="how-step">05 Clean & Validate</div>
        </div>
        <div style="font-size:12px; color:#cbd5e1; margin-top:8px;">
            DataPulse automatically detects Structured, Semi-Structured, or Unstructured input data, triggers a 10-Agent AI pipeline, assesses DA-DQI/Document Quality, executes safe cleaning, and validates improvement.
        </div>
        <div class="how-footer">
            <span><b>Supported Formats:</b> CSV, XLSX, JSON, XML, TXT, PDF, DOCX</span>
            <span>┬╖</span>
            <span><b>Engine:</b> 10-Agent AI Pipeline</span>
            <span>┬╖</span>
            <span><b>Output:</b> Clean CSV & PostgreSQL Export</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

