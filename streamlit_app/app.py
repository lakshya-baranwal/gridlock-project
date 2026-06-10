"""
streamlit_app/app.py  —  GridLock Intelligence Dashboard
Run: streamlit run streamlit_app/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="GridLock Intelligence",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp { background: #0a0e1a; color: #e2e8f0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #1a1f35 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}
[data-testid="stSidebar"] .stRadio label { color: #a0aec0 !important; font-size: 14px; }

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(26,31,53,0.9) 0%, rgba(15,22,41,0.95) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(99,179,237,0.5); }
.kpi-value { font-size: 2.4rem; font-weight: 800; margin: 8px 0 4px; }
.kpi-label { font-size: 0.8rem; color: #718096; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-sub   { font-size: 0.78rem; color: #4a9eff; margin-top: 4px; }

/* Section headers */
.section-header {
    font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
    border-left: 4px solid #4a9eff;
    padding-left: 14px; margin: 32px 0 16px;
}

/* Prediction result */
.pred-box {
    background: linear-gradient(135deg, #1a1f35 0%, #0f1629 100%);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.pred-demand { font-size: 4rem; font-weight: 800; }
.pred-level  { font-size: 1.4rem; margin-top: 8px; }

/* Tabs */
.stTabs [data-baseweb="tab"] { color: #718096; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #4a9eff !important; }

/* Model status badge */
.status-ok  { color: #48bb78; font-weight: 600; }
.status-off { color: #f6ad55; font-weight: 600; }

/* Hide default header */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## GridLock Intelligence")
    st.markdown("<p style='color:#718096;font-size:13px;'>Stacking Ensemble V8 · Traffic Demand</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Dashboard", "Live Prediction", "Model Architecture"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("<p style='color:#4a5568;font-size:12px;'>Competition Score</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4a9eff;font-size:1.8rem;font-weight:800;'>95.61</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#718096;font-size:12px;'>R² = 0.9561 · 5-fold CV</p>", unsafe_allow_html=True)

# ── Route pages ──────────────────────────────────────────────────────────────
if page == "Dashboard":
    from streamlit_app.pages.dashboard import render
    render()
elif page == "Live Prediction":
    from streamlit_app.pages.prediction import render
    render()
elif page == "Model Architecture":
    from streamlit_app.pages.architecture import render
    render()
