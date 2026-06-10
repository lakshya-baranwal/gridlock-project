"""
streamlit_app/pages/architecture.py — Model Architecture Page
"""
import streamlit as st
import plotly.graph_objects as go

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,22,41,0.6)",
    font=dict(family="Inter", color="#a0aec0"),
    margin=dict(l=10, r=10, t=40, b=10),
)


def render():
    st.markdown("<h1 style='color:#e2e8f0;font-size:2rem;font-weight:800;margin-bottom:4px;'>Model Architecture</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#718096;font-size:14px;margin-bottom:24px;'>Stacking Ensemble V8 — Technical Deep Dive</p>", unsafe_allow_html=True)

    # ── Architecture Diagram (Sankey) ─────────────────────────────────────────
    st.markdown("<div class='section-header'>Ensemble Flow</div>", unsafe_allow_html=True)
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=22,
            line=dict(color="rgba(99,179,237,0.3)", width=1),
            label=[
                "Raw Features (11)",      # 0
                "LGB Features (28)",       # 1
                "XGB Features (35)",       # 2
                "CB Features (31)",        # 3
                "LightGBM\n(5-fold)",      # 4
                "XGBoost\n(5-fold)",       # 5
                "CatBoost\n(5-fold)",      # 6
                "Meta Features\n(base preds + orig)", # 7
                "Meta LightGBM\n(5-fold)", # 8
                "Final Prediction",        # 9
            ],
            color=[
                "#4a5568","#4fd1c5","#f6ad55","#9f7aea",
                "#4fd1c5","#f6ad55","#9f7aea",
                "#e2e8f0","#4a9eff","#48bb78",
            ],
            x=[0.0, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.75, 0.88, 1.0],
            y=[0.5, 0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.5,  0.5,  0.5],
        ),
        link=dict(
            source=[0,0,0, 1,2,3, 4,5,6, 7],
            target=[1,2,3, 4,5,6, 7,7,7, 8],
            value =[8,8,8, 8,8,8, 3,3,3, 9],
            color =[
                "rgba(79,209,197,0.15)","rgba(246,173,85,0.15)","rgba(159,122,234,0.15)",
                "rgba(79,209,197,0.2)","rgba(246,173,85,0.2)","rgba(159,122,234,0.2)",
                "rgba(226,232,240,0.15)","rgba(226,232,240,0.15)","rgba(226,232,240,0.15)",
                "rgba(74,158,255,0.3)",
            ],
        ),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=400,
                      title="Data flow through the Stacking Ensemble V8")
    st.plotly_chart(fig, use_container_width=True)

    # ── Model Scores Table ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Performance Metrics</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        data = {
            "Model":    ["LightGBM", "XGBoost", "CatBoost", "**Stacking V8**"],
            "OOF R²":   ["0.95055", "0.95578", "0.95354", "**0.95612**"],
            "Features": ["28 (+ native cat)", "35 (OHE)", "31 (+ raw cat)", "All + meta"],
            "Tuning":   ["15 trials", "15 trials", "15 trials", "15 trials"],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=["LightGBM","XGBoost","CatBoost","Stack V8"],
            y=[0.95055, 0.95578, 0.95354, 0.956115],
            marker_color=["#4fd1c5","#f6ad55","#9f7aea","#4a9eff"],
            text=["0.95055","0.95578","0.95354","0.95612"],
            textposition="outside", width=0.5,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(range=[0.945, 0.962],
                          gridcolor="rgba(99,179,237,0.08)"),
                          height=280)
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature Engineering ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Feature Engineering Pipeline</div>", unsafe_allow_html=True)
    fe_cols = st.columns(2)
    groups = [
        ("Temporal", ["Minute of day (0-1439)", "Cyclic encoding: sin(2pi*t/1440) & cos(2pi*t/1440)", "15-min time slots", "Lag-1D demand from day 48"]),
        ("Spatial", ["Geohash to lat/lon decode", "Geohash prefix levels L4 & L5", "Smooth target encoding (5-fold, k=20)", "8-directional neighbor target encoding"]),
        ("Environment", ["Weather one-hot encoding (4 types)", "Temperature imputation (geo x day median)", "Missing value indicators", "Temperature bin features"]),
        ("Road", ["Road type ordinal (Residential < Street < Highway)", "Lane count x road type interaction", "High-volume lane flag (>=4 lanes)", "Road+lane combination target encoding"]),
    ]
    for i, (title, items) in enumerate(groups):
        with fe_cols[i % 2]:
            st.markdown(f"**{title}**")
            for item in items:
                st.markdown(f"- {item}")
            st.markdown("")

    # ── Hyper-parameter Tuning ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Optuna Tuning Details</div>", unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown("**LightGBM** — Best R²: 0.94489")
        st.markdown("""
- learning_rate · num_leaves · max_depth
- feature_fraction · bagging_fraction
- lambda_l1 · lambda_l2
- min_child_samples · bagging_freq
        """)
    with tc2:
        st.markdown("**XGBoost** — Best R²: 0.95193")
        st.markdown("""
- learning_rate · max_depth
- min_child_weight
- subsample · colsample_bytree
        """)
    with tc3:
        st.markdown("**CatBoost** — Best R²: 0.94766")
        st.markdown("""
- learning_rate · depth
- l2_leaf_reg · random_strength
- bagging_temperature
        """)

    # ── Training Setup ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Training Configuration</div>", unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, label, val in [
        (sc1, "Training Rows", "77,299"),
        (sc2, "CV Folds", "5-Fold KFold"),
        (sc3, "GPU", "Enabled (fallback CPU)"),
        (sc4, "Total Time", "~34 minutes"),
    ]:
        col.metric(label, val)
