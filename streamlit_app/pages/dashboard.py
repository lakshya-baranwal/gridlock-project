"""
streamlit_app/pages/dashboard.py — Analytics Dashboard
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from streamlit_app.utils.data_loader import (
    get_hourly_trend, get_weather_impact, get_peak_heatmap,
    get_geo_demand, get_feature_importance, get_model_scores,
    get_road_type_stats, get_temp_bins, get_dataset_stats,
)

COLORS = {"primary": "#4a9eff", "success": "#48bb78", "warn": "#f6ad55",
          "danger": "#fc8181", "purple": "#9f7aea", "teal": "#4fd1c5"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,22,41,0.6)",
    font=dict(family="Inter", color="#a0aec0"),
    margin=dict(l=10, r=10, t=40, b=10),
)
_AXIS = dict(gridcolor="rgba(99,179,237,0.08)", zerolinecolor="rgba(99,179,237,0.1)")


def kpi_card(value, label, sub="", color="#4a9eff"):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def render():
    st.markdown("<h1 style='color:#e2e8f0;font-size:2rem;font-weight:800;margin-bottom:4px;'>🚦 Traffic Demand Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#718096;font-size:14px;margin-bottom:24px;'>Stacking Ensemble V8 · LightGBM + XGBoost + CatBoost + LightGBM Meta-Learner</p>", unsafe_allow_html=True)

    scores = get_model_scores()
    stats  = get_dataset_stats()

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, f"{scores.get('competition_score', 0):.2f}", "Competition Score", "out of 100", "#4a9eff"),
        (c2, f"{scores.get('stacking_r2', 0):.4f}", "Stacking R²",  "Meta-LightGBM",  "#48bb78"),
        (c3, f"{scores.get('xgb_r2', 0):.4f}",      "Best Base R²", "XGBoost",         "#9f7aea"),
        (c4, f"{stats.get('total_rows', 0):,}",       "Training Rows", "77,300 samples", "#f6ad55"),
        (c5, f"{stats.get('unique_geohashes', 0):,}", "Geohash Zones", "spatial coverage","#4fd1c5"),
    ]
    for col, val, label, sub, color in cards:
        col.markdown(kpi_card(val, label, sub, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Hourly Trend + Weather Impact ──────────────────────────────────
    st.markdown("<div class='section-header'>Temporal & Weather Analysis</div>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        df_h = get_hourly_trend()
        if not df_h.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_h["hour"], y=df_h["mean_demand"],
                mode="lines+markers",
                line=dict(color=COLORS["primary"], width=3),
                marker=dict(size=7, color=COLORS["primary"]),
                fill="tozeroy",
                fillcolor="rgba(74,158,255,0.1)",
                name="Avg Demand",
            ))
            # Rush hour bands
            for x0, x1, label in [(6,9,"Morning Rush"), (17,20,"Evening Rush")]:
                fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(252,129,129,0.08)",
                              line_width=0, annotation_text=label,
                              annotation_position="top left",
                              annotation_font_size=10, annotation_font_color="#fc8181")
            fig.update_layout(**PLOTLY_LAYOUT,
                              title="Average Demand by Hour of Day",
                              xaxis=dict(**_AXIS, title="Hour", tickvals=list(range(0,24,2))),
                              yaxis=dict(**_AXIS, title="Avg Demand"))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        df_w = get_weather_impact()
        if not df_w.empty:
            w_colors = {"Sunny":"#f6ad55","Rainy":"#4a9eff","Foggy":"#9f7aea","Snowy":"#4fd1c5"}
            fig = go.Figure()
            for _, row in df_w.iterrows():
                fig.add_trace(go.Bar(
                    x=[row["weather"]], y=[row["mean_demand"]],
                    name=row["weather"],
                    marker_color=w_colors.get(row["weather"], COLORS["primary"]),
                    error_y=dict(type="data", array=[row["std_demand"]], visible=True,
                                 color="rgba(255,255,255,0.3)"),
                    width=0.5,
                ))
            fig.update_layout(**PLOTLY_LAYOUT,
                              title="Weather Impact on Traffic Demand",
                              xaxis=dict(**_AXIS, title="Weather"),
                              yaxis=dict(**_AXIS, title="Avg Demand"),
                              showlegend=False, bargap=0.3)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Peak Heatmap + Road Type ──────────────────────────────────────
    st.markdown("<div class='section-header'>Peak Traffic Patterns</div>", unsafe_allow_html=True)
    col_l2, col_r2 = st.columns([2, 1])

    with col_l2:
        df_p = get_peak_heatmap()
        if not df_p.empty:
            pivot = df_p.pivot(index="road_type", columns="hour", values="mean_demand")
            fig = px.imshow(
                pivot,
                color_continuous_scale=[[0,"#0a0e1a"],[0.3,"#1e3a5f"],
                                        [0.6,"#2d6fa3"],[0.8,"#f6ad55"],[1,"#fc8181"]],
                labels=dict(x="Hour of Day", y="Road Type", color="Avg Demand"),
                title="Demand Heatmap: Road Type x Hour",
                aspect="auto",
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        df_rt = get_road_type_stats()
        if not df_rt.empty:
            rt_colors = {"Highway": COLORS["danger"], "Street": COLORS["warn"], "Residential": COLORS["teal"]}
            fig = go.Figure()
            for _, row in df_rt.iterrows():
                fig.add_trace(go.Bar(
                    x=[row["road_type"]], y=[row["mean"]],
                    name=row["road_type"],
                    marker_color=rt_colors.get(row["road_type"], COLORS["primary"]),
                    width=0.5,
                ))
            fig.update_layout(**PLOTLY_LAYOUT,
                              title="Avg Demand by Road Type",
                              xaxis=_AXIS, yaxis=_AXIS,
                              showlegend=False, bargap=0.3)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Feature Importance + Temperature ───────────────────────────────
    st.markdown("<div class='section-header'>Feature Analysis</div>", unsafe_allow_html=True)
    col_l3, col_r3 = st.columns(2)

    with col_l3:
        df_fi = get_feature_importance()
        if not df_fi.empty:
            cat_colors = {"spatial":"#4a9eff","temporal":"#48bb78",
                          "environment":"#f6ad55","road":"#9f7aea"}
            df_fi = df_fi.sort_values("importance")
            fig = go.Figure(go.Bar(
                x=df_fi["importance"],
                y=df_fi["feature"],
                orientation="h",
                marker_color=[cat_colors.get(c, COLORS["primary"]) for c in df_fi["category"]],
                text=[f"{v:.1%}" for v in df_fi["importance"]],
                textposition="outside",
                textfont=dict(size=10),
            ))
            # Legend annotations
            for i, (cat, color) in enumerate(cat_colors.items()):
                fig.add_annotation(x=0.85+i*0.04, y=1.05, xref="paper", yref="paper",
                                   text=f"● {cat}", showarrow=False,
                                   font=dict(color=color, size=10))
            fig.update_layout(**PLOTLY_LAYOUT,
                              title="Feature Importance (LightGBM Meta)",
                              xaxis=dict(**_AXIS, title="Importance"),
                              yaxis=_AXIS, height=480)
            st.plotly_chart(fig, use_container_width=True)

    with col_r3:
        df_tb = get_temp_bins()
        if not df_tb.empty:
            fig = go.Figure(go.Bar(
                x=df_tb["temp_bin"].astype(str),
                y=df_tb["mean_demand"],
                marker=dict(
                    color=df_tb["mean_demand"],
                    colorscale=[[0,"#0a2a5e"],[0.5,"#2d6fa3"],[1,"#fc8181"]],
                    showscale=True,
                    colorbar=dict(title="Demand"),
                ),
                text=[f"{v:.3f}" for v in df_tb["mean_demand"]],
                textposition="outside",
            ))
            fig.update_layout(**PLOTLY_LAYOUT,
                              title="Temperature Range vs Avg Demand",
                              xaxis=dict(**_AXIS, title="Temperature Range"),
                              yaxis=dict(**_AXIS, title="Avg Demand"))
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Geo Map ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Spatial Demand Map</div>", unsafe_allow_html=True)
    df_geo = get_geo_demand()
    if not df_geo.empty:
        df_geo_plot = df_geo.dropna(subset=["lat","lon"])
        fig = px.scatter_mapbox(
            df_geo_plot, lat="lat", lon="lon",
            color="mean_demand", size="mean_demand",
            color_continuous_scale=[[0,"#0a2a5e"],[0.4,"#2d6fa3"],[0.7,"#f6ad55"],[1,"#fc8181"]],
            size_max=14, zoom=10,
            hover_data={"geohash": True, "mean_demand": ":.3f", "lat": False, "lon": False},
            title="Average Traffic Demand by Geohash Zone",
            mapbox_style="carto-darkmatter",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=40,b=0),
                          font=dict(family="Inter", color="#a0aec0"),
                          coloraxis_colorbar=dict(title="Demand"))
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
    else:
        st.info("Run `scripts/precompute_analytics.py` to generate analytics data.")

    # ── Model Score Comparison ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Model Performance Comparison</div>", unsafe_allow_html=True)
    model_names  = ["LightGBM", "XGBoost", "CatBoost", "Stacking (V8)"]
    model_r2s    = [scores.get("lgbm_r2",0), scores.get("xgb_r2",0),
                    scores.get("catboost_r2",0), scores.get("stacking_r2",0)]
    model_colors = [COLORS["teal"], COLORS["warn"], COLORS["purple"], COLORS["primary"]]

    fig = go.Figure(go.Bar(
        x=model_names, y=model_r2s,
        marker_color=model_colors,
        text=[f"{v:.5f}" for v in model_r2s],
        textposition="outside",
        width=0.5,
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="R² Score per Model (5-Fold OOF)",
                      xaxis=_AXIS,
                      yaxis=dict(**_AXIS, title="R² Score", range=[0.945, 0.960]))
    st.plotly_chart(fig, use_container_width=True)
