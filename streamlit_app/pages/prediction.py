"""
streamlit_app/pages/prediction.py — Live Prediction Page
"""
import streamlit as st
import plotly.graph_objects as go
import requests
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_URL = "http://localhost:8000"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,22,41,0.6)",
    font=dict(family="Inter", color="#a0aec0"),
    margin=dict(l=10, r=10, t=40, b=10),
)


def _check_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _model_loaded() -> bool:
    try:
        r = requests.get(f"{API_URL}/predict/status", timeout=2)
        return r.json().get("loaded", False)
    except Exception:
        return False


def _model_message() -> str:
    try:
        r = requests.get(f"{API_URL}/predict/status", timeout=2)
        return r.json().get("message", "")
    except Exception:
        return "API unreachable"


def _do_predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict/", json=payload, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _demand_gauge(demand: float, level: str):
    color = "#48bb78" if demand < 0.3 else ("#f6ad55" if demand < 0.6 else "#fc8181")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(demand * 100, 1),
        title={"text": f"Traffic Demand  ·  {level}", "font": {"size": 14, "color": "#a0aec0"}},
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a5568", "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(15,22,41,0.8)",
            "bordercolor": "rgba(99,179,237,0.2)",
            "steps": [
                {"range": [0, 20],  "color": "rgba(72,187,120,0.1)"},
                {"range": [20, 40], "color": "rgba(246,173,85,0.1)"},
                {"range": [40, 60], "color": "rgba(246,173,85,0.15)"},
                {"range": [60, 100],"color": "rgba(252,129,129,0.15)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": demand * 100},
        },
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280)
    return fig


def _model_agreement_chart(lgb, xgb, cb, stack):
    fig = go.Figure(go.Bar(
        x=["LightGBM", "XGBoost", "CatBoost", "Stack (V8)"],
        y=[lgb, xgb, cb, stack],
        marker_color=["#4fd1c5", "#f6ad55", "#9f7aea", "#4a9eff"],
        text=[f"{v:.3f}" for v in [lgb, xgb, cb, stack]],
        textposition="outside",
        width=0.5,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Model Agreement", yaxis_title="Demand Score",
        yaxis=dict(**PLOTLY_LAYOUT.get("yaxis", {}),
                   gridcolor="rgba(99,179,237,0.08)", range=[0, 1.1]),
        height=260,
    )
    return fig


def render():
    st.markdown("<h1 style='color:#e2e8f0;font-size:2rem;font-weight:800;margin-bottom:4px;'>Live Traffic Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#718096;font-size:14px;margin-bottom:24px;'>Enter road and environment parameters to get an instant demand forecast from the Stacking Ensemble V8.</p>", unsafe_allow_html=True)

    # API Status Banner
    api_ok = _check_api()
    if not api_ok:
        st.warning("FastAPI backend not running. Start it with: `uvicorn backend.main:app --reload --port 8000`")
    else:
        loaded = _model_loaded()
        msg = _model_message()
        if loaded:
            st.success(msg)
        else:
            st.warning(msg)

    st.markdown("---")

    # ── Input Form ────────────────────────────────────────────────────────────
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### Input Parameters")
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            timestamp = c1.text_input("Timestamp (H:MM)", value="08:30",
                                      help="Format: H:MM e.g. 8:30 for 8:30 AM")
            day       = c2.number_input("Day", min_value=1, max_value=60, value=49)

            geohash   = st.text_input("Geohash", value="qp09d9",
                                      help="6-char geohash of the road location")

            c3, c4 = st.columns(2)
            road_type  = c3.selectbox("Road Type", ["Highway", "Street", "Residential"])
            n_lanes    = c4.slider("Number of Lanes", 1, 10, 4)

            c5, c6 = st.columns(2)
            large_veh  = c5.selectbox("Large Vehicles", ["Allowed", "Not Allowed"])
            landmarks  = c6.selectbox("Landmarks Nearby", ["Yes", "No"])

            c7, c8 = st.columns(2)
            weather     = c7.selectbox("Weather", ["Sunny", "Rainy", "Foggy", "Snowy"])
            temperature = c8.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=28.5, step=0.5)

            submitted = st.form_submit_button("Predict Traffic Demand", use_container_width=True, type="primary")

    with col_result:
        st.markdown("### Prediction Result")
        placeholder = st.empty()

        if submitted:
            payload = {
                "timestamp": timestamp, "geohash": geohash, "day": int(day),
                "RoadType": road_type, "NumberofLanes": int(n_lanes),
                "LargeVehicles": large_veh, "Landmarks": landmarks,
                "Weather": weather, "Temperature": float(temperature),
            }
            if not api_ok:
                placeholder.error("Cannot connect to the FastAPI backend. Please start it first.")
            else:
                with st.spinner("Running ensemble prediction..."):
                    result = _do_predict(payload)

                if "error" in result:
                    placeholder.error(f"**Error:** {result['error']}")
                elif result.get("status") == "unavailable":
                    placeholder.warning(f"**{result['message']}**\n\nUpload your .pkl model files to `model_artifacts/` to enable predictions.")
                else:
                    demand = result["demand"]
                    level  = result["traffic_level"]
                    lgb_p  = result["lgb_pred"]
                    xgb_p  = result["xgb_pred"]
                    cb_p   = result["cb_pred"]
                    ci_lo  = result["confidence_low"]
                    ci_hi  = result["confidence_high"]
                    agree  = result["model_agreement"]

                    with placeholder.container():
                        st.plotly_chart(_demand_gauge(demand, level), use_container_width=True)

                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Confidence Low",  f"{ci_lo:.3f}")
                        mc2.metric("Model Agreement", f"{agree:.1%}")
                        mc3.metric("Confidence High", f"{ci_hi:.3f}")

                        st.plotly_chart(
                            _model_agreement_chart(lgb_p, xgb_p, cb_p, demand),
                            use_container_width=True,
                        )
        else:
            placeholder.info("Fill in the form and click **Predict** to get a real-time demand forecast.")

    # ── What-If Simulator ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>What-If Simulator</div>", unsafe_allow_html=True)
    st.caption("Adjust sliders to explore how each factor influences the prediction (uses the last submitted geohash & road type).")

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    sim_hour  = sim_c1.slider("Hour of Day", 0, 23, 8, key="sim_hour")
    sim_temp  = sim_c2.slider("Temperature (°C)", -10, 45, 25, key="sim_temp")
    sim_lanes = sim_c3.slider("Number of Lanes", 1, 8, 3, key="sim_lanes")

    sim_c4, sim_c5 = st.columns(2)
    sim_weather = sim_c4.selectbox("Weather Condition", ["Sunny","Rainy","Foggy","Snowy"], key="sim_weather")
    sim_road    = sim_c5.selectbox("Road Type", ["Highway","Street","Residential"], key="sim_road")

    if st.button("Run Simulation", use_container_width=True) and api_ok:
        sim_payload = {
            "timestamp": f"{sim_hour}:00", "geohash": "qp09d9", "day": 49,
            "RoadType": sim_road, "NumberofLanes": sim_lanes,
            "LargeVehicles": "Allowed", "Landmarks": "Yes",
            "Weather": sim_weather, "Temperature": float(sim_temp),
        }
        with st.spinner("Simulating..."):
            sim_result = _do_predict(sim_payload)
        if "demand" in sim_result:
            d = sim_result["demand"]
            level = sim_result["traffic_level"]
            st.plotly_chart(_demand_gauge(d, level), use_container_width=False)
            st.success(f"Simulated demand: **{d:.3f}** ({level})")
        elif sim_result.get("status") == "unavailable":
            st.warning(sim_result.get("message", "Models not loaded."))
