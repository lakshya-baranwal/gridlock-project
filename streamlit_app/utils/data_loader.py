"""
streamlit_app/utils/data_loader.py
====================================
Loads analytics.json and exposes typed accessors.
"""
import json
import pandas as pd
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
ANALYTICS_PATH = ROOT / "model_artifacts" / "analytics.json"


@st.cache_data(ttl=3600)
def load_analytics() -> dict:
    if not ANALYTICS_PATH.exists():
        return {}
    with open(ANALYTICS_PATH) as f:
        return json.load(f)


def get_hourly_trend() -> pd.DataFrame:
    data = load_analytics()
    return pd.DataFrame(data.get("hourly_trend", []))


def get_weather_impact() -> pd.DataFrame:
    data = load_analytics()
    return pd.DataFrame(data.get("weather_impact", []))


def get_peak_heatmap() -> pd.DataFrame:
    data = load_analytics()
    return pd.DataFrame(data.get("peak_heatmap", []))


def get_geo_demand() -> pd.DataFrame:
    data = load_analytics()
    return pd.DataFrame(data.get("geo_demand", []))


def get_feature_importance() -> pd.DataFrame:
    data = load_analytics()
    return pd.DataFrame(data.get("feature_importance", []))


def get_model_scores() -> dict:
    data = load_analytics()
    return data.get("model_scores", {})


def get_road_type_stats() -> pd.DataFrame:
    data = load_analytics()
    return pd.DataFrame(data.get("road_type_stats", []))


def get_temp_bins() -> pd.DataFrame:
    data = load_analytics()
    return pd.DataFrame(data.get("temp_bins", []))


def get_demand_distribution() -> dict:
    data = load_analytics()
    return data.get("demand_distribution", {})


def get_dataset_stats() -> dict:
    data = load_analytics()
    return data.get("dataset_stats", {})


def get_architecture() -> dict:
    return {
        "base_models": [
            {"name": "LightGBM",  "cv_r2": 0.95055, "features": 28, "folds": 5},
            {"name": "XGBoost",   "cv_r2": 0.95578, "features": 35, "folds": 5},
            {"name": "CatBoost",  "cv_r2": 0.95354, "features": 31, "folds": 5},
        ],
        "meta_learner": {"name": "LightGBM (Feature-Aware)", "stacking_r2": 0.956115},
        "feature_engineering": [
            "Cyclic time encoding (sin/cos)",
            "Geohash spatial decoding (lat/lon)",
            "Geohash prefix levels L4, L5",
            "Neighbor target encoding (8-directional)",
            "Smooth target encoding (CV-safe, 5-fold)",
            "Lag-1D demand feature",
            "Weather one-hot encoding",
            "Road type ordinal + lane interaction",
        ],
    }
