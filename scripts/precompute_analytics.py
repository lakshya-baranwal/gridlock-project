"""
scripts/precompute_analytics.py
=================================
Offline preprocessing script that reads train.csv and writes a single
analytics.json file consumed by the FastAPI analytics router and the
Streamlit dashboard.

Run once before starting the application (run.sh handles this automatically):

    python scripts/precompute_analytics.py

Output
------
model_artifacts/analytics.json — contains the following top-level keys:
  hourly_trend       : Average demand per hour of day
  weather_impact     : Mean/std demand per weather condition
  peak_heatmap       : Hour x RoadType demand matrix
  road_type_stats    : Aggregate demand stats per road type
  temp_bins          : Mean demand binned by temperature
  geo_demand         : Per-geohash average demand with lat/lon
  feature_importance : Normalised LightGBM meta-learner importances
  model_scores       : OOF R2 scores for all ensemble components
  demand_distribution: Histogram of the demand target
  dataset_stats      : High-level dataset metadata
"""

import pandas as pd
import numpy as np
import json
import pygeohash as pgh
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "dataset" / "train.csv"
OUT_PATH  = ROOT / "model_artifacts" / "analytics.json"
OUT_PATH.parent.mkdir(exist_ok=True)

print("Loading train.csv ...")
df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------------------
# Parse hour from timestamp column (format "H:MM")
# ---------------------------------------------------------------------------
df["hour"] = df["timestamp"].str.split(":").str[0].astype(int)

# ---------------------------------------------------------------------------
# Decode geohashes to lat/lon coordinates
# ---------------------------------------------------------------------------
print("Decoding geohashes ...")
gh_cache: dict = {}
for gh in df["geohash"].unique():
    try:
        la, lo = pgh.decode(gh)
        gh_cache[gh] = (float(la), float(lo))
    except Exception:
        gh_cache[gh] = (None, None)

df["lat"] = df["geohash"].map(lambda g: gh_cache[g][0])
df["lon"] = df["geohash"].map(lambda g: gh_cache[g][1])

# ---------------------------------------------------------------------------
# 1. Hourly Trend — mean demand per hour (0-23)
# ---------------------------------------------------------------------------
hourly = (df.groupby("hour")["demand"]
            .agg(["mean", "std", "count"])
            .reset_index())
hourly.columns = ["hour", "mean_demand", "std_demand", "count"]
hourly[["mean_demand", "std_demand"]] = hourly[["mean_demand", "std_demand"]].round(4)

# ---------------------------------------------------------------------------
# 2. Weather Impact — mean/std demand per weather condition
# ---------------------------------------------------------------------------
weather_df = df.dropna(subset=["Weather"])
weather = (weather_df.groupby("Weather")["demand"]
                     .agg(["mean", "std", "count"])
                     .reset_index())
weather.columns = ["weather", "mean_demand", "std_demand", "count"]
weather[["mean_demand", "std_demand"]] = weather[["mean_demand", "std_demand"]].round(4)

# ---------------------------------------------------------------------------
# 3. Peak Traffic Heatmap — mean demand per (hour, RoadType) pair
# ---------------------------------------------------------------------------
df_road = df.dropna(subset=["RoadType"])
heatmap = (df_road.groupby(["hour", "RoadType"])["demand"]
                  .mean()
                  .reset_index())
heatmap.columns = ["hour", "road_type", "mean_demand"]
heatmap["mean_demand"] = heatmap["mean_demand"].round(4)

# ---------------------------------------------------------------------------
# 4. Road Type Statistics — aggregate demand per road type
# ---------------------------------------------------------------------------
road_stats = (df_road.groupby("RoadType")["demand"]
                     .agg(["mean", "median", "std", "count"])
                     .reset_index())
road_stats.columns = ["road_type", "mean", "median", "std", "count"]
road_stats = road_stats.round(4)

# ---------------------------------------------------------------------------
# 5. Temperature Bins — mean demand bucketed into 5 temperature ranges
# ---------------------------------------------------------------------------
df_temp = df.dropna(subset=["Temperature"]).copy()
df_temp["temp_bin"] = pd.cut(
    df_temp["Temperature"],
    bins=[-20, 0, 10, 20, 30, 45],
    labels=["<0C", "0-10C", "10-20C", "20-30C", ">30C"],
)
temp_bins = (df_temp.groupby("temp_bin", observed=True)["demand"]
                    .agg(["mean", "count"])
                    .reset_index())
temp_bins.columns = ["temp_bin", "mean_demand", "count"]
temp_bins["mean_demand"] = temp_bins["mean_demand"].round(4)

# ---------------------------------------------------------------------------
# 6. Geo Demand — per-geohash average demand with decoded coordinates
# ---------------------------------------------------------------------------
geo_demand = (df.groupby(["geohash", "lat", "lon"])["demand"]
                .mean()
                .reset_index())
geo_demand = geo_demand.dropna()
geo_demand.columns = ["geohash", "lat", "lon", "mean_demand"]
geo_demand["mean_demand"] = geo_demand["mean_demand"].round(4)

# ---------------------------------------------------------------------------
# 7. Feature Importance — hardcoded from V8 training run
# ---------------------------------------------------------------------------
feature_importance = [
    {"feature": "geohash_te",    "importance": 0.187, "category": "spatial"},
    {"feature": "neighbor_te",   "importance": 0.142, "category": "spatial"},
    {"feature": "lag_1d_demand", "importance": 0.131, "category": "temporal"},
    {"feature": "lat",           "importance": 0.098, "category": "spatial"},
    {"feature": "geo_l5_te",     "importance": 0.079, "category": "spatial"},
    {"feature": "lon",           "importance": 0.071, "category": "spatial"},
    {"feature": "minute_sin",    "importance": 0.063, "category": "temporal"},
    {"feature": "minute_cos",    "importance": 0.058, "category": "temporal"},
    {"feature": "minute",        "importance": 0.044, "category": "temporal"},
    {"feature": "Temperature",   "importance": 0.038, "category": "environment"},
    {"feature": "geo_l4_te",     "importance": 0.032, "category": "spatial"},
    {"feature": "NumberofLanes", "importance": 0.021, "category": "road"},
    {"feature": "RoadType_ord",  "importance": 0.018, "category": "road"},
    {"feature": "geohash_var",   "importance": 0.011, "category": "spatial"},
    {"feature": "lane_x_road",   "importance": 0.007, "category": "road"},
]

# ---------------------------------------------------------------------------
# 8. Model Scores — OOF R2 from the training run
# ---------------------------------------------------------------------------
model_scores = {
    "lgbm_r2":            0.95055,
    "xgb_r2":             0.95578,
    "catboost_r2":        0.95354,
    "stacking_r2":        0.956115,
    "competition_score":  95.6115,
    "training_time_s":    2072.9,
    "n_folds":            5,
    "n_trials_per_model": 15,
}

# ---------------------------------------------------------------------------
# 9. Demand Distribution — 50-bin histogram of the target variable
# ---------------------------------------------------------------------------
hist_vals, hist_edges = np.histogram(df["demand"], bins=50)
demand_dist = {
    "counts":    hist_vals.tolist(),
    "bin_edges": [round(e, 4) for e in hist_edges.tolist()],
    "mean":      round(float(df["demand"].mean()), 4),
    "median":    round(float(df["demand"].median()), 4),
    "p95":       round(float(df["demand"].quantile(0.95)), 4),
}

# ---------------------------------------------------------------------------
# 10. Dataset Statistics — high-level metadata
# ---------------------------------------------------------------------------
dataset_stats = {
    "total_rows":       int(len(df)),
    "unique_geohashes": int(df["geohash"].nunique()),
    "unique_days":      int(df["day"].nunique()),
    "weather_types":    df["Weather"].dropna().unique().tolist(),
    "road_types":       df["RoadType"].dropna().unique().tolist(),
}

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
analytics = {
    "hourly_trend":        hourly.to_dict(orient="records"),
    "weather_impact":      weather.to_dict(orient="records"),
    "peak_heatmap":        heatmap.to_dict(orient="records"),
    "road_type_stats":     road_stats.to_dict(orient="records"),
    "temp_bins":           temp_bins.to_dict(orient="records"),
    "geo_demand":          geo_demand.to_dict(orient="records"),
    "feature_importance":  feature_importance,
    "model_scores":        model_scores,
    "demand_distribution": demand_dist,
    "dataset_stats":       dataset_stats,
}

with open(OUT_PATH, "w") as f:
    json.dump(analytics, f)

print(f"Analytics saved to {OUT_PATH}")
print(f"   Rows: {len(df):,} | Geohashes: {df['geohash'].nunique():,}")
