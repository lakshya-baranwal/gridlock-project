"""
backend/core/model_wrapper.py
==============================
Loads the Stacking Ensemble V8 model artifacts from disk and serves
real-time traffic demand predictions.

The inference pipeline mirrors the training pipeline in stack_model_v8.py:
  1. Feature engineering on a single input row
  2. Forward pass through 5-fold base models (LGB, XGB, CatBoost)
  3. Concatenation of base predictions with original features
  4. Forward pass through 5-fold meta LightGBM models
  5. Clipping final prediction to [0, 1]

Feature layout (29 base features + 3 OOF preds = 32 meta features):
  Spatial     : lat, lon, geohash_te, geo_l4_te, geo_l5_te, neighbor_te
  Road        : RoadType_ord, NumberofLanes, IsHighVolumeLane, lane_x_road,
                LargeVehicles_bin, Landmarks_bin, road_lanes_key_te
  Temporal    : minute, time_slot, minute_sin, minute_cos,
                is_peak_am, is_night, is_evening
  Environment : Temperature, weather_Sunny, weather_Rainy, weather_Foggy, weather_Snowy
  Indicators  : RoadType_missing, Temp_missing, Weather_missing
  Advanced    : geohash_var, lag_1d_demand
"""

import pickle
import numpy as np
import pygeohash as pgh
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT / "model_artifacts"

# ---------------------------------------------------------------------------
# Global model registry — populated once at import time by _try_load()
# ---------------------------------------------------------------------------
MODEL_STATUS = {
    "lgb_models":  [],
    "xgb_models":  [],
    "cb_models":   [],
    "meta_models": [],
    "ohe":         None,
    "loaded":      False,
    "message":     "Models not yet loaded.",
}


def _load_pkl(path: Path):
    """
    @brief Deserialise a pickle file from disk.

    @param path  Absolute path to the .pkl file.
    @return      The unpickled Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def _try_load() -> None:
    """
    @brief Scan model_artifacts/ and load all fold-level model files.

    Expects the following naming convention (produced by stack_model_v8.py):
      - lgb_fold_{1..5}.pkl
      - xgb_fold_{1..5}.pkl
      - cb_fold_{1..5}.pkl
      - meta_lgb_fold_{1..5}.pkl
      - ohe_encoder.pkl   (optional, not used at inference)

    Updates MODEL_STATUS in-place.  Does nothing if files are missing.
    """
    lgb_files  = sorted(MODEL_DIR.glob("lgb_fold_*.pkl"))
    xgb_files  = sorted(MODEL_DIR.glob("xgb_fold_*.pkl"))
    cb_files   = sorted(MODEL_DIR.glob("cb_fold_*.pkl"))
    meta_files = sorted(MODEL_DIR.glob("meta_lgb_fold_*.pkl"))
    ohe_file   = MODEL_DIR / "ohe_encoder.pkl"

    if not lgb_files or not meta_files:
        MODEL_STATUS["message"] = (
            "No model .pkl files found in model_artifacts/. "
            "Upload lgb_fold_*.pkl, xgb_fold_*.pkl, cb_fold_*.pkl, "
            "meta_lgb_fold_*.pkl, and ohe_encoder.pkl to enable predictions."
        )
        return

    MODEL_STATUS["lgb_models"]  = [_load_pkl(f) for f in lgb_files]
    MODEL_STATUS["xgb_models"]  = [_load_pkl(f) for f in xgb_files]
    MODEL_STATUS["cb_models"]   = [_load_pkl(f) for f in cb_files]
    MODEL_STATUS["meta_models"] = [_load_pkl(f) for f in meta_files]
    if ohe_file.exists():
        MODEL_STATUS["ohe"] = _load_pkl(ohe_file)

    MODEL_STATUS["loaded"]  = True
    MODEL_STATUS["message"] = (
        f"Loaded {len(lgb_files)} LGB + {len(xgb_files)} XGB + "
        f"{len(cb_files)} CB + {len(meta_files)} Meta folds."
    )
    print(MODEL_STATUS["message"])


# Attempt load at import time
_try_load()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROAD_ORDER = {"Residential": 0, "Street": 1, "Highway": 2}

# Must exactly match FEATURE_COLS in stack_model_v8.py (29 features)
FEATURE_COLS = [
    # Spatial (6)
    "lat", "lon", "geohash_te", "geo_l4_te", "geo_l5_te", "neighbor_te",
    # Road properties (7)
    "RoadType_ord", "NumberofLanes", "IsHighVolumeLane", "lane_x_road",
    "LargeVehicles_bin", "Landmarks_bin", "road_lanes_key_te",
    # Time properties (7)
    "minute", "time_slot",
    "minute_sin", "minute_cos",
    "is_peak_am", "is_night", "is_evening",
    # Environment (5)
    "Temperature", "weather_Sunny", "weather_Rainy", "weather_Foggy", "weather_Snowy",
    # Missing indicators (3)
    "RoadType_missing", "Temp_missing", "Weather_missing",
    # Advanced targets (2) — fallback values used at inference time
    "geohash_var", "lag_1d_demand",
]
# Meta input dimensionality: 29 base features + 3 OOF preds = 32


def _engineer(row: dict) -> dict:
    """
    @brief Reproduce the feature engineering from stack_model_v8.py for one row.

    Target-encoded features (geohash_te, geo_l4_te, etc.) cannot be computed
    at inference without the full training fold statistics, so a neutral
    fallback value of 0.05 (near the global mean) is used instead.

    @param row  Dict with keys: timestamp, geohash, RoadType, NumberofLanes,
                LargeVehicles, Landmarks, Weather, Temperature.
    @return     Dict mapping each feature name in FEATURE_COLS to its value.
    """
    ts = str(row.get("timestamp", "0:0"))
    parts = ts.split(":")
    hour_part   = int(parts[0])
    minute_part = int(parts[1]) if len(parts) > 1 else 0
    minute = hour_part * 60 + minute_part

    gh = row.get("geohash", "")
    try:
        la, lo = pgh.decode(gh)
        lat, lon = float(la), float(lo)
    except Exception:
        lat, lon = 0.0, 0.0

    road_type   = row.get("RoadType", "Residential") or "Residential"
    weather     = row.get("Weather", "Sunny") or "Sunny"
    n_lanes     = int(row.get("NumberofLanes", 2))
    large_veh   = 1 if row.get("LargeVehicles") == "Allowed" else 0
    landmarks   = 1 if row.get("Landmarks") == "Yes" else 0
    temperature = float(row.get("Temperature", 20.0) or 20.0)
    road_ord    = ROAD_ORDER.get(road_type, 0)
    time_slot   = minute // 15

    return {
        # Spatial
        "lat":               lat,
        "lon":               lon,
        "geohash_te":        0.05,
        "geo_l4_te":         0.05,
        "geo_l5_te":         0.05,
        "neighbor_te":       0.05,
        # Road
        "RoadType_ord":      road_ord,
        "NumberofLanes":     n_lanes,
        "IsHighVolumeLane":  1 if n_lanes >= 4 else 0,
        "lane_x_road":       n_lanes * (road_ord + 1),
        "LargeVehicles_bin": large_veh,
        "Landmarks_bin":     landmarks,
        "road_lanes_key_te": 0.05,
        # Temporal
        "minute":            minute,
        "time_slot":         time_slot,
        "minute_sin":        np.sin(2 * np.pi * minute / 1440),
        "minute_cos":        np.cos(2 * np.pi * minute / 1440),
        "is_peak_am":        1 if 600 <= minute <= 839 else 0,   # 10:00-13:59
        "is_night":          1 if minute <= 359 else 0,           # 00:00-05:59
        "is_evening":        1 if 1020 <= minute <= 1319 else 0,  # 17:00-21:59
        # Environment
        "Temperature":       temperature,
        "weather_Sunny":     1 if weather == "Sunny" else 0,
        "weather_Rainy":     1 if weather == "Rainy" else 0,
        "weather_Foggy":     1 if weather == "Foggy" else 0,
        "weather_Snowy":     1 if weather == "Snowy" else 0,
        # Missing indicators (all present at inference)
        "RoadType_missing":  0,
        "Temp_missing":      0,
        "Weather_missing":   0,
        # Advanced targets — fallback
        "geohash_var":       0.05,
        "lag_1d_demand":     0.05,
    }


def predict(row: dict) -> dict:
    """
    @brief Run the full stacking ensemble inference for a single input row.

    Pipeline:
      1. Engineer 29 features from raw input fields.
      2. Average predictions across 5-fold LGB, XGB, and CatBoost models.
      3. Build meta-feature vector: [29 base features | lgb_pred | xgb_pred | cb_pred].
      4. Average predictions across 5-fold meta LightGBM models.
      5. Clip result to [0, 1] and attach confidence interval + level label.

    @param row  Raw input dict (same schema as PredictionRequest).
    @return     Dict with keys: demand, confidence_low, confidence_high,
                lgb_pred, xgb_pred, cb_pred, traffic_level, model_agreement.
                Returns {"error": <message>} if models are not loaded.
    """
    if not MODEL_STATUS["loaded"]:
        return {"error": MODEL_STATUS["message"]}

    feats  = _engineer(row)
    x_base = np.array([[feats[c] for c in FEATURE_COLS]])   # shape (1, 29)

    # Base model predictions
    lgb_preds = float(np.mean([m.predict(x_base)[0] for m in MODEL_STATUS["lgb_models"]]))
    xgb_preds = (float(np.mean([m.predict(x_base)[0] for m in MODEL_STATUS["xgb_models"]]))
                 if MODEL_STATUS["xgb_models"] else lgb_preds)
    cb_preds  = (float(np.mean([m.predict(x_base)[0] for m in MODEL_STATUS["cb_models"]]))
                 if MODEL_STATUS["cb_models"] else lgb_preds)

    pred_std = float(np.std([lgb_preds, xgb_preds, cb_preds]))

    # Meta-learner input: 29 base features + 3 OOF preds = 32 total
    # Matches: X_train_meta = np.column_stack([X_train, oof_lgb, oof_xgb, oof_cb])
    x_meta = np.column_stack([x_base, [[lgb_preds, xgb_preds, cb_preds]]])  # (1, 32)

    stack_pred = float(np.mean([m.predict(x_meta)[0] for m in MODEL_STATUS["meta_models"]]))
    stack_pred = float(np.clip(stack_pred, 0, 1))

    # Human-readable traffic level
    if stack_pred >= 0.6:
        level = "Very High"
    elif stack_pred >= 0.4:
        level = "High"
    elif stack_pred >= 0.2:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "demand":           round(stack_pred, 4),
        "confidence_low":   round(float(np.clip(stack_pred - pred_std, 0, 1)), 4),
        "confidence_high":  round(float(np.clip(stack_pred + pred_std, 0, 1)), 4),
        "lgb_pred":         round(float(np.clip(lgb_preds, 0, 1)), 4),
        "xgb_pred":         round(float(np.clip(xgb_preds, 0, 1)), 4),
        "cb_pred":          round(float(np.clip(cb_preds,  0, 1)), 4),
        "traffic_level":    level,
        "model_agreement":  round(max(0.0, 1 - pred_std * 5), 3),
    }
