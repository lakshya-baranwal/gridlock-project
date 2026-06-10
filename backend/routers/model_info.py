"""
backend/routers/model_info.py
==============================
FastAPI router exposing model metadata, performance scores, and
architecture details for the Stacking Ensemble V8.

Endpoints
---------
GET /model-info/feature-importance  — Feature importance from LightGBM meta-learner
GET /model-info/scores              — OOF R2 scores for all models
GET /model-info/architecture        — Full pipeline architecture descriptor
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from backend.core.model_wrapper import MODEL_STATUS

router = APIRouter(prefix="/model-info", tags=["Model Info"])

ROOT = Path(__file__).parent.parent.parent
ANALYTICS_PATH = ROOT / "model_artifacts" / "analytics.json"

_cache: dict | None = None


def _load() -> dict | None:
    """
    @brief Load analytics.json into memory (cached after first read).

    @return  Parsed analytics dict, or None if the file is absent.
    """
    global _cache
    if _cache is None:
        if not ANALYTICS_PATH.exists():
            return None
        with open(ANALYTICS_PATH) as f:
            _cache = json.load(f)
    return _cache


@router.get("/feature-importance")
async def feature_importance():
    """
    @brief Return ranked feature importances from the LightGBM meta-learner.

    Importance values are normalised to sum to 1.0.

    @return  List of dicts with keys: feature, importance, category.
    @raises HTTPException(503)  If analytics.json has not been generated.
    """
    data = _load()
    if data is None:
        raise HTTPException(503, "Analytics not computed yet. Run scripts/precompute_analytics.py")
    return data["feature_importance"]


@router.get("/scores")
async def model_scores():
    """
    @brief Return OOF R2 scores for all models plus the current load status.

    @return  Dict with keys: lgbm_r2, xgb_r2, catboost_r2, stacking_r2,
             competition_score, training_time_s, n_folds, n_trials_per_model,
             model_status (live status string from model registry).
    @raises HTTPException(503)  If analytics.json has not been generated.
    """
    data = _load()
    if data is None:
        raise HTTPException(503, "Analytics not computed yet.")
    scores = data["model_scores"]
    scores["model_status"] = MODEL_STATUS["message"]
    return scores


@router.get("/architecture")
async def architecture():
    """
    @brief Return a structured description of the full stacking pipeline.

    Includes base model configurations, meta-learner details, Optuna tuning
    settings, and all feature engineering steps applied during training.

    @return  Dict describing the Stacking Ensemble V8 architecture.
    """
    return {
        "name": "Stacking Ensemble V8",
        "base_models": [
            {"name": "LightGBM",  "cv_r2": 0.95055, "features": 29, "folds": 5},
            {"name": "XGBoost",   "cv_r2": 0.95578, "features": 29, "folds": 5},
            {"name": "CatBoost",  "cv_r2": 0.95354, "features": 29, "folds": 5},
        ],
        "meta_learner": {
            "name": "LightGBM (Feature-Aware)",
            "stacking_r2": 0.956115,
            "input_features": 32,
            "description": "29 base features + 3 OOF predictions (LGB, XGB, CB)",
        },
        "hyperparameter_tuning": {
            "library": "Optuna",
            "trials_per_model": 15,
            "sampler": "TPE (Tree-structured Parzen Estimator)",
            "cv_folds_during_tuning": 3,
        },
        "feature_engineering": [
            "Cyclic time encoding: sin/cos of minute-of-day",
            "Geohash spatial decoding to lat/lon",
            "Geohash prefix levels L4 and L5",
            "8-directional neighbor target encoding",
            "Smooth target encoding (CV-safe, 5-fold, smoothing=20)",
            "Lag-1D demand feature from day 48",
            "Weather one-hot encoding (Sunny/Rainy/Foggy/Snowy)",
            "Road type ordinal encoding and lane interaction",
            "Missing value indicators for RoadType, Temperature, Weather",
            "Geohash demand variance encoding",
        ],
    }
