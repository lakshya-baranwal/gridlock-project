"""
backend/routers/predict.py
===========================
FastAPI router for real-time traffic demand prediction.

Endpoints
---------
POST /predict/
    Accept a PredictionRequest payload and return a stacking ensemble forecast.

GET /predict/status
    Return current model loading status and fold counts.
"""

from fastapi import APIRouter, HTTPException
from backend.schemas.predict import PredictionRequest, PredictionResponse
from backend.core.model_wrapper import predict, MODEL_STATUS

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/", response_model=None)
async def make_prediction(request: PredictionRequest):
    """
    @brief Run the Stacking Ensemble V8 on a single road/time snapshot.

    If models are not loaded (pkl files absent), returns HTTP 503 with a
    descriptive message rather than raising an unhandled exception.

    @param request  PredictionRequest body containing road, time, and
                    environment features.
    @return         PredictionResponse dict with demand score, confidence
                    interval, per-model predictions, and traffic level label.
    @raises HTTPException(503)  If models failed to load.
    """
    if not MODEL_STATUS["loaded"]:
        return {"status": "unavailable", "message": MODEL_STATUS["message"]}
    result = predict(request.model_dump())
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/status")
async def model_status():
    """
    @brief Return the current model registry status.

    @return  Dict with keys: loaded (bool), message (str),
             n_lgb, n_xgb, n_cb, n_meta (fold counts).
    """
    return {
        "loaded":  MODEL_STATUS["loaded"],
        "message": MODEL_STATUS["message"],
        "n_lgb":   len(MODEL_STATUS["lgb_models"]),
        "n_xgb":   len(MODEL_STATUS["xgb_models"]),
        "n_cb":    len(MODEL_STATUS["cb_models"]),
        "n_meta":  len(MODEL_STATUS["meta_models"]),
    }
