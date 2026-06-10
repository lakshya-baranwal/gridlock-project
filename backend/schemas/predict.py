"""
backend/schemas/predict.py
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class PredictionRequest(BaseModel):
    timestamp:     str   = Field("08:00", example="08:30",  description="HH:MM format")
    geohash:       str   = Field("qp09d9", example="qp09d9")
    day:           int   = Field(49, example=49)
    RoadType:      Optional[Literal["Residential", "Street", "Highway"]] = "Highway"
    NumberofLanes: int   = Field(4, ge=1, le=10)
    LargeVehicles: Optional[Literal["Allowed", "Not Allowed"]] = "Allowed"
    Landmarks:     Optional[Literal["Yes", "No"]] = "Yes"
    Weather:       Optional[Literal["Sunny", "Rainy", "Foggy", "Snowy"]] = "Sunny"
    Temperature:   Optional[float] = Field(28.0, description="Celsius")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "08:30",
                "geohash": "qp09d9",
                "day": 49,
                "RoadType": "Highway",
                "NumberofLanes": 5,
                "LargeVehicles": "Allowed",
                "Landmarks": "No",
                "Weather": "Sunny",
                "Temperature": 28.5,
            }
        }


class PredictionResponse(BaseModel):
    demand:          float
    confidence_low:  float
    confidence_high: float
    lgb_pred:        float
    xgb_pred:        float
    cb_pred:         float
    traffic_level:   str
    model_agreement: float
