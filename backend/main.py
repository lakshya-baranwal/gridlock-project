"""
backend/main.py
================
FastAPI application entry point for GridLock Intelligence.

Wires together three routers:
  - /predict/*     — Real-time stacking ensemble inference
  - /analytics/*   — Pre-computed dataset analytics
  - /model-info/*  — Model architecture and performance metadata

Usage
-----
Start the development server from the project root:

    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Interactive API docs are available at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import predict, analytics, model_info

app = FastAPI(
    title="GridLock Intelligence API",
    description=(
        "Traffic Demand Prediction API powered by a Stacking Ensemble of "
        "LightGBM, XGBoost, and CatBoost with a Feature-Aware LightGBM meta-learner. "
        "Achieves R² = 0.9561 (competition score: 95.61/100) on 5-fold cross-validation."
    ),
    version="8.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(analytics.router)
app.include_router(model_info.router)


@app.get("/health", tags=["Health"])
async def health():
    """
    @brief Liveness probe endpoint.

    @return  Dict confirming the service is up and its version string.
    """
    return {"status": "ok", "service": "GridLock Intelligence API v8"}


@app.get("/", tags=["Root"])
async def root():
    """
    @brief Root endpoint — returns a summary of available API routes.

    @return  Dict with service name, docs URL, and available endpoint prefixes.
    """
    return {
        "message": "GridLock Intelligence API",
        "docs": "/docs",
        "endpoints": ["/predict", "/analytics/*", "/model-info/*"],
    }
