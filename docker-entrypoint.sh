#!/bin/bash
# docker-entrypoint.sh
# Starts FastAPI and Streamlit inside the HF Spaces container.

set -e

echo "============================================="
echo "  GridLock Intelligence — HF Spaces Start"
echo "============================================="

# Download models at runtime if not already present (fallback)
if [ ! -f "model_artifacts/lgb_fold_1.pkl" ]; then
    echo "[1/3] Downloading model artifacts from HF Hub..."
    python scripts/download_models.py
else
    echo "[1/3] Model artifacts already present."
fi

# Start FastAPI in the background on port 8000
echo "[2/3] Starting FastAPI on port 8000..."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for FastAPI to be ready (up to 60s — model loading takes time)
echo "      Waiting for models to load..."
for i in $(seq 1 60); do
    sleep 1
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "      FastAPI ready."
        break
    fi
    printf "."
done
echo ""

# Start Streamlit on port 7860 (HF Spaces public port)
echo "[3/3] Starting Streamlit on port 7860..."
streamlit run streamlit_app/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false

# Cleanup
kill $FASTAPI_PID 2>/dev/null
