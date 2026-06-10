#!/bin/bash
# GridLock Intelligence — Start Script (with virtualenv)
# Usage: bash run.sh

set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/venv"

echo "========================================================"
echo "  GridLock Intelligence  —  Stacking Ensemble V8"
echo "========================================================"

# ── Step 1: Create virtual environment if needed ─────────────────────
echo ""
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtual environment ..."
    python3 -m venv "$VENV_DIR"
    echo "      [OK] venv created at $VENV_DIR"
else
    echo "[1/4] Virtual environment found — skipping creation."
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "      [OK] venv activated: $(which python3)"

# ── Step 2: Install dependencies ─────────────────────────────────────
echo ""
echo "[2/4] Installing / verifying dependencies ..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "      [OK] Dependencies ready."

# ── Step 3: Precompute analytics if not done ──────────────────────────
echo ""
if [ ! -f "model_artifacts/analytics.json" ]; then
    echo "[3/4] Precomputing analytics from train.csv ..."
    python3 scripts/precompute_analytics.py
    echo "      [OK] analytics.json generated."
else
    echo "[3/4] analytics.json found — skipping precompute."
fi

# ── Step 4: Start services ────────────────────────────────────────────
echo ""
echo "[4/4] Starting services ..."
echo "  FastAPI   → http://localhost:8000"
echo "  API Docs  → http://localhost:8000/docs"
echo "  Streamlit → http://localhost:8501"
echo ""
echo "  Press Ctrl+C to stop all services."
echo ""

# Start FastAPI in background (models load here — may take ~30s)
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Cleanup trap
trap "echo ''; echo 'Shutting down...'; kill $FASTAPI_PID 2>/dev/null; exit 0" INT TERM

# Wait for FastAPI to be ready
echo "  Waiting for FastAPI to load models (~15-30s for 22 pkl files) ..."
for i in $(seq 1 30); do
    sleep 1
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "  [OK] FastAPI is ready!"
        break
    fi
    printf "."
done
echo ""

# Start Streamlit (foreground)
streamlit run streamlit_app/app.py --server.port 8501 --server.headless false

# Kill FastAPI when Streamlit exits
kill $FASTAPI_PID 2>/dev/null
