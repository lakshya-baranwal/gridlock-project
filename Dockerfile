# GridLock Intelligence — Docker image for Hugging Face Spaces
#
# HF Spaces constraint: the public-facing port MUST be 7860.
# We run Streamlit on 7860 (public) and FastAPI on 8000 (internal).
#
# Build locally:
#   docker build -t gridlock .
#   docker run -p 7860:7860 -e HF_MODEL_REPO=<your-hf-repo> gridlock

FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (install before copying code for better layer caching) ───────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir huggingface_hub

# ── Copy project source ──────────────────────────────────────────────────────
COPY backend/        ./backend/
COPY streamlit_app/  ./streamlit_app/
COPY scripts/        ./scripts/
COPY model_artifacts/analytics.json ./model_artifacts/analytics.json

# ── Download model artifacts from HF Hub at build time (if repo is set) ─────
# Set HF_MODEL_REPO as a build arg (e.g. "yourusername/gridlock-models")
ARG HF_MODEL_REPO=""
ENV HF_MODEL_REPO=${HF_MODEL_REPO}

RUN if [ -n "$HF_MODEL_REPO" ]; then \
        python scripts/download_models.py; \
    fi

# ── HF Spaces: expose port 7860 ──────────────────────────────────────────────
EXPOSE 7860 8000

# ── Startup script ───────────────────────────────────────────────────────────
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

CMD ["/docker-entrypoint.sh"]
