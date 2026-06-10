"""
scripts/download_models.py
===========================
Downloads trained model artifacts from a Hugging Face Hub repository.

Used by the Docker container at startup to fetch .pkl files that are
too large to commit to the Spaces git repository directly (XGBoost
folds are 33-42 MB each; total ~280 MB).

Usage
-----
Set the HF_MODEL_REPO environment variable to your model repository:

    export HF_MODEL_REPO="yourusername/gridlock-models"
    python scripts/download_models.py

The script is idempotent — it skips files that already exist locally.

Environment variables
---------------------
HF_MODEL_REPO   (required) Hugging Face repo id, e.g. "alice/gridlock-models"
HF_TOKEN        (optional) Read token for private repositories
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ID   = os.environ.get("HF_MODEL_REPO", "")
HF_TOKEN  = os.environ.get("HF_TOKEN", None)       # set for private repos

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model_artifacts"
MODEL_DIR.mkdir(exist_ok=True)

# Files to download from the HF repo
MODEL_FILES = [
    "lgb_fold_1.pkl",  "lgb_fold_2.pkl",  "lgb_fold_3.pkl",
    "lgb_fold_4.pkl",  "lgb_fold_5.pkl",
    "xgb_fold_1.pkl",  "xgb_fold_2.pkl",  "xgb_fold_3.pkl",
    "xgb_fold_4.pkl",  "xgb_fold_5.pkl",
    "cb_fold_1.pkl",   "cb_fold_2.pkl",   "cb_fold_3.pkl",
    "cb_fold_4.pkl",   "cb_fold_5.pkl",
    "meta_lgb_fold_1.pkl", "meta_lgb_fold_2.pkl", "meta_lgb_fold_3.pkl",
    "meta_lgb_fold_4.pkl", "meta_lgb_fold_5.pkl",
    "ohe_encoder.pkl",
]


def download_models() -> None:
    """
    @brief Download all model .pkl files from the configured HF Hub repository.

    Skips files that already exist in model_artifacts/.
    Exits with code 1 if HF_MODEL_REPO is not set.
    """
    if not REPO_ID:
        print("ERROR: Set the HF_MODEL_REPO environment variable first.")
        print("  Example: export HF_MODEL_REPO='yourusername/gridlock-models'")
        sys.exit(1)

    print(f"Downloading models from: https://huggingface.co/{REPO_ID}")

    downloaded = 0
    skipped    = 0

    for filename in MODEL_FILES:
        dest = MODEL_DIR / filename
        if dest.exists():
            print(f"  [skip] {filename}")
            skipped += 1
            continue

        print(f"  [download] {filename} ...", end="", flush=True)
        try:
            local_path = hf_hub_download(
                repo_id   = REPO_ID,
                filename  = filename,
                token     = HF_TOKEN,
                local_dir = str(MODEL_DIR),
            )
            size_mb = Path(local_path).stat().st_size / 1_048_576
            print(f" {size_mb:.1f} MB")
            downloaded += 1
        except Exception as exc:
            print(f"\n  ERROR: Could not download {filename}: {exc}")
            sys.exit(1)

    print(f"\nDone. Downloaded: {downloaded}  |  Already present: {skipped}")


if __name__ == "__main__":
    download_models()
