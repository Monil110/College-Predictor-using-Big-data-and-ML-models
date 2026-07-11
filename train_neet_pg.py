"""
train_neet_pg.py
================
Standalone training script for the NEET PG predictor.
Uses XGBoost multi-class classifier — produces ~8 MB model files
(vs the old RandomForest which produced 5+ GB).

Usage
-----
    python train_neet_pg.py

Environment variable overrides
-------------------------------
    NEET_PG_DATA_PATH  – path to the allotment Excel / CSV  (default: data/neet_pg/allotment.xlsx)
    NEET_PG_MODEL_DIR  – output directory for .pkl artefacts (default: models/neet_pg)

Output artefacts (in MODEL_DIR)
--------------------------------
    neet_pg_model.pkl             – XGBClassifier (multi:softprob, ~8 MB)
    neet_pg_category_encoder.pkl  – LabelEncoder for Allotted Category
    neet_pg_label_encoder.pkl     – LabelEncoder for "college|||course" combined label
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

try:
    import xgboost as xgb
except ImportError:
    print("[ERROR] xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT: Path = Path(__file__).resolve().parent

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH: str = os.environ.get(
    "NEET_PG_DATA_PATH",
    str(REPO_ROOT / "data" / "neet_pg" / "allotment.xlsx"),
)
MODEL_DIR: str = os.environ.get(
    "NEET_PG_MODEL_DIR",
    str(REPO_ROOT / "models" / "neet_pg"),
)

LABEL_SEP        = "|||"
SAFE_THRESHOLD   = 0.01    # ≥1%  → Safe
LIKELY_THRESHOLD = 0.003   # ≥0.3% → Likely

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    logger.info("=" * 60)
    logger.info("  NEET PG Model Training  (XGBoost)")
    logger.info("=" * 60)
    logger.info("Data  : %s", DATA_PATH)
    logger.info("Output: %s", MODEL_DIR)

    # 1 — Check data file
    if not Path(DATA_PATH).exists():
        logger.error("Data file not found: %s", DATA_PATH)
        return 1

    # 2 — Load
    logger.info("Loading data ...")
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    logger.info("  Rows: %d   Cols: %s", len(df), list(df.columns))

    RANK_COL     = "All India Rank"
    COLLEGE_COL  = "Name of the College Allotted."
    COURSE_COL   = "Course Name"
    CATEGORY_COL = "Allotted Category"

    df = df.dropna(subset=[RANK_COL, COLLEGE_COL, COURSE_COL, CATEGORY_COL]).reset_index(drop=True)
    df[RANK_COL] = pd.to_numeric(df[RANK_COL], errors="coerce")
    df = df.dropna(subset=[RANK_COL]).reset_index(drop=True)
    logger.info("  After cleaning: %d rows", len(df))

    # 3 — Features & labels
    X_rank = df[RANK_COL].astype(int).values.reshape(-1, 1)

    categories = df[CATEGORY_COL].astype(str).str.strip().str.upper().values
    cat_enc = LabelEncoder()
    X_cat = cat_enc.fit_transform(categories).reshape(-1, 1)
    logger.info("  Unique categories: %d", len(cat_enc.classes_))

    X = np.hstack([X_rank, X_cat])

    combined_labels = (
        df[COLLEGE_COL].astype(str).str.strip()
        + LABEL_SEP
        + df[COURSE_COL].astype(str).str.strip()
    ).values
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(combined_labels)
    n_classes = len(label_enc.classes_)
    logger.info("  Unique college+course combinations: %d", n_classes)

    # 4 — Train on full dataset
    # (With 787 classes and 3k rows a train/test split always leaves some
    #  classes unseen in the train fold — train on all data for production.)
    logger.info("Training XGBoost (n_estimators=100, max_depth=4) ...")
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.15,
        subsample=0.8,
        colsample_bytree=1.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y, verbose=False)

    train_acc = accuracy_score(y, model.predict(X))
    logger.info("  Train accuracy (top-1): %.1f%%", train_acc * 100)

    # 5 — Save
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(model,     Path(MODEL_DIR) / "neet_pg_model.pkl",            compress=3)
    joblib.dump(cat_enc,   Path(MODEL_DIR) / "neet_pg_category_encoder.pkl", compress=3)
    joblib.dump(label_enc, Path(MODEL_DIR) / "neet_pg_label_encoder.pkl",    compress=3)

    total_mb = 0
    for fname in ["neet_pg_model.pkl", "neet_pg_category_encoder.pkl", "neet_pg_label_encoder.pkl"]:
        mb = (Path(MODEL_DIR) / fname).stat().st_size / 1024 / 1024
        total_mb += mb
        logger.info("  %-40s  %.2f MB", fname, mb)
    logger.info("  %-40s  %.2f MB", "TOTAL", total_mb)

    # 6 — Smoke test
    logger.info("-" * 60)
    logger.info("Smoke-test predictions ...")

    test_cases = [
        (1000, "GM"), (3000, "OPN"), (5000, "2AG"), (500, "NRI"), (2000, "MM"),
    ]
    all_passed = True
    for rank, cat in test_cases:
        cu = cat.strip().upper()
        if cu not in cat_enc.classes_:
            cu = cat_enc.classes_[0]
        ce = int(cat_enc.transform([cu])[0])
        proba = model.predict_proba(np.array([[rank, ce]]))[0]

        safe   = [(label_enc.classes_[i], round(float(p)*100, 2)) for i, p in enumerate(proba) if p >= SAFE_THRESHOLD]
        likely = [(label_enc.classes_[i], round(float(p)*100, 2)) for i, p in enumerate(proba) if LIKELY_THRESHOLD <= p < SAFE_THRESHOLD]
        safe.sort(key=lambda x: x[1], reverse=True)

        total = len(safe) + len(likely)
        ok = total > 0
        if not ok:
            all_passed = False

        logger.info(
            "[%s] rank=%-6d  cat=%-5s  Safe=%-3d  Likely=%-3d",
            "PASS" if ok else "FAIL", rank, cat, len(safe), len(likely),
        )
        if safe:
            top_lbl, top_pct = safe[0]
            college, course = (top_lbl.split(LABEL_SEP, 1) if LABEL_SEP in top_lbl else (top_lbl, "?"))
            logger.info("       Top: %s | %s (%.2f%%)", college.strip()[:55], course.strip()[:30], top_pct)

    logger.info("=" * 60)
    if all_passed:
        logger.info("All smoke-tests PASSED. Model is ready for serving.")
        return 0
    else:
        logger.error("One or more smoke-tests FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
