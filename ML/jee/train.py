"""
jee_train.py
─────────────────────────────────────────────────────────────────────────────
Stage 3: Train two CatBoost classifiers (IIT and NIT) to predict whether
         a student is eligible for each (institute, program) combination.

Target  : eligible  (1 = student's rank ≤ closing_rank_max, 0 otherwise)
          We generate synthetic negative samples by pairing each feature
          row with a random rank from the range [closing_rank_max+1 .. 500000].

Run locally (after exporting features from HDFS):
    python jee_train.py

Or via spark-submit for distributed pandas-on-spark reads:
    spark-submit /app/jee_train.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
)
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────────────

IIT_PARQUET  = os.path.normpath("data/processed/jee_features/iit")
NIT_PARQUET  = os.path.normpath("data/processed/jee_features/nit")
MODEL_DIR    = os.path.normpath("models/jee")

CATEGORICAL_FEATURES = [
    "institute_short",
    "program_name",
    "degree_short",
    "category",
    "quota",
    "pool",
    "trend_direction",
]

NUMERIC_FEATURES = [
    "closing_rank_max",
    "opening_rank_min",
    "closing_rank_avg",
    "closing_rank_std",
    "rank_spread_avg",
    "rank_pressure",
    "difficulty_pct",
    "years_available",
    "yoy_rank_change",
    "student_rank",       # injected during sample generation
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATBOOST_PARAMS = dict(
    iterations=800,
    learning_rate=0.05,
    depth=7,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    early_stopping_rounds=50,
    verbose=10,
    cat_features=CATEGORICAL_FEATURES,
    auto_class_weights="Balanced",   # handles class imbalance automatically
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_parquet(path: str) -> pd.DataFrame:
    """Read a Spark-written partitioned parquet directory into pandas."""
    import glob
    files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(path, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found at: {path}")
    
    dfs = [pd.read_parquet(f) for f in tqdm(files, desc=f"Loading Parquet from {os.path.basename(path)}")]
    return pd.concat(dfs, ignore_index=True)


def generate_samples(df: pd.DataFrame, neg_ratio: float = 1.5) -> pd.DataFrame:
    """
    For each feature row create:
      - Positive sample : student_rank ~ Uniform(opening_rank_min, closing_rank_max)
      - Negative samples: student_rank ~ Uniform(closing_rank_max+1, 500_000)
    """
    rng = np.random.default_rng(42)
    rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Generating training samples"):
        lo = max(1, int(r["opening_rank_min"]))
        hi = int(r["closing_rank_max"])
        if lo > hi:
            continue

        # Positive
        rank = int(rng.integers(lo, hi + 1))
        rows.append({**r.to_dict(), "student_rank": rank, "eligible": 1})

        # Negatives
        n_neg = max(1, int(neg_ratio))
        upper_limit = max(500_001, hi + 2)
        for _ in range(n_neg):
            neg_rank = int(rng.integers(hi + 1, upper_limit))
            rows.append({**r.to_dict(), "student_rank": neg_rank, "eligible": 0})

    return pd.DataFrame(rows)


def prepare_X_y(df: pd.DataFrame):
    df = df.copy()

    # Fill missing numeric cols with median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Fill missing categoricals
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)

    # Keep only columns that exist
    available = [c for c in ALL_FEATURES if c in df.columns]
    X = df[available]
    y = df["eligible"].astype(int)
    return X, y, available


def train_model(name: str, parquet_path: str) -> CatBoostClassifier:
    print(f"\n{'-'*60}")
    print(f"  Training {name} model")
    print(f"{'-'*60}")

    print("  Loading features ...")
    features_df = load_parquet(parquet_path)
    print(f"  Feature rows: {len(features_df):,}")

    print("  Generating train samples ...")
    sampled = generate_samples(features_df, neg_ratio=1.5)
    print(f"  Total samples: {len(sampled):,}  "
          f"(+ve: {sampled['eligible'].sum():,}  "
          f"-ve: {(sampled['eligible']==0).sum():,})")

    X, y, used_cols = prepare_X_y(sampled)
    cat_cols_in_use = [c for c in CATEGORICAL_FEATURES if c in used_cols]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_cols_in_use)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols_in_use)

    params = {**CATBOOST_PARAMS, "cat_features": cat_cols_in_use}
    model  = CatBoostClassifier(**params)

    print("  Training ...")
    model.fit(train_pool, eval_set=val_pool)

    # Evaluation
    y_pred_proba = model.predict_proba(val_pool)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print(f"\n  Validation AUC : {auc:.4f}")
    print(classification_report(y_val, y_pred, target_names=["Not Eligible", "Eligible"]))

    # Feature importance top-10
    fi = pd.Series(
        model.get_feature_importance(),
        index=model.feature_names_
    ).sort_values(ascending=False)
    print("\n  Top-10 feature importances:")
    print(fi.head(10).to_string())

    # Save
    model_path = os.path.join(MODEL_DIR, f"model_{name.lower()}.cbm")
    model.save_model(model_path)
    print(f"\n  Model saved → {model_path}")

    return model


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if os.path.exists(IIT_PARQUET):
        train_model("IIT", IIT_PARQUET)
    else:
        print(f"Skipping IIT - Path not found: {IIT_PARQUET}")
        
    if os.path.exists(NIT_PARQUET):
        train_model("NIT", NIT_PARQUET)
    else:
        print(f"Skipping NIT - Path not found: {NIT_PARQUET}")
        
    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()