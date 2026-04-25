"""
jee_predict.py
─────────────────────────────────────────────────────────────────────────────
Stage 4: Given a student's rank + filters, return eligible colleges ranked
         by probability.

Usage:
    python jee_predict.py \
        --model_type iit \
        --rank 5000 \
        --quota AI \
        --pool Gender-Neutral \
        --category GEN \
        --top 20
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import glob
import os

import pandas as pd
from catboost import CatBoostClassifier, Pool

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_DIR   = "/app/models"
IIT_PARQUET = "/app/data/processed/jee_features"
NIT_PARQUET = "/app/data/processed/jee_features"

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
    "student_rank",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_parquet(path: str) -> pd.DataFrame:
    files = glob.glob(f"{path}/**/*.parquet", recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files at: {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def prepare_features(df: pd.DataFrame, student_rank: int) -> pd.DataFrame:
    df = df.copy()
    df["student_rank"] = student_rank

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)

    all_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    available = [c for c in all_cols if c in df.columns]
    return df[available]


def predict_eligible(
    model_type: str,
    student_rank: int,
    quota: str,
    pool: str,
    category: str,
    top_n: int = 20,
    prob_threshold: float = 0.40,
) -> pd.DataFrame:

    model_path = os.path.join(MODEL_DIR, f"model_{model_type.lower()}.cbm")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run jee_train.py first."
        )

    model = CatBoostClassifier()
    model.load_model(model_path)

    parquet_path = IIT_PARQUET if model_type.lower() == "iit" else NIT_PARQUET
    features_df  = load_parquet(parquet_path)

    # Filter to student's quota/pool/category to avoid irrelevant rows
    mask = (
        (features_df["quota"].str.upper()    == quota.upper()) &
        (features_df["pool"].str.lower()     == pool.lower()) &
        (features_df["category"].str.upper() == category.upper())
    )
    filtered = features_df[mask].reset_index(drop=True)

    if filtered.empty:
        print(f"  No data found for quota={quota}, pool={pool}, category={category}")
        return pd.DataFrame()

    X = prepare_features(filtered, student_rank)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    pool_obj = Pool(X, cat_features=cat_cols)
    probs    = model.predict_proba(pool_obj)[:, 1]

    results = filtered[["institute_short", "program_name", "degree_short",
                         "closing_rank_max", "closing_rank_avg",
                         "trend_direction", "difficulty_pct"]].copy()
    results["eligibility_prob"] = probs.round(4)
    results["student_rank"]     = student_rank

    # Filter by threshold and sort
    results = (
        results[results["eligibility_prob"] >= prob_threshold]
        .sort_values("eligibility_prob", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    results.index += 1  # 1-based rank
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="JEE College Predictor")
    p.add_argument("--model_type", choices=["iit", "nit"], required=True)
    p.add_argument("--rank",       type=int,   required=True,
                   help="Student's JEE Advanced / Main rank")
    p.add_argument("--quota",      default="AI",
                   choices=["AI", "OS", "HS", "JK", "GO"],
                   help="Seat quota")
    p.add_argument("--pool",       default="Gender-Neutral",
                   choices=["Gender-Neutral", "Female-Only"])
    p.add_argument("--category",   default="GEN",
                   choices=["GEN", "OBC-NCL", "SC", "ST", "GEN-EWS",
                            "GEN-PWD", "OBC-NCL-PWD", "SC-PWD", "ST-PWD"])
    p.add_argument("--top",        type=int, default=20)
    p.add_argument("--threshold",  type=float, default=0.40,
                   help="Minimum eligibility probability to include")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  JEE College Predictor")
    print(f"  Model   : {args.model_type.upper()}")
    print(f"  Rank    : {args.rank:,}")
    print(f"  Quota   : {args.quota}  |  Pool: {args.pool}  |  Category: {args.category}")
    print(f"{'='*60}\n")

    results = predict_eligible(
        model_type     = args.model_type,
        student_rank   = args.rank,
        quota          = args.quota,
        pool           = args.pool,
        category       = args.category,
        top_n          = args.top,
        prob_threshold = args.threshold,
    )

    if results.empty:
        print("  No eligible colleges found for given parameters.")
    else:
        print(f"  Top {len(results)} eligible {args.model_type.upper()} colleges:\n")
        pd.set_option("display.max_colwidth", 40)
        pd.set_option("display.width", 120)
        print(results.to_string())
        print()

        # Optionally save to CSV
        out_csv = f"/app/data/predictions_{args.model_type}_{args.rank}.csv"
        results.to_csv(out_csv, index=True)
        print(f"  Saved → {out_csv}")


if __name__ == "__main__":
    main()