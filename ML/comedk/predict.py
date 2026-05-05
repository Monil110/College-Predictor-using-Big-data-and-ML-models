import os
import sys
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

MODEL_DIR = "models/comedk"
PREDICT_YEAR = 2025
LIKELY_WINDOW = 1000


# ─── Load ─────────────────────────────────────────────────────────────────────

def load_artifacts():
    model = CatBoostRegressor()
    model.load_model(f"{MODEL_DIR}/comedk_model.cbm")

    with open(f"{MODEL_DIR}/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    with open(f"{MODEL_DIR}/features.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    with open(f"{MODEL_DIR}/raw_values.pkl", "rb") as f:
        raw_values = pickle.load(f)

    with open(f"{MODEL_DIR}/lookup.pkl", "rb") as f:
        lookup = pickle.load(f)

    return model, encoders, feature_cols, raw_values, lookup


# ─── Build Input ──────────────────────────────────────────────────────────────

def build_prediction_df(encoders, feature_cols, lookup, category, branch_filter=None):
    rows = []

    for (college, course, cat), hist in lookup.items():
        if cat != category:
            continue

        if branch_filter and branch_filter.lower() not in course.lower():
            continue

        if len(hist) == 0:
            continue

        rows.append({
            "college_name": college,
            "course_name": course,
            "category": category,
            "year": PREDICT_YEAR,

            "prev_year_closing_rank": hist[-1],
            "closing_rank_mean": np.mean(hist),
            "closing_rank_std": np.std(hist),
            "closing_rank_min": min(hist),
            "closing_rank_max": max(hist),
            "rank_trend": max(hist) - min(hist),
            "years_available": len(hist),
            "latest_year": PREDICT_YEAR - 1,
            "earliest_year": PREDICT_YEAR - len(hist),

            "college_avg_rank": np.mean(hist),
            "category_avg_rank": np.mean(hist),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return None, None

    raw_rows = df.copy()

    # Encode
    for col in ["college_name", "course_name", "category"]:
        le = encoders[col]
        df[col] = df[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    df = df[(df["college_name"] != -1) & 
            (df["course_name"] != -1) & 
            (df["category"] != -1)]
            
    raw_rows = raw_rows.loc[df.index].reset_index(drop=True)
    df = df.reset_index(drop=True)

    return df[feature_cols], raw_rows


# ─── Predict ──────────────────────────────────────────────────────────────────

def predict_all(model, df):
    log_preds = model.predict(df)
    return np.expm1(log_preds).astype(int)


# ─── Main Logic ───────────────────────────────────────────────────────────────

def run_prediction(user_rank, category, branch_filter=None):
    print(f"\nRank: {user_rank} | Category: {category} | Branch: {branch_filter or 'All'}")

    model, encoders, feature_cols, raw_values, lookup = load_artifacts()

    df_encoded, raw_rows = build_prediction_df(
        encoders, feature_cols, lookup, category, branch_filter
    )

    if df_encoded is None:
        print("No data found.")
        return

    preds = predict_all(model, df_encoded)

    results = []
    for i in range(len(preds)):
        results.append({
            "college": raw_rows.iloc[i]["college_name"],
            "course": raw_rows.iloc[i]["course_name"],
            "pred_rank": preds[i]
        })

    df = pd.DataFrame(results)

    lower = max(1, user_rank - LIKELY_WINDOW)

    safe = df[df["pred_rank"] > user_rank].sort_values("pred_rank", ascending=True)
    likely = df[(df["pred_rank"] >= lower) & (df["pred_rank"] <= user_rank)] \
                .sort_values("pred_rank", ascending=True)

    print("\nSAFE:")
    if safe.empty:
        print("No safe options found.")
    for _, row in safe.head(15).iterrows():
        print(f"{row['college']} | {row['course']} | {row['pred_rank']}")

    print("\nLIKELY:")
    if likely.empty:
        print("No likely options found.")
    for _, row in likely.head(15).iterrows():
        print(f"{row['college']} | {row['course']} | {row['pred_rank']}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int)
    parser.add_argument("--category", type=str)
    parser.add_argument("--branch", type=str, default=None)
    args = parser.parse_args()

    if args.rank and args.category:
        run_prediction(args.rank, args.category.upper(), args.branch)
    else:
        rank = int(input("Enter rank: "))
        category = input("Enter category: ").upper()
        branch = input("Branch (optional): ").strip() or None
        run_prediction(rank, category, branch)


if __name__ == "__main__":
    main()