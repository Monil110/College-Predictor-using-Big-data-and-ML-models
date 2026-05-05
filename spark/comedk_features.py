import os
import pandas as pd
import numpy as np

def main():
    print("Running COMEDK feature engineering...")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_parquet("data/processed/comedk_cleaned")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["rank", "college_name", "course_name", "category", "year"])
    df = df[df["rank"] > 0]
    df["year"] = df["year"].astype(int)

    print(f"  Loaded {len(df):,} rows | "
          f"{df['year'].nunique()} years | "
          f"{df['college_name'].nunique()} colleges | "
          f"{df['course_name'].nunique()} courses")

    group_cols = ["college_name", "course_name", "category", "year"]

    # ── Per-year closing rank (used as training target) ───────────────────────
    # closing_rank = max rank admitted that year (higher rank = weaker student)
    yearly = (
        df.groupby(group_cols)
        .agg(closing_rank=("rank", "max"))
        .reset_index()
    )

    # ── Cross-year trend features (joined back on college+course+category) ────
    key_cols = ["college_name", "course_name", "category"]

    trend = (
        yearly.groupby(key_cols)
        .agg(
            closing_rank_mean=("closing_rank", "mean"),
            closing_rank_std=("closing_rank", "std"),
            closing_rank_min=("closing_rank", "min"),   # most competitive year
            closing_rank_max=("closing_rank", "max"),   # least competitive year
            years_available=("year", "nunique"),
            latest_year=("year", "max"),
            earliest_year=("year", "min"),
        )
        .reset_index()
    )

    trend["closing_rank_std"] = trend["closing_rank_std"].fillna(0)

    # Year-over-year rank change (positive = cutoff getting easier, negative = tougher)
    trend["rank_trend"] = trend["closing_rank_max"] - trend["closing_rank_min"]

    # ── Merge yearly rows with trend features ─────────────────────────────────
    features = yearly.merge(trend, on=key_cols, how="left")

    # ── Lag features: previous year's closing rank for same combo ─────────────
    features = features.sort_values(group_cols)
    features["prev_year_closing_rank"] = (
        features.groupby(key_cols)["closing_rank"].shift(1)
    )
    # Fill missing lag with the overall mean for that combo
    features["prev_year_closing_rank"] = features.groupby(key_cols)["prev_year_closing_rank"] \
        .transform(lambda x: x.fillna(x.mean()))
    # If still NaN (only 1 year of data), fill with closing_rank itself
    features["prev_year_closing_rank"] = features["prev_year_closing_rank"].fillna(
        features["closing_rank"]
    )

    # ── College-level difficulty rank (avg closing rank across all courses) ───
    college_avg = (
        features.groupby(["college_name", "year"])["closing_rank"]
        .mean()
        .reset_index()
        .rename(columns={"closing_rank": "college_avg_rank"})
    )
    features = features.merge(college_avg, on=["college_name", "year"], how="left")

    # ── Category difficulty offset ────────────────────────────────────────────
    # GM cutoffs are tighter (lower rank) vs KKR which are relaxed (higher rank)
    cat_offset = (
        features.groupby(["college_name", "course_name", "category"])["closing_rank"]
        .mean()
        .reset_index()
        .rename(columns={"closing_rank": "category_avg_rank"})
    )
    features = features.merge(cat_offset, on=["college_name", "course_name", "category"], how="left")

    # ── Final column selection and ordering ───────────────────────────────────
    feature_cols = [
        # Identifiers (used as cat features in model)
        "college_name",
        "course_name",
        "category",
        "year",

        # Target
        "closing_rank",

        # Trend/history features
        "prev_year_closing_rank",
        "closing_rank_mean",
        "closing_rank_std",
        "closing_rank_min",
        "closing_rank_max",
        "rank_trend",
        "years_available",
        "latest_year",
        "earliest_year",

        # Context features
        "college_avg_rank",
        "category_avg_rank",
    ]

    features = features[feature_cols]

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print(f"  Features shape     : {features.shape}")
    print(f"  Unique colleges    : {features['college_name'].nunique()}")
    print(f"  Unique courses     : {features['course_name'].nunique()}")
    print(f"  Unique categories  : {features['category'].nunique()}")
    print(f"  Years              : {sorted(features['year'].unique())}")
    print(f"  Target (closing_rank) — min: {features['closing_rank'].min():.0f}, "
          f"max: {features['closing_rank'].max():.0f}, "
          f"mean: {features['closing_rank'].mean():.0f}")
    print(f"  Null counts:\n{features.isnull().sum()[features.isnull().sum() > 0]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = "data/processed/comedk_features"
    os.makedirs(out_dir, exist_ok=True)
    features.to_parquet(f"{out_dir}/data.parquet", index=False)
    print(f"\n  Features saved to: {out_dir}/data.parquet")


if __name__ == "__main__":
    main()