import os
import pandas as pd
import numpy as np


def main():
    print("Running KCET feature engineering...")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_parquet("data/processed/kcet_cleaned")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["rank", "college_name", "course_name", "category",
                           "base_category", "quota", "region", "year"])
    df = df[df["rank"] > 0]
    df["year"] = df["year"].astype(int)

    full_key = ["college_name", "course_name", "category", "base_category", "quota", "region"]
    for c in full_key:
        df[c] = df[c].astype(str)

    print(f"  Loaded {len(df):,} rows | "
          f"{df['year'].nunique()} years | "
          f"{df['college_name'].nunique()} colleges | "
          f"{df['course_name'].nunique()} courses")

    group_cols = full_key + ["year"]

    # ── Per-year closing rank (used as training target) ───────────────────────
    # closing_rank = max rank admitted that year (higher rank = weaker student)
    yearly = (
        df.groupby(group_cols, observed=True)
        .agg(closing_rank=("rank", "max"))
        .reset_index()
    )

    # ── Cross-year trend features (joined back on full demographic key) ──────
    # Most (college,course,category,base_category,quota,region) combos only
    # have 1-2 years of history (KCET's categorical space is far more granular
    # than COMEDK's), so these are a real but sparse signal on their own.
    trend = (
        yearly.groupby(full_key, observed=True)
        .agg(
            closing_rank_mean=("closing_rank", "mean"),
            closing_rank_std=("closing_rank", "std"),
            closing_rank_min=("closing_rank", "min"),
            closing_rank_max=("closing_rank", "max"),
            years_available=("year", "nunique"),
            latest_year=("year", "max"),
            earliest_year=("year", "min"),
        )
        .reset_index()
    )
    trend["closing_rank_std"] = trend["closing_rank_std"].fillna(0)
    trend["rank_trend"] = trend["closing_rank_max"] - trend["closing_rank_min"]

    features = yearly.merge(trend, on=full_key, how="left")

    # ── Lag feature: previous year's closing rank for the same full combo ────
    features = features.sort_values(group_cols)
    features["prev_year_closing_rank"] = (
        features.groupby(full_key, observed=True)["closing_rank"].shift(1)
    )
    features["prev_year_closing_rank"] = features.groupby(full_key, observed=True)["prev_year_closing_rank"] \
        .transform(lambda x: x.fillna(x.mean()))
    features["prev_year_closing_rank"] = features["prev_year_closing_rank"].fillna(
        features["closing_rank"]
    )

    # ── Denser baseline signal: college+course difficulty, ignoring the ──────
    # category/quota/region split (much better data density — see docstring
    # in ML/kcet/train.py for why the full-key trend features alone are too
    # sparse: ~72% of full-key groups have only a single year of history).
    college_course_avg = (
        df.groupby(["college_name", "course_name"], observed=True)["rank"]
        .mean()
        .reset_index()
        .rename(columns={"rank": "college_course_avg_rank"})
    )
    features = features.merge(college_course_avg, on=["college_name", "course_name"], how="left")

    # ── College-level difficulty for that year ────────────────────────────────
    college_avg = (
        features.groupby(["college_name", "year"])["closing_rank"]
        .mean()
        .reset_index()
        .rename(columns={"closing_rank": "college_avg_rank"})
    )
    features = features.merge(college_avg, on=["college_name", "year"], how="left")

    # ── Category difficulty offset within a college+course ───────────────────
    cat_offset = (
        features.groupby(["college_name", "course_name", "category"], observed=True)["closing_rank"]
        .mean()
        .reset_index()
        .rename(columns={"closing_rank": "category_avg_rank"})
    )
    features = features.merge(cat_offset, on=["college_name", "course_name", "category"], how="left")

    # ── Final column selection and ordering ───────────────────────────────────
    feature_cols = [
        # Identifiers (used as cat features in model)
        "college_name", "course_name", "category", "base_category", "quota", "region", "year",

        # Target
        "closing_rank",

        # Trend/history features (sparse but real per exact demographic combo)
        "prev_year_closing_rank",
        "closing_rank_mean", "closing_rank_std",
        "closing_rank_min", "closing_rank_max",
        "rank_trend", "years_available", "latest_year", "earliest_year",

        # Denser context features
        "college_course_avg_rank", "college_avg_rank", "category_avg_rank",
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
    out_dir = "data/processed/kcet_features"
    # Wipe old Spark-partitioned output so a stale part file can't get mixed
    # in with the new single-file write when the directory is read back.
    if os.path.isdir(out_dir):
        for fname in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, fname))
    else:
        os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.parquet")
    features.to_parquet(out_path, index=False)
    print(f"\n  Features saved to: {out_path}")


if __name__ == "__main__":
    main()
