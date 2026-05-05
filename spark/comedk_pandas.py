import os
import pandas as pd

def main():
    print("Starting Pandas COMEDK Data Processing...")
    raw_dir = "data/raw/comedk"
    if not os.path.exists(raw_dir):
        raw_dir = "/app/data/raw/comedk"

    processed_dir = "data/processed/comedk_cleaned"
    if not os.path.exists("data/processed"):
        processed_dir = "/app/data/processed/comedk_cleaned"
    os.makedirs(processed_dir, exist_ok=True)

    features_dir = "data/processed/comedk_features"
    if not os.path.exists("data/processed"):
        features_dir = "/app/data/processed/comedk_features"
    os.makedirs(features_dir, exist_ok=True)

    all_dfs = []
    
    for year in [2023, 2024, 2025]:
        path = f"{raw_dir}/comedk{year}.csv"
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        print(f"Reading {path}...")
        df = pd.read_csv(path)
        
        # Determine base columns
        cols = list(df.columns)
        if len(cols) < 4:
            continue
            
        college_code_col = cols[0]
        college_name_col = cols[1]
        category_col = cols[2]
        course_cols = cols[3:]

        # Melt
        df_melted = df.melt(
            id_vars=[college_code_col, college_name_col, category_col],
            value_vars=course_cols,
            var_name="course_name",
            value_name="rank"
        )
        
        # Rename columns
        df_melted.rename(columns={
            college_code_col: "college_code",
            college_name_col: "college_name",
            category_col: "category"
        }, inplace=True)

        df_melted["year"] = year
        
        all_dfs.append(df_melted)

    if not all_dfs:
        print("No data loaded.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    # Clean text
    for col in ["college_code", "college_name", "category", "course_name"]:
        final_df[col] = final_df[col].astype(str).str.strip().str.replace('\n', ' ').replace('\r', '')
    final_df["category"] = final_df["category"].str.upper()

    # Filter categories and ranks
    final_df = final_df[final_df["category"].isin(["GM", "KKR"])]
    final_df["rank"] = pd.to_numeric(final_df["rank"], errors="coerce")
    final_df = final_df[final_df["rank"] > 0].dropna(subset=["rank"])

    # Write cleaned
    final_df.to_parquet(os.path.join(processed_dir, "data.parquet"), index=False)
    print(f"Cleaned data saved to {processed_dir}")

    # Feature Engineering
    # Group by college_name, course_name, category, year
    features = final_df.groupby(["college_name", "course_name", "category", "year"]).agg(
        closing_rank=("rank", "max"),
        closing_rank_avg=("rank", "mean"),
        closing_rank_std=("rank", "std")
    ).reset_index()

    # Fill NaN std dev
    features["closing_rank_std"] = features["closing_rank_std"].fillna(0.0)
    features["latest_recorded_year"] = features["year"]

    features.to_parquet(os.path.join(features_dir, "data.parquet"), index=False)
    print(f"Feature data saved to {features_dir}")

if __name__ == "__main__":
    main()
