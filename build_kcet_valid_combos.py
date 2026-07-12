"""
One-off: derive the real (college, course) pairs that actually exist in the
KCET training data. The API used to build predictions from the full
college x course cross-product (459 x 146 = 67,014 combos), which asked the
model for cutoffs on tens of thousands of combinations that were never
offered — only 2,820 of them are real.
"""
import pandas as pd

df = pd.read_parquet("data/processed/kcet_features")
combos = df[["college_name", "course_name"]].drop_duplicates().reset_index(drop=True)

out_path = "models/kcet/kcet_valid_combos.pkl"
combos.to_pickle(out_path, protocol=4)

print(f"Wrote {len(combos)} real college+course combos to {out_path}")
