"""
One-off: derive an institute -> course (MBBS/BDS/B.Sc. Nursing) map from the
cleaned NEET UG features parquet, so the API can report the real course
instead of hardcoding "MBBS" for every result (dental colleges were being
mislabeled).
"""
import glob
import pickle

import pandas as pd

VALID_COURSES = {"MBBS", "BDS", "B.Sc. Nursing"}

parquet_path = glob.glob("data/processed/neet_features/**/*.parquet", recursive=True)[0]
df = pd.read_parquet(parquet_path)

df = df[df["course"].isin(VALID_COURSES)].copy()
df["institute_up"] = df["institute"].astype(str).str.strip().str.upper()

mode_course = df.groupby("institute_up")["course"].agg(lambda x: x.value_counts().idxmax())
mapping = mode_course.to_dict()

out_path = "models/neet/neet_institute_course.pkl"
with open(out_path, "wb") as f:
    pickle.dump(mapping, f, protocol=4)

print(f"Wrote {len(mapping)} institute->course entries to {out_path}")
print("Sample BDS entries:", [k for k, v in mapping.items() if v == "BDS"][:5])
