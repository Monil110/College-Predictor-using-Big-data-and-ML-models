import pickle
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# -------------------------
# Anchor paths to project root
# predict.py is at: /src/jee/predict.py  (or /src/backend/jee/predict.py)
# ROOT_DIR resolves to: /src
# -------------------------
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(_FILE_DIR)  # one level up from jee/ folder

MODELS_DIR = os.path.join(ROOT_DIR, "models", "jee")
DATA_PATH  = os.path.join(ROOT_DIR, "data", "processed", "features.csv")


def engineer_features(req_df: pd.DataFrame) -> pd.DataFrame:
    req_df = req_df.copy()
    req_df["rank_spread"] = req_df["closing_rank"] - req_df["opening_rank"]

    req_df["is_dual_degree"] = req_df["degree_short"].str.contains(
        r"IDD|M\.Tech|MSc|Dual", case=False, na=False
    ).astype(int)

    req_df["is_home_state"] = (req_df["quota"].str.upper() == "HS").astype(int)
    req_df["is_female_pool"] = req_df["pool"].str.lower().str.contains("female", na=False).astype(int)
    req_df["is_pwd"] = req_df["category"].str.contains("PWD", na=False).astype(int)
    req_df["is_ews"] = req_df["category"].str.contains("EWS", na=False).astype(int)

    req_df["round_norm"] = req_df.get("round_no", 6) / 7.0

    if "year" in req_df.columns:
        req_df["year_offset"] = req_df["year"] - req_df["year"].min()
    else:
        req_df["year_offset"] = 0

    return req_df


def encode(encoders, col, val):
    le = encoders.get(col)
    if le is None:
        return 0
    if val in le.classes_:
        return int(le.transform([val])[0])
    if "__UNKNOWN__" in le.classes_:
        return int(le.transform(["__UNKNOWN__"])[0])
    return 0


def predict_admission(rank, institute, program, category, quota, pool, round_no=6, latest_year=2021):
    try:
        inst_type = "IIT" if "IIT" in institute.upper() else "NIT"
        prefix = inst_type.lower()

        with open(os.path.join(MODELS_DIR, f"{prefix}_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, f"{prefix}_encoders.pkl"), "rb") as f:
            encoders = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "feature_metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

    except FileNotFoundError as e:
        return {"error": f"Model file not found: {e}"}

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        return {"error": f"Failed to load features.csv: {e}"}

    temp = df[
        (df["institute_short"] == institute) &
        (df["program_name"] == program) &
        (df["category"] == category) &
        (df["quota"] == quota) &
        (df["pool"] == pool)
    ].copy()

    if temp.empty:
        return {"error": "Could not find corresponding historical data for these parameters in features.csv."}

    temp = engineer_features(temp).iloc[[0]]

    X_dict = {}
    for col in metadata.get("all_cols", []):
        if col in metadata.get("cat_cols", []):
            X_dict[col] = temp[col].apply(lambda x: encode(encoders, col, x))
        else:
            X_dict[col] = pd.to_numeric(temp[col], errors="coerce").fillna(0)

    X = pd.DataFrame(X_dict)

    predicted_cutoff_log = model.predict(X)[0]
    predicted_cutoff = int(np.maximum(np.expm1(predicted_cutoff_log), 1))

    diff = predicted_cutoff - rank

    if diff >= 0:
        tier = "Safe"
    elif diff >= -1500:
        tier = "Likely"
    else:
        tier = "Unlikely"

    return {
        "institute": institute,
        "program": program,
        "predicted_cutoff": predicted_cutoff,
        "user_rank": rank,
        "chance_bucket": tier,
    }


if __name__ == "__main__":
    print("Running sample prediction test...")
    res = predict_admission(
        rank=50,
        institute="IIT-Bombay",
        program="Computer Science and Engineering",
        category="GEN",
        quota="AI",
        pool="Gender-Neutral"
    )
    print(f"Result: {res}")