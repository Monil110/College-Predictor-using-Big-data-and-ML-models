import os
import json
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from pydantic import BaseModel
import redis
import glob

app = FastAPI(title="College Predictor API (Unified)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Base Directory (anchored to this file's location)
# Ensures paths work regardless of where Render launches the process
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# -------------------------
# Load JEE Models
# -------------------------
MODELS_DIR = os.path.join(ROOT_DIR, "models", "jee")

def load_jee_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"Failed to read parquet from {path}: {e}")
        return pd.DataFrame()

try:
    iit_model = CatBoostClassifier()
    iit_model.load_model(os.path.join(MODELS_DIR, "model_iit.cbm"))
    print("[SUCCESS] IIT model loaded")
except Exception as e:
    print(f"[ERROR] IIT model load failed: {e}")
    iit_model = None

try:
    nit_model = CatBoostClassifier()
    nit_model.load_model(os.path.join(MODELS_DIR, "model_nit.cbm"))
    print("[SUCCESS] NIT model loaded")
except Exception as e:
    print(f"[ERROR] NIT model load failed: {e}")
    nit_model = None

IIT_PARQUET = os.path.join(ROOT_DIR, "data", "processed", "jee_features", "iit")
NIT_PARQUET = os.path.join(ROOT_DIR, "data", "processed", "jee_features", "nit")

try:
    iit_features_df = load_jee_parquet(IIT_PARQUET)
    nit_features_df = load_jee_parquet(NIT_PARQUET)
    print("[SUCCESS] JEE features data loaded")
except Exception as e:
    print(f"[ERROR] JEE features load failed: {e}")
    iit_features_df = pd.DataFrame()
    nit_features_df = pd.DataFrame()


# -------------------------
# Load NEET Models
# -------------------------
NEET_MODELS_DIR = os.path.join(ROOT_DIR, "models", "neet")

try:
    with open(os.path.join(NEET_MODELS_DIR, "neet_model.pkl"), "rb") as f:
        neet_model = pickle.load(f)
    with open(os.path.join(NEET_MODELS_DIR, "neet_encoders.pkl"), "rb") as f:
        neet_encoders = pickle.load(f)
    with open(os.path.join(NEET_MODELS_DIR, "neet_feature_cols.pkl"), "rb") as f:
        neet_feature_cols = pickle.load(f)
    print("[SUCCESS] NEET model loaded")
except Exception as e:
    print(f"[ERROR] NEET model load failed: {e}")
    neet_model, neet_encoders, neet_feature_cols = None, {}, []

# -------------------------
# Load KCET Models
# -------------------------
KCET_MODELS_DIR = os.path.join(ROOT_DIR, "models", "kcet")

try:
    kcet_model = CatBoostRegressor()
    kcet_model.load_model(os.path.join(KCET_MODELS_DIR, "kcet_model.cbm"))
    with open(os.path.join(KCET_MODELS_DIR, "kcet_encoders.pkl"), "rb") as f:
        kcet_encoders = pickle.load(f)
    with open(os.path.join(KCET_MODELS_DIR, "kcet_feature_cols.pkl"), "rb") as f:
        kcet_feature_cols = pickle.load(f)
    print("[SUCCESS] KCET model loaded")
except Exception as e:
    print(f"[ERROR] KCET model load failed: {e}")
    kcet_model, kcet_encoders, kcet_feature_cols = None, {}, []

# -------------------------
# Redis (optional, graceful fallback)
# -------------------------
try:
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    r.ping()
    print("[SUCCESS] Redis connected")
except Exception as e:
    print(f"[WARN] Redis unavailable (caching disabled): {e}")
    r = None

# -------------------------
# Input Schemas
# -------------------------
class InputData(BaseModel):
    user_rank: int
    exam_type: str
    category: str
    quota: str
    pool: str

class NeetInputData(BaseModel):
    candidate_rank: int
    category: str

class KcetInputData(BaseModel):
    user_rank: int
    category: str
    base_category: str
    quota: str
    region: str

# -------------------------
# Helpers (JEE)
# -------------------------
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

def prepare_jee_features(df: pd.DataFrame, student_rank: int) -> pd.DataFrame:
    df = df.copy()
    df["student_rank"] = student_rank

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(object).fillna("UNKNOWN").astype(str)

    all_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    available = [c for c in all_cols if c in df.columns]
    return df[available]

# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "PredictMe Unified Predictor API Running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "iit": iit_model is not None,
            "nit": nit_model is not None,
            "neet": neet_model is not None,
            "kcet": kcet_model is not None,
        },
        "dataset_loaded": not iit_features_df.empty,
        "redis": r is not None,
    }

# -------------------------
# Predict JEE
# -------------------------
@app.post("/predict")
def predict_jee(data: InputData):
    key = f"jee_{data.user_rank}_{data.exam_type}_{data.category}_{data.quota}_{data.pool}"
    if r:
        cached = r.get(key)
        if cached:
            return {"source": "cache", "data": json.loads(cached)}

    if "advanced" in data.exam_type.lower():
        inst_type = "IIT"
        model = iit_model
        features_df = iit_features_df
    else:
        inst_type = "NIT"
        model = nit_model
        features_df = nit_features_df

    if model is None:
        return {"source": "error", "error": f"{inst_type} model not loaded. Check server logs."}
    if features_df.empty:
        return {"source": "error", "error": f"{inst_type} features data not loaded. Check server logs."}

    mask = (
        (features_df["quota"].str.upper()    == data.quota.upper()) &
        (features_df["pool"].str.lower()     == data.pool.lower()) &
        (features_df["category"].str.upper() == data.category.upper())
    )
    filtered = features_df[mask].reset_index(drop=True)

    structured_json = {"Safe": [], "Likely": []}
    
    if not filtered.empty:
        X = prepare_jee_features(filtered, data.user_rank)
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

        pool_obj = Pool(X, cat_features=cat_cols)
        probs = model.predict_proba(pool_obj)[:, 1]

        results = filtered[["institute_short", "program_name", "degree_short",
                             "closing_rank_max", "closing_rank_avg",
                             "trend_direction", "difficulty_pct"]].copy()
        
        results["eligibility_prob"] = probs.round(4)
        
        for _, row in results.iterrows():
            prob = row["eligibility_prob"]
            if prob >= 0.70:
                tier = "Safe"
            elif prob >= 0.40:
                tier = "Likely"
            else:
                continue

            item = {
                "institute": row["institute_short"],
                "program": row["program_name"],
                "program_duration": str(row.get("degree_short", "4 Years")),
                "degree_short": row["degree_short"],
                "predicted_cutoff": int(row.get("closing_rank_max", 0)),
                "eligibility_prob": float(prob),
                "tier": tier,
            }
            structured_json[tier].append(item)
            
        structured_json["Safe"].sort(key=lambda x: x["eligibility_prob"], reverse=True)
        structured_json["Likely"].sort(key=lambda x: x["eligibility_prob"], reverse=True)
        
        structured_json["Safe"] = structured_json["Safe"][:20]
        structured_json["Likely"] = structured_json["Likely"][:20]

    if r:
        r.setex(key, 3600, json.dumps(structured_json))

    return {"source": "model", "data": structured_json}

# -------------------------
# Predict NEET
# -------------------------
@app.post("/predict/neet")
def predict_neet(data: NeetInputData):
    key = f"neet_{data.candidate_rank}_{data.category}"
    if r:
        cached = r.get(key)
        if cached:
            return {"source": "cache", "data": json.loads(cached)}

    if neet_model is None:
        return {"source": "error", "error": "NEET model not loaded. Check server logs."}

    institutes = [i for i in neet_encoders.get("institute").classes_ if i != "__UNKNOWN__"]
    if not institutes:
        return {"source": "error", "error": "No institute encoding found in neet_encoders."}

    df_neet = pd.DataFrame({"institute": institutes})
    df_neet["category"] = data.category
    df_neet["year"] = 2026

    df_raw = df_neet.copy()
    for col in ["institute", "category"]:
        le = neet_encoders[col]
        known = set(le.classes_)
        df_neet[col] = df_neet[col].astype(str).str.strip().str.upper()
        df_neet[col] = df_neet[col].apply(lambda x: x if x in known else "__UNKNOWN__")
        df_neet[col] = le.transform(df_neet[col])

    X = df_neet[neet_feature_cols]
    log_preds = neet_model.predict(X)
    pred_ranks = np.expm1(log_preds)
    df_raw["pred_closing_rank"] = pred_ranks

    df_safe = df_raw[df_raw["pred_closing_rank"] > data.candidate_rank]
    df_likely = df_raw[
        (df_raw["pred_closing_rank"] > (data.candidate_rank - 1500)) &
        (df_raw["pred_closing_rank"] <= data.candidate_rank)
    ]

    structured_json = {"Safe": [], "Likely": []}

    for _, row in df_safe.sort_values("pred_closing_rank", ascending=False).iterrows():
        structured_json["Safe"].append({
            "institute": row["institute"],
            "predicted_cutoff": int(row["pred_closing_rank"]),
            "tier": "Safe",
            "course": "MBBS",
        })

    for _, row in df_likely.sort_values("pred_closing_rank", ascending=False).iterrows():
        structured_json["Likely"].append({
            "institute": row["institute"],
            "predicted_cutoff": int(row["pred_closing_rank"]),
            "tier": "Likely",
            "course": "MBBS",
        })

    if r:
        r.setex(key, 3600, json.dumps(structured_json))

    return {"source": "model", "data": structured_json}

# -------------------------
# Predict KCET
# -------------------------
@app.post("/predict/kcet")
def predict_kcet(data: KcetInputData):
    key = f"kcet_{data.user_rank}_{data.category}_{data.base_category}_{data.quota}_{data.region}"
    if r:
        cached = r.get(key)
        if cached:
            return {"source": "cache", "data": json.loads(cached)}

    if kcet_model is None:
        return {"source": "error", "error": "KCET model not loaded. Check server logs."}

    colleges = kcet_encoders.get("college_name", [])
    courses = kcet_encoders.get("course_name", [])

    if not colleges or not courses:
        return {"source": "error", "error": "No college/course encoding found in kcet_encoders."}

    df_grid = pd.MultiIndex.from_product(
        [colleges, courses], names=["college_name", "course_name"]
    ).to_frame(index=False)

    df_grid["category"] = data.category.upper()
    df_grid["base_category"] = data.base_category.upper()
    df_grid["quota"] = data.quota.upper()
    df_grid["region"] = data.region.upper()
    df_grid["year"] = 2026

    X = df_grid[kcet_feature_cols]
    log_preds = kcet_model.predict(X)
    pred_ranks = np.expm1(log_preds)
    df_grid["pred_closing_rank"] = pred_ranks

    df_safe = df_grid[df_grid["pred_closing_rank"] > data.user_rank]
    df_likely = df_grid[
        (df_grid["pred_closing_rank"] > (data.user_rank - 5000)) &
        (df_grid["pred_closing_rank"] <= data.user_rank)
    ]

    structured_json = {"Safe": [], "Likely": []}

    for _, row in df_safe.sort_values("pred_closing_rank", ascending=False).head(15).iterrows():
        structured_json["Safe"].append({
            "institute": row["college_name"],
            "predicted_cutoff": int(row["pred_closing_rank"]),
            "tier": "Safe",
            "course": row["course_name"],
        })

    for _, row in df_likely.sort_values("pred_closing_rank", ascending=False).head(15).iterrows():
        structured_json["Likely"].append({
            "institute": row["college_name"],
            "predicted_cutoff": int(row["pred_closing_rank"]),
            "tier": "Likely",
            "course": row["course_name"],
        })

    if r:
        r.setex(key, 3600, json.dumps(structured_json))

    return {"source": "model", "data": structured_json}