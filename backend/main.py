import os
import json
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from catboost import CatBoostRegressor
from pydantic import BaseModel
import redis

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

def model_path(*parts):
    return os.path.join(BASE_DIR, *parts)

# -------------------------
# Load JEE Models
# -------------------------
MODELS_DIR = model_path("models", "jee")

try:
    with open(os.path.join(MODELS_DIR, "iit_model.pkl"), "rb") as f:
        iit_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "iit_encoders.pkl"), "rb") as f:
        iit_encoders = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "iit_feature_cols.pkl"), "rb") as f:
        iit_feature_cols = pickle.load(f)
    print("✅ IIT model loaded")
except Exception as e:
    print(f"❌ IIT model load failed: {e}")
    iit_model, iit_encoders, iit_feature_cols = None, {}, []

try:
    with open(os.path.join(MODELS_DIR, "nit_model.pkl"), "rb") as f:
        nit_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "nit_encoders.pkl"), "rb") as f:
        nit_encoders = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "nit_feature_cols.pkl"), "rb") as f:
        nit_feature_cols = pickle.load(f)
    print("✅ NIT model loaded")
except Exception as e:
    print(f"❌ NIT model load failed: {e}")
    nit_model, nit_encoders, nit_feature_cols = None, {}, []

try:
    with open(os.path.join(MODELS_DIR, "feature_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    print("✅ JEE metadata loaded")
except Exception as e:
    print(f"❌ JEE metadata load failed: {e}")
    metadata = {}

# -------------------------
# Load NEET Models
# -------------------------
NEET_MODELS_DIR = model_path("models", "neet")

try:
    with open(os.path.join(NEET_MODELS_DIR, "neet_model.pkl"), "rb") as f:
        neet_model = pickle.load(f)
    with open(os.path.join(NEET_MODELS_DIR, "neet_encoders.pkl"), "rb") as f:
        neet_encoders = pickle.load(f)
    with open(os.path.join(NEET_MODELS_DIR, "neet_feature_cols.pkl"), "rb") as f:
        neet_feature_cols = pickle.load(f)
    print("✅ NEET model loaded")
except Exception as e:
    print(f"❌ NEET model load failed: {e}")
    neet_model, neet_encoders, neet_feature_cols = None, {}, []

# -------------------------
# Load KCET Models
# -------------------------
KCET_MODELS_DIR = model_path("models", "kcet")

try:
    kcet_model = CatBoostRegressor()
    kcet_model.load_model(os.path.join(KCET_MODELS_DIR, "kcet_model.cbm"))
    with open(os.path.join(KCET_MODELS_DIR, "kcet_encoders.pkl"), "rb") as f:
        kcet_encoders = pickle.load(f)
    with open(os.path.join(KCET_MODELS_DIR, "kcet_feature_cols.pkl"), "rb") as f:
        kcet_feature_cols = pickle.load(f)
    print("✅ KCET model loaded")
except Exception as e:
    print(f"❌ KCET model load failed: {e}")
    kcet_model, kcet_encoders, kcet_feature_cols = None, {}, []

# -------------------------
# Load Dataset (JEE)
# -------------------------
try:
    df = pd.read_csv(model_path("data", "processed", "features.csv"))
    if "institute_type" in df.columns:
        df["institute_type"] = df["institute_type"].astype(str).str.upper().str.strip()
    else:
        df["institute_type"] = np.where(
            df["institute_short"].str.upper().str.contains("IIT"),
            "IIT",
            "NIT"
        )
    print("✅ JEE dataset loaded")
except Exception as e:
    print(f"❌ JEE dataset load failed: {e}")
    df = pd.DataFrame()

# -------------------------
# Redis (optional, graceful fallback)
# -------------------------
try:
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    r.ping()
    print("✅ Redis connected")
except Exception as e:
    print(f"⚠️ Redis unavailable (caching disabled): {e}")
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
def engineer_features(temp):
    temp = temp.copy()
    temp["rank_spread"] = temp["closing_rank"] - temp["opening_rank"]
    temp["is_dual_degree"] = temp["degree_short"].astype(str).str.contains(r"IDD|M\.Tech|MSc|Dual", case=False, na=False).astype(int)
    temp["is_home_state"] = (temp["quota"].astype(str).str.upper() == "HS").astype(int)
    temp["is_female_pool"] = temp["pool"].astype(str).str.lower().str.contains("female", na=False).astype(int)
    temp["is_pwd"] = temp["category"].astype(str).str.contains("PWD", na=False).astype(int)
    temp["is_ews"] = temp["category"].astype(str).str.contains("EWS", na=False).astype(int)
    temp["round_norm"] = pd.to_numeric(temp["round_no"], errors="coerce").fillna(6) / 7.0
    if "year" in temp.columns:
        temp["year_offset"] = pd.to_numeric(temp["year"], errors="coerce").fillna(0) - pd.to_numeric(temp["year"], errors="coerce").fillna(0).min()
    else:
        temp["year_offset"] = 0
    return temp

def encode(encoders, col, val):
    le = encoders.get(col)
    if le is None:
        return 0
    if val in le.classes_:
        return int(le.transform([val])[0])
    if "__UNKNOWN__" in le.classes_:
        return int(le.transform(["__UNKNOWN__"])[0])
    return 0

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
        "dataset_loaded": not df.empty,
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
        encoders = iit_encoders
    else:
        inst_type = "NIT"
        model = nit_model
        encoders = nit_encoders

    if model is None:
        return {"source": "error", "error": f"{inst_type} model not loaded. Check server logs."}
    if df.empty:
        return {"source": "error", "error": "features.csv not loaded. Check server logs."}
    if not metadata:
        return {"source": "error", "error": "feature_metadata.pkl not loaded. Check server logs."}

    temp = df[
        (df["category"] == data.category) &
        (df["quota"] == data.quota) &
        (df["pool"] == data.pool) &
        (df["institute_type"] == inst_type)
    ].copy()

    if temp.empty:
        return {"source": "model", "data": {"Safe": [], "Likely": []}}

    temp = engineer_features(temp)

    X_dict = {}
    for col in metadata["all_cols"]:
        if col in metadata["cat_cols"]:
            X_dict[col] = temp[col].apply(lambda x: encode(encoders, col, x))
        else:
            X_dict[col] = pd.to_numeric(temp[col], errors="coerce").fillna(0)
    X = pd.DataFrame(X_dict)

    pred_log = model.predict(X)
    preds = np.expm1(pred_log).astype(int)
    preds = np.maximum(preds, 1)

    rows = []
    for i, (_, row) in enumerate(temp.iterrows()):
        pred = int(preds[i])
        diff = pred - data.user_rank
        if diff >= 0:
            tier = "Safe"
        elif diff >= -1500:
            tier = "Likely"
        else:
            continue

        rows.append({
            "institute": row["institute_short"],
            "program": row["program_name"],
            "program_duration": str(row["program_duration"]) + " Years",
            "degree_short": row["degree_short"],
            "predicted_cutoff": pred,
            "tier": tier,
        })

    best = {}
    for row in rows:
        k = (row["institute"], row["program"])
        if k not in best:
            best[k] = row
        elif row["predicted_cutoff"] > best[k]["predicted_cutoff"]:
            best[k] = row

    rows_list = sorted(
        list(best.values()),
        key=lambda x: ({"Safe": 0, "Likely": 1}[x["tier"]], x["predicted_cutoff"])
    )

    structured_json = {"Safe": [], "Likely": []}
    for item in rows_list:
        structured_json[item["tier"]].append(item)

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