import os
import json
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
# Load JEE Models
# -------------------------
MODELS_DIR = "models/jee"
try:
    with open(f"{MODELS_DIR}/iit_model.pkl", "rb") as f:
        iit_model = pickle.load(f)
    with open(f"{MODELS_DIR}/iit_encoders.pkl", "rb") as f:
        iit_encoders = pickle.load(f)
except:
    iit_model, iit_encoders = None, {}

try:
    with open(f"{MODELS_DIR}/nit_model.pkl", "rb") as f:
        nit_model = pickle.load(f)
    with open(f"{MODELS_DIR}/nit_encoders.pkl", "rb") as f:
        nit_encoders = pickle.load(f)
except:
    nit_model, nit_encoders = None, {}

try:
    with open(f"{MODELS_DIR}/feature_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
except:
    metadata = {}

# -------------------------
# Load NEET Models
# -------------------------
NEET_MODELS_DIR = "/app/models/neet" if os.path.exists("/app/models/neet") else "models/neet"

try:
    with open(f"{NEET_MODELS_DIR}/neet_model.pkl", "rb") as f:
        neet_model = pickle.load(f)
    with open(f"{NEET_MODELS_DIR}/neet_encoders.pkl", "rb") as f:
        neet_encoders = pickle.load(f)
    with open(f"{NEET_MODELS_DIR}/neet_feature_cols.pkl", "rb") as f:
        neet_feature_cols = pickle.load(f)
except:
    neet_model, neet_encoders, neet_feature_cols = None, {}, []

# -------------------------
# Load Dataset (JEE)
# -------------------------
try:
    df = pd.read_csv("data/processed/features.csv")
    if "institute_type" in df.columns:
        df["institute_type"] = df["institute_type"].astype(str).str.upper().str.strip()
    else:
        df["institute_type"] = np.where(
            df["institute_short"].str.upper().str.contains("IIT"),
            "IIT",
            "NIT"
        )
except:
    df = pd.DataFrame()

# -------------------------
# Redis
# -------------------------
try:
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    r.ping()
except:
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
    if le is None: return 0
    if val in le.classes_: return int(le.transform([val])[0])
    if "__UNKNOWN__" in le.classes_: return int(le.transform(["__UNKNOWN__"])[0])
    return 0

# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "IntelliJEE Unified Predictor API Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Validated Endpoint for JEE
# -------------------------
@app.post("/predict")
def predict_jee(data: InputData):
    key = f"jee_{data.user_rank}_{data.exam_type}_{data.category}_{data.quota}_{data.pool}"
    if r:
        cached = r.get(key)
        if cached: return {"source": "cache", "data": json.loads(cached)}

    if "advanced" in data.exam_type.lower():
        inst_type = "IIT"; model = iit_model; encoders = iit_encoders
    else:
        inst_type = "NIT"; model = nit_model; encoders = nit_encoders

    if model is None: return {"source": "error", "error": f"{inst_type} model not found"}
    if df.empty: return {"source": "error", "error": "features.csv missing"}

    temp = df[(df["category"] == data.category) & (df["quota"] == data.quota) & (df["pool"] == data.pool) & (df["institute_type"] == inst_type)].copy()
    if temp.empty: return {"source": "model", "data": {"Safe": [], "Likely": [], "Ambitious": []}}

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
        if diff >= 0: tier = "Safe"
        elif diff >= -1500: tier = "Likely"
        else: continue
        
        rows.append({
            "institute": row["institute_short"],
            "program": row["program_name"],
            "program_duration": str(row["program_duration"]) + " Years",
            "degree_short": row["degree_short"],
            "predicted_cutoff": pred,
            "tier": tier
        })

    best = {}
    for row in rows:
        k = (row["institute"], row["program"])
        if k not in best: best[k] = row
        else:
            if row["predicted_cutoff"] > best[k]["predicted_cutoff"]: best[k] = row

    rows_list = sorted(list(best.values()), key=lambda x: ({"Safe":0,"Likely":1}[x["tier"]], x["predicted_cutoff"]))
    structured_json = {"Safe": [], "Likely": []}
    for item in rows_list: structured_json[item["tier"]].append(item)

    if r: r.setex(key, 3600, json.dumps(structured_json))
    return {"source": "model", "data": structured_json}

# -------------------------
# Validated Endpoint for NEET
# -------------------------
@app.post("/predict/neet")
def predict_neet(data: NeetInputData):
    key = f"neet_{data.candidate_rank}_{data.category}"
    if r:
        cached = r.get(key)
        if cached: return {"source": "cache", "data": json.loads(cached)}

    if neet_model is None:
        return {"source": "error", "error": "NEET XGBoost model is missing from path"}

    institutes = [i for i in neet_encoders.get("institute").classes_ if i != "__UNKNOWN__"]
    if not institutes:
        return {"source": "error", "error": "No institute encoding boundaries found"}

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
    df_likely = df_raw[(df_raw["pred_closing_rank"] > (data.candidate_rank - 1500)) & (df_raw["pred_closing_rank"] <= data.candidate_rank)]

    structured_json = {"Safe": [], "Likely": []}
    
    for _, row in df_safe.sort_values("pred_closing_rank", ascending=False).iterrows():
        structured_json["Safe"].append({
            "institute": row["institute"],
            "predicted_cutoff": int(row["pred_closing_rank"]),
            "tier": "Safe",
            "course": "MBBS"
        })

    for _, row in df_likely.sort_values("pred_closing_rank", ascending=False).iterrows():
        structured_json["Likely"].append({
            "institute": row["institute"],
            "predicted_cutoff": int(row["pred_closing_rank"]),
            "tier": "Likely",
            "course": "MBBS"
        })

    if r: r.setex(key, 3600, json.dumps(structured_json))
    return {"source": "model", "data": structured_json}
