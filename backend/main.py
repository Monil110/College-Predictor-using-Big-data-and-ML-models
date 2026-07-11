import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
try:
    from catboost import CatBoostRegressor, CatBoostClassifier, Pool
    _catboost_available = True
except ModuleNotFoundError:
    print("[WARN] catboost not installed — JEE/KCET/COMEDK models will be disabled. Run: pip install catboost")
    CatBoostRegressor = None   # type: ignore[assignment,misc]
    CatBoostClassifier = None  # type: ignore[assignment,misc]
    Pool = None                # type: ignore[assignment]
    _catboost_available = False
from pydantic import BaseModel, field_validator
try:
    import redis as _redis_module
    _redis_available = True
except ModuleNotFoundError:
    print("[WARN] redis not installed — caching disabled. Run: pip install redis")
    _redis_module = None  # type: ignore[assignment]
    _redis_available = False

# ─── Make repo root importable ────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR_FOR_IMPORT = os.path.dirname(_BACKEND_DIR)
if _ROOT_DIR_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _ROOT_DIR_FOR_IMPORT)

app = FastAPI(title="College Predictor API (Unified)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Base Directories ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# ─── Load JEE Models ──────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT_DIR, "models", "jee")

def load_jee_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"Failed to read parquet from {path}: {e}")
        return pd.DataFrame()

try:
    if not _catboost_available:
        raise ImportError("catboost not installed")
    iit_model = CatBoostClassifier()
    iit_model.load_model(os.path.join(MODELS_DIR, "model_iit.cbm"))
    print("[SUCCESS] IIT model loaded")
except Exception as e:
    print(f"[ERROR] IIT model load failed: {e}")
    iit_model = None

try:
    if not _catboost_available:
        raise ImportError("catboost not installed")
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

# ─── Load NEET Models ─────────────────────────────────────────────────────────
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

# ─── Load NEET PG Model (XGBoost multi-class — retrained for small file size) ──
NEET_PG_MODELS_DIR = os.path.join(ROOT_DIR, "models", "neet_pg")

try:
    import joblib as _joblib
    neet_pg_model       = _joblib.load(os.path.join(NEET_PG_MODELS_DIR, "neet_pg_model.pkl"))
    neet_pg_cat_enc     = _joblib.load(os.path.join(NEET_PG_MODELS_DIR, "neet_pg_category_encoder.pkl"))
    neet_pg_label_enc   = _joblib.load(os.path.join(NEET_PG_MODELS_DIR, "neet_pg_label_encoder.pkl"))
    print(f"[SUCCESS] NEET PG model loaded ({type(neet_pg_model).__name__})")
except Exception as e:
    print(f"[ERROR] NEET PG model load failed: {e}")
    neet_pg_model, neet_pg_cat_enc, neet_pg_label_enc = None, None, None

# ─── Load KCET Models ─────────────────────────────────────────────────────────
KCET_MODELS_DIR = os.path.join(ROOT_DIR, "models", "kcet")

try:
    if not _catboost_available:
        raise ImportError("catboost not installed")
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

# ─── Load COMEDK Models ───────────────────────────────────────────────────────
COMEDK_MODELS_DIR = os.path.join(ROOT_DIR, "models", "comedk")

try:
    if not _catboost_available:
        raise ImportError("catboost not installed")
    comedk_model = CatBoostRegressor()
    comedk_model.load_model(os.path.join(COMEDK_MODELS_DIR, "comedk_model.cbm"))
    with open(os.path.join(COMEDK_MODELS_DIR, "encoders.pkl"), "rb") as f:
        comedk_encoders = pickle.load(f)
    with open(os.path.join(COMEDK_MODELS_DIR, "features.pkl"), "rb") as f:
        comedk_feature_cols = pickle.load(f)
    with open(os.path.join(COMEDK_MODELS_DIR, "lookup.pkl"), "rb") as f:
        comedk_lookup = pickle.load(f)
    print("[SUCCESS] COMEDK model loaded")
except Exception as e:
    print(f"[ERROR] COMEDK model load failed: {e}")
    comedk_model, comedk_encoders, comedk_feature_cols, comedk_lookup = None, {}, [], {}

# ─── Redis (optional) ─────────────────────────────────────────────────────────
try:
    if not _redis_available or _redis_module is None:
        raise ImportError("redis not installed")
    r = _redis_module.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    r.ping()
    print("[SUCCESS] Redis connected")
except Exception as e:
    print(f"[WARN] Redis unavailable (caching disabled): {e}")
    r = None

# ─── Rank Validation Helper ───────────────────────────────────────────────────
def validate_rank(rank: int, field_name: str = "rank", max_rank: int = 5_000_000):
    """Raise HTTPException 422 for invalid ranks."""
    if rank <= 0:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be greater than 0. Got {rank}."
        )
    if rank > max_rank:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} {rank} seems unrealistically high (max {max_rank:,}). Please verify."
        )

# ─── Input Schemas ────────────────────────────────────────────────────────────
class InputData(BaseModel):
    user_rank: int
    exam_type: str
    category: str
    quota: str
    pool: str

    @field_validator("user_rank")
    @classmethod
    def rank_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("user_rank must be greater than 0")
        return v


class NeetInputData(BaseModel):
    candidate_rank: int
    category: str

    @field_validator("candidate_rank")
    @classmethod
    def rank_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("candidate_rank must be greater than 0")
        return v


class NeetPGInputData(BaseModel):
    """Input schema for the /predict/neet_pg endpoint."""
    candidate_rank: int
    category: str

    @field_validator("candidate_rank")
    @classmethod
    def rank_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("candidate_rank must be greater than 0")
        return v


class KcetInputData(BaseModel):
    user_rank: int
    category: str
    base_category: str
    quota: str
    region: str

    @field_validator("user_rank")
    @classmethod
    def rank_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("user_rank must be greater than 0")
        return v


class ComedkInputData(BaseModel):
    user_rank: int
    category: str

    @field_validator("user_rank")
    @classmethod
    def rank_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("user_rank must be greater than 0")
        return v


# ─── JEE Feature Helpers ──────────────────────────────────────────────────────
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


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "PredictMe Unified Predictor API Running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "iit":     iit_model is not None,
            "nit":     nit_model is not None,
            "neet_ug": neet_model is not None,
            "neet_pg": neet_pg_model is not None,
            "kcet":    kcet_model is not None,
            "comedk":  comedk_model is not None,
        },
        "dataset_loaded": not iit_features_df.empty,
        "redis": r is not None,
    }


# ─── Predict JEE ──────────────────────────────────────────────────────────────
@app.post("/predict")
def predict_jee(data: InputData):
    validate_rank(data.user_rank, "JEE Rank", max_rank=1_500_000)

    key = f"jee3_{data.user_rank}_{data.exam_type}_{data.category}_{data.quota}_{data.pool}"
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

        if not _catboost_available or Pool is None:
            return {"source": "error", "error": "catboost not installed. Run: pip install catboost"}
        pool_obj = Pool(X, cat_features=cat_cols)
        probs = model.predict_proba(pool_obj)[:, 1]

        results = filtered[["institute_short", "program_name", "degree_short",
                             "closing_rank_max", "closing_rank_avg",
                             "trend_direction", "difficulty_pct"]].copy()
        results["eligibility_prob"] = probs.round(4)

        for _, row in results.iterrows():
            cutoff = int(row.get("closing_rank_max", 0))
            prob = row["eligibility_prob"]

            # ── Rank logic ────────────────────────────────────────────────────
            # In Indian exams: LOWER rank number = BETTER student.
            # Cutoff rank = worst rank admitted. Student is eligible only if:
            #   student_rank <= cutoff_rank  (student rank is equal or better)
            #
            # Safe  : cutoff is comfortably above user_rank (big positive margin)
            #         → prob >= 0.70 AND cutoff >= user_rank
            # Likely: cutoff is slightly above or right at user_rank (tight margin)
            #         → prob >= 0.40 AND cutoff >= user_rank
            # Exclude: cutoff < user_rank → student rank is worse than cutoff → not eligible
            # ─────────────────────────────────────────────────────────────────
            if cutoff < data.user_rank:
                # College cutoff is worse than student rank → student is NOT eligible
                continue

            if prob >= 0.70:
                tier = "Safe"
            elif prob >= 0.40:
                tier = "Likely"
            else:
                continue

            item = {
                "institute":        row["institute_short"],
                "program":          row["program_name"],
                "program_duration": str(row.get("degree_short", "4 Years")),
                "degree_short":     row["degree_short"],
                "predicted_cutoff": cutoff,
                "tier":             tier,
            }
            structured_json[tier].append(item)

        # Sort ascending by predicted_cutoff (lower cutoff = more prestigious first)
        structured_json["Safe"].sort(key=lambda x: x["predicted_cutoff"])
        structured_json["Likely"].sort(key=lambda x: x["predicted_cutoff"])

    if r:
        r.setex(key, 3600, json.dumps(structured_json))

    return {"source": "model", "data": structured_json}


# ─── Predict NEET PG ──────────────────────────────────────────────────────────
_NEET_PG_LABEL_SEP = "|||"

@app.post("/predict/neet_pg")
def predict_neet_pg(data: NeetPGInputData):
    """
    Predict NEET PG college + course allocations.
    Uses XGBoost multi-class model (~8 MB) trained on Karnataka NEET PG data.

    Tier logic (probability-based):
      Safe   : probability >= 1%
      Likely : probability >= 0.3% AND < 1%

    Response shape
    --------------
    {
        "source": "model",
        "data": {
            "Safe":   [{"institute": ..., "course": ..., "predicted_cutoff": <prob_pct>, "tier": "Safe"},   ...],
            "Likely": [{"institute": ..., "course": ..., "predicted_cutoff": <prob_pct>, "tier": "Likely"}, ...]
        }
    }
    """
    validate_rank(data.candidate_rank, "NEET PG Rank", max_rank=200_000)

    if neet_pg_model is None:
        return {
            "source": "error",
            "error": "NEET PG model not loaded. Run _retrain_neet_pg.py first.",
        }

    # Redis cache key
    cache_key = f"neet_pg3_{data.candidate_rank}_{data.category.upper()}"
    if r:
        cached = r.get(cache_key)
        if cached:
            return {"source": "cache", "data": json.loads(cached)}

    # ── Encode category ───────────────────────────────────────────────────────
    cat_upper  = str(data.category).strip().upper()
    known_cats = list(neet_pg_cat_enc.classes_)
    if cat_upper not in known_cats:
        cat_upper = known_cats[0]

    cat_encoded = int(neet_pg_cat_enc.transform([cat_upper])[0])
    X_input     = np.array([[data.candidate_rank, cat_encoded]])

    # ── Full probability vector over all 787 college+course labels ────────────
    proba = neet_pg_model.predict_proba(X_input)[0]

    SAFE_THRESHOLD   = 0.01    # ≥1%  → Safe
    LIKELY_THRESHOLD = 0.003   # ≥0.3% → Likely

    structured_json: dict = {"Safe": [], "Likely": []}

    for idx, prob in enumerate(proba):
        if prob < LIKELY_THRESHOLD:
            continue

        combined_label = neet_pg_label_enc.classes_[idx]
        if _NEET_PG_LABEL_SEP in combined_label:
            college, course = combined_label.split(_NEET_PG_LABEL_SEP, 1)
        else:
            college, course = combined_label, "Unknown"

        course  = course.replace("\n", " ").strip()
        college = college.strip()

        prob_pct = round(float(prob) * 100, 2)

        item = {
            "institute":        college,
            "course":           course,
            "predicted_cutoff": prob_pct,
            "tier":             "Safe" if prob >= SAFE_THRESHOLD else "Likely",
        }
        structured_json[item["tier"]].append(item)

    # Sort each tier descending by probability (highest confidence first)
    structured_json["Safe"].sort(key=lambda x: x["predicted_cutoff"], reverse=True)
    structured_json["Likely"].sort(key=lambda x: x["predicted_cutoff"], reverse=True)

    if r:
        r.setex(cache_key, 3600, json.dumps(structured_json))

    return {"source": "model", "data": structured_json}


# ─── Predict NEET UG ──────────────────────────────────────────────────────────
def _neet_ug_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    The NEET UG training data contains duplicate/truncated institute names
    (e.g. 'JHALAWAR' and 'JHALAWAR MEDICAL COLLEGE,' both appear).
    Keep only the longest name per predicted-cutoff group so the UI shows
    clean, unambiguous names.  Within a tied cutoff bucket, the longest
    string is the most complete name.
    """
    # Round cutoffs to the nearest 100 so near-identical predictions cluster
    df = df.copy()
    df["_bucket"] = (df["pred_closing_rank"] / 100).round() * 100

    # Within each bucket keep the row with the longest institute name
    df["_namelen"] = df["institute"].str.len()
    df = (
        df.sort_values("_namelen", ascending=False)
          .drop_duplicates(subset=["_bucket"])
          .drop(columns=["_bucket", "_namelen"])
          .reset_index(drop=True)
    )
    return df


@app.post("/predict/neet")
def predict_neet(data: NeetInputData):
    validate_rank(data.candidate_rank, "NEET Rank", max_rank=2_000_000)

    key = f"neet3_{data.candidate_rank}_{data.category}"
    if r:
        cached = r.get(key)
        if cached:
            return {"source": "cache", "data": json.loads(cached)}

    if neet_model is None:
        return {"source": "error", "error": "NEET UG model not loaded. Check server logs."}

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

    # ── Rank logic ────────────────────────────────────────────────────────────
    # NEET UG: LOWER rank number = BETTER student.
    # Cutoff = last (worst) rank admitted to that institute.
    # A student qualifies when: student_rank <= cutoff_rank.
    #
    # Safe  : cutoff >= student_rank + safe_margin
    #         → lots of headroom, very comfortable admission
    # Likely: cutoff >= student_rank AND cutoff < student_rank + safe_margin
    #         → student just makes the cut, borderline
    # Excluded: cutoff < student_rank → student rank is worse → not eligible
    #
    # Sort: DESCENDING by cutoff so the most lenient (easiest-to-get-into)
    # colleges appear first — these are the "safest bets" the user should see
    # at the top.  Most-competitive (AIIMS Delhi, rank ~1) will appear last.
    # ─────────────────────────────────────────────────────────────────────────
    safe_margin = 1500

    df_safe = df_raw[df_raw["pred_closing_rank"] >= (data.candidate_rank + safe_margin)].copy()
    df_likely = df_raw[
        (df_raw["pred_closing_rank"] >= data.candidate_rank) &
        (df_raw["pred_closing_rank"] < (data.candidate_rank + safe_margin))
    ].copy()

    # Deduplicate near-identical institute entries before building the response
    df_safe   = _neet_ug_deduplicate(df_safe)
    df_likely = _neet_ug_deduplicate(df_likely)

    structured_json = {"Safe": [], "Likely": []}

    # Sort descending: highest cutoff (most lenient) first = safest colleges on top
    for _, row in df_safe.sort_values("pred_closing_rank", ascending=False).iterrows():
        structured_json["Safe"].append({
            "institute":        row["institute"],
            "predicted_cutoff": int(row["pred_closing_rank"]),
            "tier":             "Safe",
            "course":           "MBBS",
        })

    for _, row in df_likely.sort_values("pred_closing_rank", ascending=False).iterrows():
        structured_json["Likely"].append({
            "institute":        row["institute"],
            "predicted_cutoff": int(row["pred_closing_rank"]),
            "tier":             "Likely",
            "course":           "MBBS",
        })

    if r:
        r.setex(key, 3600, json.dumps(structured_json))

    return {"source": "model", "data": structured_json}


# ─── Predict KCET ─────────────────────────────────────────────────────────────
@app.post("/predict/kcet")
def predict_kcet(data: KcetInputData):
    validate_rank(data.user_rank, "KCET Rank", max_rank=300_000)

    key = f"kcet_{data.user_rank}_{data.category}_{data.base_category}_{data.quota}_{data.region}"
    if r:
        cached = r.get(key)
        if cached:
            return {"source": "cache", "data": json.loads(cached)}

    if kcet_model is None:
        return {"source": "error", "error": "KCET model not loaded. Check server logs."}

    colleges = kcet_encoders.get("college_name", [])
    courses  = kcet_encoders.get("course_name", [])

    if not colleges or not courses:
        return {"source": "error", "error": "No college/course encoding found in kcet_encoders."}

    df_grid = pd.MultiIndex.from_product(
        [colleges, courses], names=["college_name", "course_name"]
    ).to_frame(index=False)

    df_grid["category"]      = data.category.upper()
    df_grid["base_category"] = data.base_category.upper()
    df_grid["quota"]         = data.quota.upper()
    df_grid["region"]        = data.region.upper()
    df_grid["year"]          = 2026

    X = df_grid[kcet_feature_cols]
    log_preds = kcet_model.predict(X)
    pred_ranks = np.expm1(log_preds)
    df_grid["pred_closing_rank"] = pred_ranks

    # ── Rank logic ────────────────────────────────────────────────────────────
    # student_rank <= cutoff → eligible
    # Safe  : cutoff > user_rank + safety margin
    # Likely: cutoff >= user_rank AND cutoff <= user_rank + safety margin
    # Exclude: cutoff < user_rank → NOT eligible
    # ─────────────────────────────────────────────────────────────────────────
    safe_margin = 5000

    df_safe = df_grid[df_grid["pred_closing_rank"] >= (data.user_rank + safe_margin)]
    df_likely = df_grid[
        (df_grid["pred_closing_rank"] >= data.user_rank) &
        (df_grid["pred_closing_rank"] < (data.user_rank + safe_margin))
    ]

    structured_json = {"Safe": [], "Likely": []}

    # Sort ascending: lower cutoff = more prestigious first
    for _, row in df_safe.sort_values("pred_closing_rank", ascending=True).iterrows():
        structured_json["Safe"].append({
            "institute":        row["college_name"],
            "predicted_cutoff": int(row["pred_closing_rank"]),
            "tier":             "Safe",
            "course":           row["course_name"],
        })

    for _, row in df_likely.sort_values("pred_closing_rank", ascending=True).iterrows():
        structured_json["Likely"].append({
            "institute":        row["college_name"],
            "predicted_cutoff": int(row["pred_closing_rank"]),
            "tier":             "Likely",
            "course":           row["course_name"],
        })

    if r:
        r.setex(key, 3600, json.dumps(structured_json))

    return {"source": "model", "data": structured_json}


# ─── Predict COMEDK ───────────────────────────────────────────────────────────
def build_comedk_prediction_df(encoders, feature_cols, lookup, category):
    rows = []
    for (college, course, cat), hist in lookup.items():
        if cat != category:
            continue
        if len(hist) == 0:
            continue

        rows.append({
            "college_name":           college,
            "course_name":            course,
            "category":               category,
            "year":                   2026,
            "prev_year_closing_rank": hist[-1],
            "closing_rank_mean":      np.mean(hist),
            "closing_rank_std":       np.std(hist),
            "closing_rank_min":       min(hist),
            "closing_rank_max":       max(hist),
            "rank_trend":             max(hist) - min(hist),
            "years_available":        len(hist),
            "latest_year":            2025,
            "earliest_year":          2026 - len(hist),
            "college_avg_rank":       np.mean(hist),
            "category_avg_rank":      np.mean(hist),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return None, None

    raw_rows = df.copy()
    for col in ["college_name", "course_name", "category"]:
        le = encoders[col]
        df[col] = df[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    df = df[(df["college_name"] != -1) &
            (df["course_name"]  != -1) &
            (df["category"]     != -1)]

    raw_rows = raw_rows.loc[df.index].reset_index(drop=True)
    df = df.reset_index(drop=True)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols], raw_rows


@app.post("/predict/comedk")
def predict_comedk(data: ComedkInputData):
    validate_rank(data.user_rank, "COMEDK Rank", max_rank=200_000)

    key = f"comedk_{data.user_rank}_{data.category}"
    if r:
        cached = r.get(key)
        if cached:
            return {"source": "cache", "data": json.loads(cached)}

    if comedk_model is None or not comedk_lookup:
        return {"source": "error", "error": "COMEDK model/lookup not loaded. Check server logs."}

    X, raw_rows = build_comedk_prediction_df(
        comedk_encoders, comedk_feature_cols, comedk_lookup, data.category.upper()
    )

    if X is None:
        return {"source": "error", "error": "No historical data found for the given category."}

    log_preds = comedk_model.predict(X)
    pred_ranks = np.expm1(log_preds).astype(int)
    raw_rows["pred_closing_rank"] = pred_ranks

    # ── Rank logic ────────────────────────────────────────────────────────────
    # student_rank <= cutoff → eligible
    # Safe  : cutoff > user_rank + safety margin (comfortable room)
    # Likely: cutoff >= user_rank AND cutoff <= user_rank + safety margin (tight)
    # Exclude: cutoff < user_rank → NOT eligible
    # ─────────────────────────────────────────────────────────────────────────
    safe_margin = 1000

    df_safe = raw_rows[raw_rows["pred_closing_rank"] >= (data.user_rank + safe_margin)]
    df_likely = raw_rows[
        (raw_rows["pred_closing_rank"] >= data.user_rank) &
        (raw_rows["pred_closing_rank"] < (data.user_rank + safe_margin))
    ]

    structured_json = {"Safe": [], "Likely": []}

    # Sort ascending: lower cutoff = more prestigious first
    for _, row in df_safe.sort_values("pred_closing_rank", ascending=True).iterrows():
        structured_json["Safe"].append({
            "institute":        row["college_name"],
            "predicted_cutoff": int(row["pred_closing_rank"]),
            "tier":             "Safe",
            "course":           row["course_name"],
        })

    for _, row in df_likely.sort_values("pred_closing_rank", ascending=True).iterrows():
        structured_json["Likely"].append({
            "institute":        row["college_name"],
            "predicted_cutoff": int(row["pred_closing_rank"]),
            "tier":             "Likely",
            "course":           row["course_name"],
        })

    if r:
        r.setex(key, 3600, json.dumps(structured_json))

    return {"source": "model", "data": structured_json}
