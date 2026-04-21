import sys
import os
import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import redis
import json

MODEL_DIR = "models/kcet"
CAT_COLS = ["college_name", "course_name", "category", "base_category", "quota", "region"]
NUM_COLS = ["year"]

def get_redis_client():
    try:
        host = os.environ.get("REDIS_HOST", "localhost")
        return redis.Redis(host=host, port=6379, db=0, decode_responses=True)
    except Exception:
        return None

def load_models():
    paths = [f"{MODEL_DIR}/kcet_model.cbm", f"/app/{MODEL_DIR}/kcet_model.cbm"]
    m_path = next((p for p in paths if os.path.exists(p)), None)
    if not m_path: 
        raise FileNotFoundError("Model not found. Ensure you execute python ML/kcet/train.py first!")
    
    e_path = m_path.replace("kcet_model.cbm", "kcet_encoders.pkl")
    c_path = m_path.replace("kcet_model.cbm", "kcet_feature_cols.pkl")
    
    model = CatBoostRegressor()
    model.load_model(m_path)
    
    with open(e_path, "rb") as f: encoders = pickle.load(f)
    with open(c_path, "rb") as f: feature_cols = pickle.load(f)
    
    return model, encoders, feature_cols

def predict_colleges(candidate_rank, category, base_category, quota, region):
    cache = get_redis_client()
    cache_key = f"kcet:{candidate_rank}:{category}:{base_category}:{quota}:{region}"
    
    if cache:
        try:
            cached_result = cache.get(cache_key)
            if cached_result:
                print("\n[CACHE HIT] Returning rapid cached predictions via Redis")
                print(cached_result)
                return
        except Exception:
            pass

    model, encoders, feature_cols = load_models()
    
    colleges = encoders["college_name"]
    courses = encoders["course_name"]
    
    print(f"Generating mappings for {len(colleges)} colleges and {len(courses)} courses...")
    
    df_grid = pd.MultiIndex.from_product([colleges, courses], names=["college_name", "course_name"]).to_frame(index=False)
    
    df_grid["category"] = category.upper()
    df_grid["base_category"] = base_category.upper()
    df_grid["quota"] = quota.upper()
    df_grid["region"] = region.upper()
    df_grid["year"] = 2026
    
    X = df_grid[feature_cols]
    
    log_preds = model.predict(X)
    pred_ranks = np.expm1(log_preds)
    
    df_grid["pred_closing_rank"] = pred_ranks
    
    df_safe = df_grid[df_grid["pred_closing_rank"] > candidate_rank]
    df_likely = df_grid[(df_grid["pred_closing_rank"] > (candidate_rank - 5000)) & 
                        (df_grid["pred_closing_rank"] <= candidate_rank)]
                        
    output = []
    output.append(f"\n======================================")
    output.append(f" KCET Predictor Analytical Outcome     ")
    output.append(f" Rank Constraint: {candidate_rank}")
    output.append(f" Constraints: {category.upper()}, {base_category.upper()}, {quota.upper()}, {region.upper()}")
    output.append(f"======================================\n")
    
    output.append(f"---- [ ✔ ] SAFE COLLEGES ({len(df_safe)} matches) ----")
    for _, row in df_safe.sort_values("pred_closing_rank", ascending=False).head(15).iterrows():
        output.append(f"   ► {row['college_name']} - {row['course_name']} (Projected Cutoff: {int(row['pred_closing_rank'])})")
        
    output.append(f"\n---- [ ✨ ] LIKELY COLLEGES ({len(df_likely)} matches) ----")
    for _, row in df_likely.sort_values("pred_closing_rank", ascending=False).head(15).iterrows():
        output.append(f"   ► {row['college_name']} - {row['course_name']} (Projected Cutoff: {int(row['pred_closing_rank'])})")
        
    output.append("\n*Note: Output is truncated to maximum top 15 results per bucket.")
    
    final_output = "\n".join(output)
    print(final_output)
    
    if cache:
        try:
            cache.setex(cache_key, 3600, final_output)
        except Exception:
            pass

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python predict.py <CANDIDATE_RANK> <CATEGORY> <BASE_CATEGORY> <QUOTA> <REGION>")
        print("Example: python predict.py 50000 2AK 2A Kannada General")
        sys.exit(1)
        
    rank = int(sys.argv[1])
    cat = sys.argv[2]
    base_cat = sys.argv[3]
    quota = sys.argv[4]
    region = sys.argv[5]
    
    predict_colleges(rank, cat, base_cat, quota, region)
