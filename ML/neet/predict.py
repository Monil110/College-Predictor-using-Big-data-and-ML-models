import sys
import os
import pickle
import pandas as pd
import numpy as np

MODEL_DIR = "models/neet"
CAT_COLS = ["institute", "category"]
NUM_COLS = ["year"]

def load_models():
    # Supports both Native and Docker executions
    paths = [f"{MODEL_DIR}/neet_model.pkl", f"/app/{MODEL_DIR}/neet_model.pkl"]
    m_path = next((p for p in paths if os.path.exists(p)), None)
    if not m_path: 
        raise FileNotFoundError("Model not found. Ensure you execute python ML/neet/train.py first!")
    
    e_path = m_path.replace("neet_model.pkl", "neet_encoders.pkl")
    c_path = m_path.replace("neet_model.pkl", "neet_feature_cols.pkl")
    
    with open(m_path, "rb") as f: model = pickle.load(f)
    with open(e_path, "rb") as f: encoders = pickle.load(f)
    with open(c_path, "rb") as f: feature_cols = pickle.load(f)
    return model, encoders, feature_cols

def predict_colleges(candidate_rank, candidate_category):
    model, encoders, feature_cols = load_models()
    
    # We acquire the known dictionary of colleges from the ML label encoders natively
    institutes = [i for i in encoders["institute"].classes_ if i != "__UNKNOWN__"]
    
    # Construct inference mappings for ALL available institutes for the users constraints
    df = pd.DataFrame({"institute": institutes})
    df["category"] = candidate_category
    df["year"] = 2026 # Target upcoming cycle
    
    df_raw = df.copy()
    
    # Structurally encode inputs to match mapping logic
    for col in CAT_COLS:
        le = encoders[col]
        known = set(le.classes_)
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].apply(lambda x: x if x in known else "__UNKNOWN__")
        df[col] = le.transform(df[col])
        
    X = df[feature_cols]
    
    # Unpack Logarithm arrays 
    log_preds = model.predict(X)
    pred_ranks = np.expm1(log_preds)
    
    df_raw["pred_closing_rank"] = pred_ranks
    
    # Filter by user definition requests
    df_safe = df_raw[df_raw["pred_closing_rank"] > candidate_rank]
    
    df_likely = df_raw[(df_raw["pred_closing_rank"] > (candidate_rank - 1500)) & 
                       (df_raw["pred_closing_rank"] <= candidate_rank)]
                       
    print(f"\n======================================")
    print(f" NEET Predictor Analytical Outcome     ")
    print(f" Rank Constraint: {candidate_rank}")
    print(f" Target Category: {candidate_category.upper()}")
    print(f"======================================\n")
    
    print(f"---- [ ✔ ] SAFE COLLEGES ({len(df_safe)} matches) ----")
    for _, row in df_safe.sort_values("pred_closing_rank", ascending=False).head(15).iterrows():
        print(f"   ► {row['institute']} (Projected Cutoff: {int(row['pred_closing_rank'])})")
        
    print(f"\n---- [ ✨ ] LIKELY COLLEGES ({len(df_likely)} matches) ----")
    for _, row in df_likely.sort_values("pred_closing_rank", ascending=False).head(15).iterrows():
        print(f"   ► {row['institute']} (Projected Cutoff: {int(row['pred_closing_rank'])})")
        
    print("\n*Note: Output is truncated to maximum top 15 results per bucket.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <CANDIDATE_RANK> <CATEGORY>")
        print("Example: python predict.py 5000 OBC")
        sys.exit(1)
        
    predict_colleges(int(sys.argv[1]), sys.argv[2])
