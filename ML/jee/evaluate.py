import pickle
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

def evaluate_both_models():
    print("Initiating model evaluation pipeline...")
    
    # Check if models exist
    if not os.path.exists("models/feature_metadata.pkl"):
        print("Models not found. Please run train.py first.")
        return
        
    try:
        df = pd.read_csv("data/processed/features.csv")
    except Exception as e:
        print(f"Failed to load features.csv: {e}")
        return
        
    if "institute_type" in df.columns:
        df["institute_type"] = df["institute_type"].astype(str).str.upper().str.strip()
    else:
        df["institute_type"] = np.where(df["institute_short"].str.upper().str.contains("IIT"), "IIT", "NIT")
        
    # Helper functions inline to avoid import issues
    def engineer_features(req_df: pd.DataFrame) -> pd.DataFrame:
        req_df = req_df.copy()
        req_df["rank_spread"] = req_df["closing_rank"] - req_df["opening_rank"]
        req_df["is_dual_degree"] = req_df["degree_short"].str.contains(r"IDD|M\.Tech|MSc|Dual", case=False, na=False).astype(int)
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

    with open("models/feature_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    for inst_type, prefix in [("IIT", "iit"), ("NIT", "nit")]:
        try:
            with open(f"models/{prefix}_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open(f"models/{prefix}_encoders.pkl", "rb") as f:
                encoders = pickle.load(f)
        except Exception as e:
            print(f"Skipping {inst_type} evaluation: {e}")
            continue
            
        print(f"\nLoaded {inst_type} model successfully.")
        print(f"Evaluating {inst_type} tracking metrics on the full historical dataset...")
        
        temp = df[df["institute_type"] == inst_type].copy()
        temp = temp.dropna(subset=["closing_rank", "opening_rank"])
        temp = temp[(temp["closing_rank"] > 0) & (temp["opening_rank"] > 0)]
        
        if temp.empty:
            print(f"No valid data for {inst_type}.")
            continue
            
        temp = engineer_features(temp)
        
        X_dict = {}
        for col in metadata.get("all_cols", []):
            if col in metadata.get("cat_cols", []):
                X_dict[col] = temp[col].apply(lambda x: encode(encoders, col, x))
            else:
                X_dict[col] = pd.to_numeric(temp[col], errors="coerce").fillna(0)
                
        X = pd.DataFrame(X_dict)
        y_true = temp["closing_rank"].values
        
        # Predict
        pred_log = model.predict(X)
        y_pred = np.expm1(pred_log)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        pct_errors = np.abs(y_pred - y_true) / np.maximum(y_true, 1)
        acc_10pct = (pct_errors <= 0.10).mean() * 100
        acc_20pct = (pct_errors <= 0.20).mean() * 100
        
        print("---------------------------------")
        print(f"{inst_type} Model Evaluation Report:")
        print(f"- R2 Score         : {r2:.4f}")
        print(f"- Root Mean Square : {rmse:.2f} ranks")
        print(f"- Mean Abs Error   : {mae:.2f} ranks")
        print(f"- Within 10% error : {acc_10pct:.1f}% of actual cutoffs")
        print(f"- Within 20% error : {acc_20pct:.1f}% of actual cutoffs")
        print("---------------------------------")

if __name__ == "__main__":
    evaluate_both_models()
