import os
import sys
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

MODEL_DIR = "models/comedk"
DATA_PATH = "data/processed/comedk_features"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "/app/data/processed/comedk_features"

TARGET = "closing_rank"

def evaluate_model():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Data not found at {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET, "college_name", "course_name", "category"])
    df = df[df[TARGET] > 0]

    print("Loading model and encoders...")
    model_path = os.path.join(MODEL_DIR, "comedk_model.cbm")
    features_path = os.path.join(MODEL_DIR, "comedk_feature_cols.pkl")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model = CatBoostRegressor()
    model.load_model(model_path)

    with open(features_path, "rb") as f:
        feature_cols = pickle.load(f)

    # Note: the dataset might have missing columns, we should ensure they exist
    X = df[feature_cols]
    y_true = df[TARGET].values

    print("Making predictions...")
    log_preds = model.predict(X)
    y_pred = np.expm1(log_preds)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    within_500 = np.mean(np.abs(y_true - y_pred) <= 500) * 100
    within_1000 = np.mean(np.abs(y_true - y_pred) <= 1000) * 100

    print(f"\nEvaluation Results:")
    print(f"RMSE          : {rmse:.1f}")
    print(f"MAE           : {mae:.1f}")
    print(f"R-squared     : {r2:.3f}")
    print(f"Within ±500   : {within_500:.1f}%")
    print(f"Within ±1000  : {within_1000:.1f}%")

if __name__ == "__main__":
    evaluate_model()
