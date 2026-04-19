import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
MODEL_DIR = "/app/models/neet"
CAT_COLS = ["institute", "category"]
NUM_COLS = ["year"]
TARGET = "closing_rank"

def load_data():
    paths = ["/app/data/processed/neet_features", "data/processed/neet_features"]
    path = next((p for p in paths if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError("Could not find neet_features parquet folder.")
        
    # Utilize PySpark to bridge Parquet seamlessly without missing dependencies 
    spark = SparkSession.builder.appName("NEET_ML_Train").getOrCreate()
    df_spark = spark.read.parquet(path)
    df = df_spark.toPandas()
    spark.stop()
    
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET, "institute", "category"])
    df = df[df[TARGET] > 0]
    return df

def encode_categoricals(df, encoders=None, fit=True):
    df = df.copy()
    if encoders is None: encoders = {}
    for col in CAT_COLS:
        df[col] = df[col].astype(str).str.strip().str.upper()
        if fit:
            le = LabelEncoder()
            all_vals = list(df[col].unique()) + ["__UNKNOWN__"]
            le.fit(all_vals)
            df[col] = le.transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "__UNKNOWN__")
            df[col] = le.transform(df[col])
    return df, encoders

def tune_xgboost(X_train, y_train, n_trials=10):
    # Reduced trials temporarily for rapid prototyping vs 50 trials
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "random_state": 42
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            model = xgb.XGBRegressor(**params)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = model.predict(X_train.iloc[val_idx])
            scores.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], preds)))
        return np.mean(scores)
    
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def main():
    print("Loading NEET Features Parquet...")
    df = load_data()
    print(f"Successfully Loaded {len(df):,} strictly defined feature bounds.")
    
    # As instructed, utilizing log-transformation on closing_rank to prevent model variance spikes
    y = np.log1p(df[TARGET])
    
    df, encoders = encode_categoricals(df, fit=True)
    X = df[CAT_COLS + NUM_COLS]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    print("Running Optuna parameter estimation...")
    best_params = tune_xgboost(X_train, y_train, n_trials=30)
    
    print("Deploying best structural gradient boosting trees...")
    params = {"objective": "reg:squarederror", "random_state": 42, **best_params}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Reverse log to gauge human real performance indices
    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"✅ Final Test Evaluation RMSE: {rmse:.1f} Ranks\n")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/neet_model.pkl", "wb") as f: pickle.dump(model, f)
    with open(f"{MODEL_DIR}/neet_encoders.pkl", "wb") as f: pickle.dump(encoders, f)
    with open(f"{MODEL_DIR}/neet_feature_cols.pkl", "wb") as f: pickle.dump(CAT_COLS + NUM_COLS, f)
    print("Exported securely to /models/ !")

if __name__ == "__main__":
    main()
