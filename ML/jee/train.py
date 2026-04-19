"""
train.py — JEE College Eligibility Prediction Model
=====================================================
Trains separate models for IIT (JEE Advanced) and NIT (JEE Mains).
Predicts closing rank distributions to determine eligible colleges/branches.

Input : data/processed/features.csv
Output: models/iit_model.pkl
        models/nit_model.pkl
        models/iit_encoders.pkl
        models/nit_encoders.pkl
        models/feature_metadata.pkl
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

DATA_PATH   = "data/processed/features.csv"
MODEL_DIR   = "models"

# Columns that exist in your schema
INSTITUTE_TYPE_COL = "institute_type"   # "IIT" or "NIT"
INSTITUTE_COL      = "institute_short"  # or full name — whichever is in features.csv
PROGRAM_COL        = "program_name"
CATEGORY_COL       = "category"
QUOTA_COL          = "quota"
POOL_COL           = "pool"
ROUND_COL          = "round_no"
DURATION_COL       = "program_duration"
DEGREE_COL         = "degree_short"
OPEN_RANK_COL      = "opening_rank"
CLOSE_RANK_COL     = "closing_rank"     # raw closing rank (target)
YEAR_COL           = "year"             # or "latest_recorded_year" if aggregated

# Categorical columns for encoding
CAT_COLS = [
    INSTITUTE_COL,
    PROGRAM_COL,
    CATEGORY_COL,
    QUOTA_COL,
    POOL_COL,
    DEGREE_COL,
]

# Numeric columns used as features
NUM_COLS = [
    ROUND_COL,
    DURATION_COL,
    "opening_rank",           # strong signal: where the seat opens
]


# ─────────────────────────────────────────────
# 1. Load & Validate Data
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"features.csv not found at: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("features.csv is empty.")

    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    # Coerce numeric targets
    df[CLOSE_RANK_COL] = pd.to_numeric(df[CLOSE_RANK_COL], errors="coerce")
    df[OPEN_RANK_COL]  = pd.to_numeric(df[OPEN_RANK_COL],  errors="coerce")

    # Drop invalid targets
    before = len(df)
    df = df.dropna(subset=[CLOSE_RANK_COL, OPEN_RANK_COL])
    df = df[(df[CLOSE_RANK_COL] > 0) & (df[OPEN_RANK_COL] > 0)]
    print(f"Dropped {before - len(df):,} rows with invalid ranks.")

    # Normalise institute_type column
    if INSTITUTE_TYPE_COL in df.columns:
        df[INSTITUTE_TYPE_COL] = df[INSTITUTE_TYPE_COL].str.upper().str.strip()

    return df


# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that help the model."""

    # Rank spread within a row (how competitive is this seat?)
    df["rank_spread"] = df[CLOSE_RANK_COL] - df[OPEN_RANK_COL]

    # Is this a dual-degree / integrated programme?
    df["is_dual_degree"] = df[DEGREE_COL].str.contains(
        r"IDD|M\.Tech|MSc|Dual", case=False, na=False
    ).astype(int)

    # Home-state flag (HS quota → 1, everything else → 0)
    df["is_home_state"] = (df[QUOTA_COL].str.upper() == "HS").astype(int)

    # Female-only pool flag
    df["is_female_pool"] = (
        df[POOL_COL].str.lower().str.contains("female", na=False)
    ).astype(int)

    # PWD flag derived from category
    df["is_pwd"] = df[CATEGORY_COL].str.contains("PWD", na=False).astype(int)

    # EWS flag
    df["is_ews"] = df[CATEGORY_COL].str.contains("EWS", na=False).astype(int)

    # Round normalised (higher round → seats are harder to fill)
    df["round_norm"] = df[ROUND_COL] / 7.0

    # Year offset (trend over years — more recent data weighs more implicitly)
    if YEAR_COL in df.columns:
        df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
        min_year = df[YEAR_COL].min()
        df["year_offset"] = df[YEAR_COL] - min_year
    else:
        df["year_offset"] = 0

    return df


ENGINEERED_NUM_COLS = [
    "rank_spread",
    "is_dual_degree",
    "is_home_state",
    "is_female_pool",
    "is_pwd",
    "is_ews",
    "round_norm",
    "year_offset",
]


# ─────────────────────────────────────────────
# 3. Encoding
# ─────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Label-encode categorical columns.
    If fit=True, create and fit new encoders.
    If fit=False, reuse existing encoders (for inference).
    Returns (transformed_df, encoders_dict).
    """
    if encoders is None:
        encoders = {}

    df = df.copy()

    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = "UNKNOWN"

        df[col] = df[col].astype(str).str.strip()

        if fit:
            le = LabelEncoder()
            # Add a sentinel for unknown values seen at inference
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


# ─────────────────────────────────────────────
# 4. Build Feature Matrix
# ─────────────────────────────────────────────

ALL_FEATURE_COLS = CAT_COLS + NUM_COLS + ENGINEERED_NUM_COLS

def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """Returns X (DataFrame), y (Series in log space)."""
    available = [c for c in ALL_FEATURE_COLS if c in df.columns]
    missing   = [c for c in ALL_FEATURE_COLS if c not in df.columns]

    if missing:
        print(f"  ⚠ Missing feature columns (will be skipped): {missing}")

    X = df[available].copy()

    # Coerce all numerics
    for col in available:
        if col in NUM_COLS + ENGINEERED_NUM_COLS:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    y = np.log1p(df[CLOSE_RANK_COL])   # log transform → more normal distribution

    return X, y, available


# ─────────────────────────────────────────────
# 5. Optuna Hyper-parameter Search
# ─────────────────────────────────────────────

def tune_xgboost(X_train, y_train, n_trials: int = 50) -> dict:
    """Run Optuna TPE search and return best hyper-params."""

    def objective(trial):
        params = {
            "objective":        "reg:squarederror",
            "tree_method":      "hist",
            "max_depth":        trial.suggest_int("max_depth", 4, 10),
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "random_state":     42,
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, val_idx in kf.split(X_train):
            Xtr, Xvl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            ytr, yvl = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = xgb.XGBRegressor(**params, verbosity=0)
            model.fit(
                Xtr, ytr,
                eval_set=[(Xvl, yvl)],
                verbose=False,
            )
            preds = model.predict(Xvl)
            rmse_scores.append(np.sqrt(mean_squared_error(yvl, preds)))

        return np.mean(rmse_scores)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best CV log-RMSE : {study.best_value:.5f}")
    return study.best_params


# ─────────────────────────────────────────────
# 6. Train Final Model
# ─────────────────────────────────────────────

def train_model(X_train, y_train, best_params: dict) -> xgb.XGBRegressor:
    params = {
        "objective":        "reg:squarederror",
        "tree_method":      "hist",
        "random_state":     42,
        "verbosity":        0,
        **best_params
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# 7. Evaluate
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test, label: str):
    pred_log = model.predict(X_test)
    y_pred   = np.expm1(pred_log)
    y_true   = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    # Rank-bucket accuracy: how often is the prediction within ±X% of true rank?
    pct_errors = np.abs(y_pred - y_true) / np.maximum(y_true, 1)
    acc_5pct  = (pct_errors <= 0.05).mean() * 100
    acc_10pct = (pct_errors <= 0.10).mean() * 100
    acc_20pct = (pct_errors <= 0.20).mean() * 100

    print(f"\n  [{label}] Test-set metrics")
    print(f"    RMSE                    : {rmse:>10.2f}")
    print(f"    MAE                     : {mae:>10.2f}")
    print(f"    Within  5% of true rank : {acc_5pct:>9.1f}%")
    print(f"    Within 10% of true rank : {acc_10pct:>9.1f}%")
    print(f"    Within 20% of true rank : {acc_20pct:>9.1f}%")

    # Top feature importances
    feat_imp = pd.Series(model.feature_importances_, index=X_test.columns)
    top5 = feat_imp.nlargest(5)
    print(f"\n    Top-5 feature importances:")
    for feat, imp in top5.items():
        print(f"      {feat:<30s} {imp:.4f}")


# ─────────────────────────────────────────────
# 8. Save Artifacts
# ─────────────────────────────────────────────

def save_artifacts(model, encoders, feature_cols, name: str):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path    = f"{MODEL_DIR}/{name}_model.pkl"
    encoder_path  = f"{MODEL_DIR}/{name}_encoders.pkl"
    metadata_path = f"{MODEL_DIR}/{name}_feature_cols.pkl"

    with open(model_path, "wb")    as f: pickle.dump(model,        f)
    with open(encoder_path, "wb")  as f: pickle.dump(encoders,     f)
    with open(metadata_path, "wb") as f: pickle.dump(feature_cols, f)

    print(f"\n  Saved → {model_path}")
    print(f"  Saved → {encoder_path}")
    print(f"  Saved → {metadata_path}")


# ─────────────────────────────────────────────
# 9. Main Pipeline
# ─────────────────────────────────────────────

def run_pipeline(df_subset: pd.DataFrame, name: str, n_optuna_trials: int = 50):
    """Full train pipeline for a single institute type (IIT or NIT)."""

    print(f"\n{'='*60}")
    print(f"  Training model : {name.upper()}  ({len(df_subset):,} rows)")
    print(f"{'='*60}")

    # Feature engineering
    df_subset = engineer_features(df_subset)

    # Encode categoricals
    df_subset, encoders = encode_categoricals(df_subset, fit=True)

    # Build matrices
    X, y, feature_cols = build_feature_matrix(df_subset)

    # Drop remaining NaNs
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    print(f"  Clean rows after NA drop : {len(X):,}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    # Hyper-parameter search
    print(f"\n  Running Optuna ({n_optuna_trials} trials) ...")
    best_params = tune_xgboost(X_train, y_train, n_trials=n_optuna_trials)

    # Final model
    print("  Training final model ...")
    model = train_model(X_train, y_train, best_params)

    # Evaluate
    evaluate(model, X_test, y_test, label=name)

    # Save
    save_artifacts(model, encoders, feature_cols, name=name.lower())

    return model, encoders, feature_cols


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    print("JEE College Eligibility — Model Training")
    print("─" * 60)

    # ── Load ──────────────────────────────────
    df = load_data(DATA_PATH)

    # ── Split IIT vs NIT ──────────────────────
    if INSTITUTE_TYPE_COL in df.columns:
        df_iit = df[df[INSTITUTE_TYPE_COL] == "IIT"].copy()
        df_nit = df[df[INSTITUTE_TYPE_COL] == "NIT"].copy()
    else:
        # Fallback: infer from institute name
        iit_mask = df[INSTITUTE_COL].str.upper().str.contains("IIT", na=False)
        df_iit   = df[iit_mask].copy()
        df_nit   = df[~iit_mask].copy()

    print(f"IIT rows : {len(df_iit):,}   |   NIT rows : {len(df_nit):,}")

    if len(df_iit) < 100:
        print("⚠  Too few IIT rows — skipping IIT model.")
    else:
        run_pipeline(df_iit, name="iit", n_optuna_trials=50)

    if len(df_nit) < 100:
        print("⚠  Too few NIT rows — skipping NIT model.")
    else:
        run_pipeline(df_nit, name="nit", n_optuna_trials=50)

    # ── Save shared metadata (label maps) ─────
    print("\n  Saving feature metadata ...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    metadata = {
        "cat_cols":  CAT_COLS,
        "num_cols":  NUM_COLS + ENGINEERED_NUM_COLS,
        "all_cols":  ALL_FEATURE_COLS,
        "target":    CLOSE_RANK_COL,
    }
    with open(f"{MODEL_DIR}/feature_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("\n✅  Training complete.")
    print(f"    All artifacts in : ./{MODEL_DIR}/")


if __name__ == "__main__":
    main()