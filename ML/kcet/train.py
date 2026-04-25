import os
import sys
import time
import pickle
import threading
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = "models/kcet"

CAT_COLS = [
    "college_name",
    "course_name",
    "category",
    "base_category",
    "quota",
    "region",
]

NUM_COLS = ["year"]

TARGET = "closing_rank"

DATA_PATHS = [
    "data/processed/kcet_features",
    "/app/data/processed/kcet_features",
]


# ─── Progress Utilities ───────────────────────────────────────────────────────

class ProgressBar:
    """Simple terminal progress bar with percentage and elapsed time."""

    BAR_WIDTH = 40

    def __init__(self, total: int, label: str = ""):
        self.total = max(total, 1)
        self.label = label
        self.current = 0
        self._start = time.time()

    def update(self, n: int = 1, suffix: str = ""):
        self.current = min(self.current + n, self.total)
        self._render(suffix)

    def set(self, value: int, suffix: str = ""):
        self.current = min(value, self.total)
        self._render(suffix)

    def _render(self, suffix: str = ""):
        pct = self.current / self.total
        filled = int(self.BAR_WIDTH * pct)
        bar = "█" * filled + "░" * (self.BAR_WIDTH - filled)
        elapsed = time.time() - self._start
        line = f"\r  {self.label:20s} [{bar}] {pct*100:5.1f}%  {elapsed:5.1f}s  {suffix}"
        sys.stdout.write(line)
        sys.stdout.flush()
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


class SpinnerProgress:
    """Spinner for indeterminate tasks (e.g. data loading)."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = ""):
        self.label = label
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._start = time.time()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop_event.set()
        self._thread.join()
        elapsed = time.time() - self._start
        sys.stdout.write(f"\r  ✓ {self.label:40s} {elapsed:.1f}s\n")
        sys.stdout.flush()

    def _spin(self):
        idx = 0
        while not self._stop_event.is_set():
            frame = self.FRAMES[idx % len(self.FRAMES)]
            elapsed = time.time() - self._start
            sys.stdout.write(f"\r  {frame} {self.label:40s} {elapsed:.1f}s")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1


def section(title: str):
    """Print a formatted section header."""
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = next((p for p in DATA_PATHS if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError(
            f"kcet_features parquet folder not found. Tried:\n"
            + "\n".join(f"  • {p}" for p in DATA_PATHS)
        )

    df = pd.read_parquet(path)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET] + CAT_COLS)
    df = df[df[TARGET] > 0]
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalise categorical columns and collect unique value sets."""
    encoders: dict[str, list] = {}
    for col in CAT_COLS:
        df[col] = df[col].astype(str).str.strip().str.upper()
        encoders[col] = df[col].unique().tolist()
    return df, encoders


# ─── Tuning ──────────────────────────────────────────────────────────────────

def tune_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 15,
    cat_features: list[int] | None = None,
) -> dict:
    """Run Optuna HPO with a live progress bar."""

    pb = ProgressBar(total=n_trials, label="Optuna trials")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 400),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": False,
        }

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        fold_rmses: list[float] = []

        for train_idx, val_idx in kf.split(X_train):
            model = CatBoostRegressor(**params)
            model.fit(
                X_train.iloc[train_idx],
                y_train.iloc[train_idx],
                cat_features=cat_features,
                eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
                early_stopping_rounds=20,
                verbose=False,
            )
            preds = model.predict(X_train.iloc[val_idx])
            rmse = np.sqrt(mean_squared_error(y_train.iloc[val_idx], preds))
            fold_rmses.append(rmse)

        trial_rmse = float(np.mean(fold_rmses))

        pb.update(
            suffix=f"best RMSE {min(s.value for s in trial.study.trials if s.value):.1f}"
            if trial.number > 0
            else "",
        )

        return trial_rmse

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    print(f"  Best trial #{best.number}  RMSE={best.value:.2f}  params={best.params}")
    return study.best_params


# ─── Training ─────────────────────────────────────────────────────────────────

def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    cat_features: list[int],
) -> CatBoostRegressor:
    """Train final CatBoost model with a live iteration progress bar."""

    iterations = best_params.get("iterations", 300)
    pb = ProgressBar(total=iterations, label="CatBoost iterations")

    params = {
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": False,
        **best_params,
    }

    model = CatBoostRegressor(**params)

    # CatBoost callback to stream progress
    class _IterCallback:
        def after_iteration(self, info):
            pb.set(info.iteration, suffix=f"train loss {info.metrics['learn']['RMSE'][-1]:.2f}")
            return True  # continue training

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        callbacks=[_IterCallback()],
    )

    return model


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_artifacts(model: CatBoostRegressor, encoders: dict, feature_cols: list):
    os.makedirs(MODEL_DIR, exist_ok=True)

    artifacts = {
        f"{MODEL_DIR}/kcet_encoders.pkl": encoders,
        f"{MODEL_DIR}/kcet_feature_cols.pkl": feature_cols,
    }

    pb = ProgressBar(total=len(artifacts) + 1, label="Saving artifacts")

    model.save_model(f"{MODEL_DIR}/kcet_model.cbm")
    pb.update(suffix="kcet_model.cbm")

    for path, obj in artifacts.items():
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        pb.update(suffix=os.path.basename(path))

    print(f"  └─ artifacts written to {MODEL_DIR}/")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # 1. Load data
    section("1 / 4  Loading Data")
    with SpinnerProgress("Reading parquet …"):
        df = load_data()
    print(f"  Rows loaded  : {len(df):,}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"  Target range : {df[TARGET].min():.0f} – {df[TARGET].max():.0f}")

    # 2. Prepare features
    section("2 / 4  Preparing Features")
    with SpinnerProgress("Encoding categoricals …"):
        df, encoders = encode_categoricals(df)

    y = np.log1p(df[TARGET])
    X = df[CAT_COLS + NUM_COLS]
    cat_features_indices = [X.columns.get_loc(c) for c in CAT_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f"  Train rows   : {len(X_train):,}")
    print(f"  Test rows    : {len(X_test):,}")

    # 3. Hyper-parameter search
    section("3 / 4  Optuna Hyper-parameter Search  (n_trials=15)")
    best_params = tune_catboost(
        X_train,
        y_train,
        n_trials=15,
        cat_features=cat_features_indices,
    )

    # 4. Final training
    section("4 / 4  Training Final Model")
    model = train_final_model(X_train, y_train, best_params, cat_features_indices)

    # Evaluate
    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    within_500 = np.mean(np.abs(y_true - y_pred) <= 500) * 100

    print(f"\n  ┌─ Evaluation on held-out test set ──────────")
    print(f"  │  RMSE          : {rmse:,.1f}")
    print(f"  │  MAE           : {mae:,.1f}")
    print(f"  │  Within ±500   : {within_500:.1f}%")
    print(f"  └────────────────────────────────────────────")

    # Save
    save_artifacts(model, encoders, CAT_COLS + NUM_COLS)

    total = time.time() - t0
    print(f"\n  ✅  Done in {total/60:.1f} min ({total:.0f}s)\n")


if __name__ == "__main__":
    main()