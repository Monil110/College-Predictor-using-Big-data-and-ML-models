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
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = "models/comedk"
CAT_COLS  = ["college_name", "course_name", "category"]
TARGET    = "closing_rank"

DATA_PATHS = [
    "data/processed/comedk_features",
    "/app/data/processed/comedk_features",
]

FEATURE_COLS = [
    "college_name", "course_name", "category", "year",
    "prev_year_closing_rank", "closing_rank_mean", "closing_rank_std",
    "closing_rank_min", "closing_rank_max", "rank_trend",
    "years_available", "latest_year", "earliest_year",
    "college_avg_rank", "category_avg_rank",
]

# ─── Terminal Colors ──────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m";  BOLD   = "\033[1m";  GREEN  = "\033[92m"
    YELLOW = "\033[93m"; CYAN   = "\033[96m";  RED    = "\033[91m"
    DIM    = "\033[2m";  WHITE  = "\033[97m"

# ─── Progress Bar ─────────────────────────────────────────────────────────────

class ProgressBar:
    BAR_WIDTH = 38

    def __init__(self, total, label="", color=C.GREEN):
        self.total   = max(total, 1)
        self.label   = label
        self.color   = color
        self.current = 0
        self._start  = time.time()
        self._lock   = threading.Lock()
        self._done   = False

    def update(self, n=1, suffix=""):
        with self._lock:
            self.current = min(self.current + n, self.total)
            self._render(suffix)

    def set(self, value, suffix=""):
        with self._lock:
            self.current = min(max(value, 0), self.total)
            self._render(suffix)

    def done(self, msg=""):
        with self._lock:
            self.current = self.total
            self._render(msg)
            if not self._done:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._done = True

    def _render(self, suffix=""):
        pct     = self.current / self.total
        filled  = int(self.BAR_WIDTH * pct)
        bar     = f"{self.color}{'█'*filled}{C.DIM}{'░'*(self.BAR_WIDTH-filled)}{C.RESET}"
        elapsed = time.time() - self._start
        eta     = (elapsed / self.current * (self.total - self.current)) if self.current > 0 else 0
        eta_str = f"ETA {eta:4.0f}s" if self.current < self.total else f"{elapsed:5.1f}s ✓"
        pad     = len(str(self.total))
        line    = (
            f"\r  {C.BOLD}{self.label:<22}{C.RESET} [{bar}] "
            f"{C.WHITE}{pct*100:5.1f}%{C.RESET}  "
            f"{C.DIM}{self.current:>{pad}}/{self.total}  {eta_str}{C.RESET}"
            f"  {C.YELLOW}{suffix}{C.RESET}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()


class Spinner:
    FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, label=""):
        self.label       = label
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._spin, daemon=True)
        self._start      = time.time()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop_event.set()
        self._thread.join()
        elapsed = time.time() - self._start
        sys.stdout.write(f"\r  {C.GREEN}✔{C.RESET}  {self.label:<45} {C.DIM}{elapsed:.1f}s{C.RESET}\n")
        sys.stdout.flush()

    def _spin(self):
        idx = 0
        while not self._stop_event.is_set():
            f = self.FRAMES[idx % len(self.FRAMES)]
            e = time.time() - self._start
            sys.stdout.write(f"\r  {C.CYAN}{f}{C.RESET}  {self.label:<45} {C.DIM}{e:.1f}s{C.RESET}")
            sys.stdout.flush()
            time.sleep(0.08)
            idx += 1


def section(title):
    print(f"\n{C.BOLD}{C.CYAN}{'─'*62}{C.RESET}")
    print(f"  {C.BOLD}{C.WHITE}{title}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'─'*62}{C.RESET}")

def pm(label, value, unit=""):
    print(f"  {C.DIM}{label:<28}{C.RESET} {C.WHITE}{C.BOLD}{value}{C.RESET} {C.DIM}{unit}{C.RESET}")

# ─── Data ─────────────────────────────────────────────────────────────────────

def load_data():
    path = None
    for p in DATA_PATHS:
        c = os.path.join(p, "data.parquet")
        if os.path.exists(c):
            path = c
            break
    if not path:
        raise FileNotFoundError("data.parquet not found. Run comedk_features.py first.")

    with Spinner("Reading parquet file"):
        df = pd.read_parquet(path)
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=[TARGET] + CAT_COLS)
        df = df[df[TARGET] > 0]

    pm("Rows loaded",         f"{len(df):,}")
    pm("Colleges",            f"{df['college_name'].nunique():,}")
    pm("Courses (branches)",  f"{df['course_name'].nunique():,}")
    pm("Categories",          f"{df['category'].nunique()}")
    pm("Years",               str(sorted(df["year"].unique().tolist())))
    pm("Rank range",          f"{df[TARGET].min():.0f} – {df[TARGET].max():.0f}")

    missing = [c for c in FEATURE_COLS if c not in df.columns and c not in CAT_COLS + ["year"]]
    if missing:
        print(f"\n  {C.RED}[WARN] Missing feature columns: {missing}{C.RESET}")
        print(f"  {C.YELLOW}  → Re-run comedk_features.py to regenerate.{C.RESET}")
    return df

# ─── Encoding ─────────────────────────────────────────────────────────────────

def encode_categoricals(df):
    df = df.copy()
    encoders, raw_values = {}, {}
    pb = ProgressBar(total=len(CAT_COLS), label="Encoding categories", color=C.CYAN)
    for c in CAT_COLS:
        le = LabelEncoder()
        df[c]         = le.fit_transform(df[c].astype(str))
        encoders[c]   = le
        raw_values[c] = list(le.classes_)
        pb.update(suffix=c)
    pb.done()
    return df, encoders, raw_values

# ─── Tuning ───────────────────────────────────────────────────────────────────

def tune_catboost(X_train, y_train, n_trials=30):
    pb = ProgressBar(total=n_trials, label="Optuna trials", color=C.YELLOW)

    def objective(trial):
        params = {
            "iterations":       trial.suggest_int("iterations", 300, 800),
            "depth":            trial.suggest_int("depth", 4, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 8.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 10),
            "loss_function": "RMSE", "random_seed": 42, "verbose": False,
        }
        kf, scores = KFold(n_splits=5, shuffle=True, random_state=42), []
        for ti, vi in kf.split(X_train):
            m = CatBoostRegressor(**params)
            m.fit(X_train.iloc[ti], y_train.iloc[ti],
                  eval_set=(X_train.iloc[vi], y_train.iloc[vi]),
                  early_stopping_rounds=50, verbose=False)
            preds = m.predict(X_train.iloc[vi])
            scores.append(np.sqrt(mean_squared_error(
                np.expm1(y_train.iloc[vi]), np.expm1(preds)
            )))
        trial_rmse  = float(np.mean(scores))
        completed   = [t.value for t in trial.study.trials if t.value is not None]
        best_so_far = min(completed) if completed else trial_rmse
        pb.update(suffix=f"best RMSE {best_so_far:,.0f}")
        return trial_rmse

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    pb.done()

    print()
    pm("Best CV RMSE",    f"{study.best_value:,.1f}", "ranks")
    pm("Best iterations", str(study.best_params.get("iterations")))
    pm("Best depth",      str(study.best_params.get("depth")))
    pm("Best lr",         f"{study.best_params.get('learning_rate'):.4f}")
    return study.best_params

# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, params):
    iterations = params.get("iterations", 500)
    pb = ProgressBar(total=iterations, label="Training CatBoost", color=C.GREEN)

    model = CatBoostRegressor(**params, loss_function="RMSE", random_seed=42, verbose=False)

    class _CB:
        def after_iteration(self, info):
            rmse_list = info.metrics.get("learn", {}).get("RMSE", [])
            suffix    = f"train RMSE {rmse_list[-1]:,.0f}" if rmse_list else ""
            pb.set(info.iteration + 1, suffix=suffix)
            return True

    model.fit(X_train, y_train, callbacks=[_CB()])
    pb.done()
    return model

# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test_log):
    with Spinner("Running evaluation on test set"):
        y_true = np.expm1(y_test_log)
        y_pred = np.expm1(model.predict(X_test))

    rmse        = np.sqrt(mean_squared_error(y_true, y_pred))
    mae         = np.mean(np.abs(y_true - y_pred))
    within_500  = np.mean(np.abs(y_true - y_pred) <= 500)  * 100
    within_1000 = np.mean(np.abs(y_true - y_pred) <= 1000) * 100
    within_2000 = np.mean(np.abs(y_true - y_pred) <= 2000) * 100

    print()
    pm("RMSE",               f"{rmse:,.1f}",   "ranks")
    pm("MAE",                f"{mae:,.1f}",    "ranks")
    pm("Within ±500 ranks",  f"{within_500:.1f}",  "%")
    pm("Within ±1000 ranks", f"{within_1000:.1f}", "%")
    pm("Within ±2000 ranks", f"{within_2000:.1f}", "%")

    importance = pd.Series(
        model.get_feature_importance(), index=X_test.columns
    ).sort_values(ascending=False)

    print(f"\n  {C.BOLD}Top 5 important features:{C.RESET}")
    for feat, score in importance.head(5).items():
        bar = "█" * int(score / 2)
        print(f"    {feat:<30} {C.GREEN}{bar:<25}{C.RESET} {C.DIM}{score:.1f}{C.RESET}")

# ─── Save ─────────────────────────────────────────────────────────────────────

def save(model, encoders, raw_values, feature_cols, df_original):
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifacts = {
        f"{MODEL_DIR}/encoders.pkl":   encoders,
        f"{MODEL_DIR}/features.pkl":   feature_cols,
        f"{MODEL_DIR}/raw_values.pkl": raw_values,
    }
    pb = ProgressBar(total=len(artifacts) + 2, label="Saving artifacts", color=C.CYAN)

    model.save_model(f"{MODEL_DIR}/comedk_model.cbm")
    pb.update(suffix="comedk_model.cbm")

    for fpath, obj in artifacts.items():
        with open(fpath, "wb") as f:
            pickle.dump(obj, f)
        pb.update(suffix=os.path.basename(fpath))

    lookup = (
        df_original
        .groupby(["college_name", "course_name", "category"])["closing_rank"]
        .apply(list).to_dict()
    )
    with open(f"{MODEL_DIR}/lookup.pkl", "wb") as f:
        pickle.dump(lookup, f)
    pb.update(suffix="lookup.pkl")
    pb.done()
    pm("All artifacts saved to", MODEL_DIR)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    section("1 / 6  Loading Data")
    df          = load_data()
    df_original = df.copy()

    section("2 / 6  Encoding")
    df_enc, encoders, raw_values = encode_categoricals(df)

    available_cols = [c for c in FEATURE_COLS if c in df_enc.columns]
    missing_cols   = [c for c in FEATURE_COLS if c not in df_enc.columns]
    if missing_cols:
        print(f"  {C.YELLOW}[WARN] Skipping missing columns: {missing_cols}{C.RESET}")

    X = df_enc[available_cols]
    y = np.log1p(df_enc[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    pm("Train samples", f"{len(X_train):,}")
    pm("Test samples",  f"{len(X_test):,}")

    section("3 / 6  Hyperparameter Tuning")
    best_params = tune_catboost(X_train, y_train, n_trials=30)

    section("4 / 6  Training Final Model")
    model = train_model(X_train, y_train, best_params)

    section("5 / 6  Evaluation")
    evaluate(model, X_test, y_test)

    section("6 / 6  Saving Artifacts")
    save(model, encoders, raw_values, available_cols, df_original)

    print(f"\n  {C.GREEN}{C.BOLD}✅  Pipeline complete in {time.time()-t0:.1f}s{C.RESET}\n")


if __name__ == "__main__":
    main()