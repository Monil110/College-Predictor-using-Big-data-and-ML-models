"""
multi_exam_predictor.py
=======================
Production-ready, fully-typed multi-exam prediction module.

Architecture
------------
* BaseExamPredictor  – Abstract Base Class enforcing a common interface.
* NEETPGPredictor    – Concrete implementation using a scikit-learn
                       RandomForest pipeline (college+course combined target).
* NEETUGPredictor    – Placeholder stub; swap in data/model later.
* JEEPredictor       – Placeholder stub; swap in data/model later.
* ExamPredictionPipeline – Factory-registry coordinator that routes
                           prediction requests to the correct predictor.

Standardised output (every predictor must return this shape):
    {
        "status":      "success" | "error",
        "exam":        str,
        "predictions": [{"college": str, "course": str, "probability": float}]
    }

Usage
-----
    pipeline = ExamPredictionPipeline()
    pipeline.register("neet_pg", NEETPGPredictor())
    pipeline.get_predictor("neet_pg").train("path/to/data.csv")
    result = pipeline.predict_for_exam("neet_pg", rank=1500, category="OBC")
"""

from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases for readability
# ---------------------------------------------------------------------------
PredictionResult = Dict[str, Any]
PredictionList = List[Dict[str, Any]]


# ===========================================================================
# Abstract Base Class
# ===========================================================================

class BaseExamPredictor(ABC):
    """
    Unified interface that every exam-specific predictor must implement.

    Subclasses are required to provide concrete implementations of:
        - train(data_path)  : load data and fit the underlying ML model.
        - predict(...)      : return a standardised PredictionResult dict.
    """

    # ------------------------------------------------------------------
    # Helpers shared by all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _success_response(
        exam: str,
        predictions: PredictionList,
    ) -> PredictionResult:
        """Build a standardised success response."""
        return {
            "status": "success",
            "exam": exam,
            "predictions": predictions,
        }

    @staticmethod
    def _error_response(exam: str, message: str) -> PredictionResult:
        """Build a standardised error response."""
        return {
            "status": "error",
            "exam": exam,
            "predictions": [],
            "message": message,
        }

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, data_path: str) -> None:
        """
        Load training data from *data_path* and fit the model.

        Parameters
        ----------
        data_path : str
            Absolute or relative path to a CSV or Parquet file containing
            at minimum the columns: rank, category, college, course.
        """

    @abstractmethod
    def predict(
        self,
        rank: int,
        category: str,
        **kwargs: Any,
    ) -> PredictionResult:
        """
        Predict college/course allocations for a given rank + category.

        Parameters
        ----------
        rank     : int   – Candidate's exam rank (lower = better).
        category : str   – Reservation category (e.g. "General", "OBC").
        **kwargs         – Exam-specific optional parameters.

        Returns
        -------
        PredictionResult – Standardised dict (see module docstring).
        """


# ===========================================================================
# NEET PG Predictor
# ===========================================================================

# Column name aliases — covers the exact headers from allotment-list-311913.xlsx
# as well as alternative spellings for future datasets.
_NEET_PG_RANK_ALIASES: List[str] = [
    "All India Rank",          # ← exact header in allotment-list-311913.xlsx
    "rank", "Rank", "RANK", "allotted_rank", "AIR",
]
_NEET_PG_COLLEGE_ALIASES: List[str] = [
    "Name of the College Allotted.",   # ← exact header in allotment-list-311913.xlsx
    "college_name", "College_Name", "Institute", "institute", "COLLEGE",
    "Name of the College Allotted",
]
_NEET_PG_COURSE_ALIASES: List[str] = [
    "Course Name",             # ← exact header in allotment-list-311913.xlsx
    "course", "Course", "Branch", "branch", "COURSE", "subject",
]
_NEET_PG_CATEGORY_ALIASES: List[str] = [
    "Allotted Category",       # ← exact header in allotment-list-311913.xlsx
    "category", "Category", "CATEGORY", "quota", "Quota",
]

# Separator used internally to combine college + course into one label
_LABEL_SEP: str = "|||"

# Number of top predictions to surface
_TOP_N: int = 3


class NEETPGPredictor(BaseExamPredictor):
    """
    NEET PG predictor built on a scikit-learn RandomForestClassifier.

    Strategy
    --------
    1. Combine `college` + `course` into a single target string
       (e.g. "AIIMS Delhi|||MD General Medicine") — this avoids the need
       for a multi-output model while preserving joint college-course
       allocation semantics.
    2. Features  : [rank (int), category (label-encoded int)]
    3. At predict time, decode the top-N class probabilities back into
       separate college / course fields.
    """

    EXAM_NAME: str = "neet_pg"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 20,
        random_state: int = 42,
        top_n: int = _TOP_N,
    ) -> None:
        self._n_estimators: int = n_estimators
        self._max_depth: Optional[int] = max_depth
        self._random_state: int = random_state
        self._top_n: int = top_n

        # Fitted artefacts — None until train() is called
        self._model: Optional[RandomForestClassifier] = None
        self._category_encoder: LabelEncoder = LabelEncoder()
        self._label_encoder: LabelEncoder = LabelEncoder()  # college|||course
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_column(df: pd.DataFrame, aliases: List[str]) -> str:
        """
        Return the first column name in *aliases* that exists in *df*.

        Raises
        ------
        KeyError – if none of the aliases match any column in the dataframe.
        """
        for alias in aliases:
            if alias in df.columns:
                return alias
        raise KeyError(
            f"None of the expected column aliases {aliases} found in the "
            f"dataframe. Available columns: {list(df.columns)}"
        )

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load a CSV, Excel (.xlsx/.xls), or Parquet file into a DataFrame.

        Note: files whose extension is .csv but are actually Excel workbooks
        (PK magic bytes) are handled transparently via openpyxl.

        Raises
        ------
        FileNotFoundError – path does not exist.
        ValueError        – unsupported file extension.
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        suffix = path.suffix.lower()

        # Detect true Excel files even if disguised with a .csv extension
        is_excel_magic: bool = False
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic == b"PK\x03\x04":   # ZIP/XLSX magic
                is_excel_magic = True

        if is_excel_magic or suffix in (".xlsx", ".xls"):
            df: pd.DataFrame = pd.read_excel(path, engine="openpyxl")
        elif suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
        elif suffix in (".parquet", ".pq"):
            df = pd.read_parquet(path)
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. "
                "Supported: .csv, .xlsx, .xls, .parquet, .pq"
            )

        logger.info(
            "Loaded %d rows x %d cols from '%s'",
            len(df),
            len(df.columns),
            data_path,
        )
        return df

    # ------------------------------------------------------------------
    # Model persistence (joblib)
    # ------------------------------------------------------------------

    def save(self, model_dir: str) -> None:
        """
        Persist all fitted artefacts to *model_dir* using joblib.

        Files written
        -------------
        neet_pg_model.pkl           – RandomForestClassifier
        neet_pg_category_encoder.pkl – LabelEncoder for categories
        neet_pg_label_encoder.pkl    – LabelEncoder for college|||course labels

        Raises
        ------
        RuntimeError  – if called before train().
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError(
                "Cannot save: model is not trained. Call train() first."
            )
        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._model,            out / "neet_pg_model.pkl")
        joblib.dump(self._category_encoder, out / "neet_pg_category_encoder.pkl")
        joblib.dump(self._label_encoder,    out / "neet_pg_label_encoder.pkl")

        logger.info("[NEETPGPredictor] Artefacts saved to '%s'.", model_dir)

    @classmethod
    def load(
        cls,
        model_dir: str,
        top_n: int = _TOP_N,
    ) -> "NEETPGPredictor":
        """
        Restore a previously saved predictor from *model_dir*.

        Parameters
        ----------
        model_dir : str  – Directory that contains the three .pkl files.
        top_n     : int  – Number of top predictions to return (default 3).

        Returns
        -------
        NEETPGPredictor  – A fully initialised, ready-to-predict instance.

        Raises
        ------
        FileNotFoundError – if any artefact file is missing.
        """
        base = Path(model_dir)
        required = [
            "neet_pg_model.pkl",
            "neet_pg_category_encoder.pkl",
            "neet_pg_label_encoder.pkl",
        ]
        for fname in required:
            if not (base / fname).exists():
                raise FileNotFoundError(
                    f"Missing artefact: {base / fname}. "
                    "Run train_neet_pg.py to generate it."
                )

        predictor = cls.__new__(cls)
        # Manually set all instance attributes to avoid __init__ side-effects
        predictor._n_estimators      = 200
        predictor._max_depth         = 20
        predictor._random_state      = 42
        predictor._top_n             = top_n
        predictor._model             = joblib.load(base / "neet_pg_model.pkl")
        predictor._category_encoder  = joblib.load(base / "neet_pg_category_encoder.pkl")
        predictor._label_encoder     = joblib.load(base / "neet_pg_label_encoder.pkl")
        predictor._is_trained        = True

        logger.info("[NEETPGPredictor] Loaded artefacts from '%s'.", model_dir)
        return predictor

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, data_path: str) -> None:
        """
        Load allotment data and fit the RandomForest classifier.

        Expected columns (flexible naming — see alias lists at top):
            rank, category, college_name, course

        Parameters
        ----------
        data_path : str
            Path to a CSV or Parquet allotment file.
        """
        logger.info("[NEETPGPredictor] Starting training from '%s'", data_path)

        df = self._load_data(data_path)

        # Resolve actual column names (tolerates different header spellings)
        rank_col     = self._resolve_column(df, _NEET_PG_RANK_ALIASES)
        college_col  = self._resolve_column(df, _NEET_PG_COLLEGE_ALIASES)
        course_col   = self._resolve_column(df, _NEET_PG_COURSE_ALIASES)
        category_col = self._resolve_column(df, _NEET_PG_CATEGORY_ALIASES)

        # Drop rows with any missing values in the required columns
        required = [rank_col, college_col, course_col, category_col]
        df = df.dropna(subset=required).reset_index(drop=True)

        if df.empty:
            raise ValueError("No valid rows remain after dropping NaN values.")

        # Feature: rank (numeric, cast to int)
        df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
        df = df.dropna(subset=[rank_col]).reset_index(drop=True)
        X_rank: np.ndarray = df[rank_col].astype(int).values.reshape(-1, 1)

        # Feature: category (label-encoded)
        categories: np.ndarray = (
            df[category_col].astype(str).str.strip().str.upper().values
        )
        X_cat: np.ndarray = (
            self._category_encoder.fit_transform(categories).reshape(-1, 1)
        )

        # Combined feature matrix  [rank, category_encoded]
        X: np.ndarray = np.hstack([X_rank, X_cat])

        # Target: "college|||course" combined label
        combined_labels: np.ndarray = (
            df[college_col].astype(str).str.strip()
            + _LABEL_SEP
            + df[course_col].astype(str).str.strip()
        ).values
        y: np.ndarray = self._label_encoder.fit_transform(combined_labels)

        # Fit the RandomForest
        self._model = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._random_state,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        self._is_trained = True

        logger.info(
            "[NEETPGPredictor] Training complete. "
            "%d samples, %d unique college+course combinations.",
            len(df),
            len(self._label_encoder.classes_),
        )

    def predict(
        self,
        rank: int,
        category: str,
        **kwargs: Any,
    ) -> PredictionResult:
        """
        Predict top-N college+course allocations for a given rank and category.

        Parameters
        ----------
        rank     : int  – Candidate's NEET PG rank.
        category : str  – Reservation category (e.g. "OBC", "GENERAL").

        Returns
        -------
        PredictionResult – Standardised dict with top-N predictions.
        """
        if not self._is_trained or self._model is None:
            return self._error_response(
                self.EXAM_NAME,
                "Model has not been trained yet. Call train(data_path) first.",
            )

        try:
            # Encode the input category (handle unseen labels gracefully)
            cat_upper: str = str(category).strip().upper()
            known_cats: List[str] = list(self._category_encoder.classes_)
            if cat_upper not in known_cats:
                logger.warning(
                    "[NEETPGPredictor] Unseen category '%s'. "
                    "Falling back to most frequent: '%s'.",
                    cat_upper,
                    known_cats[0],
                )
                cat_upper = known_cats[0]

            cat_encoded: int = int(
                self._category_encoder.transform([cat_upper])[0]
            )

            # Build input feature vector
            X_input: np.ndarray = np.array([[rank, cat_encoded]])

            # Obtain class probabilities from the forest
            proba: np.ndarray = self._model.predict_proba(X_input)[0]

            # Select top-N class indices by probability (descending)
            top_n: int = min(self._top_n, len(proba))
            top_indices: np.ndarray = np.argsort(proba)[::-1][:top_n]

            predictions: PredictionList = []
            for idx in top_indices:
                combined_label: str = self._label_encoder.inverse_transform(
                    [idx]
                )[0]

                # Safely split back into college / course
                if _LABEL_SEP in combined_label:
                    college, course = combined_label.split(_LABEL_SEP, 1)
                else:
                    # Defensive: if separator is missing, put everything in college
                    college, course = combined_label, "Unknown"

                predictions.append(
                    {
                        "college": college.strip(),
                        "course": course.strip(),
                        "probability": round(float(proba[idx]), 6),
                    }
                )

            return self._success_response(self.EXAM_NAME, predictions)

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "[NEETPGPredictor] predict() raised an exception: %s",
                traceback.format_exc(),
            )
            return self._error_response(self.EXAM_NAME, str(exc))


# ===========================================================================
# NEET UG Predictor  (placeholder — swap in model/data when ready)
# ===========================================================================

class NEETUGPredictor(BaseExamPredictor):
    """
    Placeholder predictor for NEET UG undergraduate admissions.

    To activate:
    1. Replace the NotImplementedError bodies with real training logic.
    2. Point train() at your NEET UG allotment CSV/Parquet file.
    3. Implement predict() following the same pattern as NEETPGPredictor.
    """

    EXAM_NAME: str = "neet_ug"

    def train(self, data_path: str) -> None:
        """Load NEET UG data and fit model — NOT YET IMPLEMENTED."""
        raise NotImplementedError(
            "NEETUGPredictor.train() is a placeholder. "
            "Provide a NEET UG allotment dataset and implement this method."
        )

    def predict(
        self,
        rank: int,
        category: str,
        **kwargs: Any,
    ) -> PredictionResult:
        """Predict NEET UG allocations — NOT YET IMPLEMENTED."""
        return self._error_response(
            self.EXAM_NAME,
            "NEETUGPredictor is not yet implemented. "
            "Call train() with a valid data path first.",
        )


# ===========================================================================
# JEE Predictor  (placeholder — swap in model/data when ready)
# ===========================================================================

class JEEPredictor(BaseExamPredictor):
    """
    Placeholder predictor for JEE (IIT / NIT) admissions.

    To activate:
    1. Load allotment data (IIT / NIT seat-allotment CSVs work well).
    2. The existing backend already has CatBoost models at
       models/jee/model_iit.cbm and model_nit.cbm — wire them in here
       instead of training from scratch if preferred.
    """

    EXAM_NAME: str = "jee"

    def train(self, data_path: str) -> None:
        """Load JEE data and fit model — NOT YET IMPLEMENTED."""
        raise NotImplementedError(
            "JEEPredictor.train() is a placeholder. "
            "Provide a JEE allotment dataset and implement this method."
        )

    def predict(
        self,
        rank: int,
        category: str,
        **kwargs: Any,
    ) -> PredictionResult:
        """Predict JEE allocations — NOT YET IMPLEMENTED."""
        return self._error_response(
            self.EXAM_NAME,
            "JEEPredictor is not yet implemented. "
            "Call train() with a valid data path first.",
        )


# ===========================================================================
# Factory-Registry Coordinator
# ===========================================================================

class ExamPredictionPipeline:
    """
    Central coordinator that manages multiple exam predictors.

    Responsibilities
    ----------------
    * Maintains a registry mapping exam names → predictor instances.
    * Routes predict_for_exam() calls to the correct predictor.
    * Handles all routing errors gracefully (unknown exam, untrained model,
      unexpected exceptions) and always returns a standardised dict.

    Example
    -------
    >>> pipeline = ExamPredictionPipeline()
    >>> pg = NEETPGPredictor()
    >>> pg.train("data/neet_pg_allotment.csv")
    >>> pipeline.register("neet_pg", pg)
    >>> result = pipeline.predict_for_exam("neet_pg", rank=1500, category="OBC")
    >>> print(result["status"])
    'success'
    """

    def __init__(self) -> None:
        # Registry: exam_name (str) → predictor instance
        self._registry: Dict[str, BaseExamPredictor] = {}
        logger.info("[ExamPredictionPipeline] Initialised (empty registry).")

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register(self, exam_name: str, predictor: BaseExamPredictor) -> None:
        """
        Register a predictor under a given exam name.

        Parameters
        ----------
        exam_name : str                – Key used for routing (e.g. "neet_pg").
        predictor : BaseExamPredictor  – A fully initialised predictor instance.

        Raises
        ------
        TypeError – if predictor does not inherit from BaseExamPredictor.
        """
        if not isinstance(predictor, BaseExamPredictor):
            raise TypeError(
                f"predictor must be an instance of BaseExamPredictor, "
                f"got {type(predictor).__name__!r} instead."
            )
        self._registry[exam_name] = predictor
        logger.info(
            "[ExamPredictionPipeline] Registered predictor '%s' → %s.",
            exam_name,
            type(predictor).__name__,
        )

    def unregister(self, exam_name: str) -> None:
        """Remove a predictor from the registry (no-op if not found)."""
        removed = self._registry.pop(exam_name, None)
        if removed is not None:
            logger.info(
                "[ExamPredictionPipeline] Unregistered '%s'.", exam_name
            )

    def list_registered(self) -> List[str]:
        """Return the list of currently registered exam names."""
        return list(self._registry.keys())

    def get_predictor(self, exam_name: str) -> BaseExamPredictor:
        """
        Retrieve a registered predictor by name.

        Raises
        ------
        KeyError – if *exam_name* is not in the registry.
        """
        if exam_name not in self._registry:
            raise KeyError(
                f"No predictor registered for exam '{exam_name}'. "
                f"Registered exams: {self.list_registered()}"
            )
        return self._registry[exam_name]

    # ------------------------------------------------------------------
    # Prediction routing
    # ------------------------------------------------------------------

    def predict_for_exam(
        self,
        exam_name: str,
        rank: int,
        category: str,
        **kwargs: Any,
    ) -> PredictionResult:
        """
        Route a prediction request to the registered predictor for *exam_name*.

        This method is intentionally defensive:
        - Unknown exam names return an error dict (no exception raised).
        - Any exception inside the predictor's predict() is caught here as
          a second safety net, so the caller always receives a valid dict.

        Parameters
        ----------
        exam_name : str  – Must match a key in the registry.
        rank      : int  – Candidate rank.
        category  : str  – Reservation category string.
        **kwargs         – Forwarded verbatim to the predictor.

        Returns
        -------
        PredictionResult – Standardised dict (status / exam / predictions).
        """
        # Guard: unknown exam
        if exam_name not in self._registry:
            msg = (
                f"Exam '{exam_name}' is not registered. "
                f"Available: {self.list_registered()}"
            )
            logger.warning("[ExamPredictionPipeline] %s", msg)
            return {
                "status": "error",
                "exam": exam_name,
                "predictions": [],
                "message": msg,
            }

        predictor: BaseExamPredictor = self._registry[exam_name]

        # Guard: invalid rank
        try:
            rank = int(rank)
        except (ValueError, TypeError) as exc:
            msg = f"Invalid rank value '{rank}': {exc}"
            logger.error("[ExamPredictionPipeline] %s", msg)
            return {
                "status": "error",
                "exam": exam_name,
                "predictions": [],
                "message": msg,
            }

        # Delegate to predictor (second safety net around any stray exception)
        try:
            result: PredictionResult = predictor.predict(
                rank=rank, category=category, **kwargs
            )
        except Exception as exc:  # pylint: disable=broad-except
            msg = (
                f"Unhandled exception in {type(predictor).__name__}.predict(): "
                f"{exc}"
            )
            logger.error(
                "[ExamPredictionPipeline] %s\n%s", msg, traceback.format_exc()
            )
            return {
                "status": "error",
                "exam": exam_name,
                "predictions": [],
                "message": msg,
            }

        # Validate that the predictor returned the correct shape
        if not isinstance(result, dict) or "status" not in result:
            msg = (
                f"{type(predictor).__name__}.predict() returned an unexpected "
                f"type: {type(result).__name__}. Expected a dict."
            )
            logger.error("[ExamPredictionPipeline] %s", msg)
            return {
                "status": "error",
                "exam": exam_name,
                "predictions": [],
                "message": msg,
            }

        return result


# ===========================================================================
# Demo / smoke-test  (run: python multi_exam_predictor.py)
# ===========================================================================

if __name__ == "__main__":
    import json
    import sys

    print("=" * 70)
    print("  Multi-Exam Prediction Pipeline — Demo")
    print("=" * 70)

    # 1. Initialise the pipeline
    pipeline = ExamPredictionPipeline()
    print(f"\nRegistered exams (empty): {pipeline.list_registered()}")

    # 2. Instantiate predictors
    neet_pg_predictor = NEETPGPredictor(n_estimators=100, max_depth=15)
    neet_ug_predictor = NEETUGPredictor()
    jee_predictor     = JEEPredictor()

    # 3. Register predictors
    pipeline.register("neet_pg", neet_pg_predictor)
    pipeline.register("neet_ug", neet_ug_predictor)
    pipeline.register("jee",     jee_predictor)
    print(f"Registered exams: {pipeline.list_registered()}")

    # 4. Attempt NEET PG training
    #    Replace the path below with your actual allotment CSV/Parquet file.
    NEET_PG_DATA_PATH: str = "data/neet_pg_allotment.csv"
    print(f"\n[NEET PG] Attempting to train from '{NEET_PG_DATA_PATH}' ...")
    try:
        pipeline.get_predictor("neet_pg").train(NEET_PG_DATA_PATH)
        trained = True
    except FileNotFoundError:
        print(
            f"  [WARN] '{NEET_PG_DATA_PATH}' not found — "
            "skipping training; predict() will return an untrained error."
        )
        trained = False

    # 5. Run a safe inference call (pipeline always returns a valid dict)
    result_pg = pipeline.predict_for_exam(
        exam_name="neet_pg",
        rank=1500,
        category="OBC",
    )
    print("\n[NEET PG] predict_for_exam result:")
    print(json.dumps(result_pg, indent=2))

    # 6. Demonstrate graceful handling of a placeholder predictor
    result_ug = pipeline.predict_for_exam(
        exam_name="neet_ug",
        rank=45000,
        category="General",
    )
    print("\n[NEET UG] predict_for_exam result (placeholder):")
    print(json.dumps(result_ug, indent=2))

    # 7. Demonstrate graceful handling of an unknown exam
    result_unknown = pipeline.predict_for_exam(
        exam_name="cat_mba",
        rank=500,
        category="General",
    )
    print("\n[Unknown exam] predict_for_exam result:")
    print(json.dumps(result_unknown, indent=2))

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)
    sys.exit(0)
