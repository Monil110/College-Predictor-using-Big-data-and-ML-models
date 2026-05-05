# PredictMe — Big Data Admission Predictor 🚀

PredictMe is a powerful, end-to-end Big Data and Machine Learning platform engineered to predict admission tier probabilities for complex nationwide entrance examinations — specifically **Engineering (JEE)**, **Medical (NEET)**, **State Engineering (KCET)**, and **COMEDK**.

The platform bridges massive raw data handling via **PySpark**, highly optimized tree-based predictive algorithms via **XGBoost, CatBoost & Optuna**, rapid and scalable serving bounds via **FastAPI** & **Redis**, and an interactive graphical layer through **React/Vite**.

---

## 🏗️ Architecture & How It Works

### 1. Data Ingestion & Engineering `(/spark)`
Because legacy CSV dumps span millions of fractional iterations across historically messy rows, the foundational architecture scales purely natively atop **Apache Spark**.
- **`spark/ingest.py`, `spark/neet_ingest.py`, `spark/kcet_ingest.py` & `spark/comedk_pandas.py`**: Drops erratic structural nulls, parses string hierarchies (like COMEDK's wide multi-line headers), injects structured DataFrames enforcing unified Schema, and finally converts constraints down to highly performant **Parquet Format** bound to local containerized nodes `/app/data/processed/`.
- **`spark/features.py`, `spark/neet_features.py`, `spark/kcet_features.py` & `spark/comedk_pandas.py`**: Executes complex cross-dimensional Spark groupings aggregating exactly what elements constitute predictive bounds — defining `{Institute, Course, Category, Quota...}` against aggregated functions to lock in actual historic endpoints (like `closing_rank` thresholds).

### 2. Machine Learning Pipeline `(/ML)`
Instead of blindly predicting the next closing rank, we calculate statistical **eligibility probabilities** dynamically mapped against dense categorical parameters.
- **`ML/jee/`, `ML/neet/`, `ML/kcet/`, `ML/comedk/`**: Extracts structured Parquets interactively. The heavily categorized architectures rely purely on highly optimized **CatBoost (Classifiers and Regressors)** securely extrapolating boundaries dynamically across massive class assignments.
- **Probabilistic Targeting**: The JEE pipeline natively generates heavily randomized synthetic negative samplings spanning outside legitimate bounds—allowing `CatBoostClassifier` instances to return a clean mathematical probability factor dynamically validating student eligibility natively.
- Evaluates AUC and RMSE accurately, automatically tracking and injecting `.cbm` models and associative targets directly into the `/models` directory natively.

### 3. Unified API & Redis Caching `(/backend)`
We utilize a monolithic unified microservice to intercept queries intelligently.
- **`backend/main.py`**: Serves as the FastAPI controller securely routing the unique domains (`/predict` for JEE, `/predict/neet` for Medical, `/predict/kcet` for State Engineering, `/predict/comedk` for COMEDK). It conditionally manages runtime executions dynamically utilizing the corresponding pre-compiled structures securely deployed under `/models`!
- **How Redis Works Here**: Predictive tree traversals computationally scale. Because millions of users share duplicate exact queries (e.g., `Rank 5000, Category: OBC`), the Fast API endpoint strictly evaluates `key = neet_5000_OBC`. If it hits the deployed Docker **Redis Cluster**, it completely skips XGBoost and dynamically serves JSON data directly bound safely inside nanoseconds.
- **Inference Algorithms**: Determines the exact bounds of a valid admission securely using domain-specific metrics.
  - **Classifiers (JEE)**: Groups dynamically via Probability Thresholds (`Safe`: likelihood >= 70%, `Likely`: likelihood >= 40%).
  - **Regressors (NEET / KCET / COMEDK)**: Maps natively against dynamic rank gaps (e.g., `Predicted Rank > Candidate Rank - 1500`).

### 4. Interactive Frontend Application `(/frontend)`
A beautifully responsive React Glassmorphism interface.
- **`App.jsx` & `PredictionForm.jsx`**: Structurally multiplexes between Domain states bridging inputs explicitly into strict payload targets specifically mapped over standard Axios handlers natively formatting cleanly against port `8000`.
- **`ResultsTable.jsx`**: Parses returned boundaries dynamically rendering Safe bindings strictly over Fallback Optional Chains gracefully managing discrepancies between Medical architectures omitting parameters actively returned natively over their Engineering counterparts.

---

## 🐳 Docker Infrastructure
The entire big-data storage and execution pipeline leverages tightly integrated Docker bindings explicitly declared inside `docker-compose.yml`.
- **Spark-Master & Worker**: Emulates pure cluster architectures to deploy `spark-submit`. Without Docker, PySpark falls dynamically vulnerable against Native Hadoop missing environments natively across Windows.
- **HDFS Nodes (Namenode/Datanode)**: Scales out to ensure long-term feature aggregation is entirely abstracted.
- **Redis Node**: Exposes port `6379` caching HTTP prediction matrices securely keeping latency < 10ms.

## 🚀 How To Run Locally

1. **Deploy Background Clusters**
   ```bash
   docker-compose up -d
   ```
2. **Train your Intelligence Limits (Windows terminal)**
   *(Ensure raw data is aggregated over PySpark limits via Spark container primarily if generating newly!)*
   ```bash
   python ML/jee/train.py
   python ML/neet/train.py
   python ML/kcet/train.py
   python ML/comedk/train.py
   ```
3. **Launch the FastAPI Server**
   ```bash
   python -m uvicorn backend.main:app --reload --port 8000
   ```
4. **Boot the React Frontend**
   ```bash
   cd frontend
   npm run dev
   ```

*Happy Predicting!*
