# PredictMe — Big Data Admission Predictor 🚀

PredictMe is a powerful, end-to-end Big Data and Machine Learning platform engineered to predict admission tier probabilities for complex nationwide entrance examinations — specifically **Engineering (JEE)**, **Medical (NEET)**, and **State Engineering (KCET)**.

The platform bridges massive raw data handling via **PySpark**, highly optimized tree-based predictive algorithms via **XGBoost, CatBoost & Optuna**, rapid and scalable serving bounds via **FastAPI** & **Redis**, and an interactive graphical layer through **React/Vite**.

---

## 🏗️ Architecture & How It Works

### 1. Data Ingestion & Engineering `(/spark)`
Because legacy CSV dumps span millions of fractional iterations across historically messy rows, the foundational architecture scales purely natively atop **Apache Spark**.
- **`spark/ingest.py`, `spark/neet_ingest.py` & `spark/kcet_ingest.py`**: Drops erratic structural nulls, parses string hierarchies, injects structured DataFrames enforcing unified Schema, and finally converts constraints down to highly performant **Parquet Format** bound to local containerized nodes `/app/data/processed/`.
- **`spark/features.py`, `spark/neet_features.py` & `spark/kcet_features.py`**: Executes complex cross-dimensional Spark groupings aggregating exactly what elements constitute predictive bounds — defining `{Institute, Course, Category, Quota...}` against aggregated functions to lock in actual historic endpoints (like `closing_rank` thresholds).

### 2. Machine Learning Pipeline `(/ML)`
Instead of blindly predicting the next rank, we algorithmically derive competitive boundaries utilizing non-linear mathematical boundaries dynamically mapped against heavy categorical architectures.
- **`ML/jee/`, `ML/neet/`, `ML/kcet/`**: Extracts structured Parquets iteratively. Leverages **Optuna** to perform Bayesian Optimization testing thousands of parameters against 5-fold iterations. While JEE and NEET handle relationships across generalized distributions via XGBoost, KCET leverages **CatBoost** explicitly to safely extrapolate its heavy class assignments.
- Logs constraints natively leveraging `np.log1p` on target elements (`closing_rank`) preventing outlier variants (obscure college jumps) from dramatically burning localized weights natively. Evaluates RMSE and reliably writes `model.pkl` and associative structural dependencies `encoders.pkl` uniquely down cleanly into `/models`.

### 3. Unified API & Redis Caching `(/backend)`
We utilize a monolithic unified microservice to intercept queries intelligently.
- **`backend/main.py`**: Serves as the FastAPI controller securely routing the unique domains (`/predict` for JEE, `/predict/neet` for Medical, `/predict/kcet` for State Engineering). It conditionally manages runtime executions dynamically utilizing the corresponding pre-compiled structures securely deployed under `/models`!
- **How Redis Works Here**: Predictive tree traversals computationally scale. Because millions of users share duplicate exact queries (e.g., `Rank 5000, Category: OBC`), the Fast API endpoint strictly evaluates `key = neet_5000_OBC`. If it hits the deployed Docker **Redis Cluster**, it completely skips XGBoost and dynamically serves JSON data directly bound safely inside nanoseconds.
- **Inference Algorithms**: Defines precisely where you map safely:
  - **Safe**: `Projected Cutoff > Candidate Rank`
  - **Likely**: `Projected Cutoff > (Candidate Rank - 1500)`

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
