# Project Plan (Phase-by-Phase Roadmap)

Each phase ends with a summary and an explicit **PAUSE for owner approval**. No phase starts
until the previous one is approved. Every phase begins by re-reading `docs/`.

| Phase | Name | Status |
|---|---|---|
| 1 | Documentation & setup | ✅ Done (approved) |
| 2 | EDA & preprocessing | ✅ Done, awaiting approval |
| 3 | Model training, evaluation & selection | ⏳ Not started |
| 4 | End-to-end ML pipeline (`src/`) | ⏳ Not started |
| 5 | Streamlit web app + deployment config | ⏳ Not started |
| 6 | Final QA & hand-off | ⏳ Not started |

---

## Phase 1 — Documentation & Setup
**Goal:** agree on scope and structure before any code.
- [x] Inspect the dataset schema and quality (`data/`).
- [x] Create `docs/project_overview.md`, `docs/plan.md`, `docs/workflow.md`.
- [x] Create directory skeleton: `data/ notebooks/ src/ tests/ app/ docs/` (+ `models/ reports/`).
- [x] Initialize a local git repository and `.gitignore`.

**Exit criteria:** owner approves the three docs.

## Phase 2 — Exploratory Data Analysis & Preprocessing
**Deliverable:** `notebooks/eda.ipynb` (clearly sectioned, markdown explanations before each step).
1. **Load & schema check** against the data dictionary; note the `soil` / `soil_texture` alias.
2. **Data quality:** missing values, types, duplicate analysis (expect 4,000 duplicates →
   1,000 unique rows), label conflicts, category clean-up (`"loamy clay "`).
3. **Univariate analysis:** distributions, summary stats, class balance before vs after dedupe.
4. **Outlier detection:** IQR and z-score per feature, per-crop box plots; decide keep vs cap
   (agronomic ranges are plausible, so the default is to *keep and document*, not drop).
5. **Bivariate / multivariate:** correlation heatmap, feature-vs-crop plots, soil vs crop
   cross-tab, feature separability.
6. **Feature engineering (candidates, evaluated not assumed):** N:P:K ratios, total NPK,
   temperature–humidity interaction. Result: 0 pp CV gain → not kept.
7. **Preprocessing design:** stratified train/test split *after* dedupe, `StandardScaler` for
   numerics, `OneHotEncoder` for `soil`, `LabelEncoder` for the target, all inside a
   `ColumnTransformer` to avoid leakage.
8. **Summary of findings and decisions** feeding Phase 3–4.

**Status:** all 8 steps done in `notebooks/eda.ipynb`; outputs in `data/processed/` and `reports/figures/`.

**Exit criteria:** notebook runs top-to-bottom cleanly; findings recorded in `docs/phase_logs.md`.

## Phase 3 — Model Training, Evaluation & Selection
**Deliverable:** `notebooks/model_training.ipynb`, `models/`, `reports/`.
1. Candidates: Dummy baseline, Logistic Regression, KNN, SVM, Random Forest, Gradient
   Boosting, XGBoost, LightGBM — each wrapped in the same preprocessing pipeline.
2. Stratified 5-fold CV on the training split; light hyperparameter tuning
   (`RandomizedSearchCV`) for the top candidates.
3. **Performance metrics:** accuracy, macro precision, macro recall, macro F1, plus top-3
   accuracy (since the app shows top 3), confusion matrix.
4. **Cost metrics:** single-row and batch inference latency (median of repeated runs), model
   file size (KB), peak memory during load + predict.
5. **Selection rule:** highest macro F1; if several are within ~1 percentage point, choose the
   cheapest (latency, size). Rationale written down.
6. Serialize the winning full pipeline with joblib + `metadata.json` (features, classes,
   metrics, versions, seed). Save comparison table to `reports/model_comparison.csv` and logs.

**Exit criteria:** comparison table + chosen model with written justification.

## Phase 4 — End-to-End ML Pipeline (`src/`)
**Deliverable:** importable package in `src/` with tests in `tests/`.
- `config.py` — paths, feature names, valid ranges, seed.
- `schema.py` — input validation shared by app, parser and tests.
- `data_ingestion.py` — load CSV, verify columns/types.
- `data_cleaning.py` — strip/normalize categories, dedupe, range checks.
- `feature_engineering.py` — builds the `ColumnTransformer` (+ any approved features).
- `train.py` — CLI to retrain and save model + metadata reproducibly.
- `evaluate.py` — metrics and cost measurement.
- `inference.py` — `CropPredictor`: load once, validate, return top-3 with probabilities.
- `report_parser.py` — text extraction (PDF/DOCX/DOC/TXT) + regex/keyword extraction of the 8 inputs.
- `recommendations.py` — farming tips per crop, plus input-driven tips (e.g. low N, acidic pH).
- Tests for cleaning, schema validation, inference contract, and parser (sample reports in `tests/fixtures/`).

**Exit criteria:** `python -m src.train` reproduces the Phase 3 model; `pytest` passes.

## Phase 5 — Production-Ready Web Application
**Deliverable:** `app/` Streamlit app, `requirements.txt`, `Dockerfile`, run instructions.
1. **Manual input form:** sliders/number inputs bounded by the valid ranges, soil dropdown.
2. **File upload parser:** accepts PDF, DOC/DOCX, TXT; shows extracted values for the user to
   review/edit before predicting; clearly flags missing values.
3. **Prediction & insights:** top-3 crops with confidence bars, actionable tips, input warnings.
4. Clean, responsive layout; friendly error handling; model cached with `st.cache_resource`.
5. `requirements.txt` (pinned), `Dockerfile`, `.dockerignore`, `README.md` run instructions.

**Exit criteria:** app works locally and in Docker with manual input and all three file types.

## Phase 6 — Final QA & Hand-off
- Full test run, notebook re-execution, lint pass, README polish, final summary of results.

---

## Change log
| Date | Change |
|---|---|
| 2026-11-18 | Plan created (Phase 1). |
| 2026-11-18 | Phase 2 done. Added `data/processed/` for derived data; engineered features dropped (no gain). |
