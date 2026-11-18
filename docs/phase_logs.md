# Phase Logs

One entry per phase: what was done, decisions, assumptions, key numbers. Numbers here must
match the saved outputs (`docs/workflow.md` §2).

---

## Phase 1 — Documentation & Setup (2026-11-18)
- Created `docs/project_overview.md`, `docs/plan.md`, `docs/workflow.md`.
- Created skeleton `notebooks/ src/ tests/ app/ models/ reports/` and a local git repo.
- Dataset inspection: 5,000 rows, only 1,000 unique; balanced after dedupe; `soil_texture`/`soil`
  alias; `"loamy clay "` trailing space.
- Approved by owner on 2026-11-18.

## Phase 2 — EDA & Preprocessing (2026-11-18)

**Deliverables**
- `notebooks/eda.ipynb`: 10 sections, executed top to bottom with no errors.
- `data/processed/crop_clean.csv` (1,000 rows), `train.csv` (800), `test.csv` (200),
  `preprocessing_summary.json`.
- `reports/figures/01–09_*.png`: all EDA figures.
- `requirements.txt` (EDA stack). Project virtualenv `.venv/` created by the owner.

**Key findings**
| Topic | Result |
|---|---|
| Missing values | 0 |
| Duplicates | 4,000 / 5,000 exact duplicates removed; 0 label conflicts |
| Class balance | 100 rows per crop after dedupe |
| Outliers | Global IQR flags: humidity 122, rainfall 91, pH 11; **0 within-crop** IQR outliers |
| Correlation | Max \|r\| = 0.75 (P–humidity); no pair > 0.8 |
| Soil vs crop | Cramér's V = 1.0; each crop appears on exactly one soil |
| Mutual information | rainfall 1.60, humidity 1.51, K 1.44, temperature 1.02, P 0.98, N 0.98, pH 0.59 |
| Separability | Logistic Regression and Random Forest on base inputs: CV macro-F1 ≈ 1.00 |
| Engineered features | 0 pp gain → not kept |

**Decisions**
1. De-duplicate before splitting (leakage guard). Verified 0 rows shared between train and test.
2. Keep outliers (plausible and crop-specific); the app warns on out-of-range inputs.
3. Keep all 8 inputs; no engineered features.
4. Preprocessor: numerics `SimpleImputer(median)` → `StandardScaler`; soil
   `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`; target `LabelEncoder`.
5. 80/20 stratified split, `random_state = 42`.

**Assumptions and risks raised for the owner**
- The soil–crop pairing and the near-perfect separability suggest a clean, possibly synthetic
  dataset. Real-world accuracy would be lower, and the model will never recommend a crop on a
  soil it was not paired with in training. To be disclosed in the app.
- Because accuracy will likely saturate, Phase 3 will rely on cost metrics (and top-3 behaviour)
  to choose between models.
- Approved by owner on 2026-11-18.

## Phase 3 — Model Training, Evaluation & Selection (2026-11-18)

**Deliverables**
- `notebooks/model_training.ipynb`: 11 sections, executed top to bottom with no errors.
- `models/crop_model.joblib` (full pipeline, 2.9 KB), `models/label_encoder.joblib`, `models/metadata.json`.
  The `.joblib` files are git-ignored (Phase 1 rule). Regenerate them by re-running the notebook (or with `src/train.py` from Phase 4).
- `reports/model_comparison.csv`, `cv_results.csv`, `evaluation_log.json`, `feature_importance_selected.csv`.
- `reports/figures/10–14_*.png`.

**Candidates:** Dummy baseline, Logistic Regression, KNN, SVM (RBF), Random Forest, Gradient
Boosting, XGBoost 3.4.1, LightGBM 4.7.0. The top 4 by CV log loss (KNN, LightGBM, RF, GB) were tuned with
`RandomizedSearchCV` (20 iterations, objective `neg_log_loss`).

**Key numbers (5-fold CV on 800 training rows; test = 200 held-out rows)**
| Model | CV macro-F1 | CV log loss | Test F1 | 1-row latency | Size | Acc @10 % noise |
|---|---|---|---|---|---|---|
| **LogisticRegression (selected)** | **1.0000** | 0.0366 | **1.0000** | **2.78 ms** | **2.9 KB** | **1.000** |
| SVM (RBF) (runner-up) | 1.0000 | 0.0962 | 1.0000 | 2.76 ms | 18.2 KB | 1.000 |
| KNN | 1.0000 | ≈0 | 1.0000 | 3.12 ms | 62.9 KB | 1.000 |
| LightGBM (tuned) | 1.0000 | 0.0005 | 1.0000 | 3.31 ms | 281 KB | 0.990 |
| RandomForest | 0.9987 | 0.0084 | 1.0000 | 13.3 ms | 147 KB | 0.994 |
| XGBoost | 0.9987 | 0.0161 | 1.0000 | 3.23 ms | 65 KB | 0.969 |
| GradientBoosting | 0.9937 | 0.0125 | 1.0000 | 3.73 ms | 227 KB | 0.946 |
| Dummy baseline | 0.0950 | 32.6 | 0.0989 | — | — | 0.100 |

Full table: `reports/model_comparison.csv`.

**Selection**
- Rule (refined from `plan.md`): shortlist the models within 1 pp of the best CV macro-F1 (11 qualified). Apply
  guards: accuracy at 10 % noise within 2 pp of the best, and CV log loss ≤ 0.15 (XGBoost, default LightGBM and
  default GB failed the noise guard). Then take the lowest summed cost rank (latency, size, memory), where values
  within 10 % of each other share a rank. Ties go to the lower log loss.
- **Selected: Logistic Regression** (default C = 1). It ties for best accuracy, is the smallest model (2.9 KB), is
  among the fastest, and is the most robust to input noise. Its confidence scores are moderate rather than extreme
  (median top-1 confidence 0.98, minimum 0.65), which suits a top-3 display.
- Memory: every small model shows about a 20 MB RSS floor from scikit-learn's lazy imports. Tree and boosting models
  add 7 to 21 MB on top of that.

**Feature importance (permutation, test F1 drop):** soil 0.163 > N 0.141 > K 0.132 > rainfall 0.118 >
P 0.067 > temperature 0.036 > humidity 0.017 > pH 0.015. Soil is the top feature, which confirms the Phase 2 limitation.

**Risks and notes**
- Every model reaches 100 % on the test set because the data is near-perfectly separable. These scores will not
  carry over to real field data.
- Smoke test with the data dictionary's example row (loamy soil) returned rice 0.44, pepper 0.26 and maize 0.14. That
  input sits outside the training patterns, and the model's lower confidence correctly reflects this. Phase 5 should
  surface low confidence to the user.
- Approved by owner on 2026-11-18.

## Phase 4 — End-to-End ML Pipeline (2026-11-18)

**Deliverables:** the `src/` package (10 modules), `tests/` (59 tests, all passing), `tests/fixtures/` sample
reports, and `pytest.ini`. `requirements.txt` now also lists pdfplumber, python-docx, pytest and streamlit.

| Module | Responsibility |
|---|---|
| `config.py` | Paths, features, soil categories, crop classes, training ranges (warn) and hard limits (error), seed, model params, top-k, low-confidence threshold (0.60) |
| `schema.py` | `validate_input()`: errors vs warnings; `normalize_soil()` maps free text such as "clay loam" or "sandy-loam" to the 4 categories |
| `data_ingestion.py` | `load_raw_data()` + `validate_schema()` (`soil_texture` alias) |
| `data_cleaning.py` | normalise strings, drop missing and impossible rows, de-duplicate, fail on label conflicts; `CleaningReport` |
| `feature_engineering.py` | Phase 2 `ColumnTransformer`; `build_model_pipeline()` |
| `train.py` | CLI `python -m src.train [--full-data] [--output-dir] [--skip-cost]`; saves model, encoder and metadata (now including per-crop `crop_profiles`) |
| `evaluate.py` | performance metrics, confusion matrix, latency, size, memory |
| `inference.py` | `CropPredictor.predict()` returns top 3 with confidence, warnings and a `low_confidence` flag |
| `report_parser.py` | text from TXT/PDF/DOCX/DOC; regex extraction with synonyms; unit conversion; reports missing fields |
| `recommendations.py` | general crop tips (advisory) + data-derived comparison with crop ranges + soil health rules |

**Verification**
- `python -m src.train` gives the same split and model as Phase 3. Max probability difference vs the notebook model
  is 0.0, and test accuracy is 1.0. `models/` was regenerated by the CLI.
- Parser: all 8 values extracted correctly from the TXT, DOCX, PDF and DOC fixtures and from the US-units report
  (°F, inches, P2O5, K2O, "clay loam"). The incomplete report correctly lists 5 missing fields.
- Edge cases tested: "pH (H2O)" does not read the 2, "Nitrate" is not N, pH is not confused with P, decimal comma
  vs thousands separator, table rows.

**Notes**
- Legacy `.doc` files need `antiword`, `catdoc` or LibreOffice on the host. This machine has `catdoc`; the Phase 5
  Dockerfile must install one.
- Nutrient units are not converted (ppm, mg/kg and kg/ha are taken as-is). The dataset does not state its nutrient
  units, so the app shows parsed values for the user to review.
- Approved by owner on 2026-11-18.

## Phase 5 — Production-Ready Web Application (2026-11-18)

**Deliverables**
- `app/streamlit_app.py`: UI only; all logic comes from `src/`.
  - **Manual input:** sliders for N, P, K, pH, temperature, humidity and rainfall, plus a soil texture dropdown (inside a form).
  - **Upload a report:** PDF/DOCX/DOC/TXT. Shows how many values were found, unit-conversion notes, the matched
    text and the raw extracted text, then a pre-filled **review form**. Missing fields are marked ⚠️ and must be
    filled before predicting. Unreadable files get a friendly error.
  - **Results:** top-3 cards with confidence bars; a low-confidence warning (below 0.60); out-of-training-range
    warnings; per-crop tip tabs labelled "your data vs. training data" or "general advice"; a probability chart for
    all crops; a downloadable .txt summary.
  - Sidebar: about, model details, and limitations (one soil per crop, clean/synthetic data, units, advisory).
  - Model cached with `st.cache_resource`; a clear message if `models/` is empty.
- `.streamlit/config.toml`: binds to 127.0.0.1 locally, 10 MB upload limit, theme.
- `requirements.txt` (pinned runtime) and `requirements-dev.txt` (+ notebooks, XGBoost/LightGBM, psutil, pytest).
- `Dockerfile` (python:3.12-slim + catdoc, trains the model at build time, non-root user, health check) and
  `.dockerignore`.
- `README.md` with local and Docker run instructions.
- `tests/test_app.py`: 5 headless Streamlit AppTest tests (load, manual flow, missing soil, upload-review flow,
  missing values flagged). **64 tests pass in total.**

**Verification**
- Local: app checked in a browser. The manual path gives guinea corn / maize / cassava with tips. The upload-review
  logic is covered by AppTest using real parsed fixture reports. A by-hand upload through the browser is left to
  the owner's verification steps.
- Docker: `docker build -t crop-recommender .` succeeded (1.14 GB on disk, 266 MB content). The container is
  healthy, runs as `appuser`, serves the UI, and parses the PDF, DOCX, DOC (catdoc) and TXT fixtures correctly.

**Fix made during testing:** the upload tab cleared the parsed report on every rerun when no file object was
present. Clearing now happens only when the uploaded file changes or is removed (`on_change` callback).
