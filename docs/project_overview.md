# Project Overview

**Title:** A Data-Driven Crop Recommendation Web Application Using Machine Learning
**Source spec:** [`docs/instructions.md`](instructions.md)
**Last updated:** 2026-11-18 (Phase 6)

---

## 1. Goal

Recommend the most suitable crops for a field from its soil nutrients, soil texture and
climate readings. A user either types the values into a form or uploads a soil/lab report
(PDF, DOC/DOCX, TXT); the app returns the **top 3 crops**, a **confidence score** for each,
and **actionable farming tips**.

## 2. Scope

### In scope
- Exploratory data analysis and preprocessing in `notebooks/eda.ipynb`.
- Training and comparing several classifiers (Random Forest, XGBoost, LightGBM, Gradient
  Boosting, SVM, plus a simple baseline), judged on **performance** (accuracy, macro
  precision / recall / F1) **and cost** (inference latency, model file size, memory footprint).
- A reusable, tested ML pipeline in `src/` (ingestion, cleaning, transformation, inference).
- A Streamlit web app in `app/` with manual input, report-file parsing, and predictions.
- Deployment configuration: `requirements.txt` and a `Dockerfile`.

### Out of scope
- Collecting new data, or crops outside the 10 classes in the dataset.
- User accounts, a database, or storing uploaded reports beyond the request.
- OCR of scanned/image-only PDFs (text-based PDFs only; a scanned PDF gets a clear error
  telling the user to enter values manually).
- Cloud deployment itself (we ship the config; hosting is the owner's choice).

## 3. Dataset (grounded in `data/`)

| File | Contents |
|---|---|
| `data/original_dataset.csv` | 5,000 rows × 9 columns, raw training data |
| `data/data_dictionary.csv` | Column descriptions, types and example values |

### 3.1 Schema

| Column | Type | Observed range | Meaning |
|---|---|---|---|
| `N` | numeric | 0 – 120 | Soil nitrogen |
| `P` | numeric | 5 – 80 | Soil phosphorus |
| `K` | numeric | 5 – 85 | Soil potassium |
| `temperature` | numeric | 10.01 – 34.95 °C | Ambient temperature |
| `humidity` | numeric | 14.26 – 95.00 % | Relative humidity |
| `ph` | numeric | 5.01 – 8.87 | Soil pH |
| `rainfall` | numeric | 20.21 – 298.56 mm | Rainfall |
| `soil` | categorical (4) | sandy loam, loamy, clay, loamy clay | Soil texture |
| `crop_label` | categorical (10) — **target** | see below | Crop to recommend |

**Target classes (10):** beans, cassava, guinea_corn, maize, orange, pepper, rice, soybean,
tomatoes, yam.

### 3.2 Data-quality findings from the Phase 1 inspection

These shape the preprocessing plan and must be confirmed in the Phase 2 notebook.

1. **No missing values and no unparseable numbers** in any column.
2. **Heavy exact duplication: only 1,000 of the 5,000 rows are unique.** Each unique row
   appears between 1 and 23 times. No two identical feature rows carry different labels.
   *Consequence:* duplicates must be removed **before** the train/test split. Otherwise the
   same row lands in both sets and test scores are inflated (data leakage).
3. **After de-duplication the classes are perfectly balanced** (100 rows per crop). The raw
   file looks imbalanced (cassava 299 vs soybean 587) only because of the duplication.
4. **Column-name mismatch:** the data dictionary calls the soil column `soil_texture`; the CSV
   calls it `soil`. The pipeline uses the CSV name `soil` and the docs note the alias.
5. **Dirty category value:** `"loamy clay "` has a trailing space. Category strings must be
   stripped and lower-cased during cleaning.
6. Unique soil distribution: sandy loam 400, loamy 300, clay 200, loamy clay 100.
7. *(Phase 2)* **Soil type perfectly determines a crop group** (Cramér's V = 1.0): every crop
   occurs on exactly one soil. This is a dataset limitation that the app must disclose.
8. *(Phase 2)* The classes are almost perfectly separable (reference models reach CV macro-F1 ≈ 1.0),
   so model selection in Phase 3 will be driven mainly by cost metrics.
9. *(Phase 3)* Confirmed: 11 of 12 model variants reach CV macro-F1 ≥ 0.99; the selected model is a
   **Logistic Regression** pipeline (2.9 KB, ~2.8 ms per prediction, 100 % test accuracy).

With ~1,000 unique samples, evaluation will use **stratified k-fold cross-validation** plus a
held-out stratified test split, rather than relying on a single split.

## 4. Architecture

```
                ┌────────────────────────── Offline (training) ──────────────────────────┐
 data/*.csv ──► src/data_ingestion ──► src/data_cleaning ──► src/feature_engineering ──►
                (load + schema check)   (strip, dedupe,       (scale numerics, one-hot
                                         validate ranges)      soil, encode labels)
                                                                         │
                                                       src/train / src/evaluate
                                               (CV, metrics + latency/size/memory)
                                                                         │
                                         models/  (best pipeline .joblib, label encoder,
                                                   metadata.json)  +  reports/ (eval logs)
                └─────────────────────────────────────────────────────────────────────────┘

                ┌────────────────────────── Online (serving) ─────────────────────────────┐
 Manual form ───────────────────────────┐
                                        ├──► src/inference (load model once, validate
 Uploaded PDF / DOCX / DOC / TXT ──►    │      input, predict_proba → top-3)
   src/report_parser (text extraction   │                 │
   + regex/keyword parameter extraction)┘                 ▼
                                              src/recommendations (farming tips per crop)
                                                          │
                                              app/ (Streamlit UI: results, confidence, tips)
                └─────────────────────────────────────────────────────────────────────────┘
```

Key design choices:
- **One serialized scikit-learn `Pipeline`** (preprocessing + model) so training and serving
  can never apply different transformations.
- **The app holds no ML logic.** It calls `src/` functions only, so the same code is unit-tested
  and reused.
- **Input validation in one place** (`src/` schema module) shared by form, parser and tests.

## 5. Planned repository layout

```
Crop Recomendation System/
├── app/                 # streamlit_app.py: Streamlit UI only (manual form, report upload/review, results)
├── .streamlit/          # Streamlit server/theme config (localhost by default, 10 MB uploads)
├── data/                # Raw data (read-only) + data dictionary
│   └── processed/       # Cleaned data + train/test split (generated in Phase 2)
├── docs/                # instructions, overview, plan, workflow, phase logs
├── models/              # crop_model.joblib, label_encoder.joblib, metadata.json (generated, Phase 3)
├── notebooks/           # eda.ipynb (Phase 2), model_training.ipynb (Phase 3)
├── reports/             # Evaluation logs, comparison tables, figures (generated)
├── src/                 # Pipeline package: config, schema, data_ingestion, data_cleaning,
│                        #   feature_engineering, train, evaluate, inference, report_parser, recommendations
├── tests/               # pytest suite (59 tests) + fixtures/ sample reports (TXT, DOCX, PDF, DOC)
├── pytest.ini           # pytest config (adds project root to the import path)
├── requirements.txt     # Pinned runtime deps (app + pipeline)
├── requirements-dev.txt # + notebooks, XGBoost/LightGBM, pytest
├── README.md            # Setup and run instructions
└── Dockerfile           # python:3.12-slim + catdoc; trains the model at build time
```

`models/` and `reports/` were added beyond the six folders in the spec to keep generated
artifacts out of `src/` and `data/`.

## 6. Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.12 |
| Data / EDA | pandas, NumPy, matplotlib, seaborn, Jupyter |
| Modelling | scikit-learn, XGBoost, LightGBM |
| Serialization | joblib |
| Cost measurement | `time.perf_counter`, file size on disk, `tracemalloc` / `psutil` |
| Report parsing | pdfplumber (PDF), python-docx (DOCX), `antiword` → `catdoc` → LibreOffice for legacy DOC, built-in for TXT; regex + synonym and unit matching (°F→°C, cm/in→mm, P2O5/K2O→P/K) |
| Web app | Streamlit |
| Testing | pytest |
| Deployment | `requirements.txt`, `Dockerfile` |
| Version control | Local git repository (no remote yet) |

## 7. Success criteria

- Notebook and pipeline reproduce the same metrics from a fixed random seed.
- Selected model reaches strong macro-F1 on de-duplicated cross-validation and held-out test
  (target ≥ 0.90; the real number is reported honestly whatever it is).
- Single-prediction latency well under 100 ms on a laptop CPU.
- Parser extracts all 8 inputs from well-formed sample reports in PDF, DOCX and TXT, and
  clearly lists any value it could not find so the user can fill it in.
- `pytest` passes; `docker build` and `streamlit run` both work.
