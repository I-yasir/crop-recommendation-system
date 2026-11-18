# 🌱 Crop Recommendation System

A data-driven web app that recommends the **top 3 crops** for a field from its soil nutrients
(N, P, K), soil pH, soil texture and climate (temperature, humidity, rainfall). Values can be
typed in or **parsed automatically from a soil/lab report** (PDF, DOCX, DOC or TXT). Each
recommendation comes with a confidence score and actionable farming tips.

| Component | Where |
|---|---|
| Exploratory analysis & preprocessing | [`notebooks/eda.ipynb`](notebooks/eda.ipynb) |
| Model training, evaluation & selection | [`notebooks/model_training.ipynb`](notebooks/model_training.ipynb) |
| Modular ML pipeline | [`src/`](src/) |
| Streamlit web app | [`app/streamlit_app.py`](app/streamlit_app.py) |
| Tests (pytest) | [`tests/`](tests/) |
| Project docs, plan and phase logs | [`docs/`](docs/) |

## Quick start (local)

Requires Python 3.12. Optional: `catdoc` or `antiword` to read legacy `.doc` files.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # app + pipeline
python -m src.train                      # builds models/ from data/original_dataset.csv
streamlit run app/streamlit_app.py       # opens http://localhost:8501
```

For notebooks and tests, install the development extras: `pip install -r requirements-dev.txt`.

## Run with Docker

```bash
docker build -t crop-recommender .
docker run --rm -p 8501:8501 crop-recommender
```

Then open <http://localhost:8501>. The image trains the model from the raw CSV during the build, so it does
not depend on local `models/` files. It includes `catdoc` for `.doc` support.

## Using the app

* **Manual input:** set the sliders and soil texture, then click **Recommend crops**.
* **Upload a report:** drop in a PDF, DOCX, DOC or TXT soil report. The parser finds values such as
  "Nitrogen (N): 40 mg/kg", "Soil pH (H2O) 5.9", "P2O5 = 16 ppm" or "Rainfall 2.2 inches", converts units
  (°F→°C, cm/in→mm, P2O5/K2O→P/K) and pre-fills a review form. Any value it can't find is flagged for you to
  enter. Sample reports are in [`tests/fixtures/`](tests/fixtures/).
* **Results:** the top 3 crops with confidence bars, warnings for inputs outside the training range, a
  low-confidence notice when the inputs don't resemble the training data, tips for each crop, a chart of
  probabilities for all crops, and a downloadable text summary.

## Pipeline CLI and tests

```bash
python -m src.train                  # reproduce the selected model (80/20 split, seed 42)
python -m src.train --full-data      # retrain on all 1,000 unique rows
pytest                               # pipeline, parser, inference and app tests
```

## Model summary

* Data: 5,000 rows, of which **1,000 are unique** (duplicates are removed before splitting to avoid leakage).
  10 crops, 100 rows each.
* Candidates: Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting, XGBoost and LightGBM, compared on
  accuracy, macro precision, recall and F1, top-3 accuracy, log loss, latency, size, memory and robustness to noise.
* **Selected: Logistic Regression.** It ties for best accuracy (100 % on the held-out test set), is the smallest
  model (about 3 KB) and the fastest (about 3 ms per prediction), and is the most robust to noisy inputs. See
  [`reports/model_comparison.csv`](reports/model_comparison.csv) and [`docs/phase_logs.md`](docs/phase_logs.md).

## Results at a glance

| | |
|---|---|
| Unique training rows | 1,000 (800 train / 200 test, stratified, seed 42) |
| Selected model | Logistic Regression, scikit-learn pipeline (impute → scale / one-hot → model) |
| Held-out test | accuracy 100 %, macro-F1 1.000, top-3 accuracy 100 % |
| Cost | ~3 KB on disk, ~3 ms per prediction (1 CPU core) |
| Robustness | 100 % accuracy with 10 % Gaussian noise added to the inputs |
| Most important inputs | soil texture > N > K > rainfall > P > temperature > humidity > pH |
| Tests | 64 pytest tests (pipeline, parser, inference, tips, Streamlit app) |

Figures from the analysis are in [`reports/figures/`](reports/figures/).

## Project structure

```
├── app/streamlit_app.py      Streamlit UI (manual form, report upload + review, results)
├── src/                      Pipeline package
│   ├── config.py             paths, features, valid ranges, seed, model settings
│   ├── schema.py             input validation (errors vs warnings), soil-name normalisation
│   ├── data_ingestion.py     load CSV + schema check
│   ├── data_cleaning.py      normalise, de-duplicate, label-conflict and range checks
│   ├── feature_engineering.py  preprocessing ColumnTransformer
│   ├── train.py              CLI: train, evaluate, save model + metadata
│   ├── evaluate.py           metrics and cost measurement
│   ├── inference.py          CropPredictor: top-3 with confidence
│   ├── report_parser.py      PDF/DOCX/DOC/TXT text extraction + regex parameter extraction
│   └── recommendations.py    farming tips
├── notebooks/                eda.ipynb, model_training.ipynb
├── data/                     original_dataset.csv, data_dictionary.csv, processed/
├── models/                   crop_model.joblib (generated), label_encoder.joblib (generated), metadata.json
├── reports/                  model comparison, evaluation log, figures
├── tests/                    pytest suite + fixtures/ sample reports
├── docs/                     project_overview, plan, workflow, phase_logs
├── requirements.txt          pinned runtime dependencies
├── requirements-dev.txt      + notebooks, XGBoost/LightGBM, pytest
└── Dockerfile
```

## Documentation

* [`docs/project_overview.md`](docs/project_overview.md): scope, architecture, data schema, tech stack
* [`docs/plan.md`](docs/plan.md): phase-by-phase roadmap and change log
* [`docs/workflow.md`](docs/workflow.md): validation, testing and context-tracking rules
* [`docs/phase_logs.md`](docs/phase_logs.md): what each phase did, the decisions made and the key numbers

## Limitations

* In this dataset **each crop occurs on exactly one soil texture**, so soil strongly constrains the
  recommendations. Real agronomy is more flexible.
* The data is very clean and probably synthetic. Expect lower accuracy on real field data.
* Nutrient values are used as reported; ppm, mg/kg and kg/ha are not converted, because the dataset does not state
  its unit.
* Scanned (image-only) PDFs are not supported (no OCR).
* Tips are advisory and not a substitute for local agronomic advice.
