You are an expert Machine Learning Engineer and Full-Stack Developer. We are going to build a production-grade project titled: **"A Data-Driven Crop Recommendation Web Application Using Machine Learning"**. 

The dataset is already located in the `data/` folder. 

### CRITICAL RULES & CONSTRAINTS (Read Carefully):
1. **Docs First (No Code Yet):** Before writing any implementation code, you must initialize a `docs/` folder and create three core markdown files:
   - `docs/project_overview.md` (Scope, architecture, tech stack, data schema)
   - `docs/plan.md` (Detailed phase-by-phase roadmap)
   - `docs/workflow.md` (Rules for validation, testing, and context tracking)
   After creating these files, **STOP** and wait for my approval before writing any code.
2. **Phase-by-Phase Execution:** We will execute this project strictly phase-by-phase. After completing each phase, you must pause, summarize what was accomplished, and wait for my explicit instruction before starting the next phase.
3. **No Hallucinations / Strict Grounding:** Refer to the files in `docs/` before every phase to ensure alignment with our original goals.

---

### Project Requirements & Scope:

#### Phase 1: Documentation & Setup (Do this FIRST)
- Create `docs/project_overview.md`, `docs/plan.md`, and `docs/workflow.md`.
- Set up the initial project directory structure (`data/`, `notebooks/`, `src/`, `tests/`, `app/`, `docs/`).

#### Phase 2: Exploratory Data Analysis & Preprocessing
- Analyze the dataset inside `data/` in a Jupyter Notebook (`notebooks/eda.ipynb`).
- Handle missing values, outlier detection, feature engineering, scaling, and encoding.
- Ensure the notebook is organized into clearly explained, modular sections.

#### Phase 3: Model Training, Evaluation & Selection
- Train multiple candidate models (e.g., Random Forest, XGBoost, LightGBM, Gradient Boosting, SVM).
- Evaluate models based on **both performance metrics** (Accuracy, F1-Score, Precision, Recall) **and cost** (inference latency, model size, memory footprint).
- Select the optimal model, serialize it (using joblib/pickle), and save evaluation logs.

#### Phase 4: End-to-End ML Pipeline
- Build modular Python scripts in `src/` for data ingestion, cleaning, feature transformation, and inference pipeline wrapping.

#### Phase 5: Production-Ready Web Application
- Build a clean, responsive web application (e.g., using Streamlit or FastAPI + Streamlit) that includes:
  1. **Manual Input Form:** Sliders/inputs for environmental and soil parameters (e.g., Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall).
  2. **File Upload Parser:** Ability for users to upload soil/lab report files in **PDF, DOC/DOCX, or TXT** format, automatically parsing the text to extract the required parameters using regex/NLP.
  3. **Prediction & Insights:** Displays top three recommended crop, confidence score, and actionable farming tips.
- Include deployment configuration (`requirements.txt`, `Dockerfile`, or deployment scripts).

---

Let's begin. Please start exclusively by creating the `docs/` folder and the required markdown tracking files (`project_overview.md`, `plan.md`, `workflow.md`), then present them for my review.
