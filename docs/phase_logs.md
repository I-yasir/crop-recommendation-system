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
