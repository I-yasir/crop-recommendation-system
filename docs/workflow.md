# Workflow, Validation & Context-Tracking Rules

These rules apply to every phase. They exist to keep the project grounded in the spec,
reproducible, and reviewable.

## 1. Phase protocol

1. **Start of phase — ground in context.** Re-read `docs/instructions.md`,
   `docs/project_overview.md`, `docs/plan.md`, this file and `docs/phase_logs.md` (once it
   exists). Confirm the phase's tasks and exit criteria from `plan.md` before writing code.
2. **Docs first.** If the phase changes scope, architecture or decisions, update the docs
   *before* the code.
3. **Do only this phase's work.** No work from later phases, no unrequested features.
4. **End of phase — summarize and pause.** Report: what was done, files created/modified,
   key results/decisions, open questions. Update `plan.md` status and `docs/phase_logs.md`.
   Then **stop and wait for explicit approval**.
5. **Commit** at the end of each phase to the local git repo
   (message: `Phase N: <summary>`). No remote push unless the owner asks.

## 2. Grounding rules (no hallucination)

- Every claim about the data (counts, ranges, classes, distributions) must come from code
  run on `data/original_dataset.csv`, not from assumptions or general knowledge of other
  crop datasets.
- Metrics reported in docs must match the saved files in `reports/`.
- Farming tips must be general, conservative agronomic guidance, labelled as advisory,
  and never presented as data-derived unless they are.
- If something is ambiguous, record the assumption in `docs/phase_logs.md` and raise it at
  the phase pause.

## 3. Data & ML validation rules

- `data/` is **read-only**. Cleaned data is produced by code, never by editing the CSV.
- **Leakage guards:** de-duplicate *before* splitting; fit scalers/encoders only on training
  folds (via `Pipeline` / `ColumnTransformer`); the test split is used once, at the end.
- **Reproducibility:** one global `RANDOM_STATE = 42`; library versions recorded in
  `models/metadata.json`.
- **Evaluation:** stratified CV for model choice; macro-averaged metrics (all 10 crops count
  equally); report mean ± std across folds; cost metrics measured with repeated runs.
- **Input validation:** all inputs (form, parsed report, API) pass through the same schema
  with the numeric ranges from the data; out-of-range values produce a warning, not a crash.

## 4. Code quality rules

- Modular, single-responsibility functions; type hints; docstrings on public functions.
- No hard-coded paths outside `src/config.py`.
- The app (`app/`) contains UI only; all logic lives in `src/`.
- Errors are handled with clear, user-facing messages (especially file parsing).
- Notebooks: a markdown cell explains each section's purpose and conclusions; they must
  run top-to-bottom on a clean kernel.

## 5. Testing rules

- `pytest` for `src/`: cleaning, schema validation, feature pipeline shape, inference
  contract (returns 3 crops, probabilities in [0, 1], sorted), and report parsing for
  TXT / DOCX / PDF fixtures, including missing-value and unit-variant cases.
- Tests must pass before a phase is marked done.
- Manual check of the Streamlit app for each input path before Phase 5 sign-off.

## 6. Context tracking

| File | Purpose | Updated |
|---|---|---|
| `docs/instructions.md` | Original spec from the owner (source of truth) | Only by owner |
| `docs/project_overview.md` | Scope, architecture, stack, data schema | When any of these change |
| `docs/plan.md` | Roadmap, status per phase, change log | Start/end of each phase |
| `docs/workflow.md` | These rules | When a rule changes |
| `docs/phase_logs.md` | Per-phase summary, decisions, assumptions, metrics | End of each phase (created in Phase 2) |

## 7. Definition of done (per phase)

- All tasks in `plan.md` for the phase are checked.
- Exit criteria met and verified (notebook runs / tests pass / app runs).
- Docs updated, commit made, summary delivered, approval received.
