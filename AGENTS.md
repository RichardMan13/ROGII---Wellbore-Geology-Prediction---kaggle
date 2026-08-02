## Agent skills

### Issue tracker

Issues and PRDs for this repo live as markdown files in `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

The default 5 canonical roles are mapped 1:1. See `docs/agents/triage-labels.md`.

### Architecture & Domain Rules

This is a Kaggle Express Template project. Please strictly follow these architecture rules:
- **Data Immutability**: NEVER modify files in `data/raw/`. Write intermediate and final processed data to `data/interim/` and `data/processed/` respectively.
- **Code Organization**: All production code goes into `src/`.
  - `src/config.py`: Global variables, paths, and hyperparameters.
  - `src/features.py`: Feature engineering functions.
  - `src/models.py`: Model architectures (LightGBM, XGBoost, etc.).
  - `src/train.py`, `src/predict.py`, `src/tune.py`, `src/blend.py`: Core pipeline scripts.
- **EDA vs Production**: Use the `notebooks/` directory exclusively for rapid testing and Exploratory Data Analysis. Consolidated pipeline code must reside in `src/`.
- **Task Automation**: Use Invoke (`tasks.py`) for running tasks (e.g., `inv train`, `inv predict`, `inv check`) rather than executing Python scripts manually.
- **Data Leakage**: Maintain strict cross-validation integrity when creating features. Avoid data leakage by applying transformations correctly within the fold loop or via scikit-learn pipelines.
