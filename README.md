# 🛡️ Sentinel – Weighted Chronic Disease Risk Engine

## Project Structure
```
sentinel/
├── data/                        # Place your raw Kaggle CSVs here
│   ├── heart_disease.csv
│   ├── chronic_kidney_disease.csv
│   ├── lung_cancer.csv
│   ├── diabetes.csv
│   └── sleep_health.csv
├── models/                      # Auto-created; stores trained .pkl files
├── src/
│   ├── config.py                # All thresholds, feature maps, ordinal encodings
│   ├── submodels.py             # Sleep quality sub-model (Ridge Regression)
│   ├── train.py                 # Weight extraction + model training
│   ├── engine.py                # Scoring engine, normaliser, disease score fn
│   └── validate_columns.py     # Pre-flight CSV column checker
└── app.py                       # Streamlit UI
```

## Quickstart

### 1. Install dependencies
```bash
pip install streamlit scikit-learn pandas numpy joblib
```

### 2. Place CSVs in `sentinel/data/` using exact filenames above.

### 3. Validate your CSVs
```bash
cd sentinel/src
python validate_columns.py
```

### 4. Train all models
```bash
python train.py --disease all
```
This trains the sleep sub-model first, then all four disease models.
Trained pipelines are saved to `sentinel/models/`.

### 5. Launch the app
```bash
cd sentinel
streamlit run app.py
```

---

## Architecture Overview

```
User Inputs (22 questions)
         │
         ▼
┌─────────────────────┐
│  Sleep Sub-Model    │  ← Ridge Regression on Sleep Health dataset
│  (submodels.py)     │  → sleep_score [0.0–1.0]
└─────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│                   engine.py                          │
│  normalise_input() → [0.0, 1.0] per feature          │
│  _build_<disease>_vector() → numpy array             │
│  LogisticRegression.predict_proba() → probability    │
│  map_to_risk_label() → High / Medium / Low           │
│  get_top_factors()  → top 3 contributing features    │
└──────────────────────────────────────────────────────┘
         │
         ▼
  Streamlit UI (app.py)
  4 × disease cards with risk badge + progress bar
  + sleep score callout + top factors + description
```

## How Weights Are Derived
1. Each Kaggle CSV is loaded and features are mapped to user-facing questions.
2. Missing values are imputed with the **population mean** ("Not Sure" = mean).
3. All features are scaled to **[0.0, 1.0]** with `MinMaxScaler`.
4. A `LogisticRegression(class_weight='balanced')` pipeline is fitted.
5. `model.coef_[0]` gives the raw weights — these are saved inside the `.pkl`.
6. At inference time, `coef * normalised_feature_value` = contribution.

## Subjective vs. Objective Inputs
| Input Type      | Example              | Handling                                   |
|-----------------|----------------------|--------------------------------------------|
| Objective       | Age, BMI, Height     | Min-Max scaled to [0,1]                    |
| Ordinal         | Stress, Sugar        | Fixed mapping e.g. Low=0.0, Med=0.5, Hi=1.0|
| Self-reported   | Blood Pressure       | Yes/No/Not Sure → 1.0 / 0.0 / mean(0.4)   |
| Sub-model output| Sleep Quality Score  | Ridge regression output clipped to [0,1]   |

## Adding a New Disease
1. Add CSV path to `CSV_FILES` in `config.py`.
2. Add feature list constant (e.g. `ALZHEIMERS_FEATURES`) in `config.py`.
3. Add a `load_alzheimers()` function in `train.py`.
4. Add a `_build_alzheimers_vector()` function in `engine.py`.
5. Register both in the `LOADERS` dict (`train.py`) and `disease_configs` (`engine.py`).

## Ethical Note on Race Feature
Race is included to align with dataset features (e.g., African American patients
face statistically higher CKD risk in clinical literature). This should be
treated as a **demographic risk stratifier**, not a deterministic factor.
Outputs should always be accompanied by the disclaimer shown in the app.