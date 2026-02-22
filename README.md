# Sentinel – Weighted Chronic Disease Risk Engine

Sentinel is a self-screening tool that estimates your exposure to six chronic diseases based on lifestyle, medical history, and demographic inputs. It uses a set of trained machine learning models to produce a risk rating of High, Medium, or Low for each condition — along with the top factors driving that rating.

This is a screening tool only. It does not constitute medical advice. Always consult a qualified healthcare professional for any health concerns.

---

## Diseases Covered

- Heart Disease
- Chronic Kidney Disease
- Lung Cancer
- Type 2 Diabetes
- Stroke
- Alzheimer's Disease

---

## Project Structure

```
├── data/                        # CSV datasets
│   ├── heart_disease.csv
│   ├── chronic_kidney_disease.csv
│   ├── lung_cancer.csv
│   ├── diabetes.csv
│   ├── sleep_health.csv
│   ├── stroke.csv
│   └── alzheimers.csv
├── models/                      # Trained .pkl files (auto-created on first training run)
├── src/
│   ├── config.py                # Thresholds, feature maps, and ordinal encodings
│   ├── submodels.py             # Sleep quality sub-model (Ridge Regression)
│   ├── train.py                 # Model training and weight extraction
│   ├── engine.py                # Scoring engine and disease score functions
│   └── validate_columns.py      # Pre-flight CSV column checker
└── app.py                       # Streamlit UI
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

### 2. Validate your CSVs

Run this from the `src` directory to confirm all expected columns are present before training:

```bash
cd src
python validate_columns.py
```

### 3. Train all models

```bash
python src/train.py --disease all
```

This trains the sleep sub-model first, then all six disease models. Trained pipelines are saved to the `models/` folder.

### 4. Launch the app

```bash
streamlit run app.py
```

---

## How It Works

When you submit the form, Sentinel takes your 27 answers and runs them through a pipeline for each disease:

1. Your sleep quality is estimated by a separate Ridge Regression sub-model trained on the Sleep Health dataset. This produces a `sleep_score` between 0.0 and 1.0.
2. Your inputs are normalized to a [0.0, 1.0] scale using the same encoding scheme that was used during training.
3. Each disease model (Logistic Regression) produces a probability score.
4. That probability maps to a risk label: High (>= 0.65), Medium (>= 0.35), or Low.
5. The top three features contributing to each score are surfaced in the UI.

---

## How Weights Are Derived

1. Each dataset is loaded and columns are mapped to the user-facing questions.
2. Missing values are filled with the population mean. "Not Sure" answers also use the mean.
3. All features are scaled to [0.0, 1.0] using `MinMaxScaler`.
4. A `LogisticRegression(class_weight='balanced')` pipeline is fitted on an 80/20 train/test split.
5. `model.coef_[0]` gives the raw feature weights, which are saved inside the `.pkl` file.
6. At inference time, `coef * normalised_feature_value` gives each feature's contribution to the score.

---

## Input Types

| Input Type       | Example              | Handling                                     |
|------------------|----------------------|----------------------------------------------|
| Objective        | Age, BMI, Height     | Min-Max scaled to [0, 1]                     |
| Ordinal          | Stress, Sugar        | Fixed mapping e.g. Low=0.0, Med=0.5, Hi=1.0 |
| Self-reported    | Blood Pressure       | Yes/No/Not Sure -> 1.0 / 0.0 / mean (0.4)   |
| Sub-model output | Sleep Quality Score  | Ridge regression output clipped to [0, 1]    |

---

## A Note on Race

Race is included to align with the clinical datasets used for training, where certain demographic groups face statistically different risk profiles. For example, African American patients face higher CKD risk according to clinical literature. This field is treated as a demographic risk stratifier, not a deterministic factor. Results should always be read alongside the disclaimer shown in the app.