# =============================================================================
# sentinel/src/submodels.py
# Sleep Quality sub-model — trained on Sleep Health & Lifestyle dataset.
# Predicts a sleep_score in [0.0, 1.0] from user lifestyle inputs.
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

from config import CSV_FILES, MODELS_DIR, SLEEP_FEATURES, ORDINAL_MAPS

SLEEP_MODEL_PATH = os.path.join(MODELS_DIR, "sleep_model.pkl")


def _load_and_clean_sleep(path: str) -> pd.DataFrame:
    """
    Load Sleep Health & Lifestyle CSV.
    Exact columns present:
      Person ID, Gender, Age, Occupation, Sleep Duration, Quality of Sleep,
      Physical Activity Level, Stress Level, BMI Category, Blood Pressure,
      Heart Rate, Daily Steps, Sleep Disorder
    """
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    out = pd.DataFrame()

    # sleep_duration  (hours — numeric)
    out["sleep_duration"] = pd.to_numeric(df["sleep_duration"], errors="coerce")

    # sleep_quality  (1-10 → target, also used as feature proxy during training)
    out["sleep_quality"] = pd.to_numeric(df["quality_of_sleep"], errors="coerce")

    # stress_level_num  (1-10 numeric in this dataset)
    out["stress_level_num"] = pd.to_numeric(df["stress_level"], errors="coerce")

    # physical_activity_mins  (minutes/day numeric)
    out["physical_activity_mins"] = pd.to_numeric(
        df["physical_activity_level"], errors="coerce"
    )

    # bmi_category_num
    bmi_map = {
        "underweight": 0.1,
        "normal": 0.3,
        "normal weight": 0.3,
        "overweight": 0.65,
        "obese": 1.0,
    }
    out["bmi_category_num"] = (
        df["bmi_category"].str.lower().str.strip().map(bmi_map).fillna(0.3)
    )

    # heart_rate  (numeric bpm)
    out["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")

    # age
    out["age"] = pd.to_numeric(df["age"], errors="coerce")

    # gender_num
    out["gender_num"] = (
        df["gender"].str.lower().str.strip()
        .map(ORDINAL_MAPS["gender"]).fillna(0.5)
    )

    # Scale target quality of sleep to [0, 1]
    out["sleep_quality_scaled"] = np.clip(
        (out["sleep_quality"] - 1) / 9.0, 0.0, 1.0
    )

    out = out.dropna(subset=SLEEP_FEATURES + ["sleep_quality_scaled"])
    return out


def train_sleep_model(save: bool = True) -> Pipeline:
    """Train Ridge Regression sleep sub-model and optionally save to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = _load_and_clean_sleep(CSV_FILES["sleep"])
    X  = df[SLEEP_FEATURES].values
    y  = df["sleep_quality_scaled"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  MinMaxScaler()),
        ("ridge",   Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)

    preds = np.clip(pipeline.predict(X_test), 0.0, 1.0)
    mae   = mean_absolute_error(y_test, preds)
    print(f"  [Sleep Sub-model] MAE on test set: {mae:.4f}")

    if save:
        joblib.dump(pipeline, SLEEP_MODEL_PATH)
        print(f"  [Sleep Sub-model] Saved → {SLEEP_MODEL_PATH}")

    return pipeline


def get_sleep_score(
    sleep_duration:       float,
    sleep_quality_rating: float,
    stress_level:         str,
    physical_activity:    float,   # hrs/week from user → converted to mins/day
    bmi:                  float,
    heart_rate:           float,
    age:                  float,
    gender:               str,
    model:                Pipeline = None,
) -> float:
    """Return a sleep quality score in [0.0, 1.0] for a single user."""
    if model is None:
        if not os.path.exists(SLEEP_MODEL_PATH):
            raise FileNotFoundError(
                "Sleep model not found. Run train_sleep_model() first."
            )
        model = joblib.load(SLEEP_MODEL_PATH)

    # Convert stress string → numeric midpoint on 1-10 scale
    stress_map = {"low": 2.0, "medium": 5.0, "high": 8.0}
    stress_num = stress_map.get(str(stress_level).lower().strip(), 5.0)

    # hrs/week → mins/day
    activity_mins = (physical_activity / 7.0) * 60.0

    gender_num = ORDINAL_MAPS["gender"].get(str(gender).lower().strip(), 0.5)

    # BMI → category bucket
    if bmi < 18.5:
        bmi_cat = 0.1
    elif bmi < 25.0:
        bmi_cat = 0.3
    elif bmi < 30.0:
        bmi_cat = 0.65
    else:
        bmi_cat = 1.0

    # sleep_quality_rating 1-10 passed as-is (model was trained on 1-10 scale)
    feature_vector = np.array([[
        float(sleep_duration),
        float(sleep_quality_rating),
        stress_num,
        activity_mins,
        bmi_cat,
        float(heart_rate),
        float(age),
        gender_num,
    ]])

    score = float(np.clip(model.predict(feature_vector)[0], 0.0, 1.0))
    return score
