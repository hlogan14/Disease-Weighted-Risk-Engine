# =============================================================================
# sentinel/src/submodels.py
# Sleep Quality sub-model.
#   - Trains a Ridge Regression on the Sleep Health & Lifestyle dataset.
#   - Saves the model + scaler to models/sleep_model.pkl
#   - Exposes get_sleep_score(user_inputs) → float [0.0, 1.0]
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

from config import CSV_FILES, MODELS_DIR, SLEEP_FEATURES, ORDINAL_MAPS

SLEEP_MODEL_PATH = os.path.join(MODELS_DIR, "sleep_model.pkl")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_clean_sleep(path: str) -> pd.DataFrame:
    """Load the Sleep Health and Lifestyle CSV and standardise column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename to internal names
    rename_map = {
        "sleep_duration":         "sleep_duration",
        "quality_of_sleep":       "sleep_quality",
        "stress_level":           "stress_level_num",
        "physical_activity_level":"physical_activity_mins",
        "heart_rate":             "heart_rate",
        "age":                    "age",
        "gender":                 "gender_num",
        "bmi_category":           "bmi_category_num",
    }
    df = df.rename(columns=rename_map)

    # Encode gender
    df["gender_num"] = df["gender_num"].str.lower().map(
        ORDINAL_MAPS["gender"]
    ).fillna(0.5)

    # Encode BMI category → numeric
    bmi_map = {"underweight": 0.1, "normal": 0.3, "normal weight": 0.3,
               "overweight": 0.65, "obese": 1.0}
    df["bmi_category_num"] = df["bmi_category_num"].str.lower().map(
        bmi_map
    ).fillna(0.3)

    # Scale target to [0, 1]
    df["sleep_quality"] = (df["sleep_quality"] - 1) / 9.0

    # Drop rows with nulls in required columns
    required = SLEEP_FEATURES + ["sleep_quality"]
    df = df.dropna(subset=required)

    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sleep_model(save: bool = True) -> Pipeline:
    """
    Train a Ridge Regression sub-model to predict sleep quality score.

    Returns
    -------
    sklearn Pipeline  (MinMaxScaler → Ridge)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    df   = _load_and_clean_sleep(CSV_FILES["sleep"])
    X    = df[SLEEP_FEATURES].values
    y    = df["sleep_quality"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    preds = np.clip(preds, 0.0, 1.0)
    mae   = mean_absolute_error(y_test, preds)
    print(f"[Sleep Sub-model] MAE on test set: {mae:.4f}")

    if save:
        joblib.dump(pipeline, SLEEP_MODEL_PATH)
        print(f"[Sleep Sub-model] Saved → {SLEEP_MODEL_PATH}")

    return pipeline


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_sleep_score(
    sleep_duration:       float,
    sleep_quality_rating: float,   # user 1-10 → normalised inside
    stress_level:         str,     # "low" | "medium" | "high"
    physical_activity:    float,   # hrs/week (user) → converted to mins/day proxy
    bmi:                  float,   # raw BMI value
    heart_rate:           float,
    age:                  float,
    gender:               str,
    model: Pipeline = None,
) -> float:
    """
    Given user inputs, return a sleep quality score in [0.0, 1.0].

    If model is None the saved .pkl is loaded automatically.
    """
    if model is None:
        if not os.path.exists(SLEEP_MODEL_PATH):
            raise FileNotFoundError(
                "Sleep model not found. Run train_sleep_model() first."
            )
        model = joblib.load(SLEEP_MODEL_PATH)

    # Normalise user inputs to match training feature scale
    stress_num      = ORDINAL_MAPS["stress_level"][stress_level.lower()]
    # Convert hrs/week → mins/day (dataset uses mins/day)
    activity_mins   = (physical_activity / 7.0) * 60.0
    gender_num      = ORDINAL_MAPS["gender"][gender.lower()]

    # BMI → category bucket (matches training encoding)
    if bmi < 18.5:
        bmi_cat = 0.1
    elif bmi < 25.0:
        bmi_cat = 0.3
    elif bmi < 30.0:
        bmi_cat = 0.65
    else:
        bmi_cat = 1.0

    # sleep_quality_rating is 1-10, scale to [0,1] to match training target range
    # (we pass it as a feature here, NOT the target)
    sq_norm = (sleep_quality_rating - 1) / 9.0

    feature_vector = np.array([[
        sleep_duration,   # sleep_duration  (hours)
        sq_norm,          # sleep_quality   (scaled 1-10 → 0-1)
        stress_num,       # stress_level_num
        activity_mins,    # physical_activity_mins
        bmi_cat,          # bmi_category_num
        heart_rate,       # heart_rate
        age,              # age
        gender_num,       # gender_num
    ]])

    score = float(np.clip(model.predict(feature_vector)[0], 0.0, 1.0))
    return score