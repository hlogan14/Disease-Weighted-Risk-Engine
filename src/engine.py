# =============================================================================
# sentinel/src/engine.py
# The Sentinel Scoring Engine.
#
# Responsibilities:
#   1. Accept raw user inputs (strings, integers, floats).
#   2. Normalise / encode each input to [0.0, 1.0].
#   3. Call the sleep sub-model to get sleep_score.
#   4. For each disease: build a feature vector and run predict_proba().
#   5. Map probability → High / Medium / Low.
#   6. Return per-disease results + top contributing factors.
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from config import (
    MODELS_DIR, ORDINAL_MAPS, THRESHOLDS,
    HEART_FEATURES, CKD_FEATURES, LUNG_FEATURES, DIABETES_FEATURES,
    BMI_MIN, BMI_MAX,
)
from submodels import get_sleep_score

# ---------------------------------------------------------------------------
# Model cache  (loaded once per session)
# ---------------------------------------------------------------------------
_MODEL_CACHE: Dict[str, object] = {}


def _load_model(name: str):
    if name not in _MODEL_CACHE:
        path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model '{name}' not found at {path}. "
                "Run `python train.py` first."
            )
        _MODEL_CACHE[name] = joblib.load(path)
    return _MODEL_CACHE[name]


# ---------------------------------------------------------------------------
# Normaliser  (the core utility described in the project spec)
# ---------------------------------------------------------------------------

def normalise_input(value, input_type: str, **kwargs) -> float:
    """
    Normalise a single raw user input to [0.0, 1.0].

    Parameters
    ----------
    value      : raw user value (int, float, or str)
    input_type : one of "age" | "bmi" | "ordinal" | "bool" | "range" | "numeric"
    **kwargs   :
        For "ordinal"  → mapping=dict
        For "range"    → lo=float, hi=float
        For "numeric"  → mean=float, std=float  (Z-score then clip)
        For "bool"     → (no extra args needed; Yes/1→1.0, No/0→0.0)

    Returns
    -------
    float in [0.0, 1.0]
    """
    if value is None:
        return float(kwargs.get("default", 0.4))

    if input_type == "ordinal":
        mapping = kwargs["mapping"]
        return float(mapping.get(str(value).lower().strip(),
                                 kwargs.get("default", 0.4)))

    elif input_type == "bool":
        return 1.0 if str(value).lower().strip() in ["yes", "1", "true"] else 0.0

    elif input_type == "range":
        lo, hi = kwargs["lo"], kwargs["hi"]
        return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))

    elif input_type == "numeric":
        # Z-score normalisation then map to [0, 1] via clip
        mean, std = kwargs["mean"], kwargs["std"]
        if std == 0:
            return 0.5
        z = (float(value) - mean) / std
        return float(np.clip((z + 3) / 6.0, 0.0, 1.0))

    elif input_type == "age":
        return float(np.clip((float(value) - 0) / 120.0, 0.0, 1.0))

    elif input_type == "bmi":
        return float(np.clip((float(value) - BMI_MIN) / (BMI_MAX - BMI_MIN),
                             0.0, 1.0))

    raise ValueError(f"Unknown input_type: {input_type}")


def compute_bmi_from_inputs(height_in: float, weight_lbs: float) -> float:
    """Compute raw BMI from imperial inputs."""
    if height_in <= 0:
        return 25.0
    return (weight_lbs / (height_in ** 2)) * 703.0


# ---------------------------------------------------------------------------
# Feature vector builders
# One function per disease.  Each returns an ordered numpy array aligned
# to the DISEASE_FEATURES list in config.py.
# ---------------------------------------------------------------------------

def _build_heart_vector(u: dict, sleep_score: float) -> np.ndarray:
    age_n    = normalise_input(u["age"],               "age")
    gender_n = normalise_input(u["gender"],            "ordinal", mapping=ORDINAL_MAPS["gender"])
    bp_n     = normalise_input(u["blood_pressure"],    "ordinal", mapping=ORDINAL_MAPS["blood_pressure"])
    chol_n   = normalise_input(u["cholesterol"],       "ordinal", mapping=ORDINAL_MAPS["cholesterol"])
    smk_n    = normalise_input(u["smoking"],           "ordinal", mapping=ORDINAL_MAPS["smoking"])
    bmi_raw  = compute_bmi_from_inputs(u["height_in"], u["weight_lbs"])
    bmi_n    = normalise_input(bmi_raw,                "bmi")
    alc_n    = normalise_input(u["alcohol"],           "range",   lo=0, hi=20)
    sug_n    = normalise_input(u["sugar_consumption"], "ordinal", mapping=ORDINAL_MAPS["sugar_consumption"])
    str_n    = normalise_input(u["stress_level"],      "ordinal", mapping=ORDINAL_MAPS["stress_level"])
    cp_n     = normalise_input(u["chest_pain"],        "ordinal", mapping=ORDINAL_MAPS["chest_pain"])
    strk_n   = normalise_input(u["stroke"],            "ordinal", mapping=ORDINAL_MAPS["yes_no_notsure"])
    anm_n    = normalise_input(u["anemia"],            "ordinal", mapping=ORDINAL_MAPS["yes_no_notsure"])
    fhh_n    = normalise_input(u["family_history_heart"], "ordinal", mapping=ORDINAL_MAPS["yes_no_notsure"])
    slp_n    = float(sleep_score)

    return np.array([[age_n, gender_n, bp_n, chol_n, smk_n, bmi_n,
                      alc_n, sug_n, str_n, cp_n, strk_n, anm_n, fhh_n, slp_n]])

def _build_ckd_vector(u: dict, sleep_score: float) -> np.ndarray:
    age_n    = normalise_input(u["age"],               "age")
    gender_n = normalise_input(u["gender"],            "ordinal", mapping=ORDINAL_MAPS["gender"])
    bp_n     = normalise_input(u["blood_pressure"],    "ordinal", mapping=ORDINAL_MAPS["blood_pressure"])
    smk_n    = normalise_input(u["smoking"],           "ordinal", mapping=ORDINAL_MAPS["smoking"])
    bmi_raw  = compute_bmi_from_inputs(u["height_in"], u["weight_lbs"])
    bmi_n    = normalise_input(bmi_raw,                "bmi")
    alc_n    = normalise_input(u["alcohol"],           "range",   lo=0, hi=20)
    sug_n    = normalise_input(u["sugar_consumption"], "ordinal", mapping=ORDINAL_MAPS["sugar_consumption"])
    race_n   = normalise_input(u["race"],              "ordinal", mapping=ORDINAL_MAPS["race"])
    pa_n     = normalise_input(u["physical_activity"], "range",   lo=0, hi=10)
    fhk_n    = normalise_input(u["family_history_kidney"], "ordinal", mapping=ORDINAL_MAPS["yes_no_notsure"])
    str_n    = normalise_input(u["stress_level"],      "ordinal", mapping=ORDINAL_MAPS["stress_level"])
    anm_n    = normalise_input(u["anemia"],            "ordinal", mapping=ORDINAL_MAPS["yes_no_notsure"])
    dm_n     = normalise_input(u["diabetes_diagnosed"],"ordinal", mapping=ORDINAL_MAPS["yes_no_notsure"])
    slp_n    = float(sleep_score)

    return np.array([[age_n, gender_n, bp_n, smk_n, bmi_n,
                      alc_n, sug_n, race_n, pa_n, fhk_n,
                      str_n, anm_n, dm_n, slp_n]])


def _build_lung_vector(u: dict, sleep_score: float) -> np.ndarray:
    age_n    = normalise_input(u["age"],                "age")
    gender_n = normalise_input(u["gender"],             "ordinal", mapping=ORDINAL_MAPS["gender"])
    smk_n    = normalise_input(u["smoking"],            "ordinal", mapping=ORDINAL_MAPS["smoking"])
    bmi_raw  = compute_bmi_from_inputs(u["height_in"],  u["weight_lbs"])
    bmi_n    = normalise_input(bmi_raw,                 "bmi")
    alc_n    = normalise_input(u["alcohol"],            "range",   lo=0, hi=20)
    cp_n     = normalise_input(u["chest_pain"],         "ordinal", mapping=ORDINAL_MAPS["chest_pain"])
    gr_n     = normalise_input(u["genetic_risk_lung"],  "ordinal", mapping=ORDINAL_MAPS["yes_no_notsure"])
    oc_n     = normalise_input(u["occupational_hazards"],"ordinal",mapping=ORDINAL_MAPS["yes_no_notsure"])
    slp_n    = float(sleep_score)

    return np.array([[age_n, gender_n, smk_n, bmi_n,
                      alc_n, cp_n, gr_n, oc_n, slp_n]])


def _build_diabetes_vector(u: dict, sleep_score: float) -> np.ndarray:
    age_n    = normalise_input(u["age"],               "age")
    gender_n = normalise_input(u["gender"],            "ordinal", mapping=ORDINAL_MAPS["gender"])
    bp_n     = normalise_input(u["blood_pressure"],    "ordinal", mapping=ORDINAL_MAPS["blood_pressure"])
    smk_n    = normalise_input(u["smoking"],           "ordinal", mapping=ORDINAL_MAPS["smoking"])
    bmi_raw  = compute_bmi_from_inputs(u["height_in"], u["weight_lbs"])
    bmi_n    = normalise_input(bmi_raw,                "bmi")
    sug_n    = normalise_input(u["sugar_consumption"], "ordinal", mapping=ORDINAL_MAPS["sugar_consumption"])
    str_n    = normalise_input(u["stress_level"],      "ordinal", mapping=ORDINAL_MAPS["stress_level"])
    fhd_n    = normalise_input(u["family_history_diabetes"], "ordinal", mapping=ORDINAL_MAPS["yes_no_notsure"])
    hba1c_n  = normalise_input(u["hba1c_high"],        "ordinal", mapping=ORDINAL_MAPS["yes_no_notsure"])
    slp_n    = float(sleep_score)

    return np.array([[age_n, gender_n, bp_n, smk_n, bmi_n,
                      sug_n, str_n, fhd_n, hba1c_n, slp_n]])

# ---------------------------------------------------------------------------
# Probability → Risk Label
# ---------------------------------------------------------------------------

def map_to_risk_label(probability: float) -> str:
    if probability >= THRESHOLDS["high"]:
        return "High"
    elif probability >= THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Top Contributing Factors
# ---------------------------------------------------------------------------

def get_top_factors(
    feature_names: List[str],
    feature_values: np.ndarray,
    model_coefs: np.ndarray,
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    """
    Returns the top_n features with the highest contribution (coef * value).

    Parameters
    ----------
    feature_names  : list of feature name strings
    feature_values : 1-D array of normalised feature values
    model_coefs    : 1-D array of logistic regression coefficients
    top_n          : how many top features to return

    Returns
    -------
    List of (feature_name, contribution_score) tuples, sorted descending.
    """
    contributions = model_coefs * feature_values
    pairs = list(zip(feature_names, contributions))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]


# Friendly display names for the UI
FEATURE_DISPLAY_NAMES = {
    "age":                  "Age",
    "gender":               "Gender",
    "blood_pressure":       "Blood Pressure",
    "cholesterol":          "Cholesterol Level",
    "smoking":              "Smoking",
    "bmi":                  "Body Mass Index (BMI)",
    "alcohol":              "Alcohol Consumption",
    "sugar_consumption":    "Sugar / Diet",
    "race":                 "Race / Ethnicity",
    "physical_activity":    "Physical Activity",
    "chest_pain":           "Chest Pain History",
    "stroke":               "Prior Stroke",
    "anemia":               "Anemia",
    "sleep_score":          "Sleep Quality",
    "family_history_kidney":"Family History of Kidney Disease",
    "stress_level":         "Stress Level",
    "diabetes_diagnosed":   "Diabetes Diagnosis",
    "genetic_risk_lung":    "Family History of Lung Cancer",
    "occupational_hazards": "Occupational Chemical Exposure",
    "hba1c_high":           "Elevated Blood Sugar / A1C",
    "family_history_heart": "Family History of Heart Disease",
}


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_all_diseases(user_inputs: dict) -> Dict[str, dict]:
    """
    Score all four diseases from a single dict of user inputs.

    Parameters
    ----------
    user_inputs : dict with keys matching the full question set (see app.py)

    Returns
    -------
    dict keyed by disease name, each containing:
        {
            "probability":  float,
            "risk_label":   "High" | "Medium" | "Low",
            "top_factors":  [ ("feature_display_name", contribution), ... ]
        }
    """
    # Step 1 – get sleep score from sub-model
    bmi_raw   = compute_bmi_from_inputs(
        user_inputs["height_in"], user_inputs["weight_lbs"]
    )
    sleep_score = get_sleep_score(
        sleep_duration       = float(user_inputs["hours_of_sleep"]),
        sleep_quality_rating = float(user_inputs["sleep_quality"]),
        stress_level         = user_inputs["stress_level"],
        physical_activity    = float(user_inputs["physical_activity"]),
        bmi                  = bmi_raw,
        heart_rate           = 72.0,   # population mean; not asked in UI
        age                  = float(user_inputs["age"]),
        gender               = user_inputs["gender"],
    )

    disease_configs = {
        "Heart Disease": {
            "model_name":   "heart",
            "features":     HEART_FEATURES,
            "vec_builder":  _build_heart_vector,
        },
        "Chronic Kidney Disease": {
            "model_name":   "ckd",
            "features":     CKD_FEATURES,
            "vec_builder":  _build_ckd_vector,
        },
        "Lung Cancer": {
            "model_name":   "lung",
            "features":     LUNG_FEATURES,
            "vec_builder":  _build_lung_vector,
        },
        "Diabetes Type 2": {
            "model_name":   "diabetes",
            "features":     DIABETES_FEATURES,
            "vec_builder":  _build_diabetes_vector,
        },
    }

    results = {}

    for disease_name, cfg in disease_configs.items():
        model     = _load_model(cfg["model_name"])
        vec       = cfg["vec_builder"](user_inputs, sleep_score)
        prob      = float(model.predict_proba(vec)[0][1])
        label     = map_to_risk_label(prob)
        coefs     = model.named_steps["logreg"].coef_[0]
        top_raw   = get_top_factors(cfg["features"], vec[0], coefs)
        top_named = [
            (FEATURE_DISPLAY_NAMES.get(f, f), round(float(v), 4))
            for f, v in top_raw
        ]

        results[disease_name] = {
            "probability":  round(prob, 4),
            "risk_label":   label,
            "top_factors":  top_named,
            "sleep_score":  round(sleep_score, 4),
        }

    return results
