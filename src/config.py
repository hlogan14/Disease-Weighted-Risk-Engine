# =============================================================================
# sentinel/src/config.py
# Central configuration for the Sentinel Weighted Risk Engine.
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CSV_FILES = {
    "heart":       os.path.join(DATA_DIR, "heart_disease.csv"),
    "ckd":         os.path.join(DATA_DIR, "chronic_kidney_disease.csv"),
    "lung":        os.path.join(DATA_DIR, "lung_cancer.csv"),
    "diabetes":    os.path.join(DATA_DIR, "diabetes.csv"),
    "sleep":       os.path.join(DATA_DIR, "sleep_health.csv"),
    "stroke":      os.path.join(DATA_DIR, "stroke.csv"),
    "alzheimers":  os.path.join(DATA_DIR, "alzheimers.csv"),
}

# ---------------------------------------------------------------------------
# Risk Thresholds
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "high":   0.65,
    "medium": 0.35,
}

# ---------------------------------------------------------------------------
# Ordinal Encodings
# ---------------------------------------------------------------------------
ORDINAL_MAPS = {
    "gender": {
        "male": 1.0, "female": 0.0, "other": 0.5,
        "m": 1.0, "f": 0.0,
    },
    "blood_pressure": {
        "good": 0.0, "normal": 0.0,
        "not sure": 0.4,
        "high": 1.0, "yes": 1.0, "1": 1.0,
    },
    "stress_level": {
        "low": 0.0,  "1": 0.1, "2": 0.2,
        "medium": 0.5, "3": 0.3, "4": 0.4, "5": 0.5,
        "high": 1.0, "6": 0.6, "7": 0.7, "8": 0.8, "9": 0.9, "10": 1.0,
    },
    "cholesterol": {
        "low": 0.2, "good": 0.0, "normal": 0.0,
        "high": 1.0, "not sure": 0.4,
    },
    "smoking": {
        "no": 0.0, "0": 0.0, "never smoked": 0.0,
        "formerly smoked": 0.5,
        "yes": 1.0, "1": 1.0, "smokes": 1.0,
    },
    "sugar_consumption": {
        "low": 0.0, "medium": 0.5, "high": 1.0,
    },
    "race": {
        "caucasian": 0.25, "white": 0.25,
        "african american": 1.0, "black": 1.0,
        "asian": 0.5, "other": 0.3,
        "0": 0.25, "1": 1.0, "2": 0.5, "3": 0.3,
    },
    "yes_no_notsure": {
        "no": 0.0,  "0": 0.0,
        "not sure": 0.4,
        "yes": 1.0, "1": 1.0,
    },
    "chest_pain": {
        "no": 0.0, "sometimes": 0.5, "yes": 1.0,
    },
    "occupational": {
        "no": 0.0, "not sure": 0.3, "yes": 1.0,
    },
    "exercise_habits": {
        "never": 0.0, "low": 0.2, "rarely": 0.2,
        "sometimes": 0.5, "medium": 0.5,
        "regular": 0.8, "often": 0.8,
        "high": 1.0, "always": 1.0,
    },
    "mental_activity": {
        "rarely": 0.0, "sometimes": 0.5, "often": 1.0,
    },
}

BMI_MIN = 10.0
BMI_MAX = 60.0

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------
HEART_FEATURES = [
    "age", "gender", "blood_pressure", "cholesterol", "smoking",
    "bmi", "alcohol", "sugar_consumption", "stress_level",
    "chest_pain", "stroke", "anemia", "family_history_heart",
    "sleep_score",
]

CKD_FEATURES = [
    "age", "gender", "blood_pressure", "smoking", "bmi",
    "alcohol", "sugar_consumption", "race", "physical_activity",
    "family_history_kidney", "stress_level", "anemia",
    "diabetes_diagnosed", "sleep_score",
]

LUNG_FEATURES = [
    "age", "gender", "smoking", "bmi", "alcohol",
    "chest_pain", "genetic_risk_lung", "occupational_hazards",
    "sleep_score",
]

DIABETES_FEATURES = [
    "age", "gender", "blood_pressure", "smoking", "bmi",
    "sugar_consumption", "stress_level", "family_history_diabetes",
    "hba1c_high", "sleep_score",
]

STROKE_FEATURES = [
    "age", "gender", "blood_pressure", "heart_disease_history",
    "smoking", "bmi", "sugar_consumption", "stress_level",
    "irregular_heartbeat", "sleep_score",
]

ALZHEIMERS_FEATURES = [
    "age", "gender", "bmi", "smoking", "alcohol",
    "physical_activity", "sleep_score", "stress_level",
    "family_history_alzheimers", "mental_activity",
    "blood_pressure", "diabetes_diagnosed",
]

SLEEP_FEATURES = [
    "sleep_duration", "sleep_quality", "stress_level_num",
    "physical_activity_mins", "bmi_category_num",
    "heart_rate", "age", "gender_num",
]

# ---------------------------------------------------------------------------
# Disease targets
# ---------------------------------------------------------------------------
DISEASE_TARGETS = {
    "heart":      "heart_disease_status",
    "ckd":        "diagnosis",
    "lung":       "level_binary",
    "diabetes":   "diabetes_risk_binary",
    "stroke":     "stroke",
    "alzheimers": "diagnosis",
}
