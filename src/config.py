# =============================================================================
# sentinel/src/config.py
# Central configuration for the Sentinel Weighted Risk Engine.
# All thresholds, ordinal maps, feature lists, and paths live here.
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")

# Raw CSV file names (place these in sentinel/data/)
CSV_FILES = {
    "heart":    os.path.join(DATA_DIR, "heart_disease.csv"),
    "ckd":      os.path.join(DATA_DIR, "chronic_kidney_disease.csv"),
    "lung":     os.path.join(DATA_DIR, "lung_cancer.csv"),
    "diabetes": os.path.join(DATA_DIR, "diabetes.csv"),
    "sleep":    os.path.join(DATA_DIR, "sleep_health.csv"),
}

# ---------------------------------------------------------------------------
# Risk Thresholds  (probability → label)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "high":   0.65,   # >= 0.65  → High
    "medium": 0.35,   # 0.35–0.64 → Medium
                      # < 0.35  → Low
}

# ---------------------------------------------------------------------------
# Ordinal Encodings
# All ordinal inputs map to a float in [0.0, 1.0]
# ---------------------------------------------------------------------------
ORDINAL_MAPS = {
    # Gender
    "gender": {"male": 1.0, "female": 0.0, "other": 0.5},

    # Blood Pressure (self-reported)
    "blood_pressure": {"good": 0.0, "not sure": 0.4, "high": 1.0},

    # Stress Level
    "stress_level": {"low": 0.0, "medium": 0.5, "high": 1.0},

    # Cholesterol (self-reported)
    "cholesterol": {"low": 0.2, "good": 0.0, "high": 1.0, "not sure": 0.4},

    # Smoking
    "smoking": {"no": 0.0, "yes": 1.0},

    # Sugar Consumption
    "sugar_consumption": {"low": 0.0, "medium": 0.5, "high": 1.0},

    # Race  (kept for dataset alignment; see README ethical note)
    "race": {
        "caucasian":        0.25,
        "african american": 1.0,   # higher clinical risk in CKD/Diabetes literature
        "asian":            0.5,
        "other":            0.3,
    },

    # Yes / No / Not Sure  (used for binary medical-history questions)
    "yes_no_notsure": {"no": 0.0, "not sure": 0.4, "yes": 1.0},

    # Secondhand smoke exposure
    "passive_smoke": {"never": 0.0, "sometimes": 0.5, "often": 1.0},

    # Occupational hazard exposure
    "occupational": {"no": 0.0, "not sure": 0.3, "yes": 1.0},

    # Chest pain during activity
    "chest_pain": {"no": 0.0, "sometimes": 0.5, "yes": 1.0},
}

# Population-mean fallback value for "Not Sure" numeric fields
NOT_SURE_NUMERIC = None   # None → replaced by dataset mean at runtime

# ---------------------------------------------------------------------------
# BMI Computation helper boundaries (for [0,1] scaling after compute)
# Typical healthy BMI ~18.5–24.9; dataset range assumed 10–60
# ---------------------------------------------------------------------------
BMI_MIN = 10.0
BMI_MAX = 60.0

# ---------------------------------------------------------------------------
# Feature columns expected from each CSV after preprocessing
# These must match the renamed/cleaned column names produced by train.py
# ---------------------------------------------------------------------------
HEART_FEATURES = [
    "age", "gender", "blood_pressure", "cholesterol", "smoking",
    "bmi", "alcohol", "sugar_consumption", "race", "physical_activity",
    "chest_pain", "stroke", "anemia", "sleep_score",
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
    "sugar_consumption", "race", "hba1c_high", "sleep_score",
]

SLEEP_FEATURES = [
    "sleep_duration", "sleep_quality", "stress_level_num",
    "physical_activity_mins", "bmi_category_num",
    "heart_rate", "age", "gender_num",
]

SLEEP_TARGET = "sleep_quality"   # continuous regression target (1-10 → 0-1 scaled)

# ---------------------------------------------------------------------------
# Disease model targets
# ---------------------------------------------------------------------------
DISEASE_TARGETS = {
    "heart":    "target",
    "ckd":      "class",
    "lung":     "level_binary",   # binarised from Low/Medium/High
    "diabetes": "diabetes",
}