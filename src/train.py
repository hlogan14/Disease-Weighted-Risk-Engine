# =============================================================================
# sentinel/src/train.py
# =============================================================================

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer

from config import (
    CSV_FILES, MODELS_DIR,
    HEART_FEATURES, CKD_FEATURES, LUNG_FEATURES, DIABETES_FEATURES,
    STROKE_FEATURES, ALZHEIMERS_FEATURES,
    DISEASE_TARGETS, ORDINAL_MAPS, BMI_MIN, BMI_MAX,
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def scale_bmi(series: pd.Series) -> pd.Series:
    return np.clip((series - BMI_MIN) / (BMI_MAX - BMI_MIN), 0.0, 1.0)

def scale_age(series: pd.Series) -> pd.Series:
    return np.clip(series / 120.0, 0.0, 1.0)

def ordinal(series: pd.Series, mapping: dict, default: float = 0.4) -> pd.Series:
    return series.astype(str).str.lower().str.strip().map(mapping).fillna(default)

def binary(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip().apply(
        lambda x: 1.0 if x in ["yes", "1", "true"] else 0.0
    )

def save_model(pipeline, name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    joblib.dump(pipeline, path)
    print(f"  ✓ Saved → {path}")

def print_weights(model, feature_names: list, disease: str):
    coefs = model.coef_[0]
    pairs = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  📊 [{disease.upper()}] Feature Weights (sorted by |weight|):")
    for feat, w in pairs:
        bar  = "█" * min(int(abs(w) * 20), 40)
        sign = "+" if w >= 0 else "-"
        print(f"    {feat:<35s} {sign}{abs(w):.4f}  {bar}")

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  MinMaxScaler()),
        ("logreg",  LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

def evaluate(pipeline, X_test, y_test, name: str):
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)
    auc   = roc_auc_score(y_test, proba)
    print(f"\n  📈 [{name.upper()}] Test AUC: {auc:.4f}")
    print(classification_report(y_test, preds,
                                target_names=["No Risk", "Risk"],
                                zero_division=0))


# ---------------------------------------------------------------------------
# HEART loader
# ---------------------------------------------------------------------------

def load_heart(sleep_scores: pd.Series = None) -> tuple:
    df = pd.read_csv(CSV_FILES["heart"])
    df.columns = df.columns.str.strip()

    out = pd.DataFrame()
    out["age"]               = scale_age(pd.to_numeric(df["Age"], errors="coerce"))
    out["gender"]            = ordinal(df["Gender"], ORDINAL_MAPS["gender"])
    out["blood_pressure"]    = binary(df["High Blood Pressure"])
    ldl = binary(df["High LDL Cholesterol"])
    hdl = binary(df["Low HDL Cholesterol"])
    out["cholesterol"]       = np.clip((ldl + hdl) / 2.0, 0.0, 1.0)
    out["smoking"]           = binary(df["Smoking"])
    out["bmi"]               = scale_bmi(pd.to_numeric(df["BMI"], errors="coerce"))
    out["alcohol"]           = ordinal(df["Alcohol Consumption"],
                                      {"never": 0.0, "low": 0.2, "moderate": 0.5,
                                       "high": 1.0, "yes": 0.7, "no": 0.0}, default=0.3)
    out["sugar_consumption"] = ordinal(df["Sugar Consumption"], ORDINAL_MAPS["sugar_consumption"])
    out["stress_level"]      = ordinal(df["Stress Level"], ORDINAL_MAPS["stress_level"])
    out["chest_pain"]        = 1.0 - ordinal(df["Exercise Habits"], ORDINAL_MAPS["exercise_habits"])
    out["stroke"]            = 0.4
    out["anemia"]            = 0.4
    out["family_history_heart"] = binary(df["Family Heart Disease"])
    if sleep_scores is not None:
        out["sleep_score"]   = sleep_scores.values
    else:
        sh = pd.to_numeric(df["Sleep Hours"], errors="coerce")
        out["sleep_score"]   = np.clip((sh - 4.0) / 6.0, 0.0, 1.0)

    y = binary(df["Heart Disease Status"])
    return out[HEART_FEATURES], y


# ---------------------------------------------------------------------------
# CKD loader
# ---------------------------------------------------------------------------

def load_ckd(sleep_scores: pd.Series = None) -> tuple:
    df = pd.read_csv(CSV_FILES["ckd"])
    df.columns = df.columns.str.strip()

    out = pd.DataFrame()
    out["age"]               = scale_age(pd.to_numeric(df["Age"], errors="coerce"))
    out["gender"]            = ordinal(df["Gender"], ORDINAL_MAPS["gender"])
    sbp = pd.to_numeric(df["SystolicBP"], errors="coerce")
    out["blood_pressure"]    = np.clip((sbp - 80) / (200 - 80), 0.0, 1.0)
    out["smoking"]           = pd.to_numeric(df["Smoking"], errors="coerce").fillna(0).clip(0, 1)
    out["bmi"]               = scale_bmi(pd.to_numeric(df["BMI"], errors="coerce"))
    alc = pd.to_numeric(df["AlcoholConsumption"], errors="coerce")
    out["alcohol"]           = np.clip(alc / 20.0, 0.0, 1.0)
    fbs = pd.to_numeric(df["FastingBloodSugar"], errors="coerce")
    out["sugar_consumption"] = np.clip((fbs - 70) / 130.0, 0.0, 1.0)
    ethnicity_map = {
        "0": 0.25, "caucasian": 0.25, "white": 0.25,
        "1": 1.0,  "african american": 1.0, "black": 1.0,
        "2": 0.5,  "asian": 0.5,
        "3": 0.3,  "other": 0.3,
    }
    out["race"]              = ordinal(df["Ethnicity"], ethnicity_map, default=0.3)
    pa = pd.to_numeric(df["PhysicalActivity"], errors="coerce")
    out["physical_activity"] = np.clip(pa / 10.0, 0.0, 1.0)
    out["family_history_kidney"] = pd.to_numeric(
        df["FamilyHistoryKidneyDisease"], errors="coerce").fillna(0.4).clip(0, 1)
    fatigue = pd.to_numeric(df["FatigueLevels"], errors="coerce")
    out["stress_level"]      = np.clip(fatigue / 10.0, 0.0, 1.0)
    hemo = pd.to_numeric(df["HemoglobinLevels"], errors="coerce")
    out["anemia"]            = np.clip(1.0 - ((hemo - 8) / 9.0), 0.0, 1.0)
    out["diabetes_diagnosed"] = pd.to_numeric(
        df["FamilyHistoryDiabetes"], errors="coerce").fillna(0.4).clip(0, 1)
    if sleep_scores is not None:
        out["sleep_score"]   = sleep_scores.values
    else:
        sq = pd.to_numeric(df["SleepQuality"], errors="coerce")
        out["sleep_score"]   = np.clip((sq - 4.0) / 6.0, 0.0, 1.0)

    y = pd.to_numeric(df["Diagnosis"], errors="coerce").fillna(0).astype(int)
    return out[CKD_FEATURES], y


# ---------------------------------------------------------------------------
# LUNG loader
# ---------------------------------------------------------------------------

def load_lung(sleep_scores: pd.Series = None) -> tuple:
    df = pd.read_csv(CSV_FILES["lung"])
    df.columns = df.columns.str.strip()

    out = pd.DataFrame()
    out["age"]               = scale_age(pd.to_numeric(df["Age"], errors="coerce"))
    out["gender"]            = ordinal(df["Gender"], ORDINAL_MAPS["gender"])
    sm = pd.to_numeric(df["Smoking"], errors="coerce")
    out["smoking"]           = np.clip((sm - 1) / 7.0, 0.0, 1.0)
    ob = pd.to_numeric(df["Obesity"], errors="coerce")
    out["bmi"]               = np.clip((ob - 1) / 7.0, 0.0, 1.0)
    al = pd.to_numeric(df["Alcohol use"], errors="coerce")
    out["alcohol"]           = np.clip((al - 1) / 7.0, 0.0, 1.0)
    cp = pd.to_numeric(df["Chest Pain"], errors="coerce")
    out["chest_pain"]        = np.clip((cp - 1) / 8.0, 0.0, 1.0)
    gr = pd.to_numeric(df["Genetic Risk"], errors="coerce")
    out["genetic_risk_lung"] = np.clip((gr - 1) / 6.0, 0.0, 1.0)
    oc = pd.to_numeric(df["OccuPational Hazards"], errors="coerce")
    out["occupational_hazards"] = np.clip((oc - 1) / 7.0, 0.0, 1.0)
    if sleep_scores is not None:
        out["sleep_score"]   = sleep_scores.values
    else:
        fat = pd.to_numeric(df["Fatigue"], errors="coerce")
        out["sleep_score"]   = np.clip(1.0 - (fat - 1) / 8.0, 0.0, 1.0)

    y = df["Level"].str.strip().str.lower().apply(lambda x: 1 if x == "high" else 0)
    return out[LUNG_FEATURES], y


# ---------------------------------------------------------------------------
# DIABETES loader
# ---------------------------------------------------------------------------

def load_diabetes(sleep_scores: pd.Series = None) -> tuple:
    df = pd.read_csv(CSV_FILES["diabetes"])
    df.columns = df.columns.str.strip()

    out = pd.DataFrame()
    out["age"]               = scale_age(pd.to_numeric(df["age"], errors="coerce"))
    out["gender"]            = ordinal(df["gender"], ORDINAL_MAPS["gender"])
    bp = pd.to_numeric(df["blood_pressure"], errors="coerce")
    out["blood_pressure"]    = np.clip((bp - 60) / (180 - 60), 0.0, 1.0)
    out["smoking"]           = 0.4
    out["bmi"]               = scale_bmi(pd.to_numeric(df["bmi"], errors="coerce"))
    sug = pd.to_numeric(df["sugar_intake_grams_per_day"], errors="coerce")
    out["sugar_consumption"] = np.clip(sug / 150.0, 0.0, 1.0)
    stress = pd.to_numeric(df["stress_level"], errors="coerce")
    out["stress_level"]      = np.clip((stress - 1) / 9.0, 0.0, 1.0)
    out["family_history_diabetes"] = pd.to_numeric(
        df["family_history_diabetes"], errors="coerce").fillna(0.4).clip(0, 1)
    hba1c = pd.to_numeric(df["HbA1c_level"], errors="coerce")
    out["hba1c_high"]        = (hba1c >= 6.5).astype(float)
    if sleep_scores is not None:
        out["sleep_score"]   = sleep_scores.values
    else:
        sh = pd.to_numeric(df["sleep_hours"], errors="coerce")
        out["sleep_score"]   = np.clip((sh - 4.0) / 6.0, 0.0, 1.0)

    y = df["diabetes_risk_category"].str.strip().str.lower().apply(
        lambda x: 1 if "high" in x else 0)
    return out[DIABETES_FEATURES], y


# ---------------------------------------------------------------------------
# STROKE loader
# Confirmed columns:
#   id, gender, age, hypertension, heart_disease, ever_married,
#   work_type, Residence_type, avg_glucose_level, bmi,
#   smoking_status, stroke
# ---------------------------------------------------------------------------

def load_stroke(sleep_scores: pd.Series = None) -> tuple:
    df = pd.read_csv(CSV_FILES["stroke"])
    df.columns = df.columns.str.strip()

    out = pd.DataFrame()
    out["age"]                  = scale_age(pd.to_numeric(df["age"], errors="coerce"))
    out["gender"]               = ordinal(df["gender"], ORDINAL_MAPS["gender"])

    # hypertension — already binary 0/1
    out["blood_pressure"]       = pd.to_numeric(
        df["hypertension"], errors="coerce").fillna(0).clip(0, 1)

    # heart_disease — already binary 0/1
    out["heart_disease_history"]= pd.to_numeric(
        df["heart_disease"], errors="coerce").fillna(0).clip(0, 1)

    # smoking_status — categorical
    out["smoking"]              = ordinal(
        df["smoking_status"], ORDINAL_MAPS["smoking"], default=0.3)

    # bmi — numeric
    out["bmi"]                  = scale_bmi(
        pd.to_numeric(df["bmi"], errors="coerce"))

    # avg_glucose_level — proxy for sugar (normal ~70-140 mg/dL)
    gluc = pd.to_numeric(df["avg_glucose_level"], errors="coerce")
    out["sugar_consumption"]    = np.clip((gluc - 55) / (300 - 55), 0.0, 1.0)

    # stress_level — not in dataset; fill with population mean
    out["stress_level"]         = 0.4

    # irregular_heartbeat — not in dataset; fill with population mean
    out["irregular_heartbeat"]  = 0.4

    # sleep_score
    if sleep_scores is not None:
        out["sleep_score"]      = sleep_scores.values
    else:
        out["sleep_score"]      = 0.5

    # Target — stroke column (0/1)
    y = pd.to_numeric(df["stroke"], errors="coerce").fillna(0).astype(int)

    return out[STROKE_FEATURES], y


# ---------------------------------------------------------------------------
# ALZHEIMERS loader
# Confirmed columns:
#   PatientID, Age, Gender, Ethnicity, EducationLevel, BMI, Smoking,
#   AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality,
#   FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes,
#   Depression, HeadInjury, Hypertension, SystolicBP, DiastolicBP,
#   CholesterolTotal, CholesterolLDL, CholesterolHDL,
#   CholesterolTriglycerides, MMSE, FunctionalAssessment,
#   MemoryComplaints, BehavioralProblems, ADL, Confusion,
#   Disorientation, PersonalityChanges, DifficultyCompletingTasks,
#   Forgetfulness, Diagnosis, DoctorInCharge
# ---------------------------------------------------------------------------

def load_alzheimers(sleep_scores: pd.Series = None) -> tuple:
    df = pd.read_csv(CSV_FILES["alzheimers"])
    df.columns = df.columns.str.strip()

    out = pd.DataFrame()
    out["age"]                     = scale_age(pd.to_numeric(df["Age"], errors="coerce"))
    out["gender"]                  = ordinal(df["Gender"], ORDINAL_MAPS["gender"])
    out["bmi"]                     = scale_bmi(pd.to_numeric(df["BMI"], errors="coerce"))
    out["smoking"]                 = pd.to_numeric(
        df["Smoking"], errors="coerce").fillna(0).clip(0, 1)
    alc = pd.to_numeric(df["AlcoholConsumption"], errors="coerce")
    out["alcohol"]                 = np.clip(alc / 20.0, 0.0, 1.0)
    pa = pd.to_numeric(df["PhysicalActivity"], errors="coerce")
    out["physical_activity"]       = np.clip(pa / 10.0, 0.0, 1.0)

    # sleep_score
    if sleep_scores is not None:
        out["sleep_score"]         = sleep_scores.values
    else:
        sq = pd.to_numeric(df["SleepQuality"], errors="coerce")
        out["sleep_score"]         = np.clip((sq - 4.0) / 6.0, 0.0, 1.0)

    # Depression proxy for stress
    out["stress_level"]            = pd.to_numeric(
        df["Depression"], errors="coerce").fillna(0).clip(0, 1)

    out["family_history_alzheimers"] = pd.to_numeric(
        df["FamilyHistoryAlzheimers"], errors="coerce").fillna(0).clip(0, 1)

    # Mental activity — DietQuality as proxy (higher quality diet → more engaged lifestyle)
    dq = pd.to_numeric(df["DietQuality"], errors="coerce")
    out["mental_activity"]         = np.clip(dq / 10.0, 0.0, 1.0)

    # blood_pressure — Hypertension binary
    out["blood_pressure"]          = pd.to_numeric(
        df["Hypertension"], errors="coerce").fillna(0).clip(0, 1)

    # diabetes_diagnosed
    out["diabetes_diagnosed"]      = pd.to_numeric(
        df["Diabetes"], errors="coerce").fillna(0).clip(0, 1)

    # Target — Diagnosis (0/1)
    y = pd.to_numeric(df["Diagnosis"], errors="coerce").fillna(0).astype(int)

    return out[ALZHEIMERS_FEATURES], y


# ---------------------------------------------------------------------------
# Training orchestrator
# ---------------------------------------------------------------------------

LOADERS = {
    "heart":      (load_heart,      HEART_FEATURES),
    "ckd":        (load_ckd,        CKD_FEATURES),
    "lung":       (load_lung,       LUNG_FEATURES),
    "diabetes":   (load_diabetes,   DIABETES_FEATURES),
    "stroke":     (load_stroke,     STROKE_FEATURES),
    "alzheimers": (load_alzheimers, ALZHEIMERS_FEATURES),
}


def train_disease(name: str, sleep_scores: pd.Series = None):
    print(f"\n{'='*60}")
    print(f"  Training [{name.upper()}] model ...")
    print(f"{'='*60}")

    loader, feature_names = LOADERS[name]
    X, y = loader(sleep_scores=sleep_scores)

    print(f"  Dataset shape  : {X.shape}")
    print(f"  Positive class : {y.sum()} / {len(y)} ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=0.2,
        stratify=y.values,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    evaluate(pipeline, X_test, y_test, name)
    print_weights(pipeline.named_steps["logreg"], feature_names, name)
    save_model(pipeline, name)
    return pipeline


def train_all():
    from submodels import train_sleep_model
    print("\n" + "="*60)
    print("  [Step 1] Training Sleep Sub-model ...")
    print("="*60)
    train_sleep_model(save=True)

    print("\n  [Step 2] Training Disease Models ...")
    for name in LOADERS:
        train_disease(name)

    print("\n✅ All models trained and saved to sentinel/models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disease", type=str, default="all",
        choices=["all", "heart", "ckd", "lung", "diabetes",
                 "sleep", "stroke", "alzheimers"],
    )
    args = parser.parse_args()

    if args.disease == "all":
        train_all()
    elif args.disease == "sleep":
        from submodels import train_sleep_model
        train_sleep_model(save=True)
    else:
        train_disease(args.disease)
