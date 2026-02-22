# =============================================================================
# sentinel/src/train.py
# Trains all four disease models using the exact CSV columns confirmed
# from the uploaded Kaggle datasets.
#
# Usage:
#   python train.py                   → trains all models
#   python train.py --disease heart   → trains one model
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
    """Map Yes/1/True → 1.0, everything else → 0.0."""
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
# Confirmed columns:
#   Age, Gender, Blood Pressure, Cholesterol Level, Exercise Habits,
#   Smoking, Family Heart Disease, Diabetes, BMI, High Blood Pressure,
#   Low HDL Cholesterol, High LDL Cholesterol, Alcohol Consumption,
#   Stress Level, Sleep Hours, Sugar Consumption, Triglyceride Level,
#   Fasting Blood Sugar, CRP Level, Homocysteine Level, Heart Disease Status
# ---------------------------------------------------------------------------

def load_heart(sleep_scores: pd.Series = None) -> tuple:
    df = pd.read_csv(CSV_FILES["heart"])
    df.columns = df.columns.str.strip()

    out = pd.DataFrame()

    out["age"]              = scale_age(pd.to_numeric(df["Age"], errors="coerce"))
    out["gender"]           = ordinal(df["Gender"], ORDINAL_MAPS["gender"])
    # Use High Blood Pressure column (binary Yes/No) as primary BP signal
    out["blood_pressure"]   = binary(df["High Blood Pressure"])
    # Cholesterol — combine High LDL and Low HDL into a single risk signal
    ldl = binary(df["High LDL Cholesterol"])
    hdl = binary(df["Low HDL Cholesterol"])
    out["cholesterol"]      = np.clip((ldl + hdl) / 2.0, 0.0, 1.0)
    out["smoking"]          = binary(df["Smoking"])
    out["bmi"]              = scale_bmi(pd.to_numeric(df["BMI"], errors="coerce"))
    out["alcohol"]          = ordinal(df["Alcohol Consumption"],
                                     {"never": 0.0, "low": 0.2, "moderate": 0.5,
                                      "high": 1.0, "yes": 0.7, "no": 0.0},
                                     default=0.3)
    out["sugar_consumption"]= ordinal(df["Sugar Consumption"],
                                      ORDINAL_MAPS["sugar_consumption"])
    out["stress_level"]     = ordinal(df["Stress Level"],
                                      ORDINAL_MAPS["stress_level"])
    # Chest Pain — not directly in this dataset; use Exercise Habits as proxy
    # (low exercise → higher chest pain risk proxy)
    out["chest_pain"]       = 1.0 - ordinal(df["Exercise Habits"],
                                             ORDINAL_MAPS["exercise_habits"])
    # Stroke — not in dataset; fill with mean (0.4 = "not sure")
    out["stroke"]           = 0.4
    # Anemia — not in dataset; fill with mean
    out["anemia"]           = 0.4
    # Family History Heart
    out["family_history_heart"] = binary(df["Family Heart Disease"])

    # Sleep score — use Sleep Hours column as proxy during training
    if sleep_scores is not None:
        out["sleep_score"]  = sleep_scores.values
    else:
        sh = pd.to_numeric(df["Sleep Hours"], errors="coerce")
        out["sleep_score"]  = np.clip((sh - 4.0) / 6.0, 0.0, 1.0)

    # Target
    y = binary(df["Heart Disease Status"])

    out = out[HEART_FEATURES]
    return out, y


# ---------------------------------------------------------------------------
# CKD loader
# Confirmed columns:
#   PatientID, Age, Gender, Ethnicity, SocioeconomicStatus, EducationLevel,
#   BMI, Smoking, AlcoholConsumption, PhysicalActivity, DietQuality,
#   SleepQuality, FamilyHistoryKidneyDisease, FamilyHistoryHypertension,
#   FamilyHistoryDiabetes, PreviousAcuteKidneyInjury,
#   UrinaryTractInfections, SystolicBP, DiastolicBP, FastingBloodSugar,
#   HbA1c, SerumCreatinine, BUNLevels, GFR, ProteinInUrine, ACR,
#   SerumElectrolytesSodium, SerumElectrolytesPotassium,
#   SerumElectrolytesCalcium, SerumElectrolytesPhosphorus,
#   HemoglobinLevels, CholesterolTotal, CholesterolLDL, CholesterolHDL,
#   CholesterolTriglycerides, ACEInhibitors, Diuretics, NSAIDsUse,
#   Statins, AntidiabeticMedications, Edema, FatigueLevels,
#   NauseaVomiting, MuscleCramps, Itching, QualityOfLifeScore,
#   HeavyMetalsExposure, OccupationalExposureChemicals, WaterQuality,
#   MedicalCheckupsFrequency, MedicationAdherence, HealthLiteracy,
#   Diagnosis, DoctorInCharge
# ---------------------------------------------------------------------------

def load_ckd(sleep_scores: pd.Series = None) -> tuple:
    df = pd.read_csv(CSV_FILES["ckd"])
    df.columns = df.columns.str.strip()

    out = pd.DataFrame()

    out["age"]              = scale_age(pd.to_numeric(df["Age"], errors="coerce"))
    out["gender"]           = ordinal(df["Gender"], ORDINAL_MAPS["gender"])

    # Blood pressure — SystolicBP numeric (normal < 120; high >= 140)
    sbp = pd.to_numeric(df["SystolicBP"], errors="coerce")
    out["blood_pressure"]   = np.clip((sbp - 80) / (200 - 80), 0.0, 1.0)

    out["smoking"]          = pd.to_numeric(
                                  df["Smoking"], errors="coerce"
                              ).fillna(0).clip(0, 1)

    out["bmi"]              = scale_bmi(pd.to_numeric(df["BMI"], errors="coerce"))

    # Alcohol — numeric in this dataset (units/week assumed 0-20)
    alc = pd.to_numeric(df["AlcoholConsumption"], errors="coerce")
    out["alcohol"]          = np.clip(alc / 20.0, 0.0, 1.0)

    # Sugar consumption — FastingBloodSugar numeric (mg/dL typical 70-200)
    fbs = pd.to_numeric(df["FastingBloodSugar"], errors="coerce")
    out["sugar_consumption"]= np.clip((fbs - 70) / 130.0, 0.0, 1.0)

    # Race — Ethnicity column
    ethnicity_map = {
        "0": 0.25, "caucasian": 0.25, "white": 0.25,
        "1": 1.0,  "african american": 1.0, "black": 1.0,
        "2": 0.5,  "asian": 0.5,
        "3": 0.3,  "other": 0.3,
    }
    out["race"]             = ordinal(df["Ethnicity"], ethnicity_map, default=0.3)

    # Physical Activity — numeric (hrs/week assumed 0-10)
    pa = pd.to_numeric(df["PhysicalActivity"], errors="coerce")
    out["physical_activity"]= np.clip(pa / 10.0, 0.0, 1.0)

    out["family_history_kidney"] = pd.to_numeric(
        df["FamilyHistoryKidneyDisease"], errors="coerce"
    ).fillna(0.4).clip(0, 1)

    # Stress level — not directly in CKD dataset; use FatigueLevels as proxy
    fatigue = pd.to_numeric(df["FatigueLevels"], errors="coerce")
    out["stress_level"]     = np.clip(fatigue / 10.0, 0.0, 1.0)

    # Anemia — proxy via HemoglobinLevels (low hemoglobin → anemia)
    # Normal male ~13.5-17.5, female ~12-15.5 g/dL; low < 12 = anemic
    hemo = pd.to_numeric(df["HemoglobinLevels"], errors="coerce")
    out["anemia"]           = np.clip(1.0 - ((hemo - 8) / 9.0), 0.0, 1.0)

    # Diabetes diagnosed — FamilyHistoryDiabetes as the closest user-facing proxy
    out["diabetes_diagnosed"] = pd.to_numeric(
        df["FamilyHistoryDiabetes"], errors="coerce"
    ).fillna(0.4).clip(0, 1)

    # Sleep score
    if sleep_scores is not None:
        out["sleep_score"]  = sleep_scores.values
    else:
        sq = pd.to_numeric(df["SleepQuality"], errors="coerce")
        out["sleep_score"]  = np.clip((sq - 4.0) / 6.0, 0.0, 1.0)

    # Target — Diagnosis column (1 = CKD, 0 = no CKD)
    y = pd.to_numeric(df["Diagnosis"], errors="coerce").fillna(0).astype(int)

    out = out[CKD_FEATURES]
    return out, y


# ---------------------------------------------------------------------------
# LUNG loader
# Confirmed columns:
#   index, Patient Id, Age, Gender, Air Pollution, Alcohol use,
#   Dust Allergy, OccuPational Hazards, Genetic Risk,
#   chronic Lung Disease, Balanced Diet, Obesity, Smoking,
#   Passive Smoker, Chest Pain, Coughing of Blood, Fatigue,
#   Weight Loss, Shortness of Breath, Wheezing,
#   Swallowing Difficulty, Clubbing of Finger Nails,
#   Frequent Cold, Dry Cough, Snoring, Level
# ---------------------------------------------------------------------------

def load_lung(sleep_scores: pd.Series = None) -> tuple:
    df = pd.read_csv(CSV_FILES["lung"])
    df.columns = df.columns.str.strip()

    out = pd.DataFrame()

    out["age"]    = scale_age(pd.to_numeric(df["Age"], errors="coerce"))
    out["gender"] = ordinal(df["Gender"], ORDINAL_MAPS["gender"])

    # Smoking — 1-8 scale → normalise
    sm = pd.to_numeric(df["Smoking"], errors="coerce")
    out["smoking"]          = np.clip((sm - 1) / 7.0, 0.0, 1.0)

    # BMI proxy — Obesity column (1-8 scale)
    ob = pd.to_numeric(df["Obesity"], errors="coerce")
    out["bmi"]              = np.clip((ob - 1) / 7.0, 0.0, 1.0)

    # Alcohol (1-8 scale)
    al = pd.to_numeric(df["Alcohol use"], errors="coerce")
    out["alcohol"]          = np.clip((al - 1) / 7.0, 0.0, 1.0)

    # Chest Pain (1-9 scale)
    cp = pd.to_numeric(df["Chest Pain"], errors="coerce")
    out["chest_pain"]       = np.clip((cp - 1) / 8.0, 0.0, 1.0)

    # Genetic Risk (1-7 scale)
    gr = pd.to_numeric(df["Genetic Risk"], errors="coerce")
    out["genetic_risk_lung"]= np.clip((gr - 1) / 6.0, 0.0, 1.0)

    # Occupational Hazards (1-8 scale)
    oc = pd.to_numeric(df["OccuPational Hazards"], errors="coerce")
    out["occupational_hazards"] = np.clip((oc - 1) / 7.0, 0.0, 1.0)

    # Sleep score — Fatigue as inverse proxy (1-9 scale)
    if sleep_scores is not None:
        out["sleep_score"]  = sleep_scores.values
    else:
        fat = pd.to_numeric(df["Fatigue"], errors="coerce")
        out["sleep_score"]  = np.clip(1.0 - (fat - 1) / 8.0, 0.0, 1.0)

    # Target — binarise Level: High=1, Medium/Low=0
    y = df["Level"].str.strip().str.lower().apply(
        lambda x: 1 if x == "high" else 0
    )

    out = out[LUNG_FEATURES]
    return out, y


# ---------------------------------------------------------------------------
# DIABETES loader
# Confirmed columns:
#   Patient_ID, age, gender, bmi, blood_pressure, fasting_glucose_level,
#   insulin_level, HbA1c_level, cholesterol_level, triglycerides_level,
#   physical_activity_level, daily_calorie_intake, sugar_intake_grams_per_day,
#   sleep_hours, stress_level, family_history_diabetes,
#   waist_circumference_cm, diabetes_risk_score, diabetes_risk_category
# ---------------------------------------------------------------------------

def load_diabetes(sleep_scores: pd.Series = None) -> tuple:
    df = pd.read_csv(CSV_FILES["diabetes"])
    df.columns = df.columns.str.strip()

    out = pd.DataFrame()

    out["age"]    = scale_age(pd.to_numeric(df["age"], errors="coerce"))
    out["gender"] = ordinal(df["gender"], ORDINAL_MAPS["gender"])

    # Blood pressure — numeric column (mmHg; normal <120, high >=140)
    bp = pd.to_numeric(df["blood_pressure"], errors="coerce")
    out["blood_pressure"]   = np.clip((bp - 60) / (180 - 60), 0.0, 1.0)

    out["smoking"]          = 0.4   # not in dataset; fill with population mean

    out["bmi"]              = scale_bmi(pd.to_numeric(df["bmi"], errors="coerce"))

    # Sugar consumption — sugar_intake_grams_per_day (typical range 0-150g)
    sug = pd.to_numeric(df["sugar_intake_grams_per_day"], errors="coerce")
    out["sugar_consumption"]= np.clip(sug / 150.0, 0.0, 1.0)

    # Stress level — numeric 1-10 in this dataset
    stress = pd.to_numeric(df["stress_level"], errors="coerce")
    out["stress_level"]     = np.clip((stress - 1) / 9.0, 0.0, 1.0)

    # Family history diabetes — binary 0/1
    out["family_history_diabetes"] = pd.to_numeric(
        df["family_history_diabetes"], errors="coerce"
    ).fillna(0.4).clip(0, 1)

    # HbA1c high — threshold >=6.5 is prediabetic/diabetic
    hba1c = pd.to_numeric(df["HbA1c_level"], errors="coerce")
    out["hba1c_high"]       = (hba1c >= 6.5).astype(float)

    # Sleep score
    if sleep_scores is not None:
        out["sleep_score"]  = sleep_scores.values
    else:
        sh = pd.to_numeric(df["sleep_hours"], errors="coerce")
        out["sleep_score"]  = np.clip((sh - 4.0) / 6.0, 0.0, 1.0)

    # Target — binarise diabetes_risk_category
    # Typical values: "Low Risk", "Medium Risk", "High Risk"
    y = df["diabetes_risk_category"].str.strip().str.lower().apply(
        lambda x: 1 if "high" in x else 0
    )

    out = out[DIABETES_FEATURES]
    return out, y


# ---------------------------------------------------------------------------
# Training orchestrator
# ---------------------------------------------------------------------------

LOADERS = {
    "heart":    (load_heart,    HEART_FEATURES),
    "ckd":      (load_ckd,      CKD_FEATURES),
    "lung":     (load_lung,     LUNG_FEATURES),
    "diabetes": (load_diabetes, DIABETES_FEATURES),
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
        choices=["all", "heart", "ckd", "lung", "diabetes", "sleep"],
    )
    args = parser.parse_args()

    if args.disease == "all":
        train_all()
    elif args.disease == "sleep":
        from submodels import train_sleep_model
        train_sleep_model(save=True)
    else:
        train_disease(args.disease)
