# =============================================================================
# sentinel/src/train.py
# Weight Extraction Pipeline for all four disease models.
#
# For each disease:
#   1. Load + clean CSV
#   2. Align columns to user-facing features
#   3. Impute missing values (population mean for "Not Sure")
#   4. Scale features to [0, 1] with MinMaxScaler
#   5. Train LogisticRegression with class_weight='balanced'
#   6. Save Pipeline (scaler + model) as models/<disease>_model.pkl
#   7. Print extracted coefficients (weights)
#
# Usage:
#   python train.py              # trains all models
#   python train.py --disease heart   # trains one model
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer

from config import (
    CSV_FILES, MODELS_DIR, DATA_DIR,
    HEART_FEATURES, CKD_FEATURES, LUNG_FEATURES, DIABETES_FEATURES,
    DISEASE_TARGETS, ORDINAL_MAPS, BMI_MIN, BMI_MAX,
)

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_bmi(weight_lbs: pd.Series, height_in: pd.Series) -> pd.Series:
    """Return BMI from weight (lbs) and height (inches), scaled to [0,1]."""
    bmi = (weight_lbs / (height_in ** 2)) * 703.0
    return np.clip((bmi - BMI_MIN) / (BMI_MAX - BMI_MIN), 0.0, 1.0)


def scale_age(series: pd.Series, lo: float = 0, hi: float = 120) -> pd.Series:
    return np.clip((series - lo) / (hi - lo), 0.0, 1.0)


def ordinal_encode(series: pd.Series, mapping: dict, default: float = 0.4) -> pd.Series:
    return series.str.lower().map(mapping).fillna(default)


def save_model(pipeline: Pipeline, name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    joblib.dump(pipeline, path)
    print(f"  ✓ Saved → {path}")


def print_weights(model: LogisticRegression, feature_names: list, disease: str):
    """Print sorted feature weights extracted from the trained model."""
    coefs = model.coef_[0]
    pairs = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  📊 [{disease.upper()}] Feature Weights (sorted by |weight|):")
    for feat, w in pairs:
        bar = "█" * int(abs(w) * 20)
        sign = "+" if w >= 0 else "-"
        print(f"    {feat:<30s} {sign}{abs(w):.4f}  {bar}")


def build_pipeline() -> Pipeline:
    """Shared pipeline: MinMaxScaler → LogisticRegression."""
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


def evaluate(pipeline: Pipeline, X_test, y_test, name: str):
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)
    auc   = roc_auc_score(y_test, proba)
    print(f"\n  📈 [{name.upper()}] Test AUC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["No Risk", "Risk"]))


# ---------------------------------------------------------------------------
# Per-disease loaders
# Each loader returns (X: DataFrame, y: Series) with columns matching
# the DISEASE_FEATURES lists from config.py, BEFORE the sleep_score
# column is added (that column is injected at inference time).
# ---------------------------------------------------------------------------

def load_heart(sleep_scores: pd.Series = None) -> tuple:
    """
    Heart Disease (binary version).
    Expected CSV columns (binary/simplified version):
        HeartDisease, BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth,
        MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic,
        PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease,
        SkinCancer, HighBP, HighChol, ChestPain, Anemia
    """
    df = pd.read_csv(CSV_FILES["heart"])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    out = pd.DataFrame()

    # Age – AgeCategory is a string range like "55-59"; take midpoint
    if "agecategory" in df.columns:
        def _age_mid(s):
            try:
                parts = str(s).split("-")
                return (float(parts[0]) + float(parts[1])) / 2
            except Exception:
                return 40.0
        out["age"] = scale_age(df["agecategory"].apply(_age_mid))
    elif "age" in df.columns:
        out["age"] = scale_age(df["age"])

    # Gender
    if "sex" in df.columns:
        out["gender"] = ordinal_encode(df["sex"], ORDINAL_MAPS["gender"])

    # Blood Pressure
    bp_col = next((c for c in df.columns if "highbp" in c or "blood_pressure" in c), None)
    if bp_col:
        out["blood_pressure"] = df[bp_col].apply(
            lambda x: 1.0 if str(x).strip() in ["1", "Yes", "yes"] else 0.0
        )

    # Cholesterol
    chol_col = next((c for c in df.columns if "highchol" in c or "cholesterol" in c), None)
    if chol_col:
        out["cholesterol"] = df[chol_col].apply(
            lambda x: 1.0 if str(x).strip() in ["1", "Yes", "yes"] else 0.0
        )

    # Smoking
    out["smoking"] = df["smoking"].apply(
        lambda x: 1.0 if str(x).strip() in ["1", "Yes", "yes"] else 0.0
    ) if "smoking" in df.columns else pd.Series(np.nan, index=df.index)

    # BMI  (already computed in dataset)
    if "bmi" in df.columns:
        out["bmi"] = np.clip((df["bmi"] - BMI_MIN) / (BMI_MAX - BMI_MIN), 0.0, 1.0)

    # Alcohol
    alc_col = next((c for c in df.columns if "alcohol" in c), None)
    if alc_col:
        out["alcohol"] = df[alc_col].apply(
            lambda x: 1.0 if str(x).strip() in ["1", "Yes", "yes"] else 0.0
        )

    # Sugar consumption → proxy via diabetic or PhysicalHealth
    sugar_col = next((c for c in df.columns if "diabetic" in c or "sugar" in c), None)
    if sugar_col:
        out["sugar_consumption"] = df[sugar_col].apply(
            lambda x: 1.0 if str(x).strip() in ["1", "Yes", "yes"] else 0.0
        )

    # Race
    if "race" in df.columns:
        out["race"] = ordinal_encode(df["race"], ORDINAL_MAPS["race"], default=0.3)

    # Physical Activity
    if "physicalactivity" in df.columns:
        out["physical_activity"] = df["physicalactivity"].apply(
            lambda x: 1.0 if str(x).strip() in ["1", "Yes", "yes"] else 0.0
        )

    # Chest Pain
    cp_col = next((c for c in df.columns if "chest" in c), None)
    if cp_col:
        out["chest_pain"] = ordinal_encode(df[cp_col], ORDINAL_MAPS["chest_pain"])
    else:
        out["chest_pain"] = np.nan

    # Stroke
    if "stroke" in df.columns:
        out["stroke"] = df["stroke"].apply(
            lambda x: 1.0 if str(x).strip() in ["1", "Yes", "yes"] else 0.0
        )
    else:
        out["stroke"] = np.nan

    # Anemia
    anemia_col = next((c for c in df.columns if "anemia" in c or "anaemia" in c), None)
    if anemia_col:
        out["anemia"] = df[anemia_col].apply(
            lambda x: 1.0 if str(x).strip() in ["1", "Yes", "yes"] else 0.0
        )
    else:
        out["anemia"] = np.nan

    # Sleep score (from sub-model); use SleepTime as proxy during training
    if sleep_scores is not None:
        out["sleep_score"] = sleep_scores.values
    elif "sleeptime" in df.columns:
        out["sleep_score"] = np.clip((df["sleeptime"] - 4) / 6.0, 0.0, 1.0)
    else:
        out["sleep_score"] = np.nan

    # Target
    target_col = DISEASE_TARGETS["heart"]
    y = df[target_col].apply(
        lambda x: 1 if str(x).strip() in ["1", "Yes", "yes"] else 0
    )

    out = out[HEART_FEATURES]
    return out, y


def load_ckd(sleep_scores: pd.Series = None) -> tuple:
    """
    Chronic Kidney Disease (newer Kaggle version by rabieelkharoua).
    Expected CSV columns include named clinical + lifestyle columns.
    """
    df = pd.read_csv(CSV_FILES["ckd"])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    out = pd.DataFrame()

    out["age"] = scale_age(df["age"]) if "age" in df.columns else np.nan

    if "gender" in df.columns:
        out["gender"] = ordinal_encode(df["gender"], ORDINAL_MAPS["gender"])
    elif "sex" in df.columns:
        out["gender"] = ordinal_encode(df["sex"], ORDINAL_MAPS["gender"])

    # Blood pressure / hypertension
    bp_col = next((c for c in df.columns if "hypertension" in c or "bloodpressure" in c
                   or "blood_pressure" in c or "htn" in c), None)
    if bp_col:
        out["blood_pressure"] = df[bp_col].apply(
            lambda x: 1.0 if str(x).strip() in ["1", "yes", "Yes"] else 0.0
        )
    else:
        out["blood_pressure"] = np.nan

    # Smoking
    if "smoking" in df.columns:
        out["smoking"] = df["smoking"].apply(
            lambda x: 1.0 if str(x).strip() in ["1", "yes", "Yes"] else 0.0
        )
    else:
        out["smoking"] = np.nan

    # BMI
    if "bmi" in df.columns:
        out["bmi"] = np.clip((df["bmi"] - BMI_MIN) / (BMI_MAX - BMI_MIN), 0.0, 1.0)
    else:
        out["bmi"] = np.nan

    # Alcohol
    alc_col = next((c for c in df.columns if "alcohol" in c), None)
    out["alcohol"] = df[alc_col].apply(
        lambda x: 1.0 if str(x).strip() in ["1", "yes", "Yes"] else 0.0
    ) if alc_col else np.nan

    # Sugar consumption
    sugar_col = next((c for c in df.columns if "sugar" in c or "glucose" in c
                      or "su" == c), None)
    if sugar_col:
        out["sugar_consumption"] = ordinal_encode(
            df[sugar_col].astype(str), ORDINAL_MAPS["sugar_consumption"]
        )
    else:
        out["sugar_consumption"] = np.nan

    # Race
    if "race" in df.columns:
        out["race"] = ordinal_encode(df["race"], ORDINAL_MAPS["race"])
    else:
        out["race"] = np.nan

    # Physical Activity
    pa_col = next((c for c in df.columns if "physicalactivity" in c
                   or "physical_activity" in c), None)
    out["physical_activity"] = df[pa_col].apply(
        lambda x: 1.0 if str(x).strip() in ["1", "yes", "Yes"] else 0.0
    ) if pa_col else np.nan

    # Family history kidney
    fhk_col = next((c for c in df.columns if "familyhistory" in c
                    or "family_history" in c), None)
    out["family_history_kidney"] = df[fhk_col].apply(
        lambda x: 1.0 if str(x).strip() in ["1", "yes", "Yes"] else 0.0
    ) if fhk_col else np.nan

    # Stress level
    stress_col = next((c for c in df.columns if "stress" in c), None)
    if stress_col:
        out["stress_level"] = ordinal_encode(
            df[stress_col].astype(str), ORDINAL_MAPS["stress_level"]
        )
    else:
        out["stress_level"] = np.nan

    # Anemia
    anemia_col = next((c for c in df.columns if "anemia" in c or "anaemia" in c), None)
    out["anemia"] = df[anemia_col].apply(
        lambda x: 1.0 if str(x).strip() in ["1", "yes", "Yes"] else 0.0
    ) if anemia_col else np.nan

    # Diabetes diagnosed (new question)
    dm_col = next((c for c in df.columns if "diabetes" in c or "_dm" in c
                   or c == "dm"), None)
    out["diabetes_diagnosed"] = df[dm_col].apply(
        lambda x: 1.0 if str(x).strip() in ["1", "yes", "Yes"] else 0.0
    ) if dm_col else np.nan

    # Sleep score
    if sleep_scores is not None:
        out["sleep_score"] = sleep_scores.values
    else:
        sleep_col = next((c for c in df.columns if "sleep" in c), None)
        out["sleep_score"] = np.clip(
            (df[sleep_col] - 4) / 6.0, 0.0, 1.0
        ) if sleep_col else np.nan

    # Target
    target_col = DISEASE_TARGETS["ckd"]
    y = df[target_col].apply(
        lambda x: 1 if str(x).strip().lower() in ["1", "ckd", "yes"] else 0
    )

    out = out[CKD_FEATURES]
    return out, y


def load_lung(sleep_scores: pd.Series = None) -> tuple:
    """
    Lung Cancer (thedevastator – Cancer Patients and Air Pollution).
    Target: Level (Low/Medium/High) → binarised to High=1, else=0.
    """
    df = pd.read_csv(CSV_FILES["lung"])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    out = pd.DataFrame()

    out["age"] = scale_age(df["age"]) if "age" in df.columns else np.nan

    if "gender" in df.columns:
        out["gender"] = ordinal_encode(df["gender"], ORDINAL_MAPS["gender"])

    # Smoking (1-8 scale in this dataset → normalise)
    if "smoking" in df.columns:
        sm = pd.to_numeric(df["smoking"], errors="coerce")
        out["smoking"] = np.clip((sm - 1) / 7.0, 0.0, 1.0)
    else:
        out["smoking"] = np.nan

    # BMI proxy via Obesity column (1-8 scale)
    if "obesity" in df.columns:
        ob = pd.to_numeric(df["obesity"], errors="coerce")
        out["bmi"] = np.clip((ob - 1) / 7.0, 0.0, 1.0)
    else:
        out["bmi"] = np.nan

    # Alcohol use (1-8 scale)
    alc_col = next((c for c in df.columns if "alcohol" in c), None)
    if alc_col:
        al = pd.to_numeric(df[alc_col], errors="coerce")
        out["alcohol"] = np.clip((al - 1) / 7.0, 0.0, 1.0)
    else:
        out["alcohol"] = np.nan

    # Chest pain (1-9 scale)
    cp_col = next((c for c in df.columns if "chest" in c), None)
    if cp_col:
        cp = pd.to_numeric(df[cp_col], errors="coerce")
        out["chest_pain"] = np.clip((cp - 1) / 8.0, 0.0, 1.0)
    else:
        out["chest_pain"] = np.nan

    # Genetic Risk (1-7 scale → normalise)
    gr_col = next((c for c in df.columns if "genetic" in c), None)
    if gr_col:
        gr = pd.to_numeric(df[gr_col], errors="coerce")
        out["genetic_risk_lung"] = np.clip((gr - 1) / 6.0, 0.0, 1.0)
    else:
        out["genetic_risk_lung"] = np.nan

    # Occupational Hazards (1-8 scale)
    oc_col = next((c for c in df.columns if "occup" in c), None)
    if oc_col:
        oc = pd.to_numeric(df[oc_col], errors="coerce")
        out["occupational_hazards"] = np.clip((oc - 1) / 7.0, 0.0, 1.0)
    else:
        out["occupational_hazards"] = np.nan

    # Sleep score
    if sleep_scores is not None:
        out["sleep_score"] = sleep_scores.values
    else:
        # Fatigue as a proxy (1-9 scale; higher fatigue → lower sleep quality)
        fat_col = next((c for c in df.columns if "fatigue" in c), None)
        if fat_col:
            fat = pd.to_numeric(df[fat_col], errors="coerce")
            out["sleep_score"] = np.clip(1.0 - (fat - 1) / 8.0, 0.0, 1.0)
        else:
            out["sleep_score"] = np.nan

    # Target: binarise  High=1, Low/Medium=0
    target_col = "level"
    y = df[target_col].str.strip().str.lower().apply(
        lambda x: 1 if x == "high" else 0
    )

    out = out[LUNG_FEATURES]
    return out, y


def load_diabetes(sleep_scores: pd.Series = None) -> tuple:
    """
    Diabetes Risk Prediction (vishardmehta).
    Columns: gender, age, hypertension, heart_disease, smoking_history,
             bmi, HbA1c_level, blood_glucose_level, diabetes
    """
    df = pd.read_csv(CSV_FILES["diabetes"])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    out = pd.DataFrame()

    out["age"] = scale_age(df["age"]) if "age" in df.columns else np.nan

    if "gender" in df.columns:
        out["gender"] = ordinal_encode(df["gender"], ORDINAL_MAPS["gender"])

    # Blood pressure / hypertension
    bp_col = next((c for c in df.columns if "hypertension" in c
                   or "blood_pressure" in c), None)
    out["blood_pressure"] = df[bp_col].apply(
        lambda x: float(str(x).strip()) if str(x).strip() in ["0", "1"] else np.nan
    ) if bp_col else np.nan

    # Smoking history  → binary
    if "smoking_history" in df.columns:
        out["smoking"] = df["smoking_history"].str.lower().apply(
            lambda x: 1.0 if "current" in x or "ever" in x or "former" in x else 0.0
        )
    else:
        out["smoking"] = np.nan

    # BMI
    if "bmi" in df.columns:
        out["bmi"] = np.clip((df["bmi"] - BMI_MIN) / (BMI_MAX - BMI_MIN), 0.0, 1.0)

    # Sugar consumption proxy – HbA1c_level (typical range 3.5–9.0)
    hba1c_col = next((c for c in df.columns if "hba1c" in c), None)
    if hba1c_col:
        hba1c = pd.to_numeric(df[hba1c_col], errors="coerce")
        out["sugar_consumption"] = np.clip((hba1c - 3.5) / 5.5, 0.0, 1.0)
    else:
        out["sugar_consumption"] = np.nan

    # Race
    if "race" in df.columns:
        out["race"] = ordinal_encode(df["race"], ORDINAL_MAPS["race"])
    else:
        out["race"] = np.nan

    # hba1c_high binary flag (>= 6.5 → prediabetic/diabetic threshold)
    if hba1c_col:
        out["hba1c_high"] = (pd.to_numeric(df[hba1c_col], errors="coerce") >= 6.5
                             ).astype(float)
    else:
        out["hba1c_high"] = np.nan

    # Sleep score
    if sleep_scores is not None:
        out["sleep_score"] = sleep_scores.values
    else:
        out["sleep_score"] = np.nan

    # Target
    target_col = DISEASE_TARGETS["diabetes"]
    y = df[target_col].apply(lambda x: int(x))

    out = out[DIABETES_FEATURES]
    return out, y


# ---------------------------------------------------------------------------
# Main training orchestrator
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
        X.values, y.values, test_size=0.2, stratify=y.values, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    evaluate(pipeline, X_test, y_test, name)
    print_weights(pipeline.named_steps["logreg"], feature_names, name)
    save_model(pipeline, name)

    return pipeline


def train_all():
    from submodels import train_sleep_model
    print("\n[Step 1] Training Sleep Sub-model ...")
    train_sleep_model(save=True)

    print("\n[Step 2] Training Disease Models ...")
    for name in LOADERS:
        train_disease(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentinel disease models.")
    parser.add_argument("--disease", type=str, default="all",
                        choices=["all", "heart", "ckd", "lung", "diabetes", "sleep"])
    args = parser.parse_args()

    if args.disease == "all":
        train_all()
    elif args.disease == "sleep":
        from submodels import train_sleep_model
        train_sleep_model(save=True)
    else:
        train_disease(args.disease)