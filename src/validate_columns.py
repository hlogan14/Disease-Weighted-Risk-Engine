# =============================================================================
# sentinel/src/validate_columns.py
# =============================================================================

import pandas as pd
from config import CSV_FILES

EXPECTED = {
    "heart": [
        "Age", "Gender", "Blood Pressure", "Cholesterol Level",
        "Exercise Habits", "Smoking", "Family Heart Disease",
        "BMI", "High Blood Pressure", "Low HDL Cholesterol",
        "High LDL Cholesterol", "Alcohol Consumption", "Stress Level",
        "Sleep Hours", "Sugar Consumption", "Heart Disease Status",
    ],
    "ckd": [
        "Age", "Gender", "Ethnicity", "BMI", "Smoking",
        "AlcoholConsumption", "PhysicalActivity", "SleepQuality",
        "FamilyHistoryKidneyDisease", "FamilyHistoryDiabetes",
        "SystolicBP", "FastingBloodSugar", "HbA1c", "HemoglobinLevels",
        "FatigueLevels", "Diagnosis",
    ],
    "lung": [
        "Age", "Gender", "Smoking", "Obesity", "Alcohol use",
        "Chest Pain", "Genetic Risk", "OccuPational Hazards",
        "Fatigue", "Level",
    ],
    "diabetes": [
        "age", "gender", "bmi", "blood_pressure",
        "fasting_glucose_level", "HbA1c_level",
        "sugar_intake_grams_per_day", "sleep_hours",
        "stress_level", "family_history_diabetes",
        "diabetes_risk_category",
    ],
    "sleep": [
        "Gender", "Age", "Sleep Duration", "Quality of Sleep",
        "Physical Activity Level", "Stress Level",
        "BMI Category", "Heart Rate",
    ],
    "stroke": [
        "gender", "age", "hypertension", "heart_disease",
        "avg_glucose_level", "bmi", "smoking_status", "stroke",
    ],
    "alzheimers": [
        "Age", "Gender", "BMI", "Smoking", "AlcoholConsumption",
        "PhysicalActivity", "SleepQuality", "FamilyHistoryAlzheimers",
        "Depression", "DietQuality", "Hypertension",
        "Diabetes", "Diagnosis",
    ],
}


def validate():
    print("\n🔍 Sentinel CSV Column Validator")
    print("=" * 50)
    all_ok = True

    for name, expected_cols in EXPECTED.items():
        path = CSV_FILES[name]
        print(f"\n[{name.upper()}]  {path}")
        try:
            df = pd.read_csv(path, nrows=2)
            actual       = list(df.columns)
            actual_lower = [c.strip().lower() for c in actual]

            for col in expected_cols:
                found  = col.strip().lower() in actual_lower
                status = "  ✓" if found else "  ✗ MISSING"
                if not found:
                    all_ok = False
                print(f"{status}  {col}")

            extra = set(actual_lower) - {c.strip().lower() for c in expected_cols}
            if extra:
                print(f"  ℹ  Unused columns (ignored): {sorted(extra)}")

        except FileNotFoundError:
            print(f"  ✗ FILE NOT FOUND — place CSV at: {path}")
            all_ok = False

    print("\n" + "=" * 50)
    if all_ok:
        print("✅ All columns found. Ready to run: python train.py")
    else:
        print("❌ Some columns are missing. Paste output back to Copilot.")


if __name__ == "__main__":
    validate()
