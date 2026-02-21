# =============================================================================
# sentinel/src/validate_columns.py
# Run this BEFORE train.py to verify your CSVs have the columns the
# loaders expect.  Prints a clear ✓ / ✗ report per file.
# =============================================================================

import pandas as pd
from config import CSV_FILES


EXPECTED = {
    "heart": [
        "HeartDisease", "BMI", "Smoking", "AlcoholDrinking", "Stroke",
        "PhysicalActivity", "Sex", "AgeCategory", "Race",
        "HighBP", "HighChol", "SleepTime",
    ],
    "ckd": [
        "age", "bmi", "smoking", "alcoholconsumption", "physicalactivity",
        "dietquality", "sleephours", "familyhistorykidneydisease",
        "hypertension", "diabetes", "anemia",
    ],
    "lung": [
        "Age", "Gender", "Smoking", "Obesity", "Alcohol use",
        "Chest Pain", "Genetic Risk", "OccuPational Hazards",
        "Fatigue", "Level",
    ],
    "diabetes": [
        "gender", "age", "hypertension", "heart_disease",
        "smoking_history", "bmi", "HbA1c_level",
        "blood_glucose_level", "diabetes",
    ],
    "sleep": [
        "Gender", "Age", "Sleep Duration", "Quality of Sleep",
        "Physical Activity Level", "Stress Level",
        "BMI Category", "Heart Rate",
    ],
}


def validate():
    print("\n🔍 Sentinel CSV Column Validator\n" + "="*50)
    all_ok = True

    for name, expected_cols in EXPECTED.items():
        path = CSV_FILES[name]
        print(f"\n[{name.upper()}]  {path}")
        try:
            df = pd.read_csv(path, nrows=2)
            actual_lower = [c.strip().lower() for c in df.columns]

            for col in expected_cols:
                found = col.strip().lower() in actual_lower
                status = "  ✓" if found else "  ✗ MISSING"
                if not found:
                    all_ok = False
                print(f"  {status}  {col}")

            extra = set(actual_lower) - {c.strip().lower() for c in expected_cols}
            if extra:
                print(f"  ℹ  Extra columns (unused): {sorted(extra)}")

        except FileNotFoundError:
            print(f"  ✗ FILE NOT FOUND — place your CSV at: {path}")
            all_ok = False

    print("\n" + "="*50)
    if all_ok:
        print("✅ All expected columns found. You are ready to run train.py.")
    else:
        print("❌ Some columns are missing. Review the loader in train.py or rename CSV columns.")


if __name__ == "__main__":
    validate()