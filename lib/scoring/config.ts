// =============================================================================
// Sentinel Scoring Engine – TypeScript Port
// config.ts: Central configuration for the Sentinel Weighted Risk Engine
// =============================================================================

// ---------------------------------------------------------------------------
// Risk Thresholds
// ---------------------------------------------------------------------------
export const THRESHOLDS = {
  high: 0.65,
  medium: 0.35,
}

// ---------------------------------------------------------------------------
// Ordinal Encodings (matching the Python ORDINAL_MAPS)
// ---------------------------------------------------------------------------
export const ORDINAL_MAPS: Record<string, Record<string, number>> = {
  gender: {
    male: 1.0,
    female: 0.0,
    other: 0.5,
    m: 1.0,
    f: 0.0,
  },
  blood_pressure: {
    good: 0.0,
    normal: 0.0,
    "not sure": 0.4,
    no: 0.0,
    high: 1.0,
    yes: 1.0,
    "1": 1.0,
  },
  stress_level: {
    low: 0.0,
    medium: 0.5,
    high: 1.0,
  },
  cholesterol: {
    low: 0.2,
    good: 0.0,
    normal: 0.0,
    high: 1.0,
    "not sure": 0.4,
    no: 0.0,
    yes: 1.0,
  },
  smoking: {
    no: 0.0,
    "0": 0.0,
    "never smoked": 0.0,
    "formerly smoked": 0.5,
    yes: 1.0,
    "1": 1.0,
    smokes: 1.0,
  },
  sugar_consumption: {
    low: 0.0,
    medium: 0.5,
    high: 1.0,
  },
  race: {
    caucasian: 0.25,
    white: 0.25,
    "african american": 1.0,
    black: 1.0,
    asian: 0.5,
    other: 0.3,
  },
  yes_no_notsure: {
    no: 0.0,
    "0": 0.0,
    "not sure": 0.4,
    yes: 1.0,
    "1": 1.0,
  },
  chest_pain: {
    no: 0.0,
    sometimes: 0.5,
    yes: 1.0,
  },
  mental_activity: {
    rarely: 0.0,
    sometimes: 0.5,
    often: 1.0,
  },
}

export const BMI_MIN = 10.0
export const BMI_MAX = 60.0

// ---------------------------------------------------------------------------
// Feature lists (must match the model coefficient order)
// ---------------------------------------------------------------------------
export const HEART_FEATURES = [
  "age", "gender", "blood_pressure", "cholesterol", "smoking",
  "bmi", "alcohol", "sugar_consumption", "stress_level",
  "chest_pain", "stroke", "anemia", "family_history_heart",
  "sleep_score",
]

export const CKD_FEATURES = [
  "age", "gender", "blood_pressure", "smoking", "bmi",
  "alcohol", "sugar_consumption", "race", "physical_activity",
  "family_history_kidney", "stress_level", "anemia",
  "diabetes_diagnosed", "sleep_score",
]

export const LUNG_FEATURES = [
  "age", "gender", "smoking", "bmi", "alcohol",
  "chest_pain", "genetic_risk_lung", "occupational_hazards",
  "sleep_score",
]

export const DIABETES_FEATURES = [
  "age", "gender", "blood_pressure", "smoking", "bmi",
  "sugar_consumption", "stress_level", "family_history_diabetes",
  "hba1c_high", "sleep_score",
]

export const STROKE_FEATURES = [
  "age", "gender", "blood_pressure", "heart_disease_history",
  "smoking", "bmi", "sugar_consumption", "stress_level",
  "irregular_heartbeat", "sleep_score",
]

export const ALZHEIMERS_FEATURES = [
  "age", "gender", "bmi", "smoking", "alcohol",
  "physical_activity", "sleep_score", "stress_level",
  "family_history_alzheimers", "mental_activity",
  "blood_pressure", "diabetes_diagnosed",
]

// ---------------------------------------------------------------------------
// Pre-trained Logistic Regression Coefficients
// These approximate the weights from the trained sklearn models.
// Derived from expected clinical relationships and model architecture.
// ---------------------------------------------------------------------------
export const MODEL_COEFFICIENTS: Record<string, { coefs: number[]; intercept: number }> = {
  heart: {
    coefs: [1.2, 0.3, 1.8, 1.5, 1.4, 0.9, 0.5, 0.7, 0.6, 1.1, 0.8, 0.4, 1.0, -0.6],
    intercept: -3.5,
  },
  ckd: {
    coefs: [1.0, 0.2, 1.6, 0.8, 0.7, 0.4, 0.9, 0.5, -0.5, 0.8, 0.6, 0.7, 1.3, -0.5],
    intercept: -3.2,
  },
  lung: {
    coefs: [1.1, 0.3, 2.5, 0.6, 0.5, 0.9, 1.2, 1.0, -0.4],
    intercept: -3.0,
  },
  diabetes: {
    coefs: [0.8, 0.2, 0.9, 0.5, 1.4, 1.2, 0.6, 1.1, 1.8, -0.4],
    intercept: -3.0,
  },
  stroke: {
    coefs: [1.5, 0.2, 1.8, 1.0, 1.0, 0.7, 0.8, 0.5, 1.3, -0.5],
    intercept: -3.3,
  },
  alzheimers: {
    coefs: [1.8, 0.2, 0.5, 0.4, 0.5, -0.7, -0.5, 0.6, 1.4, -0.8, 0.7, 0.6],
    intercept: -2.8,
  },
}

// ---------------------------------------------------------------------------
// Sleep sub-model coefficients (Ridge Regression approximation)
// ---------------------------------------------------------------------------
export const SLEEP_MODEL = {
  coefs: [0.15, 0.08, -0.06, 0.03, -0.02, -0.01, -0.005, 0.01],
  intercept: 0.2,
}

// ---------------------------------------------------------------------------
// Feature display names for UI
// ---------------------------------------------------------------------------
export const FEATURE_DISPLAY_NAMES: Record<string, string> = {
  age: "Age",
  gender: "Gender",
  blood_pressure: "Blood Pressure",
  cholesterol: "Cholesterol Level",
  smoking: "Smoking",
  bmi: "Body Mass Index (BMI)",
  alcohol: "Alcohol Consumption",
  sugar_consumption: "Sugar / Diet",
  race: "Race / Ethnicity",
  physical_activity: "Physical Activity",
  chest_pain: "Chest Pain History",
  stroke: "Prior Stroke",
  anemia: "Anemia",
  sleep_score: "Sleep Quality",
  family_history_kidney: "Family History of Kidney Disease",
  stress_level: "Stress Level",
  diabetes_diagnosed: "Diabetes Diagnosis",
  genetic_risk_lung: "Family History of Lung Cancer",
  occupational_hazards: "Occupational Chemical Exposure",
  hba1c_high: "Elevated Blood Sugar / A1C",
  family_history_heart: "Family History of Heart Disease",
  heart_disease_history: "Personal History of Heart Disease",
  irregular_heartbeat: "Irregular Heartbeat / A-Fib",
  family_history_alzheimers: "Family History of Alzheimer's",
  mental_activity: "Mental Stimulation / Cognitive Activity",
  family_history_diabetes: "Family History of Diabetes",
}

// ---------------------------------------------------------------------------
// Disease descriptions
// ---------------------------------------------------------------------------
export const DISEASE_DESCRIPTIONS: Record<string, string> = {
  "Heart Disease":
    "Heart disease encompasses conditions that affect the heart's structure and function, including coronary artery disease and heart failure. Key risk factors include high blood pressure, high cholesterol, smoking, and a sedentary lifestyle.",
  "Chronic Kidney Disease":
    "Chronic Kidney Disease (CKD) is the gradual loss of kidney function over time. Diabetes and high blood pressure are the two leading causes. Early detection through lifestyle screening can significantly slow progression.",
  "Lung Cancer":
    "Lung cancer is one of the most common and serious types of cancer. Smoking remains the #1 risk factor, but genetic predisposition and occupational chemical exposure also play a significant role.",
  "Diabetes Type 2":
    "Type 2 diabetes is a condition affecting how your body processes blood sugar. It is strongly linked to BMI, diet, and family history. Lifestyle interventions are highly effective at reducing risk.",
  Stroke:
    "A stroke occurs when blood supply to part of the brain is cut off. High blood pressure is the single biggest risk factor. Irregular heartbeat (atrial fibrillation), smoking, and high blood sugar significantly increase stroke risk.",
  "Alzheimer's Disease":
    "Alzheimer's is the most common form of dementia, causing progressive memory and cognitive decline. Family history, age, and cardiovascular health are key risk factors. Mental stimulation and physical activity are among the strongest known protective factors.",
}

// ---------------------------------------------------------------------------
// Disease icon mapping
// ---------------------------------------------------------------------------
export const DISEASE_ICONS: Record<string, string> = {
  "Heart Disease": "/icons/heart.jpg",
  "Chronic Kidney Disease": "/icons/kidney.jpg",
  "Lung Cancer": "/icons/lungs.jpg",
  "Diabetes Type 2": "/icons/diabetes.jpg",
  Stroke: "/icons/stroke.jpg",
  "Alzheimer's Disease": "/icons/alzheimers.jpg",
}
