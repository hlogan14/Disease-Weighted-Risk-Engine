// =============================================================================
// Sentinel Scoring Engine – TypeScript Port
// engine.ts: Normalisation, vector building, scoring, and top-factor extraction
// =============================================================================

import {
  ORDINAL_MAPS,
  BMI_MIN,
  BMI_MAX,
  THRESHOLDS,
  HEART_FEATURES,
  CKD_FEATURES,
  LUNG_FEATURES,
  DIABETES_FEATURES,
  STROKE_FEATURES,
  ALZHEIMERS_FEATURES,
  MODEL_COEFFICIENTS,
  SLEEP_MODEL,
  FEATURE_DISPLAY_NAMES,
} from "./config"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface UserInputs {
  age: number
  gender: string
  race: string
  height_in: number
  weight_lbs: number
  blood_pressure: string
  cholesterol: string
  stroke: string
  anemia: string
  chest_pain: string
  diabetes_diagnosed: string
  hba1c_high: string
  family_history_heart: string
  family_history_kidney: string
  family_history_diabetes: string
  family_history_alzheimers: string
  genetic_risk_lung: string
  occupational_hazards: string
  irregular_heartbeat: string
  smoking: string
  alcohol: number
  sugar_consumption: string
  physical_activity: number
  stress_level: string
  hours_of_sleep: number
  sleep_quality: number
  mental_activity: string
}

export interface DiseaseResult {
  probability: number
  risk_label: "High" | "Medium" | "Low"
  top_factors: Array<{ name: string; contribution: number }>
  sleep_score: number
}

export type ScoringResults = Record<string, DiseaseResult>

// ---------------------------------------------------------------------------
// Math Helpers
// ---------------------------------------------------------------------------
function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-x))
}

// ---------------------------------------------------------------------------
// Normaliser (matches Python normalise_input)
// ---------------------------------------------------------------------------
function normaliseOrdinal(value: string, mapping: Record<string, number>, defaultVal = 0.4): number {
  const key = String(value).toLowerCase().trim()
  return mapping[key] ?? defaultVal
}

function normaliseBool(value: string): number {
  const key = String(value).toLowerCase().trim()
  return ["yes", "1", "true"].includes(key) ? 1.0 : 0.0
}

function normaliseRange(value: number, lo: number, hi: number): number {
  return clamp((value - lo) / (hi - lo), 0.0, 1.0)
}

function normaliseAge(age: number): number {
  return clamp(age / 120.0, 0.0, 1.0)
}

function normaliseBmi(bmi: number): number {
  return clamp((bmi - BMI_MIN) / (BMI_MAX - BMI_MIN), 0.0, 1.0)
}

function computeBmi(heightIn: number, weightLbs: number): number {
  if (heightIn <= 0) return 25.0
  return (weightLbs / (heightIn ** 2)) * 703.0
}

// ---------------------------------------------------------------------------
// Sleep Sub-Model (Ridge Regression approximation)
// ---------------------------------------------------------------------------
function getSleepScore(
  sleepDuration: number,
  sleepQualityRating: number,
  stressLevel: string,
  physicalActivity: number,
  bmi: number,
  heartRate: number,
  age: number,
  gender: string
): number {
  const stressMap: Record<string, number> = { low: 2.0, medium: 5.0, high: 8.0 }
  const stressNum = stressMap[stressLevel.toLowerCase().trim()] ?? 5.0
  const activityMins = (physicalActivity / 7.0) * 60.0
  const genderNum = ORDINAL_MAPS.gender[gender.toLowerCase().trim()] ?? 0.5

  let bmiCat: number
  if (bmi < 18.5) bmiCat = 0.1
  else if (bmi < 25.0) bmiCat = 0.3
  else if (bmi < 30.0) bmiCat = 0.65
  else bmiCat = 1.0

  const rawFeatures = [
    sleepDuration,
    sleepQualityRating,
    stressNum,
    activityMins,
    bmiCat,
    heartRate,
    age,
    genderNum,
  ]

  // Apply MinMaxScaler (replicates the sklearn Pipeline normalisation step)
  const { coefs, intercept, featureMin, featureMax } = SLEEP_MODEL
  const scaledFeatures = rawFeatures.map((val, i) => {
    const range = featureMax[i] - featureMin[i]
    if (range === 0) return 0.0
    return clamp((val - featureMin[i]) / range, 0.0, 1.0)
  })

  let score = intercept
  for (let i = 0; i < scaledFeatures.length; i++) {
    score += coefs[i] * scaledFeatures[i]
  }

  return clamp(score, 0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Feature Vector Builders
// ---------------------------------------------------------------------------
function buildHeartVector(u: UserInputs, sleepScore: number): number[] {
  return [
    normaliseAge(u.age),
    normaliseOrdinal(u.gender, ORDINAL_MAPS.gender),
    normaliseOrdinal(u.blood_pressure, ORDINAL_MAPS.blood_pressure),
    normaliseOrdinal(u.cholesterol, ORDINAL_MAPS.cholesterol),
    normaliseOrdinal(u.smoking, ORDINAL_MAPS.smoking),
    normaliseBmi(computeBmi(u.height_in, u.weight_lbs)),
    normaliseRange(u.alcohol, 0, 20),
    normaliseOrdinal(u.sugar_consumption, ORDINAL_MAPS.sugar_consumption),
    normaliseOrdinal(u.stress_level, ORDINAL_MAPS.stress_level),
    normaliseOrdinal(u.chest_pain, ORDINAL_MAPS.chest_pain),
    normaliseOrdinal(u.stroke, ORDINAL_MAPS.yes_no_notsure),
    normaliseOrdinal(u.anemia, ORDINAL_MAPS.yes_no_notsure),
    normaliseOrdinal(u.family_history_heart, ORDINAL_MAPS.yes_no_notsure),
    sleepScore,
  ]
}

function buildCkdVector(u: UserInputs, sleepScore: number): number[] {
  return [
    normaliseAge(u.age),
    normaliseOrdinal(u.gender, ORDINAL_MAPS.gender),
    normaliseOrdinal(u.blood_pressure, ORDINAL_MAPS.blood_pressure),
    normaliseOrdinal(u.smoking, ORDINAL_MAPS.smoking),
    normaliseBmi(computeBmi(u.height_in, u.weight_lbs)),
    normaliseRange(u.alcohol, 0, 20),
    normaliseOrdinal(u.sugar_consumption, ORDINAL_MAPS.sugar_consumption),
    normaliseOrdinal(u.race, ORDINAL_MAPS.race),
    normaliseRange(u.physical_activity, 0, 10),
    normaliseOrdinal(u.family_history_kidney, ORDINAL_MAPS.yes_no_notsure),
    normaliseOrdinal(u.stress_level, ORDINAL_MAPS.stress_level),
    normaliseOrdinal(u.anemia, ORDINAL_MAPS.yes_no_notsure),
    normaliseOrdinal(u.diabetes_diagnosed, ORDINAL_MAPS.yes_no_notsure),
    sleepScore,
  ]
}

function buildLungVector(u: UserInputs, sleepScore: number): number[] {
  return [
    normaliseAge(u.age),
    normaliseOrdinal(u.gender, ORDINAL_MAPS.gender),
    normaliseOrdinal(u.smoking, ORDINAL_MAPS.smoking),
    normaliseBmi(computeBmi(u.height_in, u.weight_lbs)),
    normaliseRange(u.alcohol, 0, 20),
    normaliseOrdinal(u.chest_pain, ORDINAL_MAPS.chest_pain),
    normaliseOrdinal(u.genetic_risk_lung, ORDINAL_MAPS.yes_no_notsure),
    normaliseOrdinal(u.occupational_hazards, ORDINAL_MAPS.yes_no_notsure),
    sleepScore,
  ]
}

function buildDiabetesVector(u: UserInputs, sleepScore: number): number[] {
  return [
    normaliseAge(u.age),
    normaliseOrdinal(u.gender, ORDINAL_MAPS.gender),
    normaliseOrdinal(u.blood_pressure, ORDINAL_MAPS.blood_pressure),
    normaliseOrdinal(u.smoking, ORDINAL_MAPS.smoking),
    normaliseBmi(computeBmi(u.height_in, u.weight_lbs)),
    normaliseOrdinal(u.sugar_consumption, ORDINAL_MAPS.sugar_consumption),
    normaliseOrdinal(u.stress_level, ORDINAL_MAPS.stress_level),
    normaliseOrdinal(u.family_history_diabetes, ORDINAL_MAPS.yes_no_notsure),
    normaliseOrdinal(u.hba1c_high, ORDINAL_MAPS.yes_no_notsure),
    sleepScore,
  ]
}

function buildStrokeVector(u: UserInputs, sleepScore: number): number[] {
  return [
    normaliseAge(u.age),
    normaliseOrdinal(u.gender, ORDINAL_MAPS.gender),
    normaliseOrdinal(u.blood_pressure, ORDINAL_MAPS.blood_pressure),
    normaliseOrdinal(u.family_history_heart, ORDINAL_MAPS.yes_no_notsure),
    normaliseOrdinal(u.smoking, ORDINAL_MAPS.smoking),
    normaliseBmi(computeBmi(u.height_in, u.weight_lbs)),
    normaliseOrdinal(u.sugar_consumption, ORDINAL_MAPS.sugar_consumption),
    normaliseOrdinal(u.stress_level, ORDINAL_MAPS.stress_level),
    normaliseOrdinal(u.irregular_heartbeat, ORDINAL_MAPS.yes_no_notsure),
    sleepScore,
  ]
}

function buildAlzheimersVector(u: UserInputs, sleepScore: number): number[] {
  return [
    normaliseAge(u.age),
    normaliseOrdinal(u.gender, ORDINAL_MAPS.gender),
    normaliseBmi(computeBmi(u.height_in, u.weight_lbs)),
    normaliseOrdinal(u.smoking, ORDINAL_MAPS.smoking),
    normaliseRange(u.alcohol, 0, 20),
    normaliseRange(u.physical_activity, 0, 10),
    sleepScore,
    normaliseOrdinal(u.stress_level, ORDINAL_MAPS.stress_level),
    normaliseOrdinal(u.family_history_alzheimers, ORDINAL_MAPS.yes_no_notsure),
    normaliseOrdinal(u.mental_activity, ORDINAL_MAPS.mental_activity),
    normaliseOrdinal(u.blood_pressure, ORDINAL_MAPS.blood_pressure),
    normaliseOrdinal(u.diabetes_diagnosed, ORDINAL_MAPS.yes_no_notsure),
  ]
}

// ---------------------------------------------------------------------------
// Probability -> Risk Label
// ---------------------------------------------------------------------------
function mapToRiskLabel(probability: number): "High" | "Medium" | "Low" {
  if (probability >= THRESHOLDS.high) return "High"
  if (probability >= THRESHOLDS.medium) return "Medium"
  return "Low"
}

// ---------------------------------------------------------------------------
// Top Contributing Factors
// ---------------------------------------------------------------------------
function getTopFactors(
  featureNames: string[],
  featureValues: number[],
  coefs: number[],
  topN = 3
): Array<{ name: string; contribution: number }> {
  const contributions = featureNames.map((name, i) => ({
    name: FEATURE_DISPLAY_NAMES[name] || name,
    contribution: coefs[i] * featureValues[i],
  }))

  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
  return contributions.slice(0, topN)
}

// ---------------------------------------------------------------------------
// Logistic Regression prediction
// ---------------------------------------------------------------------------
function predictProbability(features: number[], coefs: number[], intercept: number): number {
  let logit = intercept
  for (let i = 0; i < features.length; i++) {
    logit += coefs[i] * features[i]
  }
  return sigmoid(logit)
}

// ---------------------------------------------------------------------------
// Main Scoring Function
// ---------------------------------------------------------------------------
export function scoreAllDiseases(userInputs: UserInputs): ScoringResults {
  const bmiRaw = computeBmi(userInputs.height_in, userInputs.weight_lbs)

  const sleepScore = getSleepScore(
    userInputs.hours_of_sleep,
    userInputs.sleep_quality,
    userInputs.stress_level,
    userInputs.physical_activity,
    bmiRaw,
    72.0,
    userInputs.age,
    userInputs.gender
  )

  const diseaseConfigs: Array<{
    name: string
    modelName: string
    features: string[]
    buildVector: (u: UserInputs, sleep: number) => number[]
  }> = [
    {
      name: "Heart Disease",
      modelName: "heart",
      features: HEART_FEATURES,
      buildVector: buildHeartVector,
    },
    {
      name: "Chronic Kidney Disease",
      modelName: "ckd",
      features: CKD_FEATURES,
      buildVector: buildCkdVector,
    },
    {
      name: "Lung Cancer",
      modelName: "lung",
      features: LUNG_FEATURES,
      buildVector: buildLungVector,
    },
    {
      name: "Diabetes Type 2",
      modelName: "diabetes",
      features: DIABETES_FEATURES,
      buildVector: buildDiabetesVector,
    },
    {
      name: "Stroke",
      modelName: "stroke",
      features: STROKE_FEATURES,
      buildVector: buildStrokeVector,
    },
    {
      name: "Alzheimer's Disease",
      modelName: "alzheimers",
      features: ALZHEIMERS_FEATURES,
      buildVector: buildAlzheimersVector,
    },
  ]

  const results: ScoringResults = {}

  for (const cfg of diseaseConfigs) {
    const model = MODEL_COEFFICIENTS[cfg.modelName]
    const vector = cfg.buildVector(userInputs, sleepScore)
    const probability = predictProbability(vector, model.coefs, model.intercept)
    const riskLabel = mapToRiskLabel(probability)
    const topFactors = getTopFactors(cfg.features, vector, model.coefs)

    results[cfg.name] = {
      probability: Math.round(probability * 10000) / 10000,
      risk_label: riskLabel,
      top_factors: topFactors.map((f) => ({
        name: f.name,
        contribution: Math.round(f.contribution * 10000) / 10000,
      })),
      sleep_score: Math.round(sleepScore * 10000) / 10000,
    }
  }

  return results
}
