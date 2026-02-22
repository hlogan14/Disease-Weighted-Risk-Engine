"use server"

import { scoreAllDiseases } from "@/lib/scoring/engine"
import type { UserInputs, ScoringResults } from "@/lib/scoring/engine"

export async function calculateRisk(inputs: UserInputs): Promise<ScoringResults> {
  return scoreAllDiseases(inputs)
}
