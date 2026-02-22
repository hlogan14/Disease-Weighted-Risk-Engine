"use client"

import { useState } from "react"
import Image from "next/image"
import { calculateRisk } from "@/app/actions"
import type { UserInputs, ScoringResults } from "@/lib/scoring/engine"
import { ResultsPanel } from "./results-panel"
import { Search, ChevronRight, ChevronLeft } from "lucide-react"

const FORM_SECTIONS = [
  { id: "demographics", label: "Demographics", icon: "/icons/shield.jpg" },
  { id: "medical", label: "Medical History", icon: "/icons/heart.jpg" },
  { id: "family", label: "Family & Genetic", icon: "/icons/alzheimers.jpg" },
  { id: "lifestyle", label: "Lifestyle", icon: "/icons/lungs.jpg" },
] as const

const DEFAULT_INPUTS: UserInputs = {
  age: 35,
  gender: "Male",
  race: "Caucasian",
  height_in: 68,
  weight_lbs: 160,
  blood_pressure: "No",
  cholesterol: "No",
  stroke: "No",
  anemia: "No",
  chest_pain: "No",
  diabetes_diagnosed: "No",
  hba1c_high: "No",
  family_history_heart: "No",
  family_history_kidney: "No",
  family_history_diabetes: "No",
  family_history_alzheimers: "No",
  genetic_risk_lung: "No",
  occupational_hazards: "No",
  irregular_heartbeat: "No",
  smoking: "No",
  alcohol: 2,
  sugar_consumption: "Low",
  physical_activity: 3,
  stress_level: "Low",
  hours_of_sleep: 7,
  sleep_quality: 6,
  mental_activity: "Sometimes",
}

export function SentinelForm() {
  const [inputs, setInputs] = useState<UserInputs>(DEFAULT_INPUTS)
  const [results, setResults] = useState<ScoringResults | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [currentSection, setCurrentSection] = useState(0)

  function updateField<K extends keyof UserInputs>(key: K, value: UserInputs[K]) {
    setInputs((prev) => ({ ...prev, [key]: value }))
  }

  async function handleSubmit() {
    setIsLoading(true)
    try {
      const result = await calculateRisk(inputs)
      setResults(result)
    } finally {
      setIsLoading(false)
    }
  }

  function handleBack() {
    setResults(null)
  }

  if (results) {
    return <ResultsPanel results={results} onBack={handleBack} />
  }

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Section Navigation */}
      <nav className="flex gap-2 mb-8 overflow-x-auto pb-2">
        {FORM_SECTIONS.map((section, idx) => (
          <button
            key={section.id}
            onClick={() => setCurrentSection(idx)}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium whitespace-nowrap transition-all ${
              currentSection === idx
                ? "bg-primary text-primary-foreground shadow-sm"
                : "bg-card text-muted-foreground hover:bg-accent border border-border"
            }`}
          >
            <Image
              src={section.icon}
              alt=""
              width={20}
              height={20}
              className="rounded-sm"
            />
            {section.label}
          </button>
        ))}
      </nav>

      {/* Form Content */}
      <div className="bg-card rounded-xl border border-border p-6 md:p-8 shadow-sm">
        {currentSection === 0 && (
          <DemographicsSection inputs={inputs} updateField={updateField} />
        )}
        {currentSection === 1 && (
          <MedicalHistorySection inputs={inputs} updateField={updateField} />
        )}
        {currentSection === 2 && (
          <FamilyGeneticSection inputs={inputs} updateField={updateField} />
        )}
        {currentSection === 3 && (
          <LifestyleSection inputs={inputs} updateField={updateField} />
        )}

        {/* Navigation Buttons */}
        <div className="flex items-center justify-between mt-8 pt-6 border-t border-border">
          <button
            onClick={() => setCurrentSection((prev) => Math.max(0, prev - 1))}
            disabled={currentSection === 0}
            className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-accent disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
            Previous
          </button>

          {currentSection < FORM_SECTIONS.length - 1 ? (
            <button
              onClick={() => setCurrentSection((prev) => prev + 1)}
              className="flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
            >
              Next
              <ChevronRight className="w-4 h-4" />
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={isLoading}
              className="flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-60 transition-opacity"
            >
              <Search className="w-4 h-4" />
              {isLoading ? "Calculating..." : "Calculate My Exposure Rating"}
            </button>
          )}
        </div>
      </div>

      {/* Step indicator */}
      <div className="flex items-center justify-center gap-2 mt-6">
        {FORM_SECTIONS.map((_, idx) => (
          <div
            key={idx}
            className={`h-1.5 rounded-full transition-all ${
              idx === currentSection
                ? "w-8 bg-primary"
                : idx < currentSection
                  ? "w-4 bg-primary/40"
                  : "w-4 bg-border"
            }`}
          />
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Form Section Components
// ---------------------------------------------------------------------------

interface SectionProps {
  inputs: UserInputs
  updateField: <K extends keyof UserInputs>(key: K, value: UserInputs[K]) => void
}

function FormField({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm font-medium text-foreground">{label}</label>
      {children}
    </div>
  )
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: string[]
  onChange: (v: string) => void
}) {
  return (
    <FormField label={label}>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2.5 rounded-lg border border-input bg-background text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring/30 transition-shadow"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </FormField>
  )
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
}) {
  return (
    <FormField label={label}>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full px-3 py-2.5 rounded-lg border border-input bg-background text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring/30 transition-shadow"
      />
    </FormField>
  )
}

function SliderField({
  label,
  value,
  min,
  max,
  onChange,
  labelLeft,
  labelRight,
}: {
  label: string
  value: number
  min: number
  max: number
  onChange: (v: number) => void
  labelLeft?: string
  labelRight?: string
}) {
  return (
    <FormField label={label}>
      <div className="flex flex-col gap-2">
        <input
          type="range"
          value={value}
          min={min}
          max={max}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full accent-primary"
        />
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>{labelLeft || min}</span>
          <span className="font-medium text-foreground text-sm">{value}</span>
          <span>{labelRight || max}</span>
        </div>
      </div>
    </FormField>
  )
}

function SectionTitle({ title, description }: { title: string; description: string }) {
  return (
    <div className="mb-6">
      <h3 className="text-lg font-semibold text-foreground">{title}</h3>
      <p className="text-sm text-muted-foreground mt-1">{description}</p>
    </div>
  )
}

function DemographicsSection({ inputs, updateField }: SectionProps) {
  return (
    <div>
      <SectionTitle
        title="Demographics"
        description="Basic information about you. This helps calibrate the risk model."
      />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <NumberField
          label="Age"
          value={inputs.age}
          min={19}
          max={80}
          onChange={(v) => updateField("age", v)}
        />
        <SelectField
          label="Gender"
          value={inputs.gender}
          options={["Male", "Female", "Other"]}
          onChange={(v) => updateField("gender", v)}
        />
        <SelectField
          label="Race / Ethnicity"
          value={inputs.race}
          options={["Caucasian", "African American", "Asian", "Other"]}
          onChange={(v) => updateField("race", v)}
        />
        <NumberField
          label="Height (inches)"
          value={inputs.height_in}
          min={48}
          max={96}
          onChange={(v) => updateField("height_in", v)}
        />
        <NumberField
          label="Weight (lbs)"
          value={inputs.weight_lbs}
          min={80}
          max={500}
          onChange={(v) => updateField("weight_lbs", v)}
        />
      </div>
    </div>
  )
}

function MedicalHistorySection({ inputs, updateField }: SectionProps) {
  const yesNoOptions = ["No", "Yes", "Not Sure"]
  return (
    <div>
      <SectionTitle
        title="Medical History"
        description="Information about your past and current medical conditions."
      />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <SelectField
          label="High blood pressure?"
          value={inputs.blood_pressure}
          options={yesNoOptions}
          onChange={(v) => updateField("blood_pressure", v)}
        />
        <SelectField
          label="High cholesterol?"
          value={inputs.cholesterol}
          options={yesNoOptions}
          onChange={(v) => updateField("cholesterol", v)}
        />
        <SelectField
          label="Ever had a stroke or TIA?"
          value={inputs.stroke}
          options={yesNoOptions}
          onChange={(v) => updateField("stroke", v)}
        />
        <SelectField
          label="Anemic (low iron/blood count)?"
          value={inputs.anemia}
          options={yesNoOptions}
          onChange={(v) => updateField("anemia", v)}
        />
        <SelectField
          label="Chest pain during physical activity?"
          value={inputs.chest_pain}
          options={["No", "Sometimes", "Yes"]}
          onChange={(v) => updateField("chest_pain", v)}
        />
        <SelectField
          label="Diagnosed with diabetes?"
          value={inputs.diabetes_diagnosed}
          options={yesNoOptions}
          onChange={(v) => updateField("diabetes_diagnosed", v)}
        />
        <SelectField
          label="Blood sugar or A1C high/borderline?"
          value={inputs.hba1c_high}
          options={yesNoOptions}
          onChange={(v) => updateField("hba1c_high", v)}
        />
        <SelectField
          label="Irregular heartbeat / A-Fib?"
          value={inputs.irregular_heartbeat}
          options={yesNoOptions}
          onChange={(v) => updateField("irregular_heartbeat", v)}
        />
      </div>
    </div>
  )
}

function FamilyGeneticSection({ inputs, updateField }: SectionProps) {
  const yesNoOptions = ["No", "Yes", "Not Sure"]
  return (
    <div>
      <SectionTitle
        title="Family & Genetic History"
        description="Conditions that run in your immediate family."
      />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <SelectField
          label="Family history of heart disease?"
          value={inputs.family_history_heart}
          options={yesNoOptions}
          onChange={(v) => updateField("family_history_heart", v)}
        />
        <SelectField
          label="Family history of kidney disease?"
          value={inputs.family_history_kidney}
          options={yesNoOptions}
          onChange={(v) => updateField("family_history_kidney", v)}
        />
        <SelectField
          label="Family history of diabetes?"
          value={inputs.family_history_diabetes}
          options={yesNoOptions}
          onChange={(v) => updateField("family_history_diabetes", v)}
        />
        <SelectField
          label="Family history of lung cancer?"
          value={inputs.genetic_risk_lung}
          options={yesNoOptions}
          onChange={(v) => updateField("genetic_risk_lung", v)}
        />
        <SelectField
          label="Family history of Alzheimer's / dementia?"
          value={inputs.family_history_alzheimers}
          options={yesNoOptions}
          onChange={(v) => updateField("family_history_alzheimers", v)}
        />
        <SelectField
          label="Occupational exposure to dust/chemicals/fumes?"
          value={inputs.occupational_hazards}
          options={yesNoOptions}
          onChange={(v) => updateField("occupational_hazards", v)}
        />
      </div>
    </div>
  )
}

function LifestyleSection({ inputs, updateField }: SectionProps) {
  return (
    <div>
      <SectionTitle
        title="Lifestyle"
        description="Your daily habits and routines that impact long-term health."
      />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <SelectField
          label="Do you currently smoke?"
          value={inputs.smoking}
          options={["No", "Yes"]}
          onChange={(v) => updateField("smoking", v)}
        />
        <NumberField
          label="Alcoholic drinks per week"
          value={inputs.alcohol}
          min={0}
          max={20}
          onChange={(v) => updateField("alcohol", v)}
        />
        <SelectField
          label="Daily sugar / processed food intake"
          value={inputs.sugar_consumption}
          options={["Low", "Medium", "High"]}
          onChange={(v) => updateField("sugar_consumption", v)}
        />
        <NumberField
          label="Hours of exercise per week"
          value={inputs.physical_activity}
          min={0}
          max={10}
          onChange={(v) => updateField("physical_activity", v)}
        />
        <SelectField
          label="Overall stress level"
          value={inputs.stress_level}
          options={["Low", "Medium", "High"]}
          onChange={(v) => updateField("stress_level", v)}
        />
        <NumberField
          label="Hours of sleep per night"
          value={inputs.hours_of_sleep}
          min={4}
          max={10}
          onChange={(v) => updateField("hours_of_sleep", v)}
        />
        <div className="md:col-span-2">
          <SliderField
            label="How well-rested do you feel most mornings?"
            value={inputs.sleep_quality}
            min={1}
            max={10}
            onChange={(v) => updateField("sleep_quality", v)}
            labelLeft="Exhausted"
            labelRight="Fully rested"
          />
        </div>
        <SelectField
          label="Mentally stimulating activities? (reading, puzzles, learning)"
          value={inputs.mental_activity}
          options={["Rarely", "Sometimes", "Often"]}
          onChange={(v) => updateField("mental_activity", v)}
        />
      </div>
    </div>
  )
}
