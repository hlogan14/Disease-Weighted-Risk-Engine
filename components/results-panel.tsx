"use client"

import { useState } from "react"
import Image from "next/image"
import type { ScoringResults, DiseaseResult } from "@/lib/scoring/engine"
import { DISEASE_DESCRIPTIONS, DISEASE_ICONS } from "@/lib/scoring/config"
import { ArrowLeft, ChevronDown, TrendingUp, TrendingDown, Moon, AlertTriangle, Info } from "lucide-react"

const RISK_ORDER: Record<string, number> = { High: 0, Medium: 1, Low: 2 }

function getRiskColor(label: string) {
  switch (label) {
    case "High":
      return { bg: "bg-risk-high/10", text: "text-risk-high", border: "border-risk-high/30", bar: "bg-risk-high" }
    case "Medium":
      return { bg: "bg-risk-medium/10", text: "text-risk-medium", border: "border-risk-medium/30", bar: "bg-risk-medium" }
    case "Low":
      return { bg: "bg-risk-low/10", text: "text-risk-low", border: "border-risk-low/30", bar: "bg-risk-low" }
    default:
      return { bg: "bg-muted", text: "text-muted-foreground", border: "border-border", bar: "bg-muted-foreground" }
  }
}

function getSleepLabel(score: number): { label: string; color: string } {
  if (score >= 0.6) return { label: "Good", color: "text-risk-low" }
  if (score >= 0.35) return { label: "Fair", color: "text-risk-medium" }
  return { label: "Poor", color: "text-risk-high" }
}

interface ResultsPanelProps {
  results: ScoringResults
  onBack: () => void
}

export function ResultsPanel({ results, onBack }: ResultsPanelProps) {
  const sortedResults = Object.entries(results).sort(
    ([, a], [, b]) =>
      (RISK_ORDER[a.risk_label] ?? 2) - (RISK_ORDER[b.risk_label] ?? 2) ||
      b.probability - a.probability
  )

  const highCount = sortedResults.filter(([, d]) => d.risk_label === "High").length
  const mediumCount = sortedResults.filter(([, d]) => d.risk_label === "Medium").length
  const lowCount = sortedResults.filter(([, d]) => d.risk_label === "Low").length

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to form
        </button>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold text-foreground">Your Exposure Ratings</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Results are sorted by risk level. Click any card to see detailed breakdown.
        </p>
      </div>

      {/* Summary Bar */}
      <div className="flex items-center gap-4 mb-8 p-4 rounded-xl bg-card border border-border">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-risk-high" />
          <span className="text-sm text-foreground">
            <span className="font-semibold">{highCount}</span> High
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-risk-medium" />
          <span className="text-sm text-foreground">
            <span className="font-semibold">{mediumCount}</span> Medium
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-risk-low" />
          <span className="text-sm text-foreground">
            <span className="font-semibold">{lowCount}</span> Low
          </span>
        </div>
      </div>

      {/* Disease Cards */}
      <div className="flex flex-col gap-3">
        {sortedResults.map(([diseaseName, data]) => (
          <DiseaseCard key={diseaseName} name={diseaseName} data={data} />
        ))}
      </div>

      {/* Disclaimer */}
      <div className="mt-8 p-4 rounded-xl bg-risk-medium/5 border border-risk-medium/20">
        <div className="flex gap-3">
          <AlertTriangle className="w-5 h-5 text-risk-medium shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-foreground">Disclaimer</p>
            <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
              Sentinel is a screening tool only and does not constitute medical advice,
              diagnosis, or treatment. All results are estimates based on self-reported
              data and statistical models. Please consult a qualified healthcare
              professional for any health concerns.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Individual Disease Card with Accordion
// ---------------------------------------------------------------------------

function DiseaseCard({ name, data }: { name: string; data: DiseaseResult }) {
  const [isOpen, setIsOpen] = useState(false)
  const colors = getRiskColor(data.risk_label)
  const sleepInfo = getSleepLabel(data.sleep_score)
  const icon = DISEASE_ICONS[name]
  const description = DISEASE_DESCRIPTIONS[name]

  return (
    <div className={`rounded-xl border ${colors.border} bg-card overflow-hidden transition-all`}>
      {/* Collapsed Summary */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-4 p-4 text-left hover:bg-accent/30 transition-colors"
      >
        {icon && (
          <Image
            src={icon}
            alt={name}
            width={40}
            height={40}
            className="rounded-lg shrink-0"
          />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-1.5">
            <h3 className="text-sm font-semibold text-foreground truncate">{name}</h3>
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${colors.bg} ${colors.text}`}
            >
              {data.risk_label} Risk
            </span>
          </div>
          {/* Progress bar */}
          <div className="flex items-center gap-3">
            <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
              <div
                className={`h-full rounded-full ${colors.bar} transition-all duration-500`}
                style={{ width: `${Math.min(data.probability * 100, 100)}%` }}
              />
            </div>
            <span className={`text-xs font-semibold tabular-nums ${colors.text}`}>
              {(data.probability * 100).toFixed(1)}%
            </span>
          </div>
        </div>
        <ChevronDown
          className={`w-5 h-5 text-muted-foreground shrink-0 transition-transform duration-200 ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>

      {/* Expanded Details */}
      {isOpen && (
        <div className="px-4 pb-4 border-t border-border">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            {/* Top Contributing Factors */}
            <div className="p-3 rounded-lg bg-muted/50">
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                Top Contributing Factors
              </h4>
              <div className="flex flex-col gap-2.5">
                {data.top_factors.map((factor, i) => (
                  <div key={i} className="flex items-center gap-2">
                    {factor.contribution > 0 ? (
                      <TrendingUp className="w-3.5 h-3.5 text-risk-high shrink-0" />
                    ) : (
                      <TrendingDown className="w-3.5 h-3.5 text-risk-low shrink-0" />
                    )}
                    <span className="text-sm text-foreground">{factor.name}</span>
                    <span className="ml-auto text-xs text-muted-foreground">
                      {factor.contribution > 0 ? "Increases" : "Decreases"} risk
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Sleep Score */}
            <div className="p-3 rounded-lg bg-muted/50">
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                Sleep Quality Score
              </h4>
              <div className="flex items-center gap-3">
                <Moon className="w-5 h-5 text-primary" />
                <div className="flex-1">
                  <div className="flex items-baseline gap-1.5">
                    <span className="text-2xl font-bold text-foreground">
                      {Math.round(data.sleep_score * 100)}
                    </span>
                    <span className="text-sm text-muted-foreground">/ 100</span>
                  </div>
                  <span className={`text-xs font-medium ${sleepInfo.color}`}>
                    {sleepInfo.label}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Description */}
          {description && (
            <div className="mt-4 p-3 rounded-lg bg-primary/5 border border-primary/10">
              <div className="flex gap-2">
                <Info className="w-4 h-4 text-primary shrink-0 mt-0.5" />
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {description}
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
