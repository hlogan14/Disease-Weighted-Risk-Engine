import Image from "next/image"
import { SentinelForm } from "@/components/sentinel-form"
import { Shield, Activity, Brain, Heart } from "lucide-react"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center gap-3">
          <Image
            src="/icons/shield.jpg"
            alt="Sentinel logo"
            width={32}
            height={32}
            className="rounded-md"
          />
          <div>
            <h1 className="text-base font-bold text-foreground leading-none">
              Sentinel
            </h1>
            <p className="text-xs text-muted-foreground">
              Chronic Disease Risk Screener
            </p>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="py-16 md:py-24 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-medium mb-6">
            <Activity className="w-3.5 h-3.5" />
            Data-Driven Risk Assessment
          </div>
          <h2 className="text-3xl md:text-4xl font-bold text-foreground leading-tight text-balance">
            Know Your Risk. Take Control.
          </h2>
          <p className="text-base text-muted-foreground mt-4 max-w-xl mx-auto leading-relaxed text-pretty">
            Sentinel estimates your exposure rating for six chronic diseases
            using a weighted risk model. Answer honestly and receive your
            personalized assessment in seconds.
          </p>

          {/* Feature pills */}
          <div className="flex flex-wrap items-center justify-center gap-3 mt-8">
            <FeaturePill icon={<Heart className="w-4 h-4" />} label="6 Disease Models" />
            <FeaturePill icon={<Shield className="w-4 h-4" />} label="Privacy First" />
            <FeaturePill icon={<Brain className="w-4 h-4" />} label="Evidence Based" />
          </div>
        </div>
      </section>

      {/* Main Form */}
      <main className="pb-24 px-4">
        <SentinelForm />
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-8 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <p className="text-xs text-muted-foreground leading-relaxed">
            Sentinel is a screening tool only and does not constitute medical
            advice, diagnosis, or treatment. All results are estimates based on
            self-reported data and statistical models. Please consult a
            qualified healthcare professional for any health concerns.
          </p>
          <p className="text-xs text-muted-foreground mt-3">
            Built with the Sentinel Weighted Risk Engine
          </p>
        </div>
      </footer>
    </div>
  )
}

function FeaturePill({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-card border border-border text-sm text-foreground">
      <span className="text-primary">{icon}</span>
      {label}
    </div>
  )
}
