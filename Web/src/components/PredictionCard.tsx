import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Gauge, Timer, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { formatConfidence, type ModelInfo } from "@/lib/hnrs";
import type { InferenceResult } from "@/lib/inference-shared";

interface PredictionCardProps {
  result: InferenceResult | null;
  model: ModelInfo;
  pending: boolean;
}

export function PredictionCard({ result, model, pending }: PredictionCardProps) {
  const chartData = (result?.probabilities ?? [])
    .slice()
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 8)
    .map((entry) => ({ label: entry.label, value: Number((entry.probability * 100).toFixed(2)) }));

  return (
    <div className="panel p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-wider text-primary">
            Inference output
          </p>
          <h2 className="mt-1 text-lg font-semibold">Prediction</h2>
        </div>
        <Badge variant="outline" className="font-mono text-[11px]">
          {model.key}
        </Badge>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
        <div className="grid-canvas flex min-h-32 flex-col items-center justify-center rounded-xl p-4 text-center">
          {pending ? (
            <span className="animate-pulse font-mono text-sm text-muted-foreground">
              running inference…
            </span>
          ) : result ? (
            <>
              <span className="font-mono text-3xl sm:text-4xl font-bold tracking-wider text-primary break-all px-2">
                {result.predictedText}
              </span>
              <span className="mt-2 font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                {model.classes}
              </span>
            </>
          ) : (
            <span className="font-mono text-sm text-muted-foreground">no prediction yet</span>
          )}
        </div>

        <div className="space-y-4">
          <div>
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2 text-muted-foreground">
                <Gauge className="size-4 text-primary" /> Confidence
              </span>
              <span className="font-mono">
                {result ? formatConfidence(result.confidence) : "—"}
              </span>
            </div>
            <Progress value={(result?.confidence ?? 0) * 100} className="mt-2" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg border border-border bg-secondary/40 p-3">
              <p className="flex items-center gap-2 text-[11px] uppercase tracking-wider text-muted-foreground">
                <Timer className="size-3.5" /> Latency
              </p>
              <p className="mt-1 font-mono text-lg">
                {result ? `${result.latencyMs} ms` : "—"}
              </p>
            </div>
            <div className="rounded-lg border border-border bg-secondary/40 p-3">
              <p className="flex items-center gap-2 text-[11px] uppercase tracking-wider text-muted-foreground">
                <Sparkles className="size-3.5" /> Framework
              </p>
              <p className="mt-1 text-sm">{model.framework}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-5">
        <p className="font-mono text-[11px] uppercase tracking-wider text-accent">
          Class probabilities
        </p>
        <div className="mt-2 h-48 w-full">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                <XAxis
                  dataKey="label"
                  stroke="var(--muted-foreground)"
                  tick={{ fontSize: 12, fontFamily: "var(--font-mono)" }}
                />
                <YAxis
                  stroke="var(--muted-foreground)"
                  tick={{ fontSize: 11 }}
                  unit="%"
                  domain={[0, 100]}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--popover)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    color: "var(--popover-foreground)",
                    fontSize: 12,
                  }}
                  formatter={(value: number) => [`${value}%`, "probability"]}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell
                      key={entry.label}
                      fill={index === 0 ? "var(--chart-1)" : "var(--chart-5)"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="grid h-full place-items-center rounded-xl border border-dashed border-border">
              <span className="font-mono text-xs text-muted-foreground">
                probability distribution appears after inference
              </span>
            </div>
          )}
        </div>
      </div>

      {result?.reasoning ? (
        <p className="mt-3 rounded-lg border border-border bg-secondary/30 p-3 text-xs text-muted-foreground">
          {result.reasoning}
        </p>
      ) : null}
    </div>
  );
}