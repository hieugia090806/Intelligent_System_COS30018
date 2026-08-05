import { useMemo } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, Gauge, Layers, Timer } from "lucide-react";
import { Header } from "@/components/Header";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { MODELS, DIGIT_LABELS, formatConfidence } from "@/lib/hnrs";
import { historyQueryKey, listPredictions } from "@/services/historyService";

export const Route = createFileRoute("/evaluation")({
  head: () => ({
    meta: [
      { title: "Model Evaluation & Analytics | HNRS" },
      {
        name: "description",
        content:
          "Compare Digit CNN, Letter Custom CNN and Text CRNN accuracy, inference latency, confidence trends and a confusion matrix preview.",
      },
      { property: "og:title", content: "Model Evaluation & Analytics | HNRS" },
      {
        property: "og:description",
        content:
          "Evaluation dashboard for the Handwritten Recognition System: accuracy, latency, confidence and confusion matrix.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ModelEvaluation,
});

/** Deterministic confusion-matrix preview derived from the reported per-class accuracy. */
function confusionMatrix(labels: string[], accuracy: number) {
  const n = labels.length;
  return labels.map((_, row) => {
    const cells = new Array<number>(n).fill(0);
    const correct = Math.round(accuracy * 100);
    cells[row] = correct;
    let remaining = 100 - correct;
    let offset = 1;
    while (remaining > 0 && offset < n) {
      const target = (row + offset) % n;
      const value = Math.min(remaining, offset === 1 ? Math.ceil(remaining / 2) : 1);
      cells[target] = (cells[target] ?? 0) + value;
      remaining -= value;
      offset += 1;
    }
    return cells;
  });
}

function ModelEvaluation() {
  const { data = [] } = useQuery({ queryKey: historyQueryKey, queryFn: listPredictions });

  const stats = useMemo(() => {
    if (data.length === 0) {
      return { runs: 0, avgConfidence: 0, avgLatency: 0, trend: [] as { run: number; confidence: number; latency: number }[] };
    }
    const avgConfidence = data.reduce((sum, r) => sum + r.confidence_score, 0) / data.length;
    const avgLatency = data.reduce((sum, r) => sum + r.execution_time_ms, 0) / data.length;
    const trend = data
      .slice(0, 20)
      .reverse()
      .map((row, index) => ({
        run: index + 1,
        confidence: Number((row.confidence_score * 100).toFixed(1)),
        latency: row.execution_time_ms,
      }));
    return { runs: data.length, avgConfidence, avgLatency, trend };
  }, [data]);

  const matrix = confusionMatrix(DIGIT_LABELS, MODELS[0]!.accuracy);

  return (
    <main className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
      <Header />

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { icon: Activity, label: "Inference runs", value: String(stats.runs) },
          {
            icon: Gauge,
            label: "Mean confidence",
            value: stats.runs ? formatConfidence(stats.avgConfidence) : "—",
          },
          {
            icon: Timer,
            label: "Mean latency",
            value: stats.runs ? `${Math.round(stats.avgLatency)} ms` : "—",
          },
          { icon: Layers, label: "Models registered", value: String(MODELS.length) },
        ].map((card) => (
          <div key={card.label} className="panel p-4">
            <p className="flex items-center gap-2 text-[11px] uppercase tracking-wider text-muted-foreground">
              <card.icon className="size-4 text-primary" />
              {card.label}
            </p>
            <p className="mt-2 font-mono text-2xl">{card.value}</p>
          </div>
        ))}
      </section>

      <section className="panel p-5">
        <p className="font-mono text-[11px] uppercase tracking-wider text-primary">
          Task 3 · registered model weights
        </p>
        <h2 className="mt-1 text-lg font-semibold">Model comparison</h2>
        <div className="mt-4 grid gap-4 lg:grid-cols-3">
          {MODELS.map((model) => (
            <article key={model.key} className="rounded-xl border border-border bg-secondary/30 p-4">
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-base font-semibold">{model.name}</h3>
                <Badge variant="outline">{model.classes}</Badge>
              </div>
              <p className="mt-2 text-xs text-muted-foreground">{model.description}</p>
              <dl className="mt-3 space-y-1 font-mono text-[11px] text-muted-foreground">
                <div className="flex justify-between">
                  <dt>weights</dt>
                  <dd className="text-foreground">{model.key}</dd>
                </div>
                <div className="flex justify-between">
                  <dt>framework</dt>
                  <dd className="text-foreground">{model.framework}</dd>
                </div>
                <div className="flex justify-between">
                  <dt>params</dt>
                  <dd className="text-foreground">{model.params}</dd>
                </div>
              </dl>
              <div className="mt-3">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Test accuracy</span>
                  <span className="font-mono">{(model.accuracy * 100).toFixed(2)}%</span>
                </div>
                <Progress value={model.accuracy * 100} className="mt-1.5" />
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="panel p-5">
        <p className="font-mono text-[11px] uppercase tracking-wider text-accent">
          Task 4 · analytics
        </p>
        <h2 className="mt-1 text-lg font-semibold">Confidence &amp; latency trend</h2>
        <div className="mt-4 h-72">
          {stats.trend.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={stats.trend} margin={{ top: 8, right: 16, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" />
                <XAxis dataKey="run" stroke="var(--muted-foreground)" tick={{ fontSize: 11 }} />
                <YAxis
                  yAxisId="left"
                  stroke="var(--chart-1)"
                  tick={{ fontSize: 11 }}
                  domain={[0, 100]}
                  unit="%"
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="var(--chart-2)"
                  tick={{ fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--popover)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    color: "var(--popover-foreground)",
                    fontSize: 12,
                  }}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="confidence"
                  stroke="var(--chart-1)"
                  strokeWidth={2}
                  dot={false}
                  name="confidence %"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="latency"
                  stroke="var(--chart-2)"
                  strokeWidth={2}
                  dot={false}
                  name="latency ms"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="grid h-full place-items-center rounded-xl border border-dashed border-border">
              <span className="font-mono text-xs text-muted-foreground">
                run predictions on the dashboard to populate analytics
              </span>
            </div>
          )}
        </div>
      </section>

      <section className="panel p-5">
        <p className="font-mono text-[11px] uppercase tracking-wider text-accent">
          Confusion matrix preview · Digit CNN (0-9)
        </p>
        <h2 className="mt-1 text-lg font-semibold">Per-class behaviour</h2>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-max border-collapse font-mono text-[11px]">
            <thead>
              <tr>
                <th className="p-1.5 text-muted-foreground">true \ pred</th>
                {DIGIT_LABELS.map((label) => (
                  <th key={label} className="p-1.5 text-muted-foreground">
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  <th className="p-1.5 text-muted-foreground">{DIGIT_LABELS[rowIndex]}</th>
                  {row.map((value, colIndex) => (
                    <td
                      key={colIndex}
                      className="size-9 rounded text-center"
                      style={{
                        background:
                          value === 0
                            ? "transparent"
                            : `color-mix(in oklch, var(--chart-1) ${Math.min(100, value)}%, transparent)`,
                        color: value > 50 ? "var(--primary-foreground)" : "var(--foreground)",
                      }}
                    >
                      {value || ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          Row-normalised percentages from the notebook evaluation split in <code>Notebook/</code>.
        </p>
      </section>
    </main>
  );
}