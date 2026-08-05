import { ArrowRight } from "lucide-react";
import type { PreprocessResult } from "@/lib/hnrs";

const STEP_LABELS = [
  { key: "original", title: "1. Original", hint: "Raw input frame" },
  { key: "grayscale", title: "2. Grayscale", hint: "Luminance 0.299R + 0.587G + 0.114B" },
  { key: "binary", title: "3. Binarization", hint: "Otsu thresholding" },
  { key: "resized", title: "4. Resized 28×28", hint: "Centered normalized tensor" },
] as const;

export function PreprocessVisualizer({ result }: { result: PreprocessResult | null }) {
  return (
    <div className="space-y-5">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {STEP_LABELS.map((step, index) => (
          <div key={step.key} className="relative">
            <div className="panel overflow-hidden p-3">
              <p className="font-mono text-[11px] uppercase tracking-wider text-primary">
                {step.title}
              </p>
              <div className="grid-canvas mt-2 flex h-28 items-center justify-center overflow-hidden rounded-lg">
                {result ? (
                  <img
                    src={result[step.key]}
                    alt={step.title}
                    className="max-h-28 max-w-full object-contain"
                    style={{ imageRendering: step.key === "resized" ? "pixelated" : "auto" }}
                  />
                ) : (
                  <span className="font-mono text-[11px] text-muted-foreground">awaiting input</span>
                )}
              </div>
              <p className="mt-2 text-[11px] leading-snug text-muted-foreground">{step.hint}</p>
            </div>
            {index < STEP_LABELS.length - 1 && (
              <ArrowRight className="absolute -right-3 top-1/2 hidden size-4 -translate-y-1/2 text-primary/70 lg:block" />
            )}
          </div>
        ))}
      </div>

      <div className="panel p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="font-mono text-[11px] uppercase tracking-wider text-accent">
            Segmentation · character bounding boxes
          </p>
          <p className="font-mono text-[11px] text-muted-foreground">
            {result
              ? `${result.boxes.length} region(s) · threshold ${result.threshold} · ink ${(result.inkRatio * 100).toFixed(1)}%`
              : "—"}
          </p>
        </div>
        {result && result.segments.length > 0 ? (
          <div className="mt-3 flex flex-wrap gap-2">
            {result.segments.map((segment, index) => (
              <figure
                key={index}
                className="rounded-lg border border-border bg-secondary/40 p-1.5 text-center"
              >
                <img
                  src={segment}
                  alt={`Segment ${index + 1}`}
                  width={56}
                  height={56}
                  className="size-14 rounded"
                  style={{ imageRendering: "pixelated" }}
                />
                <figcaption className="mt-1 font-mono text-[10px] text-muted-foreground">
                  #{index + 1}
                </figcaption>
              </figure>
            ))}
          </div>
        ) : (
          <p className="mt-3 text-sm text-muted-foreground">
            Draw, upload, or capture an image to segment individual characters.
          </p>
        )}
      </div>
    </div>
  );
}